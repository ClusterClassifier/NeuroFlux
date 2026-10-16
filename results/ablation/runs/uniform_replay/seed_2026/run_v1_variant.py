import copy
import logging
import os
import pickle
import queue
import random
import shutil
import threading
from collections import deque
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
print('All imports loaded successfully.')

def setup_logger() -> logging.Logger:
    """Configures and returns a named console logger."""
    logger = logging.getLogger('CRN_Continual')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    return logger

def atomic_save(obj, filepath: str, is_torch: bool=True) -> None:
    """Writes to a .tmp file first, then performs an atomic rename to prevent corruption."""
    tmp_path = filepath + '.tmp'
    if is_torch:
        torch.save(obj, tmp_path)
    else:
        with open(tmp_path, 'wb') as f:
            pickle.dump(obj, f)
    shutil.move(tmp_path, filepath)

def save_checkpoint(task_id: int, policy_net: nn.Module, target_net: nn.Module, fisher_dict, theta_star_dict, episodic_memory, metrics_df: pd.DataFrame, save_dir: str='checkpoints') -> None:
    """
    Saves all critical training state at a task boundary.
    Components: policy/target nets, Fisher matrices, theta*, episodic memory, metrics CSV.
    """
    os.makedirs(save_dir, exist_ok=True)
    nets = {'policy': policy_net.state_dict(), 'target': target_net.state_dict()}
    atomic_save(nets, os.path.join(save_dir, f'task_{task_id}_policy.pt'), is_torch=True)
    if fisher_dict:
        atomic_save(fisher_dict, os.path.join(save_dir, f'task_{task_id}_fisher.pkl'), is_torch=True)
    if theta_star_dict:
        atomic_save(theta_star_dict, os.path.join(save_dir, f'task_{task_id}_theta_star.pkl'), is_torch=True)
    if episodic_memory and len(episodic_memory) > 0:
        atomic_save(episodic_memory.memory, os.path.join(save_dir, f'task_{task_id}_episodic.pkl'), is_torch=False)
    if metrics_df is not None and (not metrics_df.empty):
        tmp_csv = os.path.join(save_dir, 'metrics.csv.tmp')
        metrics_df.to_csv(tmp_csv, index=False)
        shutil.move(tmp_csv, os.path.join(save_dir, 'metrics.csv'))

def save_rollback_checkpoint(policy_net: nn.Module, filepath: str) -> None:
    """Saves a lightweight rollback snapshot used by ADWIN drift recovery."""
    atomic_save(policy_net.state_dict(), filepath, is_torch=True)

def load_rollback_checkpoint(policy_net: nn.Module, filepath: str) -> bool:
    """Loads an ADWIN rollback snapshot. Returns True on success, False if file is absent."""
    if os.path.exists(filepath):
        policy_net.load_state_dict(torch.load(filepath))
        return True
    return False
print('Utility functions defined successfully.')

def set_global_seed(seed: int=42, deterministic: bool=True) -> None:
    """
    Seeds Python, NumPy, and PyTorch. This does not change the algorithm;
    it makes experimental runs reproducible and enables proper multi-seed reporting.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def mean_std_ci95(values):
    """
    Returns mean, sample standard deviation, and a normal-approximation 95% CI.
    The CI is intended for evaluation episodes within a run. For the paper's
    headline statistics, aggregate across independent training seeds as well.
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    half = 1.96 * std / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return (mean, std, mean - half, mean + half)

def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum((p.numel() for p in model.parameters() if p.requires_grad)))
print('Reproducibility and statistical-reporting utilities defined.')

class CRNEnv:
    """
    Base Cognitive Radio Network environment.

    State:  Flattened SNR + occupancy history for all N channels → size (N × 2M,).
    Action: 0 … N-1  → transmit on channel k
            N        → idle (no transmission)
    Reward: +R_success on clear channel, -R_collision on occupied, -R_idle when idle.
    """

    def __init__(self, num_channels: int=8, history_length: int=5, max_steps: int=1000):
        self.num_channels = num_channels
        self.history_length = history_length
        self.max_steps = max_steps
        self.state_dim = num_channels * 2 * history_length
        self.action_space = num_channels + 1
        self.idle_action = num_channels
        self.R_success = 1.0
        self.R_collision = 1.0
        self.R_idle = 0.05
        self.pu_activity_probs = np.zeros(num_channels)
        self.snr_history = [deque(maxlen=history_length) for _ in range(num_channels)]
        self.occ_history = [deque(maxlen=history_length) for _ in range(num_channels)]
        self.current_step = 0

    def _set_probs(self, seed: int, base_low: float, base_high: float) -> None:
        """Generates reproducible, task-distinct PU activity probabilities."""
        rng = np.random.RandomState(seed)
        self.pu_activity_probs = rng.uniform(base_low, base_high, self.num_channels)
        optimal_ch = rng.randint(0, self.num_channels)
        self.pu_activity_probs[optimal_ch] = max(0.01, base_low * 0.5)

    def _generate_observation(self) -> None:
        """Advances all channels by one timestep and appends to history buffers."""
        for c in range(self.num_channels):
            is_busy = np.random.rand() < self.pu_activity_probs[c]
            if is_busy:
                snr = np.random.uniform(0.5, 1.0)
                occ = 1
            else:
                snr = np.random.uniform(0.0, 0.2)
                occ = 0
            self.snr_history[c].append(snr)
            self.occ_history[c].append(occ)

    def reset(self) -> np.ndarray:
        self.current_step = 0
        for c in range(self.num_channels):
            self.snr_history[c].clear()
            self.occ_history[c].clear()
        for _ in range(self.history_length):
            self._generate_observation()
        return self.get_state()

    def step(self, action: int):
        """
        Executes one environment step.
        Returns: (next_state, reward, done, info)
        """
        current_occ = [self.occ_history[c][-1] for c in range(self.num_channels)]
        if action == self.idle_action:
            reward = -self.R_idle
        elif 0 <= action < self.num_channels:
            reward = -self.R_collision if current_occ[action] == 1 else self.R_success
        else:
            raise ValueError(f'Invalid action: {action}')
        self._generate_observation()
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return (self.get_state(), reward, done, {})

    def get_state(self) -> np.ndarray:
        """Returns the flattened (N × 2M,) state vector."""
        state = []
        for c in range(self.num_channels):
            state.extend(self.snr_history[c])
            state.extend(self.occ_history[c])
        return np.array(state, dtype=np.float32)

class Task1Env(CRNEnv):
    """T1 | 06:00 | Early commute ramp-up       | PU load: Low–Medium (0.2–0.4)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_probs(seed=42, base_low=0.2, base_high=0.4)

class Task2Env(CRNEnv):
    """T2 | 09:00 | Peak commute traffic         | PU load: High (0.6–0.9)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_probs(seed=43, base_low=0.6, base_high=0.9)

class Task3Env(CRNEnv):
    """T3 | 12:00 | Midday office traffic        | PU load: Medium (0.4–0.6)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_probs(seed=44, base_low=0.4, base_high=0.6)

class Task4Env(CRNEnv):
    """T4 | 14:00 | Sparse IoT sensor bursts     | PU load: Low (0.05–0.2)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_probs(seed=45, base_low=0.05, base_high=0.2)

class Task5Env(CRNEnv):
    """T5 | 17:00 | Evening commute              | PU load: High (0.65–0.85)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_probs(seed=46, base_low=0.65, base_high=0.85)

class Task6Env(CRNEnv):
    """T6 | 20:00 | Heavy video streaming        | PU load: Very High (0.8–0.95)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_probs(seed=47, base_low=0.8, base_high=0.95)

class Task7Env(CRNEnv):
    """T7 | 23:00 | Night-time low activity      | PU load: Very Low (0.01–0.1)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_probs(seed=48, base_low=0.01, base_high=0.1)
print('Cognitive radio environments (T1–T7) defined successfully.')

class DQN(nn.Module):
    """
    3-layer feedforward Deep Q-Network.
    Architecture: input → 256 → 256 → output (Q-values per action).
    """

    def __init__(self, input_dim: int, output_dim: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """
    Manages the DQN policy/target network pair with Boltzmann (softmax) exploration.

    Temperature τ decays exponentially from τ₀ → τ_min over each task regime,
    and is reset at detected task boundaries.
    """

    def __init__(self, input_dim: int, output_dim: int, lr: float=0.001, gamma: float=0.99, device: str='cpu'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.device = device
        self.policy_net = DQN(input_dim, output_dim).to(device)
        self.target_net = DQN(input_dim, output_dim).to(device)
        self.update_target_network()
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.tau0 = 2.0
        self.tau_min = 0.1
        self.d = 1e-06
        self.tau = self.tau0
        self.task_step_counter = 0

    def update_target_network(self) -> None:
        """Hard-copies policy weights into the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def reset_temperature(self) -> None:
        """Resets Boltzmann temperature to τ₀ at the start of a new task."""
        self.tau = self.tau0
        self.task_step_counter = 0

    def get_action(self, state, evaluate: bool=False) -> int:
        """
        Selects an action.
        - evaluate=False → Boltzmann sampling (exploration).
        - evaluate=True  → greedy argmax (deterministic evaluation).
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state)
        if evaluate:
            return torch.argmax(q_values, dim=1).item()
        q_scaled = q_values / self.tau
        probs = F.softmax(q_scaled, dim=1).cpu().numpy()[0]
        if np.isnan(probs).any() or np.sum(probs) <= 0:
            probs = np.ones(self.output_dim, dtype=np.float64) / self.output_dim
        else:
            probs = probs.astype(np.float64)
            probs /= np.sum(probs)
        action = np.random.choice(self.output_dim, p=probs)
        self.task_step_counter += 1
        self.tau = max(self.tau_min, self.tau0 * np.exp(-self.d * self.task_step_counter))
        return action
print('DQN agent defined successfully.')

class ReplayBuffer:
    """Standard i.i.d. experience replay buffer for DQN."""

    def __init__(self, capacity: int=100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        """Adds a single SARS′d transition to the buffer."""
        self.buffer.append({'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done})

    def sample(self, batch_size: int):
        """Samples a random i.i.d. minibatch and returns PyTorch tensors."""
        return self._format_batch(random.sample(self.buffer, batch_size))

    def _format_batch(self, batch):
        states = torch.tensor(np.array([b['state'] for b in batch]), dtype=torch.float32)
        actions = torch.tensor([b['action'] for b in batch], dtype=torch.long)
        rewards = torch.tensor([b['reward'] for b in batch], dtype=torch.float32)
        next_states = torch.tensor(np.array([b['next_state'] for b in batch]), dtype=torch.float32)
        dones = torch.tensor([b['done'] for b in batch], dtype=torch.float32)
        return (states, actions, rewards, next_states, dones)

    def get_top_k_transitions(self, k: int, policy_net, target_net, gamma, device, batch_size: int=512):
        """
        Scores all buffered transitions by absolute TD-error and returns the top-k
        most informative ones. Used at task boundaries to populate episodic memory.
        """
        if len(self.buffer) == 0:
            return []
        policy_net.eval()
        buffer_list = list(self.buffer)
        td_errors = []
        for i in range(0, len(buffer_list), batch_size):
            batch = buffer_list[i:i + batch_size]
            s, a, r, ns, d = self._format_batch(batch)
            s, a, r, ns, d = (s.to(device), a.to(device), r.to(device), ns.to(device), d.to(device))
            with torch.no_grad():
                q_exp = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                q_next = target_net(ns).max(1)[0]
                q_target = r + gamma * q_next * (1 - d)
                td_errors.extend(torch.abs(q_exp - q_target).cpu().numpy())
        policy_net.train()
        scored = sorted(zip(buffer_list, td_errors), key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored[:k]]

    def flush(self) -> None:
        """Clears the buffer to prevent cross-task data contamination."""
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)

class EpisodicMemory:
    """
    Permanent continual-learning memory bank.
    Stores the highest-TD-error transitions from past tasks to augment the EWC replay mix.
    Evicts oldest 10% when capacity is exceeded (FIFO eviction preserves task diversity).
    """

    def __init__(self, capacity: int=2000):
        self.capacity = capacity
        self.memory = []

    def add_transitions(self, transitions) -> None:
        self.memory.extend(transitions)
        if len(self.memory) > self.capacity:
            evict_count = max(int(self.capacity * 0.1), len(self.memory) - self.capacity)
            self.memory = self.memory[evict_count:]

    def sample(self, batch_size: int):
        """Samples a random minibatch (or the full memory if smaller than batch_size)."""
        batch = self.memory if len(self.memory) < batch_size else random.sample(self.memory, batch_size)
        states = torch.tensor(np.array([b['state'] for b in batch]), dtype=torch.float32)
        actions = torch.tensor([b['action'] for b in batch], dtype=torch.long)
        rewards = torch.tensor([b['reward'] for b in batch], dtype=torch.float32)
        next_states = torch.tensor(np.array([b['next_state'] for b in batch]), dtype=torch.float32)
        dones = torch.tensor([b['done'] for b in batch], dtype=torch.float32)
        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        return len(self.memory)
print('Replay and episodic memory classes defined successfully.')

class EWC:
    """
    Elastic Weight Consolidation for continual reinforcement learning.

    Key contributions vs. vanilla EWC:
      1. Adaptive λ: penalty coefficient is modulated by the rolling TD-error variance,
         automatically balancing plasticity vs. stability per task.
      2. Async-compatible: FIM and θ* arrive from the MEC server after a delay;
         update_task_penalty() integrates them whenever they land.
    """

    def __init__(self, policy_net: nn.Module, lambda_max: float=4.0, beta: float=10.0, window_size: int=500):
        self.policy_net = policy_net
        self.lambda_max = lambda_max
        self.beta = beta
        self.past_tasks: list = []
        self.td_error_window = deque(maxlen=window_size)

    def update_task_penalty(self, fisher_dict: dict, theta_star_dict: dict) -> None:
        """
        Registers the FIM and optimal weights for a completed task.
        Tensors are moved to the policy network's active device.
        """
        device = next(self.policy_net.parameters()).device
        f_dict = {n: f.to(device) for n, f in fisher_dict.items()}
        t_dict = {n: t.to(device) for n, t in theta_star_dict.items()}
        self.past_tasks.append((f_dict, t_dict))

    def update_td_error(self, td_error: float) -> None:
        """Appends the latest batch-mean absolute TD-error to the rolling window."""
        self.td_error_window.append(float(abs(td_error)))

    def reset_td_window(self) -> None:
        """Clears the TD-error window at detected task boundaries."""
        self.td_error_window.clear()

    def get_adaptive_lambda(self) -> float:
        """
        Computes λ_t = λ_max / (1 + β · Var[δ]).
        Returns λ_max if fewer than 2 samples are available.
        """
        if len(self.td_error_window) < 2:
            return self.lambda_max
        variance = np.var(self.td_error_window)
        return self.lambda_max / (1.0 + self.beta * variance)

    def compute_ewc_loss(self) -> torch.Tensor:
        """
        EWC quadratic penalty: Σ_tasks (λ_t/2) · Σ_i F_i · (θ_i − θ*_i)².
        Returns a zero tensor if no past tasks have been registered yet.
        """
        device = next(self.policy_net.parameters()).device
        if not self.past_tasks:
            return torch.tensor(0.0, device=device)
        lambda_t = self.get_adaptive_lambda()
        ewc_loss = torch.tensor(0.0, device=device)
        for fisher_dict, theta_star_dict in self.past_tasks:
            for name, param in self.policy_net.named_parameters():
                if name in fisher_dict and name in theta_star_dict:
                    penalty = (fisher_dict[name] * (param - theta_star_dict[name]) ** 2).sum()
                    ewc_loss += penalty
        return lambda_t / 2.0 * ewc_loss
print('EWC module defined successfully.')

class ADWIN:
    """Pure Python ADWIN implementation (No external dependencies)."""
    _MAX_BUCKETS = 5

    class _Bucket:
        __slots__ = ('total', 'variance', 'size')

        def __init__(self):
            self.total = self.variance = 0.0
            self.size = 0

    def __init__(self, delta: float=0.002):
        self.delta = delta
        self._total = self._variance = 0.0
        self._width = 0
        self._bucket_rows = [[]]
        self.drift_detected = False

    @property
    def width(self):
        return self._width

    def update(self, value: float):
        self.drift_detected = False
        self._insert(float(value))
        self._check_drift()

    def _insert(self, value):
        b = self._Bucket()
        b.total = value
        b.size = 1
        self._bucket_rows[0].insert(0, b)
        self._width += 1
        old_mean = self._total / self._width if self._width > 1 else value
        self._variance += (value - old_mean) * (value - (self._total + value) / self._width)
        self._total += value
        self._compress()

    def _compress(self):
        level = 0
        while level < len(self._bucket_rows):
            row = self._bucket_rows[level]
            if len(row) <= self._MAX_BUCKETS:
                break
            if level + 1 >= len(self._bucket_rows):
                self._bucket_rows.append([])
            b1, b2 = (row[-2], row[-1])
            m = self._Bucket()
            m.size = b1.size + b2.size
            m.total = b1.total + b2.total
            d = b2.total / b2.size - b1.total / b1.size if b1.size > 0 and b2.size > 0 else 0.0
            m.variance = b1.variance + b2.variance + d ** 2 * b1.size * b2.size / max(m.size, 1)
            self._bucket_rows[level] = row[:-2]
            self._bucket_rows[level + 1].insert(0, m)
            level += 1

    def _check_drift(self):
        n0 = total0 = 0.0
        for row in reversed(self._bucket_rows):
            for bucket in reversed(row):
                n0 += bucket.size
                total0 += bucket.total
                n1 = self._width - n0
                if n1 <= 0 or n0 <= 0:
                    continue
                m0 = total0 / n0
                m1 = (self._total - total0) / n1
                dd = math.log(2.0 * math.log(self._width + 1) / self.delta) if self._width > 1 else 0.0
                eps = math.sqrt(dd / (2 * n0)) + math.sqrt(dd / (2 * n1))
                if abs(m0 - m1) >= eps:
                    self._shrink(int(n0))
                    self.drift_detected = True
                    return

    def _shrink(self, drop):
        removed = 0.0
        rem = drop
        for level in range(len(self._bucket_rows) - 1, -1, -1):
            row = self._bucket_rows[level]
            while row and rem > 0:
                b = row[-1]
                if b.size <= rem:
                    rem -= b.size
                    removed += b.total
                    row.pop()
                else:
                    f = rem / b.size
                    removed += b.total * f
                    b.total -= b.total * f
                    b.size -= rem
                    rem = 0
        self._width -= drop
        self._total -= removed
        if self._variance < 0:
            self._variance = 0.0

class ChangePointDetector:
    """
    ADWIN wrapper with publication-grade post-hoc event matching.

    The detector itself is unchanged. The important correction is that false
    positives and detection delay are computed by matching detections to the
    known task-transition steps after training, instead of treating ADWIN's
    internal window width as the detection delay.
    """

    def __init__(self, delta: float=0.05):
        self.delta = delta
        self.adwin = ADWIN(delta=delta)
        self.step_counter = 0
        self.detection_log = []

    def update(self, td_error: float, global_step: int=None):
        self.step_counter += 1
        self.adwin.update(float(td_error))
        if self.adwin.drift_detected:
            detected_at = int(global_step if global_step is not None else self.step_counter)
            estimated_boundary = max(0, detected_at - self.adwin.width)
            self.detection_log.append({'detected_at': detected_at, 'estimated_boundary': estimated_boundary, 'adwin_width': int(self.adwin.width)})
            return (True, estimated_boundary)
        return (False, None)

    def reset(self, reset_step_counter: bool=False) -> None:
        self.adwin = ADWIN(delta=self.delta)
        if reset_step_counter:
            self.step_counter = 0

    def get_detection_stats(self, true_boundaries=None, tolerance_steps: int=2000) -> dict:
        """
        Matches detections one-to-one to known task boundaries.

        A detection is a true positive if detected_at lies in
        [boundary, boundary + tolerance_steps]. Unmatched detections are false
        positives and unmatched boundaries are false negatives.
        """
        detections = sorted((int(e['detected_at']) for e in self.detection_log))
        boundaries = sorted((int(b) for b in true_boundaries or []))
        if not boundaries:
            return {'num_detections': len(detections), 'true_positive_count': np.nan, 'false_positive_count': np.nan, 'false_negative_count': np.nan, 'detection_rate': np.nan, 'false_positive_rate': np.nan, 'false_negative_rate': np.nan, 'mean_detection_delay': np.nan, 'std_detection_delay': np.nan}
        used = set()
        delays = []
        tp = 0
        for boundary in boundaries:
            candidates = [(idx, d) for idx, d in enumerate(detections) if idx not in used and boundary <= d <= boundary + tolerance_steps]
            if candidates:
                idx, det = min(candidates, key=lambda z: z[1])
                used.add(idx)
                tp += 1
                delays.append(det - boundary)
        fp = len(detections) - tp
        fn = len(boundaries) - tp
        return {'num_detections': len(detections), 'num_true_boundaries': len(boundaries), 'true_positive_count': tp, 'false_positive_count': fp, 'false_negative_count': fn, 'detection_rate': tp / len(boundaries) if boundaries else np.nan, 'false_positive_rate': fp / len(detections) if detections else 0.0, 'false_negative_rate': fn / len(boundaries) if boundaries else np.nan, 'mean_detection_delay': float(np.mean(delays)) if delays else np.nan, 'std_detection_delay': float(np.std(delays, ddof=1)) if len(delays) > 1 else 0.0 if len(delays) == 1 else np.nan, 'median_detection_delay': float(np.median(delays)) if delays else np.nan, 'tolerance_steps': int(tolerance_steps)}
print('Change-point detector defined successfully.')

class MECServer:
    """
    Simulates an asynchronous MEC server for Fisher Information Matrix computation.

    The MEC runs in a daemon thread. The edge radio submits tasks (frozen net + replay subset)
    and polls for results non-blockingly via check_results(). This models real MEC offloading
    latency without blocking the online training loop.
    """

    def __init__(self, device: str='cpu'):
        self.device = device
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self._stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)

    def start(self) -> None:
        """Starts the background MEC worker thread."""
        self.worker_thread.start()

    def stop(self) -> None:
        """Signals the worker to terminate and waits for clean shutdown."""
        self._stop_event.set()
        self.request_queue.put(None)
        self.worker_thread.join()

    def submit_task(self, replay_subset: list, frozen_policy_net: nn.Module) -> None:
        """
        Queues a FIM computation task.
        frozen_policy_net MUST be a cpu() deep-copy to avoid live-weight staleness.
        """
        self.request_queue.put((replay_subset, frozen_policy_net))

    def check_results(self):
        """
        Non-blocking poll for a completed FIM result.
        Returns (fisher_dict, theta_star_dict) if ready, otherwise (None, None).
        """
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return (None, None)

    def _worker_loop(self) -> None:
        """Core computation loop (diagonal empirical Fisher via per-sample backward passes)."""
        while not self._stop_event.is_set():
            task = self.request_queue.get()
            if task is None:
                continue
            replay_subset, frozen_net = task
            frozen_net = frozen_net.to(self.device)
            frozen_net.train()
            states = torch.tensor(np.array([t['state'] for t in replay_subset]), dtype=torch.float32).to(self.device)
            actions = torch.tensor([t['action'] for t in replay_subset], dtype=torch.long).to(self.device)
            fisher_dict = {n: torch.zeros_like(p) for n, p in frozen_net.named_parameters()}
            num_samples = len(states)
            for i in range(num_samples):
                frozen_net.zero_grad()
                q_vals = frozen_net(states[i:i + 1])
                q_a = q_vals.gather(1, actions[i:i + 1].unsqueeze(1)).squeeze(1)
                q_a.backward()
                for name, param in frozen_net.named_parameters():
                    if param.grad is not None:
                        fisher_dict[name] += param.grad.detach() ** 2
            for name in fisher_dict:
                fisher_dict[name] /= num_samples
            theta_star_dict = {n: p.detach().clone() for n, p in frozen_net.named_parameters()}
            self.result_queue.put((fisher_dict, theta_star_dict))
print('MEC server defined successfully.')

def evaluate_policy_on_env(agent, env, num_episodes=200):
    """
    Frozen greedy-policy evaluation on one CRN task.

    Returns episode-level reward uncertainty plus CRN coexistence metrics.
    No training state is changed.
    """
    episode_rewards = []
    total_steps = 0
    total_idle_actions = 0
    total_transmission_attempts = 0
    total_successful_transmissions = 0
    total_collisions = 0
    action_counts = np.zeros(env.action_space, dtype=np.int64)
    for _ in range(num_episodes):
        state = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.get_action(state, evaluate=True)
            next_state, reward, done, _ = env.step(action)
            total_steps += 1
            ep_reward += reward
            action_counts[action] += 1
            if action == env.idle_action:
                total_idle_actions += 1
            elif 0 <= action < env.num_channels:
                total_transmission_attempts += 1
                if reward == env.R_success:
                    total_successful_transmissions += 1
                elif reward == -env.R_collision:
                    total_collisions += 1
            state = next_state
        episode_rewards.append(ep_reward)
    reward_mean, reward_std, ci_low, ci_high = mean_std_ci95(episode_rewards)
    collision_rate = total_collisions / total_transmission_attempts if total_transmission_attempts > 0 else 0.0
    successful_transmission_rate = total_successful_transmissions / total_steps if total_steps > 0 else 0.0
    idle_rate = total_idle_actions / total_steps if total_steps > 0 else 0.0
    channel_utilization = total_transmission_attempts / total_steps if total_steps > 0 else 0.0
    transmission_success_rate = total_successful_transmissions / total_transmission_attempts if total_transmission_attempts > 0 else 0.0
    channel_actions = action_counts[:env.num_channels]
    ch_total = channel_actions.sum()
    if ch_total > 0:
        probs = channel_actions / ch_total
        probs = probs[probs > 0]
        channel_entropy = float(-np.sum(probs * np.log(probs)))
        normalized_channel_entropy = channel_entropy / np.log(env.num_channels) if env.num_channels > 1 else 0.0
    else:
        channel_entropy = 0.0
        normalized_channel_entropy = 0.0
    normalized_access_efficiency = transmission_success_rate
    return {'reward_mean': reward_mean, 'reward_std': reward_std, 'reward_ci95_low': ci_low, 'reward_ci95_high': ci_high, 'num_eval_episodes': int(num_episodes), 'total_steps': int(total_steps), 'total_transmission_attempts': int(total_transmission_attempts), 'total_successful_transmissions': int(total_successful_transmissions), 'total_collisions': int(total_collisions), 'collision_rate': float(collision_rate), 'successful_transmission_rate': float(successful_transmission_rate), 'idle_rate': float(idle_rate), 'channel_utilization': float(channel_utilization), 'transmission_success_rate': float(transmission_success_rate), 'throughput': float(successful_transmission_rate), 'pu_interference_rate': float(collision_rate), 'normalized_access_efficiency': float(normalized_access_efficiency), 'channel_entropy': float(channel_entropy), 'normalized_channel_entropy': float(normalized_channel_entropy)}

def run_evaluation(agent, environments, current_task_id, num_episodes=500):
    """
    Evaluates the frozen greedy policy on all tasks seen so far.

    Backward-compatible return signature:
      mean_rewards, collision_rates, transmission_metrics, extended_metrics
    """
    mean_rewards = []
    collision_rates = []
    transmission_metrics = []
    extended_metrics = []
    for task_id in range(current_task_id + 1):
        env = environments[task_id]
        m = evaluate_policy_on_env(agent, env, num_episodes=num_episodes)
        mean_rewards.append(m['reward_mean'])
        collision_rates.append(m['collision_rate'])
        transmission_metrics.append({'task': f'T{task_id + 1}', 'collision_rate': m['collision_rate'], 'successful_transmission_rate': m['successful_transmission_rate'], 'idle_rate': m['idle_rate'], 'channel_utilization': m['channel_utilization'], 'transmission_success_rate': m['transmission_success_rate'], 'reward_std': m['reward_std'], 'reward_ci95_low': m['reward_ci95_low'], 'reward_ci95_high': m['reward_ci95_high'], 'num_eval_episodes': m['num_eval_episodes'], 'total_steps': m['total_steps'], 'total_transmission_attempts': m['total_transmission_attempts'], 'total_successful_transmissions': m['total_successful_transmissions'], 'total_collisions': m['total_collisions']})
        extended_metrics.append({'task': f'T{task_id + 1}', 'throughput': m['throughput'], 'pu_interference_rate': m['pu_interference_rate'], 'normalized_access_efficiency': m['normalized_access_efficiency'], 'channel_entropy': m['channel_entropy'], 'normalized_channel_entropy': m['normalized_channel_entropy']})
    return (mean_rewards, collision_rates, transmission_metrics, extended_metrics)
print('Core evaluation runner defined.')

def calculate_continual_metrics(R_matrix: list, pre_task_scores: list=None, random_baseline_scores: list=None, single_task_oracle_scores: list=None) -> dict:
    """
    Publication-grade continual-learning metrics.

    R[k][j] = performance on task j after training through task k.

    Corrected definitions:
      Final Average Performance = mean_j R[K-1][j]
      Diagonal Learning Score   = mean_j R[j][j]
      Forgetting_j              = R[j][j] - R[K-1][j]
      BWT                       = mean_{j<K-1}(R[K-1][j] - R[j][j])
      FWT_j                     = pre_task_score[j] - random_baseline[j]

    FWT is reported as NaN when zero-shot pre-task evaluation was not measured;
    unevaluated future-task cells are never interpreted as zero transfer.
    Intransigence is reported only when a proper single-task oracle is supplied.
    """
    metrics = {}
    K = len(R_matrix)
    if K == 0:
        return metrics
    diag = [float(R_matrix[j][j]) for j in range(K)]
    final_row = [float(R_matrix[-1][j]) for j in range(K)]
    metrics['ACC_final'] = float(np.mean(final_row))
    metrics['ACC_diagonal'] = float(np.mean(diag))
    metrics['ACC'] = metrics['ACC_final']
    fr_abs = []
    fr_pct = []
    for j in range(K - 1):
        f = diag[j] - final_row[j]
        metrics[f'FR_T{j + 1}'] = float(f)
        denom = max(abs(diag[j]), 1e-12)
        fp = 100.0 * f / denom
        metrics[f'FRpct_T{j + 1}'] = float(fp)
        fr_abs.append(f)
        fr_pct.append(fp)
    metrics['mean_forgetting'] = float(np.mean(fr_abs)) if fr_abs else 0.0
    metrics['mean_forgetting_pct'] = float(np.mean(fr_pct)) if fr_pct else 0.0
    if K > 1:
        bwt_vals = [final_row[j] - diag[j] for j in range(K - 1)]
        metrics['BWT'] = float(np.mean(bwt_vals))
        bwt_pct = [100.0 * (final_row[j] - diag[j]) / max(abs(diag[j]), 1e-12) for j in range(K - 1)]
        metrics['BWT_pct'] = float(np.mean(bwt_pct))
    else:
        metrics['BWT'] = np.nan
        metrics['BWT_pct'] = np.nan
    fwt_values = []
    if pre_task_scores is not None and random_baseline_scores is not None:
        for j in range(1, min(K, len(pre_task_scores), len(random_baseline_scores))):
            pre = pre_task_scores[j]
            rnd = random_baseline_scores[j]
            if pre is not None and rnd is not None and np.isfinite(pre) and np.isfinite(rnd):
                fwt_j = float(pre - rnd)
                metrics[f'FWT_T{j + 1}'] = fwt_j
                fwt_values.append(fwt_j)
        metrics['FWT_mean'] = float(np.mean(fwt_values)) if fwt_values else np.nan
    else:
        metrics['FWT_mean'] = np.nan
    intransigence = []
    if single_task_oracle_scores is not None:
        for j in range(min(K, len(single_task_oracle_scores))):
            oracle = single_task_oracle_scores[j]
            if oracle is not None and np.isfinite(oracle):
                val = float(oracle - diag[j])
                metrics[f'Intransigence_T{j + 1}'] = val
                intransigence.append(val)
        metrics['Intransigence_mean'] = float(np.mean(intransigence)) if intransigence else np.nan
    else:
        metrics['Intransigence_mean'] = np.nan
    metrics['stability_score'] = float(1.0 - max(0.0, metrics['mean_forgetting_pct']) / 100.0)
    return metrics
print('Continual learning metric suite defined.')

def measure_adaptation_speed(agent, env, within_task_reference_episode_reward: float, threshold_fraction: float=0.95, max_steps: int=5000, eval_window: int=100) -> int:
    """
    Number of greedy-evaluation steps required to recover to a chosen fraction
    of the reference reward.

    FIX: the old implementation compared a rolling *per-step* reward (bounded
    near ±1) against ~95% of an *episode* reward (near 1000), which made recovery
    mathematically impossible. The reference is now normalized by env.max_steps.
    """
    reference_reward_per_step = within_task_reference_episode_reward / max(1, env.max_steps)
    target = threshold_fraction * reference_reward_per_step
    recent_rewards = deque(maxlen=eval_window)
    state = env.reset()
    for step in range(max_steps):
        action = agent.get_action(state, evaluate=True)
        next_state, reward, done, _ = env.step(action)
        recent_rewards.append(float(reward))
        state = next_state if not done else env.reset()
        if len(recent_rewards) == eval_window and np.mean(recent_rewards) >= target:
            return step + 1
    return max_steps
print('Adaptation speed metric defined.')

class RandomBaseline:
    """Uniform random action baseline (including idle)."""

    def __init__(self, action_space: int):
        self.action_space = action_space

    def get_action(self, state, evaluate: bool=False) -> int:
        return random.randint(0, self.action_space - 1)

class GreedyMyopicBaseline:
    """
    Chooses the channel with the lowest observed occupancy over the state history.
    This is a non-learning domain heuristic.
    """

    def __init__(self, num_channels: int, history_length: int):
        self.num_channels = num_channels
        self.history_length = history_length

    def get_action(self, state, evaluate: bool=False) -> int:
        occ_means = []
        for c in range(self.num_channels):
            offset = c * 2 * self.history_length + self.history_length
            occ_hist = state[offset:offset + self.history_length]
            occ_means.append(np.mean(occ_hist))
        return int(np.argmin(occ_means))

def evaluate_baseline(baseline, environments, num_episodes: int=200) -> dict:
    """
    Evaluates a non-learning baseline on all tasks with the exact same episode
    horizon and CRN metric definitions used for the proposed agent.
    """
    results = {'mean_rewards': [], 'reward_stds': [], 'reward_ci95_low': [], 'reward_ci95_high': [], 'collision_rates': [], 'throughputs': [], 'normalized_access_efficiency': []}
    for env in environments:
        ep_rewards = []
        total_steps = total_collisions = total_successes = total_attempts = 0
        for _ in range(num_episodes):
            state = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                action = baseline.get_action(state, evaluate=True)
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                total_steps += 1
                if 0 <= action < env.num_channels:
                    total_attempts += 1
                    if reward == env.R_success:
                        total_successes += 1
                    elif reward == -env.R_collision:
                        total_collisions += 1
                state = next_state
            ep_rewards.append(ep_reward)
        mean, std, lo, hi = mean_std_ci95(ep_rewards)
        results['mean_rewards'].append(mean)
        results['reward_stds'].append(std)
        results['reward_ci95_low'].append(lo)
        results['reward_ci95_high'].append(hi)
        results['collision_rates'].append(total_collisions / total_attempts if total_attempts else 0.0)
        results['throughputs'].append(total_successes / total_steps if total_steps else 0.0)
        results['normalized_access_efficiency'].append(total_successes / total_attempts if total_attempts else 0.0)
    return results
print('Baseline agents and statistically aligned evaluator defined.')

def generate_metric_report(R_matrix: list, cl_metrics: dict, transmission_metrics: list, extended_metrics: list, adwin_stats: dict, training_log: list, adaptation_steps: list=None, save_dir: str='checkpoints') -> pd.DataFrame:
    """
    Consolidates task-level and global metrics into publication-ready CSV output.
    """
    os.makedirs(save_dir, exist_ok=True)
    num_tasks = len(transmission_metrics)
    rows = []
    for t_idx in range(num_tasks):
        label = f'T{t_idx + 1}'
        tm = transmission_metrics[t_idx] if t_idx < len(transmission_metrics) else {}
        em = extended_metrics[t_idx] if t_idx < len(extended_metrics) else {}
        row = {'task': label, 'R_self': R_matrix[t_idx][t_idx] if t_idx < len(R_matrix) else np.nan, 'R_final': R_matrix[-1][t_idx] if R_matrix else np.nan, 'reward_std': tm.get('reward_std', np.nan), 'reward_ci95_low': tm.get('reward_ci95_low', np.nan), 'reward_ci95_high': tm.get('reward_ci95_high', np.nan), 'num_eval_episodes': tm.get('num_eval_episodes', np.nan), 'forgetting_abs': cl_metrics.get(f'FR_{label}', np.nan), 'forgetting_pct': cl_metrics.get(f'FRpct_{label}', np.nan), 'fwt': cl_metrics.get(f'FWT_{label}', np.nan), 'intransigence': cl_metrics.get(f'Intransigence_{label}', np.nan), 'collision_rate': tm.get('collision_rate', np.nan), 'successful_transmission_rate': tm.get('successful_transmission_rate', np.nan), 'idle_rate': tm.get('idle_rate', np.nan), 'channel_utilization': tm.get('channel_utilization', np.nan), 'transmission_success_rate': tm.get('transmission_success_rate', np.nan), 'throughput': em.get('throughput', np.nan), 'pu_interference_rate': em.get('pu_interference_rate', np.nan), 'normalized_access_efficiency': em.get('normalized_access_efficiency', np.nan), 'channel_entropy_nats': em.get('channel_entropy', np.nan), 'normalized_channel_entropy': em.get('normalized_channel_entropy', np.nan), 'total_steps': tm.get('total_steps', np.nan), 'total_transmission_attempts': tm.get('total_transmission_attempts', np.nan), 'total_successful_transmissions': tm.get('total_successful_transmissions', np.nan), 'total_collisions': tm.get('total_collisions', np.nan), 'adaptation_steps_95pct': adaptation_steps[t_idx] if adaptation_steps is not None and t_idx < len(adaptation_steps) else np.nan}
        rows.append(row)
    report_df = pd.DataFrame(rows)
    summary = {'task': 'SUMMARY', 'R_final': cl_metrics.get('ACC_final', np.nan), 'R_self': cl_metrics.get('ACC_diagonal', np.nan), 'forgetting_abs': cl_metrics.get('mean_forgetting', np.nan), 'forgetting_pct': cl_metrics.get('mean_forgetting_pct', np.nan), 'fwt': cl_metrics.get('FWT_mean', np.nan), 'intransigence': cl_metrics.get('Intransigence_mean', np.nan), 'bwt': cl_metrics.get('BWT', np.nan), 'bwt_pct': cl_metrics.get('BWT_pct', np.nan), 'stability_score': cl_metrics.get('stability_score', np.nan), 'adwin_detection_rate': adwin_stats.get('detection_rate', np.nan), 'adwin_false_positive_rate': adwin_stats.get('false_positive_rate', np.nan), 'adwin_false_negative_rate': adwin_stats.get('false_negative_rate', np.nan), 'adwin_mean_delay': adwin_stats.get('mean_detection_delay', np.nan), 'adwin_std_delay': adwin_stats.get('std_detection_delay', np.nan), 'adwin_median_delay': adwin_stats.get('median_detection_delay', np.nan), 'adwin_tp': adwin_stats.get('true_positive_count', np.nan), 'adwin_fp': adwin_stats.get('false_positive_count', np.nan), 'adwin_fn': adwin_stats.get('false_negative_count', np.nan)}
    if training_log:
        log_df = pd.DataFrame(training_log)
        for key in ['dqn_loss', 'ewc_loss', 'total_loss', 'lambda_t', 'td_error']:
            if key in log_df:
                summary[f'mean_{key}'] = float(log_df[key].mean())
                summary[f'std_{key}'] = float(log_df[key].std(ddof=1))
    report_df = pd.concat([report_df, pd.DataFrame([summary])], ignore_index=True)
    csv_path = os.path.join(save_dir, 'ieee_full_metric_report.csv')
    report_df.to_csv(csv_path, index=False)
    notes = pd.DataFrame([['throughput', 'successful transmissions / all environment steps', 'normalized per-step rate'], ['collision_rate', 'collisions / transmission attempts', 'also PU interference rate in this single-SU simulator'], ['normalized_access_efficiency', 'successful transmissions / transmission attempts', 'dimensionless proxy; NOT physical bit/s/Hz spectral efficiency'], ['channel_entropy_nats', 'Shannon entropy of channel-action distribution', 'nats'], ['normalized_channel_entropy', 'entropy / ln(number of channels)', '0..1'], ['ACC_final', 'mean final-row reward across all tasks', 'continual-learning final performance'], ['BWT', 'mean(final reward - reward after own-task learning)', 'negative indicates forgetting']], columns=['metric', 'definition', 'publication_note'])
    notes.to_csv(os.path.join(save_dir, 'metric_definitions.csv'), index=False)
    print(f'IEEE metric report saved → {csv_path}')
    print(f"Metric definitions saved → {os.path.join(save_dir, 'metric_definitions.csv')}")
    return report_df
print('IEEE metric report generator defined.')

def main():
    logger = setup_logger()
    logger.info('Initializing Continual RL Cognitive Radio Framework...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Compute device: {device}')
    EXPERIMENT_SEED = 2026
    set_global_seed(EXPERIMENT_SEED)
    logger.info(f'Experiment seed: {EXPERIMENT_SEED}')
    envs = [Task1Env(), Task2Env(), Task3Env(), Task4Env(), Task5Env(), Task6Env(), Task7Env()]
    state_dim = envs[0].state_dim
    output_dim = envs[0].action_space
    BATCH_SIZE = 64
    GAMMA = 0.97
    TARGET_UPDATE = 1000
    ROLLBACK_INTERVAL = 100
    TASK_STEPS = 10000
    agent = DQNAgent(state_dim, output_dim, device=device)
    active_buffer = ReplayBuffer(capacity=10000)
    episodic_memory = EpisodicMemory(capacity=2000)
    ewc = EWC(agent.policy_net)
    cpd = ChangePointDetector(delta=0.002)
    mec = MECServer(device=device)
    mec.start()
    global_step = 0
    R_matrix = []
    training_log = []
    pre_task_scores = [None] * len(envs)
    task_boundary_steps = [i * TASK_STEPS for i in range(1, len(envs))]
    rollback_path = 'checkpoints/rollback.pt'
    os.makedirs('checkpoints', exist_ok=True)
    try:
        for current_true_task_idx, env in enumerate(envs):
            logger.info(f'\n--- Environment → Task {current_true_task_idx + 1} ---')
            if current_true_task_idx > 0:
                pre_eval = evaluate_policy_on_env(agent, env, num_episodes=50)
                pre_task_scores[current_true_task_idx] = pre_eval['reward_mean']
                logger.info(f"Pre-task zero-shot T{current_true_task_idx + 1}: {pre_eval['reward_mean']:.3f} ± {pre_eval['reward_std']:.3f}")
            state = env.reset()
            for step in range(TASK_STEPS):
                global_step += 1
                fisher, theta_star = mec.check_results()
                if fisher is not None:
                    logger.info('MEC FIM received → injecting EWC penalty.')
                    ewc.update_task_penalty(fisher, theta_star)
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                active_buffer.push(state, action, reward, next_state, done)
                if len(active_buffer) > BATCH_SIZE:
                    num_episodic = int(BATCH_SIZE * 0.2)
                    num_active = BATCH_SIZE - num_episodic
                    if len(episodic_memory) > num_episodic:
                        s_a, a_a, r_a, ns_a, d_a = active_buffer.sample(num_active)
                        s_e, a_e, r_e, ns_e, d_e = episodic_memory.sample(num_episodic)
                        states = torch.cat([s_a, s_e]).to(device)
                        actions_t = torch.cat([a_a, a_e]).to(device)
                        rewards_t = torch.cat([r_a, r_e]).to(device)
                        next_states = torch.cat([ns_a, ns_e]).to(device)
                        dones_t = torch.cat([d_a, d_e]).to(device)
                    else:
                        states, actions_t, rewards_t, next_states, dones_t = active_buffer.sample(BATCH_SIZE)
                        states, actions_t, rewards_t = (states.to(device), actions_t.to(device), rewards_t.to(device))
                        next_states, dones_t = (next_states.to(device), dones_t.to(device))
                    q_values = agent.policy_net(states)
                    q_expected = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        q_next = agent.target_net(next_states).max(1)[0]
                        q_targets = rewards_t + GAMMA * q_next * (1 - dones_t)
                    td_errors_arr = (q_expected - q_targets).detach().cpu().numpy()
                    current_td_error = float(np.mean(np.abs(td_errors_arr)))
                    dqn_loss = nn.MSELoss()(q_expected, q_targets)
                    ewc_loss = ewc.compute_ewc_loss()
                    loss = dqn_loss + ewc_loss
                    agent.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.policy_net.parameters(), max_norm=10.0)
                    agent.optimizer.step()
                    ewc.update_td_error(current_td_error)
                    training_log.append({'global_step': global_step, 'task': current_true_task_idx + 1, 'dqn_loss': float(dqn_loss.item()), 'ewc_loss': float(ewc_loss.item()), 'total_loss': float(loss.item()), 'lambda_t': ewc.get_adaptive_lambda(), 'td_error': current_td_error, 'tau': agent.tau})
                    boundary_detected, b_index = cpd.update(current_td_error, global_step=global_step)
                    if boundary_detected:
                        logger.info(f'ADWIN drift detected | estimated boundary: step {b_index}')
                        if load_rollback_checkpoint(agent.policy_net, rollback_path):
                            logger.info('Weight rollback applied.')
                            agent.update_target_network()
                        top_k = active_buffer.get_top_k_transitions(k=286, policy_net=agent.policy_net, target_net=agent.target_net, gamma=GAMMA, device=device)
                        episodic_memory.add_transitions(top_k)
                        logger.info(f'Episodic memory: +{len(top_k)} transitions.')
                        frozen_net = copy.deepcopy(agent.policy_net).cpu()
                        mec.submit_task(list(active_buffer.buffer), frozen_net)
                        logger.info('FIM computation dispatched to MEC.')
                        active_buffer.flush()
                        cpd.reset(reset_step_counter=False)
                        agent.reset_temperature()
                        ewc.reset_td_window()
                if global_step % TARGET_UPDATE == 0:
                    agent.update_target_network()
                if global_step % ROLLBACK_INTERVAL == 0:
                    save_rollback_checkpoint(agent.policy_net, rollback_path)
                state = env.reset() if done else next_state
            logger.info(f'\n=== IEEE Evaluation Protocol: After Task {current_true_task_idx + 1} ===')
            mean_rewards, collision_rates, transmission_metrics, extended_metrics = run_evaluation(agent, envs, current_true_task_idx, num_episodes=100)
            padded = mean_rewards + [0.0] * (len(envs) - len(mean_rewards))
            R_matrix.append(padded)
            cl_metrics = calculate_continual_metrics(R_matrix)
            logger.info(f'Mean Rewards : {[round(r, 2) for r in mean_rewards]}')
            logger.info(f'Collision Rates: {[round(c, 3) for c in collision_rates]}')
            if cl_metrics:
                logger.info(f'CL Metrics : { {k: round(v, 4) for k, v in cl_metrics.items()}}')
            df = pd.DataFrame(R_matrix, columns=[f'T{i + 1}' for i in range(len(envs))])
            save_checkpoint(task_id=current_true_task_idx + 1, policy_net=agent.policy_net, target_net=agent.target_net, fisher_dict=None, theta_star_dict=None, episodic_memory=episodic_memory, metrics_df=df)
    except KeyboardInterrupt:
        logger.info('\nTraining interrupted by user.')
    finally:
        mec.stop()
        logger.info('MEC server shut down. Training complete.')
    if training_log:
        pd.DataFrame(training_log).to_csv('checkpoints/training_log.csv', index=False)
    logger.info('\n=== Generating Full IEEE Metric Report ===')
    adwin_stats = cpd.get_detection_stats(true_boundaries=task_boundary_steps, tolerance_steps=max(500, int(0.2 * TASK_STEPS)))
    logger.info(f'ADWIN Stats: {adwin_stats}')
    final_mean_rewards, final_collision_rates, final_tx_metrics, final_ext_metrics = run_evaluation(agent, envs, len(envs) - 1, num_episodes=200)
    logger.info('\n=== Evaluating Baseline Agents ===')
    random_baseline = RandomBaseline(action_space=envs[0].action_space)
    greedy_baseline = GreedyMyopicBaseline(num_channels=envs[0].num_channels, history_length=envs[0].history_length)
    random_results = evaluate_baseline(random_baseline, envs, num_episodes=200)
    greedy_results = evaluate_baseline(greedy_baseline, envs, num_episodes=200)
    final_cl_metrics = calculate_continual_metrics(R_matrix, pre_task_scores=pre_task_scores, random_baseline_scores=random_results['mean_rewards'], single_task_oracle_scores=None)
    adaptation_steps = []
    for i, env in enumerate(envs):
        ref = R_matrix[i][i]
        adaptation_steps.append(measure_adaptation_speed(agent, env, within_task_reference_episode_reward=ref, threshold_fraction=0.95, max_steps=5000, eval_window=100))
    report_df = generate_metric_report(R_matrix=R_matrix, cl_metrics=final_cl_metrics, transmission_metrics=final_tx_metrics, extended_metrics=final_ext_metrics, adwin_stats=adwin_stats, training_log=training_log, adaptation_steps=adaptation_steps, save_dir='checkpoints')
    logger.info(f"Random   Mean Rewards: {[round(r, 2) for r in random_results['mean_rewards']]}")
    logger.info(f"Greedy   Mean Rewards: {[round(r, 2) for r in greedy_results['mean_rewards']]}")
    logger.info(f'Proposed Mean Rewards: {[round(r, 2) for r in final_mean_rewards]}')
    baseline_df = pd.DataFrame({'task': [f'T{i + 1}' for i in range(len(envs))], 'random_reward_mean': random_results['mean_rewards'], 'random_reward_std': random_results['reward_stds'], 'random_reward_ci95_low': random_results['reward_ci95_low'], 'random_reward_ci95_high': random_results['reward_ci95_high'], 'random_collision': random_results['collision_rates'], 'random_throughput': random_results['throughputs'], 'greedy_reward_mean': greedy_results['mean_rewards'], 'greedy_reward_std': greedy_results['reward_stds'], 'greedy_reward_ci95_low': greedy_results['reward_ci95_low'], 'greedy_reward_ci95_high': greedy_results['reward_ci95_high'], 'greedy_collision': greedy_results['collision_rates'], 'greedy_throughput': greedy_results['throughputs'], 'proposed_reward_mean': final_mean_rewards, 'proposed_reward_std': [m['reward_std'] for m in final_tx_metrics], 'proposed_reward_ci95_low': [m['reward_ci95_low'] for m in final_tx_metrics], 'proposed_reward_ci95_high': [m['reward_ci95_high'] for m in final_tx_metrics], 'proposed_collision': final_collision_rates, 'proposed_throughput': [m['successful_transmission_rate'] for m in final_tx_metrics]})
    eps = 1e-12
    baseline_df['reward_gain_vs_greedy_pct'] = 100.0 * (baseline_df['proposed_reward_mean'] - baseline_df['greedy_reward_mean']) / baseline_df['greedy_reward_mean'].abs().clip(lower=eps)
    baseline_df['collision_reduction_vs_greedy_pct'] = 100.0 * (baseline_df['greedy_collision'] - baseline_df['proposed_collision']) / baseline_df['greedy_collision'].clip(lower=eps)
    baseline_df['throughput_gain_vs_greedy_pct'] = 100.0 * (baseline_df['proposed_throughput'] - baseline_df['greedy_throughput']) / baseline_df['greedy_throughput'].clip(lower=eps)
    baseline_csv = 'checkpoints/baseline_comparison.csv'
    baseline_df.to_csv(baseline_csv, index=False)
    logger.info(f'Baseline comparison saved → {baseline_csv}')
    complexity_df = pd.DataFrame([{'seed': EXPERIMENT_SEED, 'device': str(device), 'trainable_parameters': count_trainable_parameters(agent.policy_net), 'task_steps': TASK_STEPS, 'num_tasks': len(envs), 'evaluation_episodes_final': 200, 'batch_size': BATCH_SIZE, 'gamma': GAMMA, 'target_update_steps': TARGET_UPDATE, 'adwin_delta': cpd.delta}])
    complexity_df.to_csv('checkpoints/experiment_metadata.csv', index=False)
    print('\n' + '=' * 92)
    print('PUBLICATION-READY RESULT SUMMARY')
    print('=' * 92)
    print(f"Final average performance (ACC_final): {final_cl_metrics.get('ACC_final', np.nan):.3f}")
    print(f"Diagonal learning score:              {final_cl_metrics.get('ACC_diagonal', np.nan):.3f}")
    print(f"BWT:                                  {final_cl_metrics.get('BWT', np.nan):.3f}")
    print(f"Mean normalized forgetting:           {final_cl_metrics.get('mean_forgetting_pct', np.nan):.3f}%")
    print(f"FWT vs random baseline:               {final_cl_metrics.get('FWT_mean', np.nan):.3f}")
    print(f"ADWIN detection rate:                 {adwin_stats.get('detection_rate', np.nan):.3f}")
    print(f"ADWIN false-positive rate:            {adwin_stats.get('false_positive_rate', np.nan):.3f}")
    print(f"ADWIN false-negative rate:            {adwin_stats.get('false_negative_rate', np.nan):.3f}")
    print(f"ADWIN mean detection delay:           {adwin_stats.get('mean_detection_delay', np.nan):.1f} steps")
    print('\nNOTE: true physical spectral efficiency (bit/s/Hz) is NOT claimed because this')
    print('environment has no bandwidth / payload-bit-rate / link-SINR model. The notebook')
    print('reports normalized access efficiency instead, avoiding a duplicate/mislabelled metric.')
    print('=' * 92)
    return (report_df, baseline_df)

_v1_original_sample = random.sample
def _v1_sequence_sample(population, k, *args, **kwargs):
    if isinstance(population, deque):
        population = list(population)
    return _v1_original_sample(population, k, *args, **kwargs)
random.sample = _v1_sequence_sample

_OriginalReplayBuffer = ReplayBuffer
class ReplayBuffer(_OriginalReplayBuffer):
    def get_top_k_transitions(self, k, policy_net, target_net, gamma, device, batch_size=512):
        return random.sample(list(self.buffer), min(k, len(self.buffer)))

main()
