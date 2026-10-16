"""Notebook definitions for environments."""

from collections import deque
import numpy as np


class CRNEnv:
    """
    Base Cognitive Radio Network environment.

    State:  Flattened SNR + occupancy history for all N channels → size (N × 2M,).
    Action: 0 … N-1  → transmit on channel k
            N        → idle (no transmission)
    Reward: +R_success on clear channel, -R_collision on occupied, -R_idle when idle.
    """

    def __init__(self, num_channels: int = 8, history_length: int = 5, max_steps: int = 1000):
        self.num_channels   = num_channels
        self.history_length = history_length
        self.max_steps      = max_steps

        self.state_dim    = num_channels * 2 * history_length
        self.action_space = num_channels + 1
        self.idle_action  = num_channels


        self.R_success   = 1.0
        self.R_collision = 1.0
        self.R_idle      = 0.05


        self.pu_activity_probs = np.zeros(num_channels)


        self.snr_history = [deque(maxlen=history_length) for _ in range(num_channels)]
        self.occ_history = [deque(maxlen=history_length) for _ in range(num_channels)]

        self.current_step = 0


    def _set_probs(self, seed: int, base_low: float, base_high: float) -> None:
        """Generates reproducible, task-distinct PU activity probabilities."""
        rng = np.random.RandomState(seed)
        self.pu_activity_probs = rng.uniform(base_low, base_high, self.num_channels)
        # Keeps at least one channel relatively clear.
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
        # Prefills history to avoid cold-start edge states.
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
            raise ValueError(f"Invalid action: {action}")

        self._generate_observation()
        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self.get_state(), reward, done, {}

    def get_state(self) -> np.ndarray:
        """Returns the flattened (N × 2M,) state vector."""
        state = []
        for c in range(self.num_channels):
            state.extend(self.snr_history[c])
            state.extend(self.occ_history[c])
        return np.array(state, dtype=np.float32)


class Task1Env(CRNEnv):
    """T1 | 06:00 | Early commute ramp-up       | PU load: Low–Medium (0.2–0.4)"""
    def __init__(self, **kwargs): super().__init__(**kwargs); self._set_probs(seed=42, base_low=0.2, base_high=0.4)

class Task2Env(CRNEnv):
    """T2 | 09:00 | Peak commute traffic         | PU load: High (0.6–0.9)"""
    def __init__(self, **kwargs): super().__init__(**kwargs); self._set_probs(seed=43, base_low=0.6, base_high=0.9)

class Task3Env(CRNEnv):
    """T3 | 12:00 | Midday office traffic        | PU load: Medium (0.4–0.6)"""
    def __init__(self, **kwargs): super().__init__(**kwargs); self._set_probs(seed=44, base_low=0.4, base_high=0.6)

class Task4Env(CRNEnv):
    """T4 | 14:00 | Sparse IoT sensor bursts     | PU load: Low (0.05–0.2)"""
    def __init__(self, **kwargs): super().__init__(**kwargs); self._set_probs(seed=45, base_low=0.05, base_high=0.2)

class Task5Env(CRNEnv):
    """T5 | 17:00 | Evening commute              | PU load: High (0.65–0.85)"""
    def __init__(self, **kwargs): super().__init__(**kwargs); self._set_probs(seed=46, base_low=0.65, base_high=0.85)

class Task6Env(CRNEnv):
    """T6 | 20:00 | Heavy video streaming        | PU load: Very High (0.8–0.95)"""
    def __init__(self, **kwargs): super().__init__(**kwargs); self._set_probs(seed=47, base_low=0.8, base_high=0.95)

class Task7Env(CRNEnv):
    """T7 | 23:00 | Night-time low activity      | PU load: Very Low (0.01–0.1)"""
    def __init__(self, **kwargs): super().__init__(**kwargs); self._set_probs(seed=48, base_low=0.01, base_high=0.1)


print("Cognitive radio environments (T1–T7) defined successfully.")
