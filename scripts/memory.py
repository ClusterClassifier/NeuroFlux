"""Notebook definitions for memory."""

from collections import deque
import numpy as np
import random
import torch


class ReplayBuffer:
    """Standard i.i.d. experience replay buffer for DQN."""

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer   = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        """Adds a single SARS′d transition to the buffer."""
        self.buffer.append({'state': state, 'action': action, 'reward': reward,
                            'next_state': next_state, 'done': done})

    def sample(self, batch_size: int):
        """Samples a random i.i.d. minibatch and returns PyTorch tensors."""
        return self._format_batch(random.sample(self.buffer, batch_size))

    def _format_batch(self, batch):
        states      = torch.tensor(np.array([b['state']      for b in batch]), dtype=torch.float32)
        actions     = torch.tensor([b['action']               for b in batch], dtype=torch.long)
        rewards     = torch.tensor([b['reward']               for b in batch], dtype=torch.float32)
        next_states = torch.tensor(np.array([b['next_state'] for b in batch]), dtype=torch.float32)
        dones       = torch.tensor([b['done']                 for b in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def get_top_k_transitions(self, k: int, policy_net, target_net, gamma, device, batch_size: int = 512):
        """
        Scores all buffered transitions by absolute TD-error and returns the top-k
        most informative ones. Used at task boundaries to populate episodic memory.
        """
        if len(self.buffer) == 0:
            return []

        policy_net.eval()
        buffer_list = list(self.buffer)
        td_errors   = []

        for i in range(0, len(buffer_list), batch_size):
            batch = buffer_list[i:i + batch_size]
            s, a, r, ns, d = self._format_batch(batch)
            s, a, r, ns, d = s.to(device), a.to(device), r.to(device), ns.to(device), d.to(device)

            with torch.no_grad():
                q_exp    = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                q_next   = target_net(ns).max(1)[0]
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

    def __init__(self, capacity: int = 2000):
        self.capacity = capacity
        self.memory   = []

    def add_transitions(self, transitions) -> None:
        self.memory.extend(transitions)
        if len(self.memory) > self.capacity:
            evict_count = max(int(self.capacity * 0.10), len(self.memory) - self.capacity)
            self.memory = self.memory[evict_count:]

    def sample(self, batch_size: int):
        """Samples a random minibatch (or the full memory if smaller than batch_size)."""
        batch = self.memory if len(self.memory) < batch_size else random.sample(self.memory, batch_size)
        states      = torch.tensor(np.array([b['state']      for b in batch]), dtype=torch.float32)
        actions     = torch.tensor([b['action']               for b in batch], dtype=torch.long)
        rewards     = torch.tensor([b['reward']               for b in batch], dtype=torch.float32)
        next_states = torch.tensor(np.array([b['next_state'] for b in batch]), dtype=torch.float32)
        dones       = torch.tensor([b['done']                 for b in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.memory)


print("Replay and episodic memory classes defined successfully.")
