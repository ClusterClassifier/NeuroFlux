"""Notebook definitions for agent."""

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch


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

    def __init__(self, input_dim: int, output_dim: int, lr: float = 1e-3,
                 gamma: float = 0.99, device: str = 'cpu'):
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.gamma      = gamma
        self.device     = device

        self.policy_net = DQN(input_dim, output_dim).to(device)
        self.target_net = DQN(input_dim, output_dim).to(device)
        self.update_target_network()
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)


        self.tau0             = 2.0
        self.tau_min          = 0.1
        self.d                = 1e-6
        self.tau              = self.tau0
        self.task_step_counter = 0

    def update_target_network(self) -> None:
        """Hard-copies policy weights into the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def reset_temperature(self) -> None:
        """Resets Boltzmann temperature to τ₀ at the start of a new task."""
        self.tau              = self.tau0
        self.task_step_counter = 0

    def get_action(self, state, evaluate: bool = False) -> int:
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

        # Scales Q-values before Boltzmann sampling.
        q_scaled = q_values / self.tau
        probs    = F.softmax(q_scaled, dim=1).cpu().numpy()[0]

        # Normalizes probabilities and falls back to uniform sampling for invalid values.
        if np.isnan(probs).any() or np.sum(probs) <= 0:
            probs = np.ones(self.output_dim, dtype=np.float64) / self.output_dim
        else:
            probs = probs.astype(np.float64)
            probs /= np.sum(probs)

        action   = np.random.choice(self.output_dim, p=probs)

        # Decays the exploration temperature after each sampled action.
        self.task_step_counter += 1
        self.tau = max(self.tau_min, self.tau0 * np.exp(-self.d * self.task_step_counter))

        return action


print("DQN agent defined successfully.")
