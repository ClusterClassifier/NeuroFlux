"""Notebook definitions for ewc."""

from collections import deque
import torch.nn as nn
import numpy as np
import torch


class EWC:
    """
    Elastic Weight Consolidation for continual reinforcement learning.

    Key contributions vs. vanilla EWC:
      1. Adaptive λ: penalty coefficient is modulated by the rolling TD-error variance,
         automatically balancing plasticity vs. stability per task.
      2. Async-compatible: FIM and θ* arrive from the MEC server after a delay;
         update_task_penalty() integrates them whenever they land.
    """

    def __init__(self, policy_net: nn.Module,
                 lambda_max: float = 4.0,
                 beta: float       = 10.0,
                 window_size: int  = 500):
        self.policy_net  = policy_net
        self.lambda_max  = lambda_max
        self.beta        = beta

        # Stores Fisher estimates and reference weights for each completed task.
        self.past_tasks: list = []

        # Tracks recent absolute TD errors for the adaptive EWC coefficient.
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
                    penalty   = (fisher_dict[name] * (param - theta_star_dict[name]) ** 2).sum()
                    ewc_loss += penalty

        return (lambda_t / 2.0) * ewc_loss


print("EWC module defined successfully.")
