"""Notebook definitions for mec."""

import torch.nn as nn
import numpy as np
import queue
import threading
import torch


class MECServer:
    """
    Simulates an asynchronous MEC server for Fisher Information Matrix computation.

    The MEC runs in a daemon thread. The edge radio submits tasks (frozen net + replay subset)
    and polls for results non-blockingly via check_results(). This models real MEC offloading
    latency without blocking the online training loop.
    """

    def __init__(self, device: str = 'cpu'):
        self.device        = device
        self.request_queue = queue.Queue()
        self.result_queue  = queue.Queue()
        self._stop_event   = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)

    def start(self) -> None:
        """Starts the background MEC worker thread."""
        self.worker_thread.start()

    def stop(self) -> None:
        """Signals the worker to terminate and waits for clean shutdown."""
        self._stop_event.set()
        self.request_queue.put(None)  # Unblocks the worker while it waits for a queued request.
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
            return None, None

    def _worker_loop(self) -> None:
        """Core computation loop (diagonal empirical Fisher via per-sample backward passes)."""
        while not self._stop_event.is_set():
            task = self.request_queue.get()
            if task is None:
                continue

            replay_subset, frozen_net = task
            frozen_net = frozen_net.to(self.device)
            frozen_net.train()

            states  = torch.tensor(np.array([t['state']  for t in replay_subset]), dtype=torch.float32).to(self.device)
            actions = torch.tensor([t['action']            for t in replay_subset], dtype=torch.long).to(self.device)

            fisher_dict  = {n: torch.zeros_like(p) for n, p in frozen_net.named_parameters()}
            num_samples  = len(states)

            # Accumulates squared per-sample gradients for the diagonal Fisher estimate.
            for i in range(num_samples):
                frozen_net.zero_grad()
                q_vals = frozen_net(states[i:i+1])
                q_a    = q_vals.gather(1, actions[i:i+1].unsqueeze(1)).squeeze(1)
                q_a.backward()
                for name, param in frozen_net.named_parameters():
                    if param.grad is not None:
                        fisher_dict[name] += param.grad.detach() ** 2

            for name in fisher_dict:
                fisher_dict[name] /= num_samples

            theta_star_dict = {n: p.detach().clone() for n, p in frozen_net.named_parameters()}
            self.result_queue.put((fisher_dict, theta_star_dict))


print("MEC server defined successfully.")
