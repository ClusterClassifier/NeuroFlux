"""Notebook definitions for utils."""

import logging
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import pickle
import random
import shutil
import torch


def setup_logger() -> logging.Logger:
    """Configures and returns a named console logger."""
    logger = logging.getLogger("CRN_Continual")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    return logger


def atomic_save(obj, filepath: str, is_torch: bool = True) -> None:
    """Writes to a .tmp file first, then performs an atomic rename to prevent corruption."""
    tmp_path = filepath + ".tmp"
    if is_torch:
        torch.save(obj, tmp_path)
    else:
        with open(tmp_path, 'wb') as f:
            pickle.dump(obj, f)
    shutil.move(tmp_path, filepath)


def save_checkpoint(
    task_id: int,
    policy_net: nn.Module,
    target_net: nn.Module,
    fisher_dict,
    theta_star_dict,
    episodic_memory,
    metrics_df: pd.DataFrame,
    save_dir: str = "checkpoints"
) -> None:
    """
    Saves all critical training state at a task boundary.
    Components: policy/target nets, Fisher matrices, theta*, episodic memory, metrics CSV.
    """
    os.makedirs(save_dir, exist_ok=True)


    nets = {'policy': policy_net.state_dict(), 'target': target_net.state_dict()}
    atomic_save(nets, os.path.join(save_dir, f"task_{task_id}_policy.pt"), is_torch=True)

    if fisher_dict:
        atomic_save(fisher_dict, os.path.join(save_dir, f"task_{task_id}_fisher.pkl"), is_torch=True)

    if theta_star_dict:
        atomic_save(theta_star_dict, os.path.join(save_dir, f"task_{task_id}_theta_star.pkl"), is_torch=True)

    if episodic_memory and len(episodic_memory) > 0:
        atomic_save(episodic_memory.memory, os.path.join(save_dir, f"task_{task_id}_episodic.pkl"), is_torch=False)

    if metrics_df is not None and not metrics_df.empty:
        tmp_csv = os.path.join(save_dir, "metrics.csv.tmp")
        metrics_df.to_csv(tmp_csv, index=False)
        shutil.move(tmp_csv, os.path.join(save_dir, "metrics.csv"))


def save_rollback_checkpoint(policy_net: nn.Module, filepath: str) -> None:
    """Saves a lightweight rollback snapshot used by ADWIN drift recovery."""
    atomic_save(policy_net.state_dict(), filepath, is_torch=True)


def load_rollback_checkpoint(policy_net: nn.Module, filepath: str) -> bool:
    """Loads an ADWIN rollback snapshot. Returns True on success, False if file is absent."""
    if os.path.exists(filepath):
        policy_net.load_state_dict(torch.load(filepath))
        return True
    return False


print("Utility functions defined successfully.")


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
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
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    half = 1.96 * std / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return mean, std, mean - half, mean + half


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


print("Reproducibility and statistical-reporting utilities defined.")
