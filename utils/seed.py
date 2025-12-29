"""Seed utilities for reproducibility.

Provides functions to set random seeds across Python, NumPy, and PyTorch
for reproducible experiments.
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for Python's random module, NumPy, and PyTorch (CPU and CUDA).
    Optionally enables deterministic algorithms for full reproducibility.

    Args:
        seed: Random seed value. Defaults to 42.
        deterministic: If True, use deterministic algorithms (may impact performance).
            Defaults to True.

    Note:
        Deterministic mode may slow down training but ensures reproducibility.
        For production training, consider setting deterministic=False.
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Set environment variable for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # Enable deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch 1.8+ deterministic setting
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                # Some operations don't have deterministic implementations
                pass
    else:
        # Enable cuDNN auto-tuner for better performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """Initialize DataLoader worker with unique seed.

    This function should be passed to DataLoader's worker_init_fn parameter
    to ensure each worker has a different random seed while maintaining
    reproducibility across runs.

    Args:
        worker_id: The worker ID assigned by DataLoader.

    Example:
        >>> from torch.utils.data import DataLoader
        >>> from utils.seed import worker_init_fn, set_seed
        >>> set_seed(42)
        >>> loader = DataLoader(
        ...     dataset,
        ...     num_workers=4,
        ...     worker_init_fn=worker_init_fn
        ... )
    """
    # Get initial seed from PyTorch
    worker_seed = torch.initial_seed() % 2**32

    # Combine with worker_id for unique seed per worker
    unique_seed = worker_seed + worker_id

    # Set seeds for this worker
    random.seed(unique_seed)
    np.random.seed(unique_seed)


def get_random_state() -> dict:
    """Get current random state for checkpointing.

    Returns:
        Dictionary containing random states for Python, NumPy, and PyTorch.

    Example:
        >>> state = get_random_state()
        >>> # ... do some training ...
        >>> set_random_state(state)  # restore state
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def set_random_state(state: dict) -> None:
    """Restore random state from checkpoint.

    Args:
        state: Dictionary containing random states (from get_random_state).
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
