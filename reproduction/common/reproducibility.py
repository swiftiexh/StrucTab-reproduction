"""Random-state initialization shared by all experiment entry points."""

import os
import random

import numpy as np
import torch


DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED, deterministic_algorithms: bool = False) -> int:
    """Seed Python, NumPy and PyTorch and return the applied seed.

    Strict deterministic algorithms remain opt-in because some multimodal CUDA
    operations may not provide a deterministic implementation.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(deterministic_algorithms)
    return seed

