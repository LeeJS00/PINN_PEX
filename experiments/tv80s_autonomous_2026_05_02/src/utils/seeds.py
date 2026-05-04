"""
seeds.py — 4-way deterministic seed setter.

Single source of truth for "make this run reproducible." Sets seeds for
numpy, random, torch CPU, torch CUDA. Also flips cuDNN flags and
deterministic-algorithms flag.

Use at the very start of any entrypoint:

    from pex_v3.src.utils.seeds import set_all_seeds
    set_all_seeds(seed=42)

This will set CUBLAS_WORKSPACE_CONFIG env var if not already set; without
it `torch.use_deterministic_algorithms(True)` raises on CUDA matmul.

If `torch.compile` is in use anywhere downstream, determinism is best-effort
only — `torch.compile` introduces nondeterministic kernels. For the v3
critical-path runs we recommend disabling torch.compile (legacy used it in
`run_active_learning.py`).
"""
from __future__ import annotations
import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    """Set all four RNG sources to `seed`.

    Args:
        seed: integer seed
        deterministic: if True, also flips cuDNN/torch flags for
                       maximum determinism (slower).
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # CUBLAS workspace config required by torch.use_deterministic_algorithms
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older torch may not support warn_only
            torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker_init_fn that propagates per-worker seed.

    Pass `worker_init_fn=worker_init_fn` to DataLoader so each worker has
    a deterministic seed derived from torch's initial seed.
    """
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)
