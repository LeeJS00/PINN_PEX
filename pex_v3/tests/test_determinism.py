"""
test_determinism.py — 4-way seed reproducibility.

Verifies `set_all_seeds` reliably reproduces RNG sequences across all four
sources (random, numpy, torch CPU, torch CUDA).
"""
from __future__ import annotations
import random
import numpy as np
import torch

from src.utils.seeds import set_all_seeds


def test_set_all_seeds_repeats_random():
    set_all_seeds(123)
    a = [random.random() for _ in range(5)]
    set_all_seeds(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_all_seeds_repeats_numpy():
    set_all_seeds(123)
    a = np.random.rand(10)
    set_all_seeds(123)
    b = np.random.rand(10)
    assert np.allclose(a, b)


def test_set_all_seeds_repeats_torch_cpu():
    set_all_seeds(123)
    a = torch.randn(20)
    set_all_seeds(123)
    b = torch.randn(20)
    assert torch.allclose(a, b)


def test_different_seeds_produce_different_sequences():
    set_all_seeds(123)
    a = torch.randn(20)
    set_all_seeds(124)
    b = torch.randn(20)
    assert not torch.allclose(a, b)


def test_set_all_seeds_repeats_torch_cuda():
    if not torch.cuda.is_available():
        return
    set_all_seeds(456)
    a = torch.randn(20, device="cuda")
    set_all_seeds(456)
    b = torch.randn(20, device="cuda")
    assert torch.allclose(a, b)
