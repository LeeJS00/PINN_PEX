"""profiler.py — per-stage runtime profiler for joint-Pareto SPEF variants.

Owned by pex-runtime-owner. Used to emit a per-stage breakdown JSON next
to each experiment's SPEF output, plus a paired-seed runtime measurement
helper.

Contract:

    with StageTimer(stats := {}) as t:
        with t.stage("topology_load"):
            load_topology(...)
        with t.stage("kdtree_build"):
            build_kdtree(...)
        ...
    print(stats)   # {"topology_load_s": 12.8, "kdtree_build_s": 0.9, ...}

The 5-seed paired runtime helper below performs N independent runs of a
callable and reports mean ± stdev for each stage.
"""
from __future__ import annotations
import contextlib
import statistics
import time
from typing import Callable, Any


class StageTimer:
    """Context-manager hierarchy for per-stage wall-clock accounting."""

    def __init__(self, stats: dict | None = None):
        self.stats = stats if stats is not None else {}

    def __enter__(self):
        self._t_start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.stats["wall_clock_s"] = time.perf_counter() - self._t_start

    @contextlib.contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            key = f"{name}_s"
            self.stats[key] = self.stats.get(key, 0.0) + (time.perf_counter() - t0)


def paired_runtime_5seed(
    callable_factory: Callable[[int], Callable[[], dict]],
    n_seeds: int = 5,
) -> dict:
    """Run a variant under N independent seeds and report mean ± stdev per stage.

    `callable_factory(seed)` returns the no-arg function to invoke for that seed
    (which must return a dict of per-stage seconds; produced by StageTimer).
    """
    runs = [callable_factory(s)() for s in range(n_seeds)]
    keys = sorted({k for r in runs for k in r if k.endswith("_s")})
    out: dict = {"n_seeds": n_seeds, "per_seed": runs, "summary": {}}
    for k in keys:
        vals = [r.get(k, 0.0) for r in runs]
        out["summary"][k] = {
            "mean": statistics.mean(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }
    return out
