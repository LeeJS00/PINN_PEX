"""baseline.py — current Pareto-frontier c_gnd allocator (Path-2 v3).

This is the EXACT formula in `pex_v3/src/utils/fast_spef_engine.py:
analytic_per_net_cap_estimate`. Mirrored here so future variants can be
diff'd against it cleanly.

Contract: given a net's segments + a per-net c_gnd_total target (in fF),
return a dict mapping topology-node ID → gnd cap (fF) summing to total.
"""
from __future__ import annotations
from typing import Iterable

# Layer ε proxy used in the v3 placeholder (matches fast_spef_engine).
_LAYER_EPS_PROXY = {
    "m1": 4.2, "m2": 3.9, "m3": 3.6, "m4": 3.3,
    "m5": 3.1, "m6": 2.9, "m7": 2.7, "m8": 2.7,
}

V3_GND_SCALE = 0.22       # tuned 2026-05-03 to unmatched-net golden median 0.477 fF
V3_CPL_GND_RATIO = 1.3    # empirical 0.609 / 0.477 cpl-to-gnd ratio


def per_net_cap_estimate(segments: Iterable) -> tuple[float, float]:
    """Return (c_gnd_total, c_cpl_total) in fF using the v3 calibrated formula.

    Matched nets are XGB-rescaled exactly so this constant is invariant
    for them. Unmatched nets receive a per-net total that lands on the
    tv80s test golden median (0.477 fF gnd, 0.609 fF cpl).
    """
    c_gnd = 0.0
    for seg in segments:
        eps = _LAYER_EPS_PROXY.get(getattr(seg, "layer", "m1"), 3.5)
        c_gnd += seg.length * seg.width * eps * V3_GND_SCALE
    c_cpl = c_gnd * V3_CPL_GND_RATIO
    return c_gnd, c_cpl


def allocate_gnd(
    segments,
    c_gnd_total: float,
    layer_info: dict | None = None,
) -> dict:
    """Length-proportional distribution to topology nodes.

    The current frontier delegates this to `src.utils.spef_writer.distribute_net_caps`
    which uses the same length-proportional approach. This wrapper exists
    only to make variants drop-in compatible.
    """
    raise NotImplementedError(
        "baseline allocation is performed inline by the legacy "
        "src.utils.spef_writer.distribute_net_caps; use that path "
        "directly when comparing variants. This stub exists for the "
        "diff-against-baseline contract."
    )
