"""
Per-layer physics parameter table from intel22 layer_info.

Threads correct ε/d/t per metal layer into physics_base, replacing the
placeholder ε=3.0/d=0.2/t=0.144 used in Phase 1.0 smoke.

Output:
    LayerPhysicsTable: lookup by metal layer name (m1, m2, ..., m8) →
        {eps_above, eps_below, d_above, d_below, t_metal, lvl_idx}
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class LayerParams:
    name: str
    z_pos: float          # μm
    z_top: float          # μm
    t_metal: float        # μm
    eps_self: float       # surrounding dielectric ε (intel22 layer_info convention)
    eps_above: float
    eps_below: float
    d_above: float        # μm to next metal above
    d_below: float        # μm to next metal below (or to substrate)
    lvl_idx: int


def _series_eps_d(dielectrics: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Compute (eps_effective, d_total) for a series of dielectric layers.

    For series capacitance through stacked dielectrics:
        1/C = Σ d_i / (ε_i · A)
        C = (eps_eff · A) / d_total
        → eps_eff = d_total / Σ (d_i / ε_i)

    Args:
        dielectrics: list of (thickness, epsilon) tuples for each dielectric layer.

    Returns:
        (eps_effective, d_total)
    """
    if not dielectrics:
        return 3.0, 0.05  # placeholder
    d_total = 0.0
    inv_eps_d = 0.0
    for d, eps in dielectrics:
        if d <= 0 or eps <= 0:
            continue
        d_total += d
        inv_eps_d += d / eps
    if inv_eps_d <= 0 or d_total <= 0:
        return 3.0, 0.05
    eps_eff = d_total / inv_eps_d
    return eps_eff, max(d_total, 0.005)


class LayerPhysicsTable:
    """Lookup table built once from LayerInfoParser output.

    Phase C: computes effective ε / d for series-stacked dielectrics between
    adjacent metals (replaces single-layer placeholder).
    """

    def __init__(self, layer_info: Dict):
        # Find metal layers (type='C' with lvl_idx).
        metals: List[Tuple[str, Dict]] = []
        dielectrics: List[Tuple[str, Dict]] = []
        for k, v in layer_info.items():
            if v.get("type") == "C" and "lvl_idx" in v and k.lower().startswith("m"):
                metals.append((k, v))
            if v.get("type") == "D" and v.get("thickness", 0) > 1e-6:
                dielectrics.append((k, v))
        metals.sort(key=lambda x: x[1]["z_pos"])
        # Sort dielectrics by z_pos for stacking computation.
        dielectrics.sort(key=lambda x: x[1].get("z_pos", 0))

        self.params: Dict[str, LayerParams] = {}
        self.layer_idx_to_name: Dict[int, str] = {}

        for i, (name, v) in enumerate(metals):
            z_pos = float(v["z_pos"])
            z_top = z_pos + float(v["thickness"])
            t = float(v["thickness"])
            eps = float(v.get("epsilon", 3.0))

            # Phase C: gather all dielectrics in (z_top, next_metal.z_pos) band.
            if i + 1 < len(metals):
                next_z = float(metals[i + 1][1]["z_pos"])
                stack_above = []
                for dn, dv in dielectrics:
                    dz = float(dv.get("z_pos", 0))
                    dt = float(dv.get("thickness", 0))
                    dz_top = dz + dt
                    # Dielectric overlaps the gap (z_top, next_z).
                    if dz_top > z_top + 1e-6 and dz < next_z - 1e-6:
                        # Trim to gap.
                        slice_lo = max(dz, z_top)
                        slice_hi = min(dz_top, next_z)
                        slice_t = slice_hi - slice_lo
                        if slice_t > 1e-6:
                            stack_above.append((slice_t, float(dv.get("epsilon", eps))))
                eps_above, d_above = _series_eps_d(stack_above)
                d_above = max(0.005, d_above)
            else:
                d_above = 1.0   # top metal — large distance to passivation
                eps_above = 1.0

            if i > 0:
                prev = metals[i - 1][1]
                prev_top = float(prev["z_pos"]) + float(prev["thickness"])
                stack_below = []
                for dn, dv in dielectrics:
                    dz = float(dv.get("z_pos", 0))
                    dt = float(dv.get("thickness", 0))
                    dz_top = dz + dt
                    if dz_top > prev_top + 1e-6 and dz < z_pos - 1e-6:
                        slice_lo = max(dz, prev_top)
                        slice_hi = min(dz_top, z_pos)
                        slice_t = slice_hi - slice_lo
                        if slice_t > 1e-6:
                            stack_below.append((slice_t, float(dv.get("epsilon", eps))))
                eps_below, d_below = _series_eps_d(stack_below)
                d_below = max(0.005, d_below)
            else:
                d_below = z_pos  # M1 to substrate
                eps_below = 3.9  # bulk Si region

            params = LayerParams(
                name=name,
                z_pos=z_pos,
                z_top=z_top,
                t_metal=t,
                eps_self=eps,
                eps_above=eps_above,
                eps_below=eps_below,
                d_above=d_above,
                d_below=d_below,
                lvl_idx=int(v["lvl_idx"]),
            )
            self.params[name] = params
            li_idx = int(round(z_pos * 100))
            self.layer_idx_to_name[li_idx] = name

    def lookup(self, layer_name: str) -> Optional[LayerParams]:
        return self.params.get(layer_name)

    def lookup_by_idx(self, layer_idx: int) -> Optional[LayerParams]:
        # Find closest layer_idx (segment_extractor's z_pos*100 binning).
        if layer_idx in self.layer_idx_to_name:
            return self.params[self.layer_idx_to_name[layer_idx]]
        # Fallback: nearest neighbor.
        keys = sorted(self.layer_idx_to_name.keys())
        if not keys:
            return None
        nearest = min(keys, key=lambda k: abs(k - layer_idx))
        return self.params[self.layer_idx_to_name[nearest]]

    def metal_layer_names(self) -> List[str]:
        return list(self.params.keys())

    # --- Tensor lookup helpers for batched physics_base calls ---

    def build_seg_tensors(
        self,
        layer_idxs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Returns per-segment ε/d/t tensors broadcast-ready for gnd_base_per_segment."""
        device = layer_idxs.device
        N = layer_idxs.shape[0]
        t_metal = torch.empty(N, dtype=torch.float32)
        eps_a = torch.empty(N, dtype=torch.float32)
        eps_b = torch.empty(N, dtype=torch.float32)
        d_a = torch.empty(N, dtype=torch.float32)
        d_b = torch.empty(N, dtype=torch.float32)
        for i, idx in enumerate(layer_idxs.tolist()):
            p = self.lookup_by_idx(int(idx))
            if p is None:
                t_metal[i] = 0.144
                eps_a[i] = 3.0
                eps_b[i] = 3.0
                d_a[i] = 0.2
                d_b[i] = 0.2
            else:
                t_metal[i] = p.t_metal
                eps_a[i] = p.eps_above
                eps_b[i] = p.eps_below
                d_a[i] = p.d_above
                d_b[i] = p.d_below
        return {
            "t_metal": t_metal.to(device),
            "eps_above": eps_a.to(device),
            "eps_below": eps_b.to(device),
            "d_above": d_a.to(device),
            "d_below": d_b.to(device),
        }

    def build_pair_tensors(
        self,
        t_layer_idxs: torch.Tensor,
        a_layer_idxs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Per-edge ε_pair, t_pair (avg of two), d_pair_vert."""
        device = t_layer_idxs.device
        E = t_layer_idxs.shape[0]
        eps_pair = torch.empty(E, dtype=torch.float32)
        t_pair = torch.empty(E, dtype=torch.float32)
        for i in range(E):
            pt = self.lookup_by_idx(int(t_layer_idxs[i]))
            pa = self.lookup_by_idx(int(a_layer_idxs[i]))
            if pt is None or pa is None:
                eps_pair[i] = 3.0
                t_pair[i] = 0.144
            else:
                # ε_pair: dielectric between target and aggressor metals.
                # If same layer: use eps_self. If cross: use the one closer to gap.
                if pt.lvl_idx == pa.lvl_idx:
                    eps_pair[i] = pt.eps_self
                else:
                    # average eps_above and eps_below of the two layers
                    eps_pair[i] = 0.5 * (pt.eps_self + pa.eps_self)
                t_pair[i] = 0.5 * (pt.t_metal + pa.t_metal)
        return {
            "eps_pair": eps_pair.to(device),
            "t_pair": t_pair.to(device),
        }


def _smoke_test() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from configs import config as cfg
    from src.preprocessing.layer_parser import LayerInfoParser

    li = LayerInfoParser(str(cfg.LAYERS_INFO_PATH)).parse()
    table = LayerPhysicsTable(li)
    print(f"Metal layers: {table.metal_layer_names()}")
    for name in table.metal_layer_names():
        p = table.lookup(name)
        print(f"  {name}: t={p.t_metal:.3f} ε_a={p.eps_above:.2f} ε_b={p.eps_below:.2f} "
              f"d_a={p.d_above:.3f} d_b={p.d_below:.3f}")

    # Lookup test.
    idx_m4 = int(round(table.lookup("m4").z_pos * 100))
    p = table.lookup_by_idx(idx_m4)
    print(f"\nLookup by idx {idx_m4}: {p.name if p else None}")
    print("[layer_physics smoke] OK")


if __name__ == "__main__":
    _smoke_test()
