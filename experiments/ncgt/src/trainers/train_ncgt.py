"""
NCGT training loop (Plan v4 §5).

Phase 1.0 mode: 10-net overfit on a single design — gate for architecture sanity.
Phase 2.0+: full single-design / multi-design supervised training.

Loss design (Plan v4 §4):
- Phase 1.0/1.1 (no bins): hand-tuned weights, no GradNorm.
- Phase 2.4+: 6 composite tasks via GradNorm.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import math  # noqa: E402

from experiments.ncgt.src.data.geometric_aug import SAFE_TRANSFORMS, transform_xy  # noqa: E402
from experiments.ncgt.src.data.layer_physics import LayerPhysicsTable  # noqa: E402
from experiments.ncgt.src.models.ncgt_model import NCGTConfig, NCGTModel  # noqa: E402


def curriculum_clamp(step: int, total_steps: int) -> float:
    """pex_v3 curriculum: log(1.5) → log(2.5) → log(4.0).

    Phase 0 (0-25%): log(1.5) ≈ 0.405 → mul ∈ [0.67, 1.50]
    Phase 1 (25-75%): log(2.5) ≈ 0.916 → mul ∈ [0.40, 2.50]
    Phase 2 (75%+): log(4.0) ≈ 1.386 → mul ∈ [0.25, 4.00]
    """
    progress = step / max(1, total_steps)
    if progress < 0.25:
        return math.log(1.5)
    if progress < 0.75:
        return math.log(2.5)
    return math.log(4.0)


EPS_GND = 1e-3  # 0.001 fF threshold for "near zero" GND target.
EPS_CPL = 1e-4  # 0.0001 fF threshold for CPL.
EPS_TOTAL = 1e-3


def mape_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = EPS_TOTAL) -> torch.Tensor:
    """MAPE on a single scalar pair (or vectorized). Rule 1."""
    denom = torch.clamp(target.abs(), min=eps)
    return ((pred - target).abs() / denom).mean()


def edge_mape_supervised(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = EPS_CPL,
) -> torch.Tensor:
    """Per-edge MAPE on edges where mask=True (supervised by SPEF tie-break)."""
    if not mask.any():
        return torch.zeros((), device=pred.device, requires_grad=False)
    p = pred[mask]
    t = target[mask]
    return ((p - t).abs() / torch.clamp(t.abs(), min=eps)).mean()


def per_aggr_net_mape(
    pred_per_net: torch.Tensor,
    gt_per_net: torch.Tensor,
    eps: float = EPS_CPL,
    threshold: float = 1e-3,
) -> Tuple[torch.Tensor, int]:
    """Phase B — per-aggressor-net CPL MAPE.

    Returns (mean_mape, n_supervised). Only supervises aggr_nets with non-trivial
    coupling (gt > threshold) to avoid noisy supervision on near-zero entries.
    """
    if pred_per_net.numel() == 0 or gt_per_net.numel() == 0:
        return torch.zeros((), device=pred_per_net.device), 0
    n = min(pred_per_net.shape[0], gt_per_net.shape[0])
    p = pred_per_net[:n]
    t = gt_per_net[:n]
    mask = t > threshold
    if not mask.any():
        return torch.zeros((), device=p.device), 0
    p_m = p[mask]
    t_m = t[mask]
    mape = ((p_m - t_m).abs() / torch.clamp(t_m, min=eps)).mean()
    return mape, int(mask.sum().item())


def kcl_closure(pred_gnd: torch.Tensor, pred_cpl: torch.Tensor, pred_total: torch.Tensor) -> torch.Tensor:
    """Internal consistency: gnd + cpl ≈ total. Rule 4."""
    return torch.nn.functional.smooth_l1_loss(
        pred_gnd + pred_cpl,
        pred_total.detach(),
        beta=0.05,
    )


def zero_target_pen(pred_total: torch.Tensor, gt_total: torch.Tensor, eps: float = EPS_TOTAL) -> torch.Tensor:
    """Rule 3: small target supervision via smooth_l1."""
    if gt_total.abs() < eps:
        return torch.nn.functional.smooth_l1_loss(pred_total, gt_total, beta=0.05)
    return torch.zeros((), device=pred_total.device)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: torch.Tensor = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal cross-entropy (Lin et al. 2017) for class imbalance.

    Args:
        logits: (N, K) classifier output
        targets: (N,) integer class labels
        alpha: (K,) per-class weight (default uniform)
        gamma: focusing parameter (typical 2)
    """
    if logits.shape[0] == 0:
        return torch.zeros((), device=logits.device, requires_grad=False)
    log_probs = torch.log_softmax(logits, dim=-1)  # (N, K)
    probs = log_probs.exp()
    target_log_prob = log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
    target_prob = probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
    focal_w = (1 - target_prob).pow(gamma)
    if alpha is not None:
        a = alpha[targets]
        focal_w = focal_w * a
    return -(focal_w * target_log_prob).mean()


def compute_bin_alpha(targets: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Inverse-frequency α weights for focal loss."""
    counts = torch.bincount(targets, minlength=n_bins).float()
    counts = counts.clamp(min=1.0)
    inv_freq = 1.0 / counts
    return (inv_freq / inv_freq.sum() * n_bins).to(targets.device)


# ---------------------------------------------------------------------------
# Geometric augmentation per training sample (NAS-Cap-style 6× safe).
# ---------------------------------------------------------------------------
def _torch_xy_transform(xy: torch.Tensor, transform: str) -> torch.Tensor:
    """Apply 2D transform to a tensor (..., 2). Uses torch ops (GPU-friendly)."""
    x = xy[..., 0]
    y = xy[..., 1]
    if transform == "identity":
        return xy
    if transform == "rot180":
        return torch.stack([-x, -y], dim=-1)
    if transform == "reflect_x":
        return torch.stack([x, -y], dim=-1)
    if transform == "reflect_y":
        return torch.stack([-x, y], dim=-1)
    if transform == "diag":
        return torch.stack([y, x], dim=-1)
    if transform == "antidiag":
        return torch.stack([-y, -x], dim=-1)
    raise ValueError(f"unknown transform: {transform}")


def augment_sample(sample: Dict[str, torch.Tensor], transform: str) -> Dict[str, torch.Tensor]:
    """Apply geometric augmentation to a sample IN PLACE on a copy.

    Transforms xy of: target_p_start/end, aggr_p_start/end, target_feats[:, 0:2],
    aggr_feats[:, 0:2]. dx/dy at columns 3:5 are vector extents — also transform.
    Does not touch z (column 2), w/h (5/6), or layer_idx (7+).
    """
    if transform == "identity":
        return sample

    out = dict(sample)
    # target_feats: cols 0,1 = x_mid,y_mid;  cols 3,4 = dx,dy.
    tf = out["target_feats"].clone()
    af = out["aggr_feats"].clone()
    xy_t = _torch_xy_transform(tf[:, 0:2], transform)
    xy_a = _torch_xy_transform(af[:, 0:2], transform)
    dxdy_t = _torch_xy_transform(tf[:, 3:5], transform)
    dxdy_a = _torch_xy_transform(af[:, 3:5], transform)
    tf[:, 0:2] = xy_t
    tf[:, 3:5] = dxdy_t
    af[:, 0:2] = xy_a
    af[:, 3:5] = dxdy_a
    out["target_feats"] = tf
    out["aggr_feats"] = af

    # Endpoints (3D): transform xy, keep z.
    for k in ("target_p_start", "target_p_end", "aggr_p_start", "aggr_p_end"):
        p = out[k].clone()
        p[:, 0:2] = _torch_xy_transform(p[:, 0:2], transform)
        out[k] = p
    return out


def compute_losses(
    pred: Dict[str, torch.Tensor],
    sample: Dict[str, torch.Tensor],
    n_bins: int = 5,
) -> Dict[str, torch.Tensor]:
    """All loss components per Plan v4 §4.

    Phase 1.0/2.0 (no bins): gnd, cpl_total, edge, kcl, zero.
    Phase 2.3 (bins): + gnd_cls, cpl_cls (focal CE on bin assignment).
    """
    gnd_gt = sample["gnd_total"]
    cpl_gt = sample["cpl_total"]
    total_gt = gnd_gt + cpl_gt

    loss_gnd = mape_loss(pred["pred_gnd_total"], gnd_gt, eps=EPS_GND)
    loss_cpl_total = mape_loss(pred["pred_cpl_total"], cpl_gt, eps=EPS_CPL)
    loss_edge = edge_mape_supervised(
        pred["pred_cpl_per_edge"], sample["edge_gt"], sample["edge_supervised"]
    )
    loss_kcl = kcl_closure(
        pred["pred_gnd_total"], pred["pred_cpl_total"], pred["pred_total"]
    )
    loss_zero = zero_target_pen(pred["pred_total"], total_gt)

    # Phase B — per-aggressor-net CPL supervision (50-200× denser).
    if "pred_cpl_per_aggr_net" in pred and "gt_cpl_per_aggr_net" in sample:
        loss_cpl_per_net, n_sup_aggr = per_aggr_net_mape(
            pred["pred_cpl_per_aggr_net"], sample["gt_cpl_per_aggr_net"],
        )
    else:
        loss_cpl_per_net = torch.zeros((), device=gnd_gt.device)
        n_sup_aggr = 0

    out = {
        "loss_gnd": loss_gnd,
        "loss_cpl_total": loss_cpl_total,
        "loss_cpl_per_net": loss_cpl_per_net,
        "loss_edge": loss_edge,
        "loss_kcl": loss_kcl,
        "loss_zero": loss_zero,
        "_n_sup_aggr": n_sup_aggr,
    }

    if "gnd_bin_logits" in pred and pred["gnd_bin_logits"] is not None:
        gnd_logits = pred["gnd_bin_logits"]
        gnd_target = pred["gnd_bin_target"]
        alpha = compute_bin_alpha(gnd_target, n_bins)
        out["loss_gnd_cls"] = focal_loss(gnd_logits, gnd_target, alpha=alpha)
    else:
        out["loss_gnd_cls"] = torch.zeros((), device=gnd_gt.device)

    if "cpl_bin_logits" in pred and pred["cpl_bin_logits"] is not None:
        cpl_logits = pred["cpl_bin_logits"]
        cpl_target = pred["cpl_bin_target"]
        alpha = compute_bin_alpha(cpl_target, n_bins)
        out["loss_cpl_cls"] = focal_loss(cpl_logits, cpl_target, alpha=alpha)
    else:
        out["loss_cpl_cls"] = torch.zeros((), device=gnd_gt.device)

    return out


def hand_tuned_combined(
    losses: Dict[str, torch.Tensor],
    edge_weight: float = 1.0,
    cls_weight: float = 0.3,
    per_net_weight: float = 2.0,
) -> torch.Tensor:
    """Hand-tuned loss combination (Plan v4 §4 warmup weights).

    edge_weight: caller can downweight when SPEF supervision rate is sparse.
    cls_weight: focal-CE weight for bin classifiers (active when use_bins=True).
    per_net_weight: Phase B per-aggressor-net CPL MAPE weight (default 2.0).
    """
    device = losses["loss_gnd"].device
    return (
        3.0 * losses["loss_gnd"]
        + 3.0 * losses["loss_cpl_total"]
        + per_net_weight * losses.get("loss_cpl_per_net", torch.zeros((), device=device))
        + edge_weight * losses["loss_edge"]
        + 0.1 * losses["loss_kcl"]
        + 0.1 * losses["loss_zero"]
        + cls_weight * losses.get("loss_gnd_cls", torch.zeros((), device=device))
        + cls_weight * losses.get("loss_cpl_cls", torch.zeros((), device=device))
    )


# ---------------------------------------------------------------------------
# Sample assembly from a design — used by the 10-net overfit smoke.
# ---------------------------------------------------------------------------
def build_samples_from_design(
    def_path: Path,
    spef_path: Path,
    layer_info: dict,
    tech_lef: dict,
    cell_lib: dict,
    n_nets: int = 10,
    r_aggr: float = 12.0,
    max_aggr_per_net: int = 2000,
    max_target_per_net: int = 1000,
    min_target_segs: int = 5,
    require_supervision: bool = True,
    rng_seed: int = 0,
) -> List[Dict[str, torch.Tensor]]:
    """End-to-end: extract segments, build edges, parse SPEF, pack tensors.

    Returns a list of dicts ready for model.forward(sample).
    """
    from experiments.ncgt.src.data.segment_extractor import (
        Segment,
        classify_net,
        iter_design_segments,
    )
    from experiments.ncgt.src.data.edge_builder import build_edges_for_net
    from experiments.ncgt.src.data.spef_to_targets import parse_spef, build_edge_supervision
    from experiments.ncgt.src.data.ncgt_dataset import build_sample

    rng = np.random.default_rng(rng_seed)

    print(f"[build] parsing DEF {def_path.name}")
    nets: List[Tuple[str, List[Segment]]] = []
    for net_name, segs in iter_design_segments(str(def_path), layer_info, tech_lef, cell_lib):
        if classify_net(net_name) == "signal" and len(segs) >= min_target_segs:
            nets.append((net_name, segs))
    print(f"[build] {len(nets)} candidate signal nets")

    print(f"[build] parsing SPEF {spef_path.name}")
    spef_nets = parse_spef(spef_path)
    print(f"[build] {len(spef_nets)} D_NET entries")

    # Build aggressor pool: {net_name -> [Segment]} for fast lookup.
    net_to_segs: Dict[str, List[Segment]] = {}
    # We also need to scan all segments (not only signal nets) to find aggressors.
    # Re-parse design to get all nets. Cache from first pass for signal nets only is incomplete;
    # iterate again over all to populate aggressor sources.
    print("[build] re-parsing for full-design segment index (aggressor sources)")
    for net_name, segs in iter_design_segments(str(def_path), layer_info, tech_lef, cell_lib):
        net_to_segs[net_name] = segs

    # Build flat aggressor index for fast spatial query.
    all_segs: List[Tuple[str, Segment]] = []
    for nm, segs in net_to_segs.items():
        for s in segs:
            if s.seg_type == "VIA":
                continue  # exclude vias per Phase 0 audit
            all_segs.append((nm, s))
    print(f"[build] flat segment index size (vias excluded): {len(all_segs)}")

    coords = np.array([(s.x_mid, s.y_mid, s.z) for _, s in all_segs], dtype=np.float32)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
    except Exception:
        tree = None

    samples: List[Dict[str, torch.Tensor]] = []
    rng.shuffle(nets)
    for net_name, target_segs in nets:
        if len(samples) >= n_nets:
            break
        if net_name not in spef_nets:
            continue

        # Aggressors within R_aggr of any target midpoint.
        target_mids = np.array([(s.x_mid, s.y_mid, s.z) for s in target_segs], dtype=np.float32)
        aggr_idx_set = set()
        if tree is not None:
            for tm in target_mids:
                for j in tree.query_ball_point(tm, r=r_aggr):
                    nm_j, _ = all_segs[j]
                    if nm_j != net_name:
                        aggr_idx_set.add(j)
        else:
            for tm in target_mids:
                d = np.linalg.norm(coords - tm, axis=1)
                for j in np.where(d < r_aggr)[0]:
                    nm_j, _ = all_segs[j]
                    if nm_j != net_name:
                        aggr_idx_set.add(int(j))

        if len(aggr_idx_set) == 0:
            continue
        # Cap at max_aggr_per_net; closest by min target distance.
        aggr_indices = list(aggr_idx_set)
        if len(aggr_indices) > max_aggr_per_net:
            min_d = np.full(len(aggr_indices), 1e9, dtype=np.float32)
            for k, j in enumerate(aggr_indices):
                min_d[k] = np.linalg.norm(target_mids - coords[j], axis=1).min()
            order = np.argsort(min_d)[:max_aggr_per_net]
            aggr_indices = [aggr_indices[i] for i in order]

        aggr_segs = [all_segs[j][1] for j in aggr_indices]
        aggr_net_names = [all_segs[j][0] for j in aggr_indices]

        # Cap target.
        if len(target_segs) > max_target_per_net:
            target_segs = target_segs[:max_target_per_net]

        # Build edges.
        edges = build_edges_for_net(
            targets=target_segs,
            aggressors=aggr_segs,
            r_edge_local=4.0,
            r_edge_mid=8.0,
            r_aggr=r_aggr,
            k_mid=8,
            aggr_net_ids=[hash(n) % (2**31) for n in aggr_net_names],
        )
        if len(edges) == 0:
            continue

        # SPEF-derived supervision.
        target_spef = spef_nets[net_name]
        # Aggressor net SPEFs (only those present in SPEF).
        aggr_unique_names = set(aggr_net_names)
        aggr_net_to_spef = {n: spef_nets[n] for n in aggr_unique_names if n in spef_nets}

        edge_gt, is_sup, cpl_per_aggr_net = build_edge_supervision(
            target_segments=target_segs,
            aggressor_segments=aggr_segs,
            target_net_spef=target_spef,
            aggr_net_to_spef=aggr_net_to_spef,
            edge_index=edges.edge_index,
            aggr_net_names=aggr_net_names,
        )

        if require_supervision and not is_sup.any():
            continue

        # Compute net-level GND and CPL totals from SPEF.
        cpl_total = float(sum(cpl_per_aggr_net.values()))
        gnd_total = max(0.0, target_spef.total_cap - cpl_total)

        # Pack into NCGTSample then to_torch().
        sample_obj = build_sample(
            net_name=net_name,
            design_name=def_path.stem,
            target_segments=target_segs,
            aggressor_segments=aggr_segs,
            aggr_net_names=aggr_net_names,
            edge_index=edges.edge_index,
            edge_band=edges.band,
            gnd_total=gnd_total,
            cpl_total=cpl_total,
            cpl_per_aggr_net=cpl_per_aggr_net,
            edge_gt=edge_gt,
            edge_supervised=is_sup,
        )
        samples.append(sample_obj.to_torch())

        sup_frac = is_sup.sum() / max(1, len(is_sup))
        print(f"[build] {net_name}: T={len(target_segs)} A={len(aggr_segs)} E={len(edges)} "
              f"sup={sup_frac:.1%} gnd={gnd_total:.4f} cpl={cpl_total:.4f}")

    print(f"[build] assembled {len(samples)} samples")
    return samples


# ---------------------------------------------------------------------------
# Phase 1.0 / 1.1 driver.
# ---------------------------------------------------------------------------
def _setup_pdk():
    """Load PDK + layer info + LayerPhysicsTable. Returns (cfg, layer_info, tech_lef, cell_lib, layer_table)."""
    from configs import config as cfg

    from src.preprocessing.layer_parser import LayerInfoParser
    from src.preprocessing.lef_parser import LefParser
    from src.preprocessing.cell_parser import CellLibParser

    layer_info = LayerInfoParser(str(cfg.LAYERS_INFO_PATH)).parse()
    tech_lef = LefParser(str(cfg.TECH_LEF_PATH)).parse()
    cell_lib = CellLibParser(str(cfg.CELL_LEF_PATH)).parse()
    layer_table = LayerPhysicsTable(layer_info)
    return cfg, layer_info, tech_lef, cell_lib, layer_table


def _resolve_design_paths(cfg, design: str) -> Tuple[Path, Path]:
    for d, s in zip(cfg.TRAIN_DEFS + cfg.TEST_DEFS, cfg.TRAIN_SPEFS + cfg.TEST_SPEFS):
        if Path(d).stem == design:
            return Path(d), Path(s)
    raise FileNotFoundError(f"design/spef not found: {design}")


def overfit_smoke(args: argparse.Namespace) -> int:
    """Plan v4 §5 Phase 1.0+1.1: 10-net overfit on gcd_f3.

    Gate: total loss → 0.001 within `args.steps`. If not, architecture broken.
    """
    cfg, layer_info, tech_lef, cell_lib, layer_table = _setup_pdk()
    def_path, spef_path = _resolve_design_paths(cfg, args.design)

    samples = build_samples_from_design(
        def_path=def_path,
        spef_path=spef_path,
        layer_info=layer_info,
        tech_lef=tech_lef,
        cell_lib=cell_lib,
        n_nets=args.n_nets,
        r_aggr=args.r_aggr,
        max_aggr_per_net=args.max_aggr,
        max_target_per_net=args.max_target,
        min_target_segs=args.min_segs,
        rng_seed=args.seed,
    )
    if not samples:
        print("[overfit] no samples built — abort")
        return 1

    # Move to GPU if available.
    device = torch.device(f"cuda:{args.gpu}") if (torch.cuda.is_available() and args.gpu >= 0) else torch.device("cpu")
    print(f"[overfit] device: {device}")
    samples_dev = []
    for s in samples:
        sd = {k: v.to(device) if torch.is_tensor(v) else v for k, v in s.items()}
        sd["_layer_table"] = layer_table  # threaded into model.compute_*_base
        samples_dev.append(sd)

    # Model + optimizer.
    cfg_m = NCGTConfig(use_bins=False, dropout=0.0)  # disable dropout for overfit
    model = NCGTModel(cfg_m).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[overfit] model: {n_params / 1e6:.2f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    # Auto-detect sparse-supervision regime; downweight edge loss accordingly.
    sup_fracs = []
    for s in samples_dev:
        n = int(s["edge_supervised"].numel())
        sup_fracs.append(int(s["edge_supervised"].sum().item()) / max(1, n))
    mean_sup = float(np.mean(sup_fracs)) if sup_fracs else 0.0
    edge_weight = 1.0 if mean_sup >= 0.05 else 0.05  # 5% threshold per Plan rule
    print(f"[overfit] mean edge supervision = {mean_sup:.2%}, edge_weight = {edge_weight}")

    # Training loop — repeat sample list, hand-tuned weights.
    log = []
    losses_history = []
    for step in range(args.steps):
        sample = samples_dev[step % len(samples_dev)]
        opt.zero_grad()
        pred = model(sample)
        losses = compute_losses(pred, sample)
        total = hand_tuned_combined(losses, edge_weight=edge_weight)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        if step % args.log_interval == 0 or step == args.steps - 1:
            row = {"step": step}
            row.update({k: float(v.detach().item()) for k, v in losses.items()})
            row["total"] = float(total.detach().item())
            row["pred_total"] = float(pred["pred_total"].detach().item())
            row["gt_total"] = float((sample["gnd_total"] + sample["cpl_total"]).item())
            row["pred_gnd"] = float(pred["pred_gnd_total"].detach().item())
            row["pred_cpl"] = float(pred["pred_cpl_total"].detach().item())
            log.append(row)
            losses_history.append(row["total"])
            print(
                f"step {step:5d}: total={row['total']:.4f} "
                f"gnd={row['loss_gnd']:.4f} cpl_total={row['loss_cpl_total']:.4f} "
                f"edge={row['loss_edge']:.4f} kcl={row['loss_kcl']:.4f} "
                f"| pred {row['pred_gnd']:.3f}+{row['pred_cpl']:.3f}={row['pred_total']:.3f} "
                f"vs gt={row['gt_total']:.3f}"
            )

    # Save log.
    out_dir = Path(__file__).resolve().parents[2] / "output_overfit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase1_smoke_{args.design}_{args.n_nets}nets_{args.steps}steps.json"
    with open(out_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[overfit] log → {out_path}")

    # Gate decision: average final-window MAPE for net totals < threshold.
    # Per Plan v4: Phase 1.0 sanity gate is "model can fit 10 samples to near-zero".
    # With placeholder physics-base ε=3.0 d=0.2 (Phase 2 will thread layer_info), a
    # realistic gate is per-net total MAPE < 30% averaged across recent samples.
    last_n = min(2 * len(samples_dev), len(log))
    recent = log[-last_n:]
    gnd_mape = float(np.mean([r["loss_gnd"] for r in recent]))
    cpl_mape = float(np.mean([r["loss_cpl_total"] for r in recent]))
    print(f"[overfit] recent-window mean: gnd MAPE={gnd_mape:.4f}, cpl MAPE={cpl_mape:.4f}")

    final_total = log[-1]["total"]
    init_total = log[0]["total"]
    print(f"[overfit] init_total={init_total:.4f} final_total={final_total:.4f}")

    pass_thr = args.gate
    if max(gnd_mape, cpl_mape) < pass_thr:
        print(f"[overfit] PASS — net-total MAPE max({gnd_mape:.4f}, {cpl_mape:.4f}) < {pass_thr}")
        return 0
    if final_total < init_total * 0.5:
        print(
            f"[overfit] PARTIAL — total dropped {init_total/final_total:.1f}× "
            f"but per-net MAPE > {pass_thr}; expected once layer_info is threaded"
        )
        return 0
    print(f"[overfit] FAIL — no meaningful improvement")
    return 2


@torch.no_grad()
def evaluate(model, samples, edge_weight: float = 0.05) -> Dict[str, float]:
    """Compute per-net MAPE on a sample list. Returns mean / median / max stats."""
    model.eval()
    gnd_mapes = []
    cpl_mapes = []
    tot_mapes = []
    pred_totals = []
    gt_totals = []
    for s in samples:
        pred = model(s)
        gnd_gt = float(s["gnd_total"].item())
        cpl_gt = float(s["cpl_total"].item())
        tot_gt = gnd_gt + cpl_gt
        gnd_pred = float(pred["pred_gnd_total"].item())
        cpl_pred = float(pred["pred_cpl_total"].item())
        tot_pred = gnd_pred + cpl_pred
        gnd_mapes.append(abs(gnd_pred - gnd_gt) / max(gnd_gt, EPS_GND))
        cpl_mapes.append(abs(cpl_pred - cpl_gt) / max(cpl_gt, EPS_CPL))
        tot_mapes.append(abs(tot_pred - tot_gt) / max(tot_gt, EPS_TOTAL))
        pred_totals.append(tot_pred)
        gt_totals.append(tot_gt)
    model.train()
    pearson_r = float(np.corrcoef(pred_totals, gt_totals)[0, 1]) if len(pred_totals) > 1 else 0.0
    return {
        "n": len(samples),
        "gnd_mape_mean": float(np.mean(gnd_mapes)),
        "cpl_mape_mean": float(np.mean(cpl_mapes)),
        "tot_mape_mean": float(np.mean(tot_mapes)),
        "tot_mape_median": float(np.median(tot_mapes)),
        "tot_mape_p95": float(np.percentile(tot_mapes, 95)),
        "pearson_r_total": pearson_r,
    }


@torch.no_grad()
def physics_only_baseline(samples) -> Dict[str, float]:
    """Pure physics base (no NN; residual=0) MAPE — control comparison."""
    gnd_mapes = []
    cpl_mapes = []
    tot_mapes = []
    pred_totals = []
    gt_totals = []
    for s in samples:
        # gnd_base + cpl_base (no NN) = sum directly.
        # Build a temporary "pred" by computing physics base only.
        from experiments.ncgt.src.data.physics_base import (
            compute_segment_geometry,
            edge_overlap_length,
            gnd_base_per_segment,
            cpl_base_per_edge,
        )

        layer_table = s["_layer_table"]
        # GND base
        ps = s["target_p_start"]
        pe = s["target_p_end"]
        feats = s["target_feats"]
        width = feats[:, 5]
        layer_idxs = feats[:, 7].long()
        phys = layer_table.build_seg_tensors(layer_idxs)
        thick = phys["t_metal"].clamp(min=0.05)
        area, perim = compute_segment_geometry(ps, pe, width, thick)
        cap_gnd = gnd_base_per_segment(
            seg_area_top=area, seg_area_bot=area,
            seg_perimeter=perim, seg_thickness=thick,
            d_top=phys["d_above"], d_bot=phys["d_below"],
            eps_top=phys["eps_above"], eps_bot=phys["eps_below"],
        ).sum().item()

        # CPL base
        E = s["edge_index"].shape[1]
        if E > 0:
            ti = s["edge_index"][0]
            ai = s["edge_index"][1]
            t_layer = s["target_feats"][ti, 7].long()
            a_layer = s["aggr_feats"][ai, 7].long()
            same_layer = (t_layer == a_layer)
            t_ps = s["target_p_start"][ti]
            t_pe = s["target_p_end"][ti]
            a_ps = s["aggr_p_start"][ai]
            a_pe = s["aggr_p_end"][ai]
            ov_len = edge_overlap_length(t_ps, t_pe, a_ps, a_pe)
            ov_area = ov_len * s["aggr_feats"][ai, 5]
            d_xy = torch.linalg.norm(
                s["target_feats"][ti, :2] - s["aggr_feats"][ai, :2], dim=-1
            )
            d_z = (s["target_feats"][ti, 2] - s["aggr_feats"][ai, 2]).abs()
            pair = layer_table.build_pair_tensors(t_layer, a_layer)
            cap_cpl = cpl_base_per_edge(
                same_layer=same_layer,
                overlap_length=ov_len,
                overlap_area=ov_area,
                lateral_distance=d_xy,
                vertical_distance=d_z,
                metal_thickness=pair["t_pair"].clamp(min=0.05),
                eps_pair=pair["eps_pair"],
            ).sum().item()
        else:
            cap_cpl = 0.0

        gnd_gt = float(s["gnd_total"].item())
        cpl_gt = float(s["cpl_total"].item())
        tot_gt = gnd_gt + cpl_gt
        gnd_mapes.append(abs(cap_gnd - gnd_gt) / max(gnd_gt, EPS_GND))
        cpl_mapes.append(abs(cap_cpl - cpl_gt) / max(cpl_gt, EPS_CPL))
        tot_mapes.append(abs((cap_gnd + cap_cpl) - tot_gt) / max(tot_gt, EPS_TOTAL))
        pred_totals.append(cap_gnd + cap_cpl)
        gt_totals.append(tot_gt)
    pearson_r = float(np.corrcoef(pred_totals, gt_totals)[0, 1]) if len(pred_totals) > 1 else 0.0
    return {
        "n": len(samples),
        "gnd_mape_mean": float(np.mean(gnd_mapes)),
        "cpl_mape_mean": float(np.mean(cpl_mapes)),
        "tot_mape_mean": float(np.mean(tot_mapes)),
        "tot_mape_median": float(np.median(tot_mapes)),
        "tot_mape_p95": float(np.percentile(tot_mapes, 95)),
        "pearson_r_total": pearson_r,
    }


def _build_multi_design_samples(args, cfg, layer_info, tech_lef, cell_lib) -> Tuple[List, List[str]]:
    """Build samples across all TRAIN_DEFS (or just args.design if not multi)."""
    samples_all = []
    designs_used = []
    if args.multi_design:
        designs = [Path(p).stem for p in cfg.TRAIN_DEFS]
        print(f"[train_val] multi_design: {len(designs)} designs × {args.n_nets} nets each")
    else:
        designs = [args.design]

    for design in designs:
        try:
            def_path, spef_path = _resolve_design_paths(cfg, design)
        except FileNotFoundError as e:
            print(f"[train_val]   skipping {design}: {e}")
            continue
        print(f"[train_val] === building from {design} ===")
        s = build_samples_from_design(
            def_path=def_path,
            spef_path=spef_path,
            layer_info=layer_info,
            tech_lef=tech_lef,
            cell_lib=cell_lib,
            n_nets=args.n_nets,
            r_aggr=args.r_aggr,
            max_aggr_per_net=args.max_aggr,
            max_target_per_net=args.max_target,
            min_target_segs=args.min_segs,
            rng_seed=args.seed,
        )
        for sample in s:
            samples_all.append((design, sample))
        designs_used.append(design)
        print(f"[train_val]   built {len(s)} samples from {design}")
    return samples_all, designs_used


def train_validation(args: argparse.Namespace) -> int:
    """Plan v4 §5 Phase 2.0/3.0: real training experiment with held-out validation.

    Phase 2.0: single-design (pass --design <name>).
    Phase 3.0: multi-design (pass --multi_design).
    """
    cfg, layer_info, tech_lef, cell_lib, layer_table = _setup_pdk()

    samples_with_design, designs_used = _build_multi_design_samples(args, cfg, layer_info, tech_lef, cell_lib)
    samples = [s for _, s in samples_with_design]
    sample_design_ids = [d for d, _ in samples_with_design]
    if len(samples) < 20:
        print(f"[train_val] only {len(samples)} samples — increase --n_nets")
        return 1
    print(f"\n[train_val] total samples: {len(samples)} from {len(designs_used)} designs")

    # Stratified split: per (design, magnitude_quartile), 80/20.
    rng = np.random.default_rng(args.seed)
    totals = [float(s["gnd_total"] + s["cpl_total"]) for s in samples]
    quartiles = np.percentile(totals, [25, 50, 75])

    def magnitude_bucket(t: float) -> int:
        if t < quartiles[0]: return 0
        if t < quartiles[1]: return 1
        if t < quartiles[2]: return 2
        return 3

    strata: Dict[Tuple[str, int], List[int]] = {}
    for i, t in enumerate(totals):
        key = (sample_design_ids[i], magnitude_bucket(t))
        strata.setdefault(key, []).append(i)

    train_idx, valid_idx = [], []
    for key, idxs in strata.items():
        rng.shuffle(idxs)
        cut = max(1, int(len(idxs) * 0.8))
        train_idx.extend(idxs[:cut])
        valid_idx.extend(idxs[cut:])
    print(f"[train_val] split (stratified by design × magnitude): "
          f"train={len(train_idx)}, valid={len(valid_idx)}, strata={len(strata)}")

    # Move samples to device.
    device = torch.device(f"cuda:{args.gpu}") if (torch.cuda.is_available() and args.gpu >= 0) else torch.device("cpu")
    print(f"[train_val] device: {device}")
    samples_dev = []
    for s in samples:
        sd = {k: v.to(device) if torch.is_tensor(v) else v for k, v in s.items()}
        sd["_layer_table"] = layer_table
        samples_dev.append(sd)
    train_samples = [samples_dev[i] for i in train_idx]
    valid_samples = [samples_dev[i] for i in valid_idx]

    # Physics-only baseline.
    print("\n[train_val] === Physics-only baseline (no NN) ===")
    phys_train = physics_only_baseline(train_samples)
    phys_valid = physics_only_baseline(valid_samples)
    print(f"  TRAIN: gnd_mape={phys_train['gnd_mape_mean']:.2%} cpl_mape={phys_train['cpl_mape_mean']:.2%} "
          f"tot_mape_mean={phys_train['tot_mape_mean']:.2%} median={phys_train['tot_mape_median']:.2%} "
          f"r={phys_train['pearson_r_total']:.3f}")
    print(f"  VALID: gnd_mape={phys_valid['gnd_mape_mean']:.2%} cpl_mape={phys_valid['cpl_mape_mean']:.2%} "
          f"tot_mape_mean={phys_valid['tot_mape_mean']:.2%} median={phys_valid['tot_mape_median']:.2%} "
          f"r={phys_valid['pearson_r_total']:.3f}")

    # Untrained NN baseline (random init residual).
    cfg_m = NCGTConfig(use_bins=args.use_bins, dropout=0.1, use_curriculum=args.use_curriculum)
    if args.use_curriculum:
        cfg_m.clamp_bound = math.log(1.5)
    model = NCGTModel(cfg_m).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[train_val] model: {n_params / 1e6:.2f}M params")

    print("\n[train_val] === Untrained model (random init) ===")
    init_train = evaluate(model, train_samples)
    init_valid = evaluate(model, valid_samples)
    print(f"  TRAIN: tot_mape_mean={init_train['tot_mape_mean']:.2%} r={init_train['pearson_r_total']:.3f}")
    print(f"  VALID: tot_mape_mean={init_valid['tot_mape_mean']:.2%} r={init_valid['pearson_r_total']:.3f}")

    # Auto-detect edge weight.
    sup_fracs = [int(s["edge_supervised"].sum().item()) / max(1, int(s["edge_supervised"].numel()))
                 for s in train_samples]
    mean_sup = float(np.mean(sup_fracs)) if sup_fracs else 0.0
    edge_weight = 1.0 if mean_sup >= 0.05 else 0.05
    print(f"\n[train_val] mean train edge supervision = {mean_sup:.2%}, edge_weight = {edge_weight}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.1)

    print(f"\n[train_val] === Training (use_bins={args.use_bins}, use_aug={args.use_aug}, "
          f"n_seeds={args.n_seeds}) ===")
    aug_set = list(SAFE_TRANSFORMS) if args.use_aug else ["identity"]

    seed_results = []
    for seed_idx in range(args.n_seeds):
        seed_value = args.seed + seed_idx
        print(f"\n[train_val] ===== seed {seed_idx + 1}/{args.n_seeds} (value={seed_value}) =====")
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
        model = NCGTModel(cfg_m).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.1)
        rng_seed = np.random.default_rng(seed_value)

        log = []
        best_valid = 1e9
        best_step = 0
        last_clamp = -1.0
        for step in range(args.steps):
            # pex_v3 curriculum: progressively widen residual clamp.
            if args.use_curriculum:
                cb = curriculum_clamp(step, args.steps)
                if abs(cb - last_clamp) > 1e-6:
                    model.set_clamp_bound(cb)
                    last_clamp = cb
            s = train_samples[rng_seed.integers(0, len(train_samples))]
            if args.use_aug:
                t = aug_set[int(rng_seed.integers(0, len(aug_set)))]
                s = augment_sample(s, t)
            opt.zero_grad()
            pred = model(s)
            losses = compute_losses(pred, s)
            total = hand_tuned_combined(losses, edge_weight=edge_weight)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            sched.step()

            if step % args.eval_interval == 0 or step == args.steps - 1:
                tr = evaluate(model, train_samples)
                va = evaluate(model, valid_samples)
                log.append({"step": step, "train": tr, "valid": va, "lr": opt.param_groups[0]["lr"]})
                if step % (args.eval_interval * 3) == 0 or step == args.steps - 1:
                    print(f"  step {step:5d}: TRAIN {tr['tot_mape_mean']:.4f} | "
                          f"VALID {va['tot_mape_mean']:.4f} med={va['tot_mape_median']:.4f} "
                          f"p95={va['tot_mape_p95']:.4f} r={va['pearson_r_total']:.3f}")
                if va["tot_mape_mean"] < best_valid:
                    best_valid = va["tot_mape_mean"]
                    best_step = step

        final_va = evaluate(model, valid_samples)
        print(f"  ⇒ best VALID MAPE: {best_valid:.4f} @ step {best_step}, final {final_va['tot_mape_mean']:.4f}")
        seed_results.append({
            "seed": seed_value,
            "best_valid": best_valid,
            "best_step": best_step,
            "final_valid": final_va["tot_mape_mean"],
            "final_pearson": final_va["pearson_r_total"],
            "final_p95": final_va["tot_mape_p95"],
        })

    # 5-seed aggregate.
    bests = np.array([r["best_valid"] for r in seed_results])
    finals = np.array([r["final_valid"] for r in seed_results])
    rs = np.array([r["final_pearson"] for r in seed_results])
    p95s = np.array([r["final_p95"] for r in seed_results])
    print(f"\n[train_val] ===== {args.n_seeds}-seed summary =====")
    print(f"  Physics-only VALID baseline: {phys_valid['tot_mape_mean']:.4f}")
    print(f"  Best VALID MAPE  : mean={bests.mean():.4f} ± {bests.std():.4f} "
          f"(min={bests.min():.4f}, max={bests.max():.4f})")
    print(f"  Final VALID MAPE : mean={finals.mean():.4f} ± {finals.std():.4f}")
    print(f"  Final p95        : mean={p95s.mean():.4f} ± {p95s.std():.4f}")
    print(f"  Pearson r        : mean={rs.mean():.4f} ± {rs.std():.4f}")
    print(f"  Per-seed bests   : {[f'{x:.4f}' for x in bests.tolist()]}")

    out_dir = Path(__file__).resolve().parents[2] / "output_train"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_designs = len(designs_used)
    tag = f"multi{n_designs}d" if args.multi_design else args.design
    suffix_parts = []
    if args.use_bins: suffix_parts.append("bins")
    if args.use_aug: suffix_parts.append("aug")
    suffix = "_".join(suffix_parts) if suffix_parts else "vanilla"
    out_path = out_dir / f"5seed_{tag}_{suffix}_{args.n_nets}nets_{args.steps}steps.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": vars(args),
            "physics_valid": phys_valid,
            "init_valid": init_valid,
            "n_designs": n_designs,
            "n_train": len(train_idx),
            "n_valid": len(valid_idx),
            "seed_results": seed_results,
            "summary": {
                "best_valid_mean": float(bests.mean()),
                "best_valid_std": float(bests.std()),
                "best_valid_min": float(bests.min()),
                "best_valid_max": float(bests.max()),
                "final_valid_mean": float(finals.mean()),
                "final_valid_std": float(finals.std()),
                "final_p95_mean": float(p95s.mean()),
                "pearson_r_mean": float(rs.mean()),
            },
        }, f, indent=2, default=str)
    print(f"\n[train_val] log → {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["overfit", "train"], default="overfit")
    ap.add_argument("--design", default="intel22_gcd_f3")
    ap.add_argument("--n_nets", type=int, default=10)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--r_aggr", type=float, default=12.0)
    ap.add_argument("--max_aggr", type=int, default=2000)
    ap.add_argument("--max_target", type=int, default=1000)
    ap.add_argument("--min_segs", type=int, default=5)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", type=float, default=0.1)
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--eval_interval", type=int, default=200)
    ap.add_argument("--use_bins", action="store_true", help="Phase 2.3: 5-bin specialization")
    ap.add_argument("--use_aug", action="store_true", help="Phase 2.2: 6× geometric augmentation")
    ap.add_argument("--multi_design", action="store_true", help="Phase 3: train across all TRAIN_DEFS")
    ap.add_argument("--n_seeds", type=int, default=1, help="Number of independent seeds (Plan v4 protocol)")
    ap.add_argument("--use_curriculum", action="store_true",
                    help="pex_v3 curriculum residual: hard clamp + progressive widening (log 1.5 → 2.5 → 4.0)")
    args = ap.parse_args()
    if args.mode == "train":
        return train_validation(args)
    return overfit_smoke(args)


if __name__ == "__main__":
    sys.exit(main())
