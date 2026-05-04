#!/usr/bin/env python3
"""44_per_pair_calibrate_spef.py — overwrite per-pair c_cpl in SPEF using
the trained per-pair residual model, while preserving each net's cpl_total.

Pipeline:
  1. Stream input SPEF, collect per-net (target, aggressor) → c_cpl entries.
  2. For each net, compute per-pair predictions via per_pair_residual.predict_per_pair.
  3. Rescale predictions per net so Σ pred = current cpl_total per net (KCL preserved).
  4. Write new SPEF with overwritten c_cpl values.

Inputs:
  --in-spef:    e.g. v10's HERO output
  --design:     e.g. intel22_tv80s_f3
  --topo-dir:   /data/PINNPEX/.../intel22_tv80s_f3/topology
  --model:      pex_v3/.../residual_model.lgb
  --out-spef:   output path

Wall-clock target: ≤ 20 s on tv80s.
"""
from __future__ import annotations
import argparse
import gzip
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pex_v3" / "joint_pareto" / "allocators" / "cpl"))

from configs.config_v3 import LAYERS_INFO_PATH
from src.preprocessing.layer_parser import LayerInfoParser
import per_pair_residual as ppr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-spef", type=Path, required=True)
    p.add_argument("--design", type=str, required=True)
    p.add_argument("--topo-dir", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out-spef", type=Path, required=True)
    return p.parse_args()


def load_design_segments(topo_dir: Path):
    paths = sorted(topo_dir.glob("*.pkl.gz"))
    out = {}
    for p in paths:
        try:
            with gzip.open(p, "rb") as f:
                d = pickle.load(f)
        except Exception:
            continue
        gs = d.get("global_segments", [])
        own_net = None
        for s in gs:
            if "net_name" in s and s.get("type") == "WIRE":
                own_net = s["net_name"]
                break
        if own_net is None:
            continue
        own_segs = [s for s in gs if s.get("type") == "WIRE" and s.get("net_name") == own_net]
        arr = ppr._segs_to_arr(own_segs)
        if arr is not None:
            out[own_net] = arr
    return out


def collect_per_net_pairs(in_spef: Path):
    """Stream SPEF; for each D_NET, collect aggressor → c_cpl_current.
    Returns dict: net -> dict(aggressor_net -> running c_pair_current_fF)
    """
    out = {}
    current = None
    in_cap = False
    cur = defaultdict(float)
    with open(in_spef) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("//"):
                continue
            if s.startswith("*D_NET"):
                if current is not None:
                    out[current] = dict(cur)
                cur = defaultdict(float)
                current = s.split()[1]
                in_cap = False
                continue
            if s == "*CAP":
                in_cap = True
                continue
            if s == "*RES" or s.startswith("*END"):
                if current is not None and s.startswith("*END"):
                    out[current] = dict(cur)
                    cur = defaultdict(float)
                    current = None
                in_cap = False
                continue
            if not in_cap or current is None:
                continue
            parts = s.split()
            if len(parts) >= 4 and ":" in parts[2]:
                try:
                    c_val = float(parts[3])
                except ValueError:
                    continue
                agg_node = parts[2]
                if ":" in agg_node:
                    head, _, _ = agg_node.rpartition(":")
                    agg_net = head
                else:
                    agg_net = agg_node
                cur[agg_net] += c_val
    if current is not None:
        out[current] = dict(cur)
    return out


def compute_pair_scales(per_net_pairs, nets, metal_props, booster):
    """For each (target, aggressor) listed in per_net_pairs, predict c_pair via
    the residual model. Then per-net rescale so Σ pred = current cpl_total.
    Returns: dict {(target_net, aggressor_net): pred_c_pair_fF_rescaled}
    """
    rows = []
    keys = []
    for tgt, agg_dict in per_net_pairs.items():
        if tgt not in nets:
            continue
        for agg in agg_dict.keys():
            if agg not in nets:
                continue
            if agg == tgt:
                continue
            feat = ppr.extract_pair_features_fast(nets[tgt], nets[agg], metal_props, cutoff_um=8.0)
            if feat is None:
                continue
            feat["c_analytic_pair_fF"] = feat["c_analytic_pair_fF"]
            keys.append((tgt, agg))
            rows.append(feat)
    if not rows:
        return {}, {}
    df = pd.DataFrame(rows)
    pred = ppr.predict_per_pair(df, booster)
    raw_pred = {k: float(p) for k, p in zip(keys, pred)}

    # Per-net rescale: Σ pred over (tgt, agg) must equal current cpl_total per tgt
    # Group by tgt
    final = {}
    skipped = 0
    rescale_stats = []
    for tgt, agg_dict in per_net_pairs.items():
        cur_total = sum(agg_dict.values())
        if cur_total <= 0:
            continue
        # Sum of predictions for THIS tgt's pairs (only those we predicted)
        pred_for_tgt = {agg: raw_pred[(tgt, agg)] for agg in agg_dict if (tgt, agg) in raw_pred}
        if not pred_for_tgt:
            skipped += 1
            continue
        sum_pred = sum(pred_for_tgt.values())
        if sum_pred <= 0:
            skipped += 1
            continue
        scale = cur_total / sum_pred
        # We rescale ONLY the predicted aggressors. Aggressors without a prediction
        # keep their original c_pair, then we re-rescale globally to preserve total.
        # Implementation: assign pred_for_tgt[agg] * scale to predicted ones; for
        # non-predicted, retain agg_dict[agg] unchanged. Then compute the actual
        # final sum and apply a uniform correction.
        new_caps = {}
        for agg, v in agg_dict.items():
            if agg in pred_for_tgt:
                new_caps[agg] = pred_for_tgt[agg] * scale
            else:
                new_caps[agg] = v
        new_sum = sum(new_caps.values())
        if new_sum > 0 and abs(new_sum - cur_total) > 1e-9:
            corr = cur_total / new_sum
            for k in new_caps:
                new_caps[k] *= corr
        for agg, v in new_caps.items():
            final[(tgt, agg)] = v
        rescale_stats.append((tgt, scale, cur_total, sum_pred))
    return final, {"skipped_nets": skipped, "n_rescaled_nets": len(rescale_stats)}


def rewrite_spef(in_spef: Path, out_spef: Path, final_pairs):
    """Stream rewrite — overwrite each *CAP coupling line's c_val using final_pairs."""
    rewritten = unchanged = 0
    current = None
    in_cap = False
    with open(in_spef) as fin, open(out_spef, "w") as fout:
        for line in fin:
            s = line.rstrip("\n")
            stripped = s.strip()
            if stripped.startswith("*D_NET"):
                current = stripped.split()[1]
                in_cap = False
                fout.write(line)
                continue
            if stripped == "*CAP":
                in_cap = True
                fout.write(line)
                continue
            if stripped == "*RES" or stripped.startswith("*END"):
                in_cap = False
                if stripped.startswith("*END"):
                    current = None
                fout.write(line)
                continue
            if not in_cap or current is None or not stripped:
                fout.write(line)
                continue
            parts = stripped.split()
            if len(parts) >= 4 and ":" in parts[2]:
                # Coupling: <id> <tgt_node> <agg_node> <c_val>
                try:
                    val = float(parts[3])
                except ValueError:
                    fout.write(line)
                    continue
                agg_node = parts[2]
                if ":" in agg_node:
                    head, _, _ = agg_node.rpartition(":")
                    agg_net = head
                else:
                    agg_net = agg_node
                key = (current, agg_net)
                if key in final_pairs:
                    new_val = final_pairs[key]
                    # Many *CAP lines for one (tgt, agg) pair (one per node). We need to
                    # split the new total proportionally to the original line shares.
                    # We do this in a 2-pass approach below; for simplicity, use share:
                    # new_val_for_line = original_line_val * (new_total / original_total).
                    # This factor is computed as final_pairs[key] / per_net_pairs_orig[key].
                    # See refined helper below; here we approximate by per-line proportional.
                    fout.write(line)
                    # The actual rewrite happens in pass-2 (see main).
                    rewritten += 1
                else:
                    fout.write(line)
                    unchanged += 1
            else:
                fout.write(line)
    return rewritten, unchanged


def rewrite_spef_proportional(in_spef: Path, out_spef: Path, final_pairs, per_net_orig):
    """Stream rewrite with per-line proportional split.
    For each (tgt, agg) pair, compute scale = final_pairs[(tgt,agg)] / per_net_orig[tgt][agg].
    Apply scale to every *CAP line where node_a∈tgt and node_b∈agg.
    Also recompute *D_NET total to keep KCL.
    """
    rewritten = unchanged = 0
    current = None
    in_cap = False
    cur_gnd_sum = 0.0
    cur_cpl_sum = 0.0
    # Two-pass: first compute new per-net total = sum(final_pairs over agg) + sum(gnd unchanged)
    # We can do this in a single pass if we BUFFER each net's lines then flush at *END.
    buf_lines = []
    new_dnet_total = 0.0
    with open(in_spef) as fin, open(out_spef, "w") as fout:
        for line in fin:
            s = line.rstrip("\n")
            stripped = s.strip()
            if stripped.startswith("*D_NET"):
                # Flush previous if any
                if current is not None:
                    # Update D_NET line with new total
                    for i, bl in enumerate(buf_lines):
                        if bl.strip().startswith("*D_NET"):
                            buf_lines[i] = f"*D_NET {current} {new_dnet_total:.6f}\n"
                            break
                    fout.writelines(buf_lines)
                buf_lines = [line]
                current = stripped.split()[1]
                in_cap = False
                cur_gnd_sum = 0.0
                cur_cpl_sum = 0.0
                new_dnet_total = 0.0
                continue
            if stripped == "*CAP":
                in_cap = True
                buf_lines.append(line)
                continue
            if stripped == "*RES" or stripped.startswith("*END"):
                in_cap = False
                buf_lines.append(line)
                if stripped.startswith("*END"):
                    # Update D_NET line with new total
                    new_dnet_total = cur_gnd_sum + cur_cpl_sum
                    for i, bl in enumerate(buf_lines):
                        if bl.strip().startswith("*D_NET"):
                            buf_lines[i] = f"*D_NET {current} {new_dnet_total:.6f}\n"
                            break
                    fout.writelines(buf_lines)
                    buf_lines = []
                    current = None
                continue
            if not in_cap or current is None or not stripped:
                buf_lines.append(line)
                continue
            parts = stripped.split()
            try:
                if ":" in parts[2]:
                    # Coupling
                    val = float(parts[3])
                    agg_node = parts[2]
                    if ":" in agg_node:
                        head, _, _ = agg_node.rpartition(":")
                        agg_net = head
                    else:
                        agg_net = agg_node
                    key = (current, agg_net)
                    if key in final_pairs:
                        orig_pair_total = per_net_orig.get(current, {}).get(agg_net, val)
                        if orig_pair_total > 1e-12:
                            scale = final_pairs[key] / orig_pair_total
                        else:
                            scale = 1.0
                        new_val = val * scale
                        cur_cpl_sum += new_val
                        buf_lines.append(f"{parts[0]} {parts[1]} {parts[2]} {new_val:.6f}\n")
                        rewritten += 1
                    else:
                        cur_cpl_sum += val
                        buf_lines.append(line)
                        unchanged += 1
                else:
                    val = float(parts[2])
                    cur_gnd_sum += val
                    buf_lines.append(line)
            except (ValueError, IndexError):
                buf_lines.append(line)
        # Final flush
        if buf_lines:
            new_dnet_total = cur_gnd_sum + cur_cpl_sum
            if current is not None:
                for i, bl in enumerate(buf_lines):
                    if bl.strip().startswith("*D_NET"):
                        buf_lines[i] = f"*D_NET {current} {new_dnet_total:.6f}\n"
                        break
            fout.writelines(buf_lines)
    return rewritten, unchanged


def main():
    args = parse_args()
    layer_info = LayerInfoParser(LAYERS_INFO_PATH).parse()
    metal_props = ppr.metal_layer_props(layer_info)

    print(f">>> in_spef: {args.in_spef}")
    print(f">>> model:   {args.model}")
    t0 = time.time()
    print(">>> loading topology segments")
    nets = load_design_segments(args.topo_dir)
    print(f"    {len(nets)} nets in {time.time()-t0:.1f}s")

    t1 = time.time()
    print(">>> collecting per-net pair table from SPEF")
    per_net_pairs = collect_per_net_pairs(args.in_spef)
    n_pairs_orig = sum(len(v) for v in per_net_pairs.values())
    print(f"    {len(per_net_pairs)} D_NET blocks, {n_pairs_orig:,} (target,aggressor) entries, {time.time()-t1:.1f}s")

    t2 = time.time()
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(args.model))
    print(f"    loaded model in {time.time()-t2:.2f}s")

    t3 = time.time()
    print(">>> predicting per-pair residuals + per-net rescale")
    final_pairs, stats = compute_pair_scales(per_net_pairs, nets, metal_props, booster)
    print(f"    {len(final_pairs):,} final pair predictions ({stats}), {time.time()-t3:.1f}s")

    t4 = time.time()
    print(">>> rewriting SPEF")
    rew, unc = rewrite_spef_proportional(args.in_spef, args.out_spef, final_pairs, per_net_pairs)
    print(f"    rewrote {rew:,} *CAP lines (unchanged {unc:,}), {time.time()-t4:.1f}s")
    print(f"✅ {args.out_spef} (total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
