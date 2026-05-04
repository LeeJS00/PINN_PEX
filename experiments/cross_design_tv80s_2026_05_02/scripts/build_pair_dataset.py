"""Build per-pair training dataset for ParaGraph-style edge regression.

For each train design:
  1. Parse SPEF coupling pairs → ground-truth c_pair_fF for each (target, agg)
  2. For each net, enumerate geometric pairs (cutoff=4μm) using cuboid pkls
  3. Match: for geometric pairs that have SPEF labels, use the label
            for geometric pairs without SPEF labels, label = 0 (assume below
            SPEF threshold)
  4. Save per-design parquet with feature columns + label

Then concat all train designs into one big training set.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS))

from configs import cfg
from src.per_pair_features import extract_pairs_for_net


_p_dnet = re.compile(r"\*D_NET\s+(\S+)\s+([0-9.eE+\-]+)")
_p_section = re.compile(r"\*(CAP|RES|END)")


def _norm(s):
    return s.replace("\\", "").strip() if s else ""


def parse_spef_coupling_pairs(spef_path: Path) -> dict:
    """Returns {(target_net, aggressor_net): c_pair_fF}.
    Note: SPEF stores each pair under one net's coupling section.
    """
    out = {}
    section = None
    cur_net = None
    with open(spef_path, "r") as f:
        for line in f:
            s = line.rstrip()
            if not s: continue
            m = _p_dnet.match(s)
            if m:
                cur_net = _norm(m.group(1))
                section = None
                continue
            if cur_net is None: continue
            sm = _p_section.match(s.strip())
            if sm:
                tag = sm.group(1)
                section = None if tag == "END" else tag
                continue
            if section != "CAP": continue
            tokens = s.split()
            if len(tokens) >= 4 and tokens[0].rstrip(":").isdigit():
                # Coupling: id node1 node2 val
                try:
                    val = float(tokens[3])
                except ValueError:
                    continue
                n1 = _norm(tokens[1].split(":")[0])
                n2 = _norm(tokens[2].split(":")[0])
                aggressor = n2 if n1 == cur_net else n1
                key = (cur_net, aggressor)
                # Multiple pair entries may exist; sum
                out[key] = out.get(key, 0.0) + val
    return out


def _proc_one_net(args):
    pkls_str, target_name = args
    paths = [Path(p) for p in pkls_str]
    pair_features, target_summary = extract_pairs_for_net(paths, cutoff_um=cfg.CPL_CUTOFF_UM, target_net_name=target_name)
    rows = []
    for agg_name, feats in pair_features.items():
        feats["target_net"] = target_name
        feats["aggressor_net"] = agg_name
        rows.append(feats)
    return rows


def build_design(design: str, manifest: pd.DataFrame, force: bool, n_workers: int):
    out_dir = cfg.CACHE_DIR / "pair_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{design}.parquet"
    if out_path.exists() and not force:
        print(f"[{design}] cached → skip"); return out_path

    sub = manifest[manifest["design_name"] == design]
    if sub.empty: return None
    sub = sub.copy()
    sub["abs_path"] = str(cfg.DATA_ROOT) + "/" + sub["rel_path"].astype(str)
    grouped = sub.groupby("net_name")["abs_path"].apply(list)

    spef_path = cfg.SPEF_DIR / f"{design}_starrc.spef"
    if not spef_path.exists():
        print(f"[{design}] missing spef"); return None
    print(f"[{design}] parsing SPEF coupling: {spef_path.name}")
    t0 = time.time()
    pair_labels = parse_spef_coupling_pairs(spef_path)
    print(f"[{design}]   {len(pair_labels)} pairs in SPEF, took {time.time()-t0:.1f}s")

    work = [(grouped[n], n) for n in grouped.index]
    t0 = time.time()
    all_rows = []
    with mp.Pool(processes=n_workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(_proc_one_net, work, chunksize=4)):
            all_rows.extend(rows)
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"[{design}]  {i+1}/{len(work)} nets  {rate:.1f}/s", flush=True)

    if not all_rows:
        print(f"[{design}] no rows")
        return None

    df = pd.DataFrame(all_rows)
    # Match labels
    labels = []
    for r in all_rows:
        key = (r["target_net"], r["aggressor_net"])
        labels.append(pair_labels.get(key, 0.0))
    df["c_pair_fF"] = labels
    df["design_name"] = design

    df.to_parquet(out_path, index=False)
    n_with_label = (df["c_pair_fF"] > 0).sum()
    print(f"[{design}] → {out_path}  {len(df)} pairs, {n_with_label} with SPEF labels in {time.time()-t0:.1f}s")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    designs = args.designs or cfg.ALL_DESIGNS

    manifest = pd.read_csv(cfg.MANIFEST_PATH)
    for d in designs:
        try:
            build_design(d, manifest, args.force, args.workers)
        except Exception:
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
