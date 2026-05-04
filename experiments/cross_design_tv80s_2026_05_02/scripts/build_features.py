"""
Build per-net feature parquet for each design.

Walks the v3 manifest, groups by (design, net_name), pools tile pkls, and
emits experiments/.../cache/features/<design>.parquet with columns:
    design_name, net_name, split, total_cap_fF, c_gnd_fF, c_cpl_total_fF,
    total_res_ohm, n_aggressors_spef, cpl_p95_fF, <feature_columns...>

Parallel via multiprocessing. Skips designs whose parquet already exists
(idempotent). Re-run with `--force` to overwrite.
"""
from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS))

from configs import cfg
from src.feat_extract import (
    extract_features_for_net,
    init_layer_breaks_from_design,
    FEATURE_NAMES,
)
from src.spef_parser import parse_spef_to_targets


def _process_one_net(args):
    pkl_paths_str, net_name = args
    paths = [Path(p) for p in pkl_paths_str]
    try:
        feat = extract_features_for_net(paths, cutoff_um=cfg.CPL_CUTOFF_UM)
    except Exception:
        traceback.print_exc()
        return None
    if feat is None:
        return None
    feat["net_name"] = net_name
    return feat


def build_design(design: str, manifest: pd.DataFrame, force: bool = False, n_workers: int = 16):
    out_dir = cfg.CACHE_DIR / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{design}.parquet"
    if out_path.exists() and not force:
        print(f"[{design}] cached → skip")
        return out_path

    sub = manifest[manifest["design_name"] == design]
    if sub.empty:
        print(f"[{design}] no rows in manifest")
        return None

    # Group pkl paths by net
    sub = sub.copy()
    sub["abs_path"] = str(cfg.DATA_ROOT) + "/" + sub["rel_path"].astype(str)
    grouped = sub.groupby("net_name")["abs_path"].apply(list)

    # Init layer breakpoints from this design's pkls
    init_layer_breaks_from_design(cfg.DATA_ROOT / design, sample_n_files=200)

    # Find SPEF
    spef_path = cfg.SPEF_DIR / f"{design}_starrc.spef"
    if not spef_path.exists():
        print(f"[{design}] missing SPEF at {spef_path}")
        return None
    print(f"[{design}] parsing SPEF: {spef_path.name}")
    t0 = time.time()
    spef_targets = parse_spef_to_targets(spef_path)
    print(f"[{design}]   spef parse: {time.time()-t0:.1f}s, {len(spef_targets)} nets")

    # Filter to nets present in both manifest & SPEF
    spef_keys = set(spef_targets.keys())
    keep_nets = [n for n in grouped.index if n in spef_keys]
    print(f"[{design}] {len(grouped)} manifest nets → {len(keep_nets)} after SPEF-join")
    if not keep_nets:
        return None

    work = [(grouped[n], n) for n in keep_nets]

    rows = []
    t0 = time.time()
    with mp.Pool(processes=n_workers) as pool:
        for i, feat in enumerate(pool.imap_unordered(_process_one_net, work, chunksize=4)):
            if feat is None:
                continue
            tgt = spef_targets[feat["net_name"]]
            row = {
                "design_name": design,
                "net_name": feat["net_name"],
                "split": sub[sub["net_name"] == feat["net_name"]]["split"].iloc[0],
                "total_cap_fF": tgt["total_cap_fF"],
                "c_gnd_fF": tgt["c_gnd_fF"],
                "c_cpl_total_fF": tgt["c_cpl_total_fF"],
                "total_res_ohm": tgt["total_res_ohm"],
                "n_aggressors_spef": tgt["n_aggressors_spef"],
                "cpl_p95_fF": tgt["cpl_p95_fF"],
            }
            for k in FEATURE_NAMES:
                row[k] = feat.get(k, np.nan)
            rows.append(row)
            if (i + 1) % 500 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(work) - i - 1) / max(rate, 1e-3)
                print(f"[{design}]   {i+1}/{len(work)} nets  {rate:.1f}/s  eta {eta:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    print(f"[{design}] writing {len(df)} rows → {out_path}")
    df.to_parquet(out_path, index=False)
    print(f"[{design}] done in {time.time()-t0:.1f}s")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", nargs="+", default=None,
                    help="defaults to ALL_DESIGNS")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    designs = args.designs or cfg.ALL_DESIGNS

    print(f"loading manifest: {cfg.MANIFEST_PATH}")
    manifest = pd.read_csv(cfg.MANIFEST_PATH)
    print(f"manifest rows: {len(manifest):,}")
    print(f"design counts:\n{manifest['design_name'].value_counts()}")

    for d in designs:
        try:
            build_design(d, manifest, force=args.force, n_workers=args.workers)
        except Exception:
            traceback.print_exc()
        gc.collect()


if __name__ == "__main__":
    main()
