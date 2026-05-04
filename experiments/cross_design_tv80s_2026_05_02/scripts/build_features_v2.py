"""Build v2 per-net feature parquets (improved layer mapping + extra features)."""
from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
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
from src.feat_extract_v2 import extract_features_for_net_v2, FEATURE_NAMES_V2
from src.spef_parser import parse_spef_to_targets


def _proc_one(args):
    pkls, name = args
    try:
        f = extract_features_for_net_v2([Path(p) for p in pkls], cutoff_um=cfg.CPL_CUTOFF_UM)
    except Exception:
        traceback.print_exc()
        return None
    if f is None:
        return None
    f["net_name"] = name
    return f


def build_design_v2(design: str, manifest: pd.DataFrame, force: bool, n_workers: int):
    out_dir = cfg.CACHE_DIR / "features_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{design}.parquet"
    if out_path.exists() and not force:
        print(f"[{design}] cached → skip")
        return out_path

    sub = manifest[manifest["design_name"] == design]
    if sub.empty:
        print(f"[{design}] no rows in manifest")
        return None
    sub = sub.copy()
    sub["abs_path"] = str(cfg.DATA_ROOT) + "/" + sub["rel_path"].astype(str)
    grouped = sub.groupby("net_name")["abs_path"].apply(list)

    spef_path = cfg.SPEF_DIR / f"{design}_starrc.spef"
    if not spef_path.exists():
        print(f"[{design}] missing spef")
        return None
    print(f"[{design}] parsing SPEF: {spef_path.name}")
    t0 = time.time()
    spef_targets = parse_spef_to_targets(spef_path)
    print(f"[{design}]   spef parse: {time.time()-t0:.1f}s, {len(spef_targets)} nets")

    spef_keys = set(spef_targets.keys())
    keep = [n for n in grouped.index if n in spef_keys]
    print(f"[{design}] {len(grouped)} → {len(keep)} after SPEF-join")
    if not keep:
        return None

    work = [(grouped[n], n) for n in keep]
    rows = []
    t0 = time.time()
    with mp.Pool(processes=n_workers) as pool:
        for i, feat in enumerate(pool.imap_unordered(_proc_one, work, chunksize=4)):
            if feat is None: continue
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
            for k in FEATURE_NAMES_V2:
                row[k] = feat.get(k, np.nan)
            rows.append(row)
            if (i + 1) % 1000 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(work) - i - 1) / max(rate, 1e-3)
                print(f"[{design}]  {i+1}/{len(work)}  {rate:.1f}/s eta {eta:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"[{design}] → {out_path}  ({len(df)} rows)  in {time.time()-t0:.1f}s")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    designs = args.designs or cfg.ALL_DESIGNS
    print(f"loading manifest: {cfg.MANIFEST_PATH}")
    manifest = pd.read_csv(cfg.MANIFEST_PATH)
    for d in designs:
        try:
            build_design_v2(d, manifest, args.force, args.workers)
        except Exception:
            traceback.print_exc()
        gc.collect()


if __name__ == "__main__":
    main()
