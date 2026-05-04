"""Build v3 features for a design WITHOUT SPEF labels (inference-only).

Use case: a new design where SPEF doesn't exist yet — we just need the
feature parquet for downstream prediction.

Inputs:
  cuboid_pkl_dir  : directory containing per-net cuboid pkls (*.pkl.gz)
  manifest_path   : CSV with columns (sample_filename, net_name, design_name, ...)
                    pointing to the pkls
  out_path        : output parquet

Outputs a parquet with columns:
  design_name, net_name, [145 v3 features]

(No total_cap_fF, c_gnd_fF, c_cpl_total_fF, total_res_ohm columns —
those come from SPEF in the labeled version.)
"""
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


def _setup_paths():
    _HERE = Path(__file__).resolve().parent
    _WS = _HERE.parent
    if str(_WS) not in sys.path:
        sys.path.insert(0, str(_WS))


def _proc_one(args):
    _setup_paths()
    from src.feat_extract_v3 import extract_features_for_net_v3
    pkls, name, cutoff_um = args
    try:
        f = extract_features_for_net_v3([Path(p) for p in pkls], cutoff_um=cutoff_um)
    except (KeyError, AttributeError, TypeError, Exception):
        return None
    if f is None:
        return None
    f["net_name"] = name
    return f


def build_features_inference(design_name: str,
                              cuboid_pkl_dir: Path,
                              out_path: Path,
                              manifest_df: pd.DataFrame = None,
                              cutoff_um: float = 4.0,
                              n_workers: int = 12) -> Path:
    """Build v3 features for a design from cuboid pkls (no SPEF labels)."""
    _setup_paths()
    from src.feat_extract_v3 import FEATURE_NAMES_V3

    if manifest_df is None:
        # Auto-discover from pkl filenames
        pkl_files = sorted(Path(cuboid_pkl_dir).rglob("*.pkl.gz"))
        if not pkl_files:
            raise RuntimeError(f"No pkl.gz files found in {cuboid_pkl_dir}")
        # Group by net_name from pkl metadata
        # Each pkl typically has a 'net_name' field; we'll group by parsing the file
        # For speed, we use the manifest path naming convention if available
        net_to_pkls = {}
        for p in pkl_files:
            try:
                import gzip, pickle
                with gzip.open(p, "rb") as f:
                    rec = pickle.load(f)
                if not isinstance(rec, dict) or "cuboids" not in rec:
                    continue
                net_name = rec.get("net_name", p.stem)
                net_to_pkls.setdefault(net_name, []).append(str(p))
            except Exception:
                continue
    else:
        sub = manifest_df[manifest_df["design_name"] == design_name].copy()
        if "abs_path" not in sub.columns:
            if "rel_path" in sub.columns:
                # Fall back to relative path interpretation
                sub["abs_path"] = sub["rel_path"].astype(str)
        net_to_pkls = sub.groupby("net_name")["abs_path"].apply(list).to_dict()

    print(f"[{design_name}] {len(net_to_pkls)} nets discovered from cuboid pkls")
    work = [(pkls, name, cutoff_um) for name, pkls in net_to_pkls.items()]

    rows = []
    t0 = time.time()
    with mp.Pool(processes=n_workers) as pool:
        for i, feat in enumerate(pool.imap_unordered(_proc_one, work, chunksize=4)):
            if feat is None:
                continue
            row = {"design_name": design_name, "net_name": feat["net_name"]}
            for k in FEATURE_NAMES_V3:
                row[k] = feat.get(k, np.nan)
            rows.append(row)
            if (i + 1) % 500 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(work) - i - 1) / max(rate, 1e-3)
                print(f"[{design_name}]  {i+1}/{len(work)}  {rate:.1f}/s eta {eta:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[{design_name}] → {out_path} ({len(df)} rows) in {time.time()-t0:.1f}s")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design_name", required=True, help="design tag (e.g., intel22_tv80s_f3)")
    ap.add_argument("--cuboid_pkl_dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="optional CSV manifest with abs_path or rel_path column")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cutoff_um", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest) if args.manifest else None
    build_features_inference(args.design_name, args.cuboid_pkl_dir, args.out,
                              manifest_df=manifest, cutoff_um=args.cutoff_um,
                              n_workers=args.workers)


if __name__ == "__main__":
    main()
