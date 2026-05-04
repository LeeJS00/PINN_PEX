"""Build per-(target,aggressor) pair features from cuboid pkls (no SPEF labels).

Inference variant of scripts/build_pair_dataset.py.
"""
from __future__ import annotations

import argparse
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
    from src.per_pair_features import extract_pairs_for_net
    pkls_str, target_name, cutoff_um = args
    paths = [Path(p) for p in pkls_str]
    try:
        pair_features, _ = extract_pairs_for_net(paths, cutoff_um=cutoff_um, target_net_name=target_name)
    except (KeyError, AttributeError, TypeError) as e:
        # Skip nets where some pkls have an unexpected schema
        return []
    rows = []
    for agg_name, feats in pair_features.items():
        feats["target_net"] = target_name
        feats["aggressor_net"] = agg_name
        rows.append(feats)
    return rows


def build_pair_features_inference(design_name: str,
                                    cuboid_pkl_dir: Path,
                                    out_path: Path,
                                    manifest_df: pd.DataFrame = None,
                                    cutoff_um: float = 4.0,
                                    n_workers: int = 12) -> Path:
    _setup_paths()

    if manifest_df is None:
        pkl_files = sorted(Path(cuboid_pkl_dir).rglob("*.pkl.gz"))
        if not pkl_files:
            raise RuntimeError(f"No pkl.gz in {cuboid_pkl_dir}")
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
        if "abs_path" not in sub.columns and "rel_path" in sub.columns:
            sub["abs_path"] = sub["rel_path"].astype(str)
        net_to_pkls = sub.groupby("net_name")["abs_path"].apply(list).to_dict()

    print(f"[{design_name}] {len(net_to_pkls)} nets")
    work = [(pkls, name, cutoff_um) for name, pkls in net_to_pkls.items()]

    all_rows = []
    t0 = time.time()
    with mp.Pool(processes=n_workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(_proc_one, work, chunksize=4)):
            all_rows.extend(rows)
            if (i + 1) % 500 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(work) - i - 1) / max(rate, 1e-3)
                print(f"[{design_name}]  {i+1}/{len(work)}  {rate:.1f}/s eta {eta:.0f}s", flush=True)

    if not all_rows:
        print(f"[{design_name}] no pairs")
        return None

    df = pd.DataFrame(all_rows)
    df["design_name"] = design_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[{design_name}] → {out_path} ({len(df)} pairs) in {time.time()-t0:.1f}s")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design_name", required=True)
    ap.add_argument("--cuboid_pkl_dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cutoff_um", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest) if args.manifest else None
    build_pair_features_inference(args.design_name, args.cuboid_pkl_dir, args.out,
                                    manifest_df=manifest, cutoff_um=args.cutoff_um,
                                    n_workers=args.workers)


if __name__ == "__main__":
    main()
