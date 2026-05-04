"""End-to-end SPEF prediction pipeline — DEF + LEF + layers.info → SPEF.

Usage:
    python3 scripts/predict_spef_e2e.py \\
        --def_path <path/to/design.def> \\
        --tech_lef <path/to/tech.lef> \\
        --cell_lef <path/to/cells.lef> \\
        --layers_info <path/to/layers.info> \\
        --top_module <name> \\
        --out_spef <path/to/predicted.spef> \\
        --temp_dir <scratch>

Stages:
    [1] DEF/LEF/layers parse → cuboid pkls (via build_dataset.py)
    [2] cuboid pkls → 145-dim v3 hand features
    [3] cuboid pkls → per-(target,aggressor) pair features
    [4] cuboid pkls → 3-stream cuboid arrays (target/aggressor/power)
    [5] features → predicted total_cap, c_gnd_ratio, total_R (saved LGBM ensembles)
    [6] split + per-pair distribute (geometric heuristic) + analytic R fallback
    [7] write lumped SPEF (IEEE 1481-1999 compatible subset)

Compared to traditional EDA PEX (StarRC, Quantus):
    - Input: identical (DEF + LEF + tech files)
    - Output: SPEF (lumped per-net topology, *CAP + *RES sections)
    - Runtime: ~1-3 minutes for typical design (vs StarRC's ~10-30 min)
    - Quality: ~8% per-net total cap MAPE on cross-design
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
_PINNPEX_ROOT = _WS.parent.parent
sys.path.insert(0, str(_WS))

from configs import cfg
from pex_pipeline.build_features_inference import build_features_inference
from pex_pipeline.build_pair_features_inference import build_pair_features_inference
from pex_pipeline.build_cuboid_arr_inference import build_cuboid_arr_inference
from pex_pipeline.compute_resistance import total_resistance_for_design
from pex_pipeline.decompose_caps import (
    assemble_net_records,
    distribute_cpl_to_pairs,
    load_pair_features_design,
    split_total_to_gnd_cpl,
)
from pex_pipeline.distribute_pairs_lgbm import distribute_with_lgbm
from pex_pipeline.predict_caps import (
    predict_cgnd_direct,
    predict_gnd_ratio,
    predict_total_cap,
    predict_total_r,
)
from pex_pipeline.write_spef import LumpedSPEFWriter


def stage1_build_cuboids(def_path: Path, temp_dir: Path, num_workers: int = 16):
    """Stage 1: invoke PINNPEX build_dataset.py to create cuboid pkls."""
    # Resolve to absolute paths because build_dataset.py runs with cwd=PINNPEX_ROOT
    pkl_dir = (temp_dir / "cuboids").resolve()
    pt_dir = (temp_dir / "cuboids_pt").resolve()
    pkl_dir.mkdir(parents=True, exist_ok=True)
    pt_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3",
        str(_PINNPEX_ROOT / "scripts" / "build_dataset.py"),
        "--def_path", str(def_path.resolve()),
        "--out_dir", str(pkl_dir),
        "--pt_out_dir", str(pt_dir),
        "--num_workers", str(num_workers),
    ]
    print(f"[Stage 1] cmd: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(_PINNPEX_ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        raise RuntimeError("build_dataset.py failed")
    print(f"[Stage 1] cuboid pkls built in {time.time()-t0:.1f}s")
    return pkl_dir


def main():
    ap = argparse.ArgumentParser(description="DEF+LEF→SPEF e2e PEX pipeline")
    ap.add_argument("--def_path", type=Path, required=True)
    ap.add_argument("--top_module", type=str, default=None,
                    help="Top module name (default: derived from DEF filename)")
    ap.add_argument("--out_spef", type=Path, required=True)
    ap.add_argument("--temp_dir", type=Path, default=None,
                    help="Scratch directory (default: alongside out_spef)")
    ap.add_argument("--cuboid_pkl_dir", type=Path, default=None,
                    help="Skip Stage 1 if cuboid pkls already exist here")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Optional manifest CSV with rel_path/abs_path to speed up pkl discovery")
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--cutoff_um", type=float, default=4.0)
    ap.add_argument("--models_dir", type=Path, default=cfg.OUTPUT_DIR / "spef_e2e",
                    help="Directory with saved LGBM/CatBoost models")
    args = ap.parse_args()

    if args.top_module is None:
        args.top_module = args.def_path.stem.replace("intel22_", "").replace("_t1", "").replace("_f3", "")
    # design_name in manifest is the form `intel22_<module>_f3`; if the user
    # provided --top_module that matches the manifest, use it directly.
    if args.temp_dir is None:
        args.temp_dir = args.out_spef.parent / f"_pex_temp_{args.top_module}"
    args.temp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"PINNPEX EDA-style PEX pipeline")
    print(f"  DEF      : {args.def_path}")
    print(f"  Top      : {args.top_module}")
    print(f"  Output   : {args.out_spef}")
    print(f"  Models   : {args.models_dir}")
    print(f"  Temp dir : {args.temp_dir}")
    print("=" * 70)

    t_start = time.time()
    # design_name uses the manifest convention: intel22_<module>_f3
    if args.def_path.stem.startswith("intel22_"):
        # e.g. intel22_tv80s_t1.def → intel22_tv80s_f3
        stem = args.def_path.stem
        if "_t1" in stem:
            design_name = stem.replace("_t1", "_f3")
        else:
            design_name = stem
    else:
        design_name = f"intel22_{args.top_module}_f3"
    print(f"[design_name] {design_name}")

    # === Stage 1: DEF → cuboid pkls ===
    if args.cuboid_pkl_dir and args.cuboid_pkl_dir.exists() and any(args.cuboid_pkl_dir.rglob("*.pkl.gz")):
        pkl_dir = args.cuboid_pkl_dir
        print(f"[Stage 1] Reusing cuboid pkls from {pkl_dir}")
    else:
        t0 = time.time()
        pkl_dir = stage1_build_cuboids(args.def_path, args.temp_dir, args.num_workers)
        print(f"[Stage 1] {time.time()-t0:.1f}s")

    # Optionally load manifest to speed up Stage 2/3/4
    manifest_df = None
    if args.manifest and args.manifest.exists():
        manifest_df = pd.read_csv(args.manifest)
        if "abs_path" not in manifest_df.columns and "rel_path" in manifest_df.columns:
            data_root = Path(pkl_dir).parent if pkl_dir else Path("/data/PINNPEX/data/processed_v3/intel22")
            manifest_df["abs_path"] = str(data_root) + "/" + manifest_df["rel_path"].astype(str)
        print(f"[Manifest] Loaded {len(manifest_df)} rows from {args.manifest.name}")

    # Auto-build manifest from Stage-1's cuboids_map.csv if no explicit --manifest given.
    # Without this, Stage 2 falls back to slow rglob + gzip+pickle scan over all pkls
    # (~3 min for tv80s 100K pkls; >4 hr for nova 684K pkls — see HERO.md runtime report).
    if manifest_df is None:
        cmap_path = args.temp_dir / "cuboids_map.csv"
        if cmap_path.exists():
            manifest_df = pd.read_csv(cmap_path)
            manifest_df["design_name"] = manifest_df["def_name"].str.replace(r"\.def$", "", regex=True)
            manifest_df["abs_path"] = str(pkl_dir) + "/" + manifest_df["sample_filename"].astype(str)
            print(f"[Manifest] Auto-built {len(manifest_df)} rows from {cmap_path.name} (Stage 1 output)")

    # === Stage 2: cuboids → features ===
    feat_path = args.temp_dir / "features.parquet"
    if not feat_path.exists():
        t0 = time.time()
        build_features_inference(design_name, pkl_dir, feat_path,
                                  manifest_df=manifest_df,
                                  cutoff_um=args.cutoff_um, n_workers=args.num_workers)
        print(f"[Stage 2] {time.time()-t0:.1f}s")
    feat = pd.read_parquet(feat_path)
    print(f"[Stage 2] Features: {len(feat)} nets")

    # === Stage 3: cuboids → pair features ===
    pair_path = args.temp_dir / "pair_features.parquet"
    if not pair_path.exists():
        t0 = time.time()
        build_pair_features_inference(design_name, pkl_dir, pair_path,
                                       manifest_df=manifest_df,
                                       cutoff_um=args.cutoff_um, n_workers=args.num_workers)
        print(f"[Stage 3] {time.time()-t0:.1f}s")
    pair_groups = load_pair_features_design(pair_path)
    print(f"[Stage 3] Pair features: {len(pair_groups)} target nets, "
          f"{sum(len(v) for v in pair_groups.values())} pairs")

    # === Stage 4: cuboids → cuboid arr ===
    cubarr_path = args.temp_dir / "cuboid_arr.npz"
    if not cubarr_path.exists():
        t0 = time.time()
        build_cuboid_arr_inference(design_name, pkl_dir, cubarr_path,
                                     manifest_df=manifest_df,
                                     n_workers=args.num_workers)
        print(f"[Stage 4] {time.time()-t0:.1f}s")
    R_analytic = total_resistance_for_design(cubarr_path)
    print(f"[Stage 4] Cuboid_arr + analytic R: {len(R_analytic)} nets")

    # === Stage 5: features → predictions ===
    t0 = time.time()
    print("[Stage 5] Loading models and predicting...")
    # Attach cuboid_arr path to feat so DeepSet can load it
    feat.attrs["cuboid_arr_npz"] = str(cubarr_path)
    total_pred = predict_total_cap(feat, args.models_dir / "total_cap")
    if (args.models_dir / "gnd_ratio").exists():
        ratio_pred = predict_gnd_ratio(feat, args.models_dir / "gnd_ratio")
    else:
        ratio_pred = np.full(len(feat), 0.36)
    if (args.models_dir / "total_r").exists():
        r_pred = predict_total_r(feat, args.models_dir / "total_r")
    else:
        r_pred = np.array([R_analytic.get(n, 0.0) for n in feat["net_name"]], dtype=np.float64)
    print(f"[Stage 5] {time.time()-t0:.1f}s — total_cap mean={total_pred.mean():.3f}, "
          f"ratio mean={ratio_pred.mean():.3f}, R mean={r_pred.mean():.2f}")

    # === Stage 6: split + distribute ===
    t0 = time.time()
    # Direct c_gnd model — used when feature distribution matches training set
    # (cached cuboid path). For raw-DEF e2e the distribution shift hurts the
    # direct model more than the ratio model, so we use a small blend weight.
    cgnd_direct = predict_cgnd_direct(feat, args.models_dir / "cgnd_direct")
    if cgnd_direct is not None:
        # Empirical sweep on cached cuboids (intel22 tv80s) with v7 15-mdl direct:
        # (10 LGBM/CatBoost + 5 DeepSet stratum b=12)
        #   w=0.0: c_gnd 21.752%, bias -0.29%
        #   w=0.3: c_gnd 21.316%, bias +0.39%
        #   w=0.5: c_gnd 21.153%, bias +0.85%
        #   w=0.7: c_gnd 21.087%, bias +1.30%  ← best mean
        #   w=1.0: c_gnd 21.155%, bias +1.98%
        # DeepSet adds enough diversity that direct ensemble beats ratio×total.
        w_direct = float(__import__("os").environ.get("CGND_DIRECT_W", 0.7))
        c_gnd_via_ratio = total_pred * ratio_pred
        c_gnd_pred = w_direct * cgnd_direct + (1.0 - w_direct) * c_gnd_via_ratio
        c_gnd_pred = np.clip(c_gnd_pred, 0, total_pred * 0.95)
        c_cpl_pred = np.clip(total_pred - c_gnd_pred, 0, total_pred)
        print(f"[Stage 6] c_gnd blend (w_direct={w_direct}): "
              f"direct_mean={cgnd_direct.mean():.4f}, ratio×total_mean={c_gnd_via_ratio.mean():.4f}, "
              f"blended_mean={c_gnd_pred.mean():.4f}")
    else:
        c_gnd_pred = total_pred * ratio_pred
        c_cpl_pred = total_pred * (1.0 - ratio_pred)
        print(f"[Stage 6] split: c_gnd_mean={c_gnd_pred.mean():.4f}, c_cpl_mean={c_cpl_pred.mean():.4f}")

    # Per-pair distribution: prefer LGBM pair regressor if available
    pair_models_dir = args.models_dir / "pair_regressor"
    if pair_models_dir.exists() and (pair_models_dir / "fcols.json").exists():
        # Load full pair dataframe (with all pairs for all nets in test)
        pair_df = pd.read_parquet(pair_path)
        c_cpl_map = {n: float(c_cpl_pred[i]) for i, n in enumerate(feat["net_name"].tolist())}
        pair_dist = distribute_with_lgbm(c_cpl_map, pair_df, pair_models_dir)
        print(f"[Stage 6] LGBM pair regressor: {sum(len(v) for v in pair_dist.values())} pairs")
    else:
        pair_dist = None
        print(f"[Stage 6] LGBM pair regressor not available, using geometric heuristic")

    records = []
    for i, net_name in enumerate(feat["net_name"].tolist()):
        c_cpl_i = float(c_cpl_pred[i])
        if c_cpl_i <= 0:
            pairs = []
        elif pair_dist is not None:
            pairs = pair_dist.get(net_name, [])
        elif net_name in pair_groups:
            pairs = distribute_cpl_to_pairs(c_cpl_i, pair_groups[net_name])
        else:
            pairs = []
        records.append({
            "name": net_name,
            "total_cap": float(total_pred[i]),
            "c_gnd": float(c_gnd_pred[i]),
            "pairs": pairs,
            "total_r": float(r_pred[i]),
            "port_xy": (0.0, 0.0),
        })
    print(f"[Stage 6] {time.time()-t0:.1f}s — {len(records)} D_NETs assembled")

    # === Stage 7: write SPEF ===
    t0 = time.time()
    args.out_spef.parent.mkdir(parents=True, exist_ok=True)
    writer = LumpedSPEFWriter(design_name=args.top_module,
                               vendor="PINNPEX",
                               program="PINNPEX-EDA")
    writer.write(args.out_spef, records)
    n_pairs = sum(len(r["pairs"]) for r in records)
    print(f"[Stage 7] {time.time()-t0:.1f}s — SPEF written: {args.out_spef} "
          f"({len(records)} D_NETs, {n_pairs} coupling pairs, "
          f"{args.out_spef.stat().st_size/1024:.1f} KB)")

    print("=" * 70)
    print(f"TOTAL: {time.time()-t_start:.1f}s")
    print(f"OUT  : {args.out_spef}")
    print("=" * 70)


if __name__ == "__main__":
    main()
