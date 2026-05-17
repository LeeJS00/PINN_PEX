"""00_build_asap7_features.py — Build 41-D base features for ASAP7 (parallel).

Mirrors intel22 pipeline but with ASAP7 paths inline + design-level
multiprocessing (8 workers) for ~5× speedup vs sequential.

Outputs:
    /data/PINNPEX/data/processed_v3/asap7/features/<design>.csv  (per-design)
    /data/PINNPEX/data/processed_v3/asap7/features/all_designs.csv  (concatenated)
"""
from __future__ import annotations

import gc
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "tv80s_autonomous_2026_05_02"

# ASAP7 paths
DATA_ROOT = Path("/data/PINNPEX/data/processed_v3/asap7")
MANIFEST = DATA_ROOT / "dataset_manifest.csv"
DEF_DIR = Path("/home/jslee/projects/PINNPEX/data/raw/def/asap7")
SPEF_DIR = Path("/home/jslee/projects/PINNPEX/golden_data/spef_data/asap7")
PDK_DIR = ROOT / "tool" / "pdk" / "7nm"
LAYERS_INFO = PDK_DIR / "layers" / "layers.info"
TECH_LEF = PDK_DIR / "lef" / "asap7_tech_1x_201209_JS.lef"
CELL_LEF = PDK_DIR / "lef" / "asap7sc7p5t_28_R_1x_220121a.lef"
OUT_DIR = DATA_ROOT / "features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF_UM = 4.0
MAX_AGGR_PER_NET = 768  # L9 2026-05-16: match intel22 historical cap (was default 256)
MAX_WORKERS = 8
SKIP_DESIGNS = {
    "asap7_ldpc_decoder_802_3an_x1",  # handled by 00_build_ldpc_parallel.py
    "asap7_nova_x1",  # test-only design; cold-eval extracts from scratch, no training value
}
FORCE_REBUILD = os.environ.get("FORCE_REBUILD", "0") == "1"  # L9: skip cache check when set


def _build_one(design: str) -> tuple[str, int, float]:
    """Process one design. Returns (design, n_rows, elapsed_sec)."""
    # Worker-local imports so module load happens once per worker process.
    sys.path.insert(0, str(EXP))
    sys.path.insert(0, str(EXP / "src"))
    from baselines.feature_dataset import write_feature_dataset_for_design

    manifest = pd.read_csv(MANIFEST)
    sub = manifest[manifest["design_name"] == design]
    def_path = DEF_DIR / f"{design}.def"
    spef_path = SPEF_DIR / f"{design}_starrc.spef"
    out_path = OUT_DIR / f"{design}.csv"

    if out_path.exists() and not FORCE_REBUILD:
        return (design, -1, 0.0)  # cached, no work

    t0 = time.time()
    n = write_feature_dataset_for_design(
        def_path=def_path,
        spef_path=spef_path,
        manifest_subset=sub,
        out_path=out_path,
        cutoff_um=CUTOFF_UM,
        max_aggr_per_net=MAX_AGGR_PER_NET,
        layers_info_path=LAYERS_INFO,
        tech_lef_path=TECH_LEF,
        cell_lef_path=CELL_LEF,
    )
    return (design, n, time.time() - t0)


def main():
    manifest = pd.read_csv(MANIFEST)
    print(f"manifest: {len(manifest):,} rows, {manifest['design_name'].nunique()} designs", flush=True)

    # Pre-sort designs by manifest row count descending — biggest first so the
    # longest pole (nova) gets a worker slot immediately, dominating wall-clock.
    sizes = manifest.groupby("design_name").size().sort_values(ascending=False)
    designs = list(sizes.index)
    print(">>> Design queue (largest first):", flush=True)
    for d in designs:
        cached = (OUT_DIR / f"{d}.csv").exists()
        mark = "(cached)" if cached else ""
        print(f"    {d:40} {sizes[d]:>8,} rows  {mark}", flush=True)

    todo = [d for d in designs
            if (FORCE_REBUILD or not (OUT_DIR / f"{d}.csv").exists())
            and d not in SKIP_DESIGNS]
    print(f"\n>>> {len(todo)} designs to build, max_workers={MAX_WORKERS}, "
          f"FORCE_REBUILD={FORCE_REBUILD}", flush=True)
    if SKIP_DESIGNS:
        print(f"    (skipping {SKIP_DESIGNS} — handled by dedicated script)", flush=True)

    t0_total = time.time()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_build_one, d): d for d in todo}
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                design, n, elapsed = fut.result()
                print(f"✅ {design}: {n:,} rows in {elapsed:.0f}s", flush=True)
            except Exception as e:
                print(f"❌ {d}: {e}", flush=True)
                import traceback; traceback.print_exc()
    print(f"\n>>> All designs done in {time.time() - t0_total:.0f}s total wall", flush=True)

    # Concatenate
    written = sorted(OUT_DIR.glob("asap7_*.csv"))
    print(f"\n=== Concatenating {len(written)} per-design CSVs → all_designs.csv ===", flush=True)
    dfs = [pd.read_csv(p) for p in written]
    all_df = pd.concat(dfs, ignore_index=True)
    out_csv = OUT_DIR / "all_designs.csv"
    all_df.to_csv(out_csv, index=False)
    print(f"final: {len(all_df):,} rows → {out_csv}", flush=True)
    print(all_df["design_name"].value_counts().to_string())
    print(all_df["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
