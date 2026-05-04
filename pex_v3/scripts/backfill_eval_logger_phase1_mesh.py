#!/usr/bin/env python3
"""
backfill_eval_logger_phase1_mesh.py — back-fill eval_logger parquet for the
locked phase1_mesh_5seed run (which pre-dates the eval_logger contract).

For each seed in `pex_v3/output/phase1_mesh_5seed/seed{0..4}/`, loads model.pt,
runs valid + test inference, joins with v3 features, and writes:
    seed{S}/eval_logger_valid.parquet
    seed{S}/eval_logger_test.parquet

This unlocks the per-net paired MWU path in `aggregate_ablation_summary.py`
when treating the locked baseline as the reference distribution.
"""
from __future__ import annotations
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "pex_v3"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.models.hybrid_v3_mesh import HybridPexV3Mesh  # noqa: E402
from src.data.cuboid_set_dataset import (  # noqa: E402
    PerNetCuboidStore, CuboidAugmentedDataset, collate_cuboid_batch,
)
from src.trainers.finetune_hybrid_v3 import (  # noqa: E402
    split_by_manifest_column, _SELF_FEATURE_COLS, _PAIR_FEATURE_COLS,
)
from src.baselines.calibration_v3 import (  # noqa: E402
    fit_per_layer_calibration, apply_per_layer_calibration,
)
from src.utils.eval_logger import collect_per_net_predictions, write_eval_parquet  # noqa: E402


SEED_DIRS = [
    _PROJECT_ROOT / "pex_v3" / "output" / "phase1_mesh_5seed" / f"seed{s}"
    for s in [0, 1, 2, 3, 4]
]
DEVICE = "cuda:7" if torch.cuda.is_available() else "cpu"
FEATURES = "/data/PINNPEX/data/processed_v3/intel22/features/all_designs.csv"
CUBOID_DIR = "/data/PINNPEX/data/processed_v3/intel22/per_net_cuboids"


def main() -> None:
    print(f">>> Loading features ...")
    df = pd.read_csv(FEATURES)
    train_df, valid_df, test_df = split_by_manifest_column(df)
    for d in (train_df, valid_df, test_df):
        d.drop(d[(d["c_gnd_fF"] + d["c_cpl_total_fF"]) <= 1e-4].index, inplace=True)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Apply same per-layer calibration as training (NNLS on train only)
    calib = fit_per_layer_calibration(train_df)
    valid_df = apply_per_layer_calibration(valid_df, calib)
    test_df  = apply_per_layer_calibration(test_df,  calib)

    print(f">>> splits: valid={len(valid_df):,}  test={len(test_df):,}")

    store = PerNetCuboidStore(Path(CUBOID_DIR))
    valid_ds = CuboidAugmentedDataset(valid_df, store, _SELF_FEATURE_COLS, _PAIR_FEATURE_COLS)
    test_ds  = CuboidAugmentedDataset(test_df,  store, _SELF_FEATURE_COLS, _PAIR_FEATURE_COLS)
    valid_loader = DataLoader(valid_ds, batch_size=256, num_workers=2, collate_fn=collate_cuboid_batch)
    test_loader  = DataLoader(test_ds,  batch_size=256, num_workers=2, collate_fn=collate_cuboid_batch)

    for sd in SEED_DIRS:
        ckpt = sd / "model.pt"
        if not ckpt.exists():
            print(f"  [skip] {ckpt} missing")
            continue
        print(f">>> {sd.name} → inference + eval_logger parquets")
        m = HybridPexV3Mesh().to(DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        v_df = collect_per_net_predictions(m, valid_loader, DEVICE, valid_df)
        write_eval_parquet(v_df, sd / "eval_logger_valid.parquet")
        t_df = collect_per_net_predictions(m, test_loader, DEVICE, test_df)
        write_eval_parquet(t_df, sd / "eval_logger_test.parquet")
        print(f"    wrote {sd}/eval_logger_{{valid,test}}.parquet ({len(v_df)} valid, {len(t_df)} test)")
    print(">>> done")


if __name__ == "__main__":
    main()
