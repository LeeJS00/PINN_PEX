#!/usr/bin/env python3
"""
24_mesh_5seed_ensemble_eval.py — 5-seed Mesh ensemble inference.

For each of the 5 mesh-curriculum seed models, run inference on
test split, average predictions per net. Save ensemble CSV +
report MAPE.

Codex Round 6 expected gain: best 6.26% → ~5.8-6.0%, last 8.27% → ~6.8-7.4%.
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


SEED_DIRS = [
    Path(f"/home/jslee/projects/PINNPEX/pex_v3/output/phase1_mesh_5seed/seed{s}")
    for s in [0, 1, 2, 3, 4]
]
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
FEATURES = "/data/PINNPEX/data/processed_v3/intel22/features/all_designs.csv"
CUBOID_DIR = Path("/data/PINNPEX/data/processed_v3/intel22/per_net_cuboids")
OUT_DIR = _PROJECT_ROOT / "pex_v3" / "output" / "phase1_mesh_5seed_ensemble"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f">>> Loading + splitting features ...")
    df = pd.read_csv(FEATURES)
    train_df, valid_df, test_df = split_by_manifest_column(df)
    for d in (train_df, valid_df, test_df):
        d.drop(d[(d["c_gnd_fF"] + d["c_cpl_total_fF"]) <= 1e-4].index, inplace=True)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f">>> splits: train={len(train_df):,}  valid={len(valid_df):,}  test={len(test_df):,}")

    # NNLS calibration (same as training)
    calib = fit_per_layer_calibration(train_df)
    valid_df = apply_per_layer_calibration(valid_df, calib)
    test_df  = apply_per_layer_calibration(test_df,  calib)

    print(f">>> Loading cuboid store ...")
    store = PerNetCuboidStore(CUBOID_DIR)
    print(f">>> store: {len(store):,}")

    valid_ds = CuboidAugmentedDataset(valid_df, store, _SELF_FEATURE_COLS, _PAIR_FEATURE_COLS)
    test_ds  = CuboidAugmentedDataset(test_df,  store, _SELF_FEATURE_COLS, _PAIR_FEATURE_COLS)
    print(f">>> datasets: valid={len(valid_ds):,}  test={len(test_ds):,}")

    valid_loader = DataLoader(valid_ds, batch_size=256, num_workers=2, collate_fn=collate_cuboid_batch)
    test_loader  = DataLoader(test_ds,  batch_size=256, num_workers=2, collate_fn=collate_cuboid_batch)

    # Per-seed predictions
    valid_pred_per_seed: list[dict] = []
    test_pred_per_seed: list[dict] = []
    print()
    for seed_dir in SEED_DIRS:
        seed = int(seed_dir.name.replace("seed", ""))
        ckpt = seed_dir / "model.pt"
        if not ckpt.exists():
            print(f"  seed {seed}: MISSING {ckpt}")
            continue
        m = HybridPexV3Mesh().to(DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        m.eval()
        print(f">>> seed {seed}: inferring valid + test")

        for split_name, loader, df_split, store_list in [
            ("valid", valid_loader, valid_df, valid_pred_per_seed),
            ("test",  test_loader,  test_df,  test_pred_per_seed),
        ]:
            preds_gnd = []
            preds_cpl = []
            net_keys = []
            with torch.no_grad():
                for b in loader:
                    ag, ac = b["analytic_gnd"].to(DEVICE), b["analytic_cpl"].to(DEVICE)
                    sf, pf = b["self_features"].to(DEVICE), b["pair_features"].to(DEVICE)
                    cb, mk = b["cuboids"].to(DEVICE), b["padding_mask"].to(DEVICE)
                    pg = m.predict_gnd(ag, sf, cb, mk).cpu().numpy()
                    pc = m.predict_cpl(ac, pf, cb, mk).cpu().numpy()
                    preds_gnd.append(pg)
                    preds_cpl.append(pc)
                    for d, n in zip(b["design_name"], b["net_name"]):
                        net_keys.append((d, n))
            pg_arr = np.concatenate(preds_gnd)
            pc_arr = np.concatenate(preds_cpl)
            store_list.append({
                "seed": seed, "keys": net_keys,
                "gnd": pg_arr, "cpl": pc_arr,
                "split_df": df_split,
            })

    # Ensemble: average across seeds (same net order)
    print()
    print(">>> Building ensemble predictions ...")
    for split_name, per_seed_list, df_split in [
        ("valid", valid_pred_per_seed, valid_df),
        ("test",  test_pred_per_seed,  test_df),
    ]:
        assert all(per_seed_list[0]["keys"] == s["keys"] for s in per_seed_list), \
            "Net key order mismatch across seeds"
        gnd_stack = np.stack([s["gnd"] for s in per_seed_list], axis=0)
        cpl_stack = np.stack([s["cpl"] for s in per_seed_list], axis=0)
        gnd_ens = gnd_stack.mean(axis=0)
        cpl_ens = cpl_stack.mean(axis=0)
        keys = per_seed_list[0]["keys"]

        # Match golden by net key (preserve order from dataset)
        # df_split rows order match dataset order if dataset built directly from df_split
        # But CuboidAugmentedDataset may have filtered some. Use net_keys as ground truth.
        df_idx = df_split.set_index(["design_name", "net_name"])
        rows = []
        for i, (d, n) in enumerate(keys):
            if (d, n) not in df_idx.index:
                continue
            rec = df_idx.loc[(d, n)]
            rows.append({
                "design_name": d, "net_name": n,
                "pred_gnd_fF": float(gnd_ens[i]), "pred_cpl_fF": float(cpl_ens[i]),
                "pred_total_fF": float(gnd_ens[i] + cpl_ens[i]),
                "golden_gnd_fF": float(rec["c_gnd_fF"]),
                "golden_cpl_fF": float(rec["c_cpl_total_fF"]),
                "golden_total_fF": float(rec["c_gnd_fF"] + rec["c_cpl_total_fF"]),
            })
        out = pd.DataFrame(rows)
        out_path = OUT_DIR / f"ensemble_predictions_{split_name}.csv"
        out.to_csv(out_path, index=False)

        def mape(p, g): return np.median(np.abs(p - g) / np.clip(np.abs(g), 1e-3, None))
        print(f">>> {split_name.upper()} ensemble: total {mape(out.pred_total_fF, out.golden_total_fF)*100:.3f}%  "
              f"gnd {mape(out.pred_gnd_fF, out.golden_gnd_fF)*100:.3f}%  "
              f"cpl {mape(out.pred_cpl_fF, out.golden_cpl_fF)*100:.3f}%")
        if split_name == "test":
            for design, sub in out.groupby("design_name"):
                m_d = mape(sub.pred_total_fF, sub.golden_total_fF)
                print(f"  per-design {design}: total {m_d*100:.3f}%")
        print(f"   wrote {out_path}")

    print()
    print("✅ ensemble inference complete")


if __name__ == "__main__":
    main()
