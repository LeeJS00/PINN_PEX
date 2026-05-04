"""Blend general ensemble preds with specialist (large-net) preds via routing.

Router: use general prediction itself as the threshold (since we don't know
true). For samples where general predicts >= threshold, blend in specialist
predictions.

Soft blend: w_specialist = sigmoid((log(general_pred) - log(threshold)) / scale)
            yhat = (1 - w) * general + w * specialist
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS))

from configs import cfg
from src.data_loader import _select_feature_cols


def _load_test(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.set_index(["design_name", "net_name"])


def main():
    # General preds: ENS_median across all models (best from final_stack_and_report)
    final_pipe = cfg.OUTPUT_DIR / "final_pipe"
    resmlp = cfg.OUTPUT_DIR / "resmlp_v2"
    specialist = cfg.OUTPUT_DIR / "specialist_large"

    if not specialist.exists():
        print("No specialist preds")
        return

    # Aggregate general preds
    general_preds = []
    for csv in sorted(final_pipe.rglob("*__test.csv")):
        general_preds.append(pd.read_csv(csv).set_index(["design_name","net_name"])["y_pred"])
    for csv in sorted(resmlp.rglob("*__test.csv")):
        general_preds.append(pd.read_csv(csv).set_index(["design_name","net_name"])["y_pred"])

    if not general_preds:
        print("no general preds"); return

    # Median across all general models
    general_df = pd.concat(general_preds, axis=1)
    general_med = general_df.median(axis=1)

    # Specialist preds (3 seeds, average)
    spec_preds = []
    for csv in sorted(specialist.rglob("*__test.csv")):
        spec_preds.append(pd.read_csv(csv).set_index(["design_name","net_name"])["y_pred"])
    spec_df = pd.concat(spec_preds, axis=1)
    spec_mean = spec_df.mean(axis=1)

    # Get true and metal area for routing
    test_csv = list(final_pipe.rglob("*__test.csv"))[0]
    base = pd.read_csv(test_csv).set_index(["design_name","net_name"])
    yt = base["y_true"]

    # Blend
    print(f"general n={len(general_med)}, specialist n={len(spec_mean)}, true n={len(yt)}")
    blended = pd.DataFrame({"y_true": yt, "general": general_med, "specialist": spec_mean})
    blended = blended.dropna()
    print(f"after dropna: {len(blended)}")

    # Hard route by general prediction
    for threshold in [0.5, 1.0, 2.0]:
        for method in ["hard", "soft"]:
            yhat = blended["general"].copy()
            if method == "hard":
                mask = blended["general"] >= threshold
                yhat[mask] = blended.loc[mask, "specialist"]
            else:  # soft
                w = 1.0 / (1.0 + np.exp(-(np.log(blended["general"].clip(lower=1e-3)) - np.log(threshold)) * 2.0))
                yhat = (1 - w) * blended["general"] + w * blended["specialist"]
            ape = 100.0 * np.abs(yhat - blended["y_true"]) / np.maximum(blended["y_true"], 1e-3)
            print(f"  threshold={threshold} method={method}: mean MAPE = {ape.mean():.3f}%, median={ape.median():.3f}%")

    # Compare to general only
    ape0 = 100.0 * np.abs(blended["general"] - blended["y_true"]) / np.maximum(blended["y_true"], 1e-3)
    print(f"\nGeneral only: mean={ape0.mean():.3f}%, median={ape0.median():.3f}%")

    ape1 = 100.0 * np.abs(blended["specialist"] - blended["y_true"]) / np.maximum(blended["y_true"], 1e-3)
    print(f"Specialist only (all): mean={ape1.mean():.3f}%, median={ape1.median():.3f}%")


if __name__ == "__main__":
    main()
