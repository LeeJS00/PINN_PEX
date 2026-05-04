"""LGBM-based per-pair coupling distributor.

Replaces the geometric heuristic (1/d² × overlap × ε) with predictions from
the trained pair regressor (output/spef_e2e/pair_regressor/).

Workflow:
  1. Predict raw c_pair_pred per pair using LGBM ensemble.
  2. For each target net, sum predicted c_pair → Σc_pair_raw.
  3. Rescale: c_pair_final = c_pair_raw × (c_cpl_total_pred / Σc_pair_raw).
     Σ(rescaled) = c_cpl_total_pred ✓ (matches our total cap prediction).

This decouples per-pair allocation accuracy from total cap prediction
accuracy.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _setup_paths():
    _HERE = Path(__file__).resolve().parent
    _WS = _HERE.parent
    if str(_WS) not in sys.path:
        sys.path.insert(0, str(_WS))


def predict_raw_pair_caps(pair_df: pd.DataFrame, pair_models_dir: Path) -> np.ndarray:
    """Predict raw c_pair using LGBM ensemble. Output shape = (len(pair_df),)."""
    _setup_paths()
    fcols = json.load(open(pair_models_dir / "fcols.json"))
    X = pair_df[fcols].to_numpy(np.float32)
    preds = []
    for f in sorted(pair_models_dir.glob("seed*.pkl")):
        with open(f, "rb") as fh:
            booster = pickle.load(fh)
        preds.append(np.exp(booster.predict(X, num_iteration=booster.best_iteration)))
    if not preds:
        return np.zeros(len(X))
    return np.mean(preds, axis=0)


def distribute_with_lgbm(c_cpl_total_per_net: Dict[str, float],
                          pair_df: pd.DataFrame,
                          pair_models_dir: Path) -> Dict[str, List[Tuple[str, float]]]:
    """Distribute c_cpl_total across pairs using LGBM-predicted weights.

    Returns: {target_net: [(aggressor_net, c_pair_final), ...]}
    """
    if pair_df.empty:
        return {}

    raw = predict_raw_pair_caps(pair_df, pair_models_dir)
    pair_df = pair_df.copy()
    pair_df["c_pair_raw"] = raw

    out: Dict[str, List[Tuple[str, float]]] = {}
    for tgt, sub in pair_df.groupby("target_net"):
        c_total = c_cpl_total_per_net.get(tgt, 0.0)
        if c_total <= 0:
            out[tgt] = []
            continue
        s = sub["c_pair_raw"].sum()
        if s <= 0:
            n = len(sub)
            share = c_total / n
            out[tgt] = [(row["aggressor_net"], share) for _, row in sub.iterrows()]
        else:
            scale = c_total / s
            out[tgt] = [(row["aggressor_net"], float(row["c_pair_raw"] * scale)) for _, row in sub.iterrows()]
    return out
