"""Cap inference: load saved models, predict total_cap, c_gnd, c_cpl per net.

Models loaded from output/spef_e2e/{total_cap, gnd_ratio, total_r}/.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def _setup_paths():
    _HERE = Path(__file__).resolve().parent
    _WS = _HERE.parent
    if str(_WS) not in sys.path:
        sys.path.insert(0, str(_WS))


def _load_mlp_preds(features_df, mlp_dir, fcols):
    """Load MLP .pt models and predict. Returns list of arrays, or [] if no MLPs."""
    if not mlp_dir.exists():
        return []
    pt_files = sorted(mlp_dir.glob("seed*.pt"))
    if not pt_files:
        return []
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return []

    class ResMLP(nn.Module):
        def __init__(self, in_dim, hidden, depth, dropout):
            super().__init__()
            self.input = nn.Linear(in_dim, hidden)
            self.blocks = nn.ModuleList([nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden),
            ) for _ in range(depth)])
            self.norm = nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden, 1)
            self.bias = nn.Parameter(torch.tensor(0.0))
        def forward(self, x):
            h = self.input(x)
            for blk in self.blocks: h = h + blk(h)
            h = self.norm(h)
            return self.head(h).squeeze(-1) + self.bias

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = []
    for pt in pt_files:
        ckpt = torch.load(pt, map_location=device, weights_only=False)
        # Use the saved feature_cols if available (must match)
        ckpt_fcols = ckpt.get("feature_cols", fcols)
        if ckpt_fcols != fcols:
            # Fall back: only use shared columns in fcols order
            X = features_df[ckpt_fcols].to_numpy(np.float32)
        else:
            X = features_df[fcols].to_numpy(np.float32)
        mu = ckpt["mu"]; sd = ckpt["sd"]
        X_n = (X - mu) / sd
        model = ResMLP(in_dim=len(ckpt_fcols), hidden=ckpt.get("hidden", 384),
                        depth=ckpt.get("depth", 6), dropout=ckpt.get("dropout", 0.10)).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_n).to(device)
            pred = torch.exp(model(X_t)).cpu().numpy()
        preds.append(pred)
    return preds


def predict_total_cap(features_df: pd.DataFrame, models_dir: Path) -> np.ndarray:
    """Predict per-net total_cap_fF using ensemble of saved models.

    Loads all model classes (LGBM + CatBoost + optional MLPs from
    `total_cap_mlp/`). If `stratum_weights.json` exists in models_dir,
    applies pre-fit per-bucket weights; else uses uniform mean.
    """
    _setup_paths()
    fcols_path = models_dir / "fcols.json"
    with open(fcols_path) as f:
        fcols = json.load(f)
    X = features_df[fcols].to_numpy(np.float32)

    # Track per-model predictions with names matching stratum_weights.json keys
    pred_dict = {}
    # LightGBM (key: lgbm_lgbm_seedN)
    for f in sorted(models_dir.glob("lgbm_seed*.pkl")):
        with open(f, "rb") as fh:
            booster = pickle.load(fh)
        pred_dict[f"lgbm_{f.stem}"] = np.exp(booster.predict(X, num_iteration=booster.best_iteration))

    # CatBoost (key: cat_cat_seedN)
    try:
        from catboost import CatBoostRegressor
        for f in sorted(models_dir.glob("cat_seed*.cbm")):
            mdl = CatBoostRegressor()
            mdl.load_model(str(f))
            pred_dict[f"cat_{f.stem}"] = np.exp(mdl.predict(X))
    except Exception:
        pass

    # MLP (key: mlp_seedN) — loaded from sibling directory
    mlp_dir = models_dir.parent / "total_cap_mlp"
    if mlp_dir.exists():
        import torch  # noqa  (ensures torch is available before _load_mlp_preds)
        pt_files = sorted(mlp_dir.glob("seed*.pt"))
        mlp_preds = _load_mlp_preds(features_df, mlp_dir, fcols)
        for pt, p in zip(pt_files, mlp_preds):
            pred_dict[f"mlp_{pt.stem}"] = p

    # DeepSet (key: deepset_seedN) — loaded from sibling directory.
    # Requires cuboid_arr.npz alongside the features.
    deepset_dir = models_dir.parent / "total_cap_deepset"
    cubarr_path = features_df.attrs.get("cuboid_arr_npz") if hasattr(features_df, "attrs") else None
    if deepset_dir.exists() and cubarr_path is not None and Path(cubarr_path).exists():
        try:
            from pex_pipeline.deepset_inference import predict_deepset_ensemble
            ds_preds = predict_deepset_ensemble(features_df, Path(cubarr_path), deepset_dir, fcols)
            for i, p in enumerate(ds_preds):
                pred_dict[f"deepset_seed{i}"] = p
        except Exception as e:
            print(f"  predict_total_cap: DeepSet inference failed ({e}), skipping")

    if not pred_dict:
        raise RuntimeError(f"No saved models in {models_dir}")

    # If stratum weights available, apply per-bucket blending
    sw_path = models_dir / "stratum_weights.json"
    if sw_path.exists():
        sw = json.load(open(sw_path))
        keys = sw["model_keys"]
        if all(k in pred_dict for k in keys):
            P = np.stack([pred_dict[k] for k in keys], axis=1)
            # Bucket assigner: geomean of all model preds
            eps = 1e-4
            assigner = np.exp(np.log(np.clip(P, eps, None)).mean(axis=1))
            boundaries = np.array(sw["boundaries"])
            test_b = np.digitize(assigner, boundaries)
            yhat = np.zeros(len(P))
            for b, w in enumerate(sw["bucket_weights"]):
                if w is None: w = [1.0/len(keys)] * len(keys)
                w = np.array(w, dtype=np.float64)
                if w.sum() == 0: w = np.ones(len(keys)) / len(keys)
                w = w / w.sum()
                m = test_b == b
                yhat[m] = P[m] @ w
            print(f"  predict_total_cap: stratum blend over {len(keys)} models, {sw['n_buckets']} buckets")
            return yhat

    print(f"  predict_total_cap: uniform mean over {len(pred_dict)} models")
    return np.mean(list(pred_dict.values()), axis=0)


def predict_gnd_ratio(features_df: pd.DataFrame, models_dir: Path) -> np.ndarray:
    """Predict per-net c_gnd / total ratio using ensemble of LGBM ratio models.

    Applies val-calibrated scale (from calibration.json) to reduce systematic bias.
    """
    _setup_paths()
    fcols_total = json.load(open(Path(__file__).parent.parent /
                                    "output" / "spef_e2e" / "total_cap" / "fcols.json"))
    X = features_df[fcols_total].to_numpy(np.float32)
    preds = []
    for f in sorted(models_dir.glob("seed*.pkl")):
        with open(f, "rb") as fh:
            booster = pickle.load(fh)
        preds.append(booster.predict(X, num_iteration=booster.best_iteration))
    if not preds:
        return np.full(len(X), 0.36, dtype=np.float64)
    z = np.mean(preds, axis=0)  # logit
    ratio = 1.0 / (1.0 + np.exp(-z))

    # Apply val-fitted calibration if available
    calib_path = models_dir / "calibration.json"
    if calib_path.exists():
        scale = json.load(open(calib_path)).get("ratio_scale", 1.0)
        ratio = np.clip(ratio * scale, 0.05, 0.95)
    return ratio


def predict_total_r(features_df: pd.DataFrame, models_dir: Path) -> np.ndarray:
    """Predict per-net total_res in ohms with stratum blend (if available)."""
    _setup_paths()
    fcols_total = json.load(open(Path(__file__).parent.parent /
                                    "output" / "spef_e2e" / "total_cap" / "fcols.json"))
    X = features_df[fcols_total].to_numpy(np.float32)
    pred_dict = {}

    has_v2 = any(models_dir.glob("lgbm_seed*.pkl"))
    if has_v2:
        for f in sorted(models_dir.glob("lgbm_seed*.pkl")):
            with open(f, "rb") as fh:
                booster = pickle.load(fh)
            pred_dict[f"lgbm_{f.stem}"] = np.exp(booster.predict(X, num_iteration=booster.best_iteration))
        try:
            from catboost import CatBoostRegressor
            for f in sorted(models_dir.glob("cat_seed*.cbm")):
                mdl = CatBoostRegressor()
                mdl.load_model(str(f))
                pred_dict[f"cat_{f.stem}"] = np.exp(mdl.predict(X))
        except Exception:
            pass
    else:
        for f in sorted(models_dir.glob("seed*.pkl")):
            with open(f, "rb") as fh:
                booster = pickle.load(fh)
            pred_dict[f.stem] = np.exp(booster.predict(X, num_iteration=booster.best_iteration))

    if not pred_dict:
        return np.zeros(len(X), dtype=np.float64)

    # Stratum blend if available
    sw_path = models_dir / "stratum_weights.json"
    if sw_path.exists():
        sw = json.load(open(sw_path))
        keys = sw["model_keys"]
        if all(k in pred_dict for k in keys):
            P = np.stack([pred_dict[k] for k in keys], axis=1)
            eps = 1e-4
            assigner = np.exp(np.log(np.clip(P, eps, None)).mean(axis=1))
            boundaries = np.array(sw["boundaries"])
            test_b = np.digitize(assigner, boundaries)
            yhat = np.zeros(len(P))
            for b, w in enumerate(sw["bucket_weights"]):
                if w is None: w = [1.0/len(keys)] * len(keys)
                w = np.array(w, dtype=np.float64)
                if w.sum() == 0: w = np.ones(len(keys)) / len(keys)
                w = w / w.sum()
                m = test_b == b
                yhat[m] = P[m] @ w
            R = yhat
            print(f"  predict_total_r: stratum blend over {len(keys)} models, {sw['n_buckets']} buckets")
        else:
            R = np.mean(list(pred_dict.values()), axis=0)
            print(f"  predict_total_r: uniform mean over {len(pred_dict)} models (stratum keys mismatch)")
    else:
        R = np.mean(list(pred_dict.values()), axis=0)

    # Optional calibration scale
    calib_path = models_dir / "calibration.json"
    if calib_path.exists():
        scale = json.load(open(calib_path)).get("r_scale", 1.0)
        R = R * scale
    return R


def predict_cgnd_direct(features_df: pd.DataFrame, models_dir: Path) -> np.ndarray | None:
    """Predict c_gnd directly using LGBM+CatBoost+DeepSet ensemble (stratum if available)."""
    _setup_paths()
    fcols_path = models_dir / "fcols.json"
    if not fcols_path.exists():
        return None
    fcols = json.load(open(fcols_path))
    X = features_df[fcols].to_numpy(np.float32)

    pred_dict = {}
    for f in sorted(models_dir.glob("lgbm_seed*.pkl")):
        with open(f, "rb") as fh:
            booster = pickle.load(fh)
        pred_dict[f"lgbm_{f.stem}"] = np.exp(booster.predict(X, num_iteration=booster.best_iteration))
    try:
        from catboost import CatBoostRegressor
        for f in sorted(models_dir.glob("cat_seed*.cbm")):
            mdl = CatBoostRegressor()
            mdl.load_model(str(f))
            pred_dict[f"cat_{f.stem}"] = np.exp(mdl.predict(X))
    except Exception:
        pass

    # DeepSet c_gnd from sibling cgnd_deepset directory
    cgnd_ds_dir = models_dir.parent / "cgnd_deepset"
    cubarr_path = features_df.attrs.get("cuboid_arr_npz") if hasattr(features_df, "attrs") else None
    if cgnd_ds_dir.exists() and cubarr_path is not None and Path(cubarr_path).exists():
        try:
            from pex_pipeline.deepset_inference import predict_deepset_ensemble
            ds_preds = predict_deepset_ensemble(features_df, Path(cubarr_path), cgnd_ds_dir, fcols)
            for i, p in enumerate(ds_preds):
                pred_dict[f"deepset_seed{i}"] = p
        except Exception as e:
            print(f"  predict_cgnd_direct: DeepSet inference failed ({e}), skip")

    if not pred_dict:
        return None

    sw_path = models_dir / "stratum_weights.json"
    if sw_path.exists():
        sw = json.load(open(sw_path))
        keys = sw["model_keys"]
        if all(k in pred_dict for k in keys):
            P = np.stack([pred_dict[k] for k in keys], axis=1)
            eps = 1e-4
            assigner = np.exp(np.log(np.clip(P, eps, None)).mean(axis=1))
            boundaries = np.array(sw["boundaries"])
            test_b = np.digitize(assigner, boundaries)
            yhat = np.zeros(len(P))
            for b, w in enumerate(sw["bucket_weights"]):
                if w is None: w = [1.0/len(keys)] * len(keys)
                w = np.array(w, dtype=np.float64)
                if w.sum() == 0: w = np.ones(len(keys)) / len(keys)
                w = w / w.sum()
                m = test_b == b
                yhat[m] = P[m] @ w
            return yhat
    return np.mean(list(pred_dict.values()), axis=0)
