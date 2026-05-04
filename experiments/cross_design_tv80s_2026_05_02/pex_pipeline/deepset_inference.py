"""DeepSet inference helper — load saved .pt and predict total_cap.

Reuses model classes from train_deepset_v2.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS))


def predict_deepset(features_df: pd.DataFrame, cuboid_arr_npz: Path,
                     pt_path: Path, fcols_total) -> np.ndarray:
    """Run DeepSet inference. Returns predictions in fF for each net in features_df.

    Aligns features_df with cuboid_arr by net_name.
    """
    import torch
    from scripts.train_deepset_v2 import DeepSetModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    fcols = ckpt.get("feature_cols", fcols_total)
    mu = np.array(ckpt["mu"], dtype=np.float32) if isinstance(ckpt["mu"], list) else ckpt["mu"]
    sd = np.array(ckpt["sd"], dtype=np.float32) if isinstance(ckpt["sd"], list) else ckpt["sd"]

    arr = np.load(cuboid_arr_npz, allow_pickle=True)
    cuboid_names = arr["net_names"]
    name_to_idx = {n: i for i, n in enumerate(cuboid_names)}

    # Align — keep features_df order, fill any missing nets with zeros (rare)
    feat_names = features_df["net_name"].values
    pred_out = np.zeros(len(feat_names), dtype=np.float32)
    keep_idx = [i for i, n in enumerate(feat_names) if n in name_to_idx]
    if not keep_idx:
        return pred_out

    arr_idx = np.array([name_to_idx[feat_names[i]] for i in keep_idx])
    T = arr["target"][arr_idx]
    A = arr["aggressor"][arr_idx]
    P = arr["power"][arr_idx]
    n_t = arr["n_target"][arr_idx]
    n_a = arr["n_agg"][arr_idx]
    n_p = arr["n_pwr"][arr_idx]

    hand = features_df[fcols].iloc[keep_idx].to_numpy(np.float32)
    hand = ((hand - mu) / (sd + 1e-6)).clip(-8, 8)

    # Build model
    model = DeepSetModel(hand_dim=len(fcols)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    T_t = torch.from_numpy(T).to(device)
    A_t = torch.from_numpy(A).to(device)
    P_t = torch.from_numpy(P).to(device)
    nT_t = torch.from_numpy(n_t.astype(np.int64)).to(device)
    nA_t = torch.from_numpy(n_a.astype(np.int64)).to(device)
    nP_t = torch.from_numpy(n_p.astype(np.int64)).to(device)
    h_t = torch.from_numpy(hand).to(device)

    T_max = T.shape[1]; A_max = A.shape[1]; P_max = P.shape[1]
    rngT = torch.arange(T_max, device=device)
    rngA = torch.arange(A_max, device=device)
    rngP = torch.arange(P_max, device=device)

    chunk = 2048
    n = T_t.shape[0]
    preds = []
    with torch.no_grad():
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            mT = rngT.unsqueeze(0) < nT_t[s:e].unsqueeze(1)
            mA = rngA.unsqueeze(0) < nA_t[s:e].unsqueeze(1)
            mP = rngP.unsqueeze(0) < nP_t[s:e].unsqueeze(1)
            p = torch.exp(model(T_t[s:e], A_t[s:e], P_t[s:e], mT, mA, mP, h_t[s:e]))
            preds.append(p.cpu().numpy())
    yhat = np.concatenate(preds, axis=0)

    for i, k in enumerate(keep_idx):
        pred_out[k] = yhat[i]
    return pred_out


def predict_deepset_ensemble(features_df, cuboid_arr_npz, models_dir: Path,
                              fcols_total) -> list:
    """Run all DeepSet seeds; return list of arrays."""
    preds = []
    for pt in sorted(models_dir.glob("seed*.pt")):
        p = predict_deepset(features_df, cuboid_arr_npz, pt, fcols_total)
        preds.append(p)
    return preds
