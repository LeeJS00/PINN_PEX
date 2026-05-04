"""Phase 14 — c_gnd Phase 1 hybrid: analytic parallel-plate + bounded MLP residual.

Implements pex_v3's Phase 1 paradigm (analytic Green's function + bounded
neural residual) for full-net c_gnd, in our isolated session. Does NOT
modify pex_v3 files.

Architecture:
  C_analytic[net] = Σ_layer  k[layer] × wire_area_M{layer}
                  + k_cell × cell_OBS_signal_area

  log_residual = MLP(features), clamp(-log(2), log(2))
  C_pred = C_analytic × exp(log_residual)

Why bounded multiplicative:
  - Day 1 (zero-init last layer) → multiplier = 1 → C_pred = C_analytic
  - Cannot deviate >2× from physics → regularized by analytic
  - "ResCap-style" data efficiency (no Q3D pretrain needed for full-net)

Compares to:
  - v7 ML legacy: 21.09% (paradigm-baseline)
  - pex_v3 hand-feature ceiling: 20.6-21.2%
  - v3 NNLS Stage 1: 26.47%

Goal: break the 21% ceiling.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent

# Canonical split (TRAIN includes ldpc, TEST = nova + tv80s)
DESIGNS_TRAIN = [
    "intel22_aes_cipher_top_f3", "intel22_gcd_f3", "intel22_ibex_core_f3",
    "intel22_ldpc_decoder_802_3an_f3",
    "intel22_mc_top_f3", "intel22_spi_top_f3", "intel22_usbf_top_f3",
    "intel22_vga_enh_top_f3", "intel22_wb_conmax_top_f3",
]
DESIGNS_TEST = ["intel22_nova_f3", "intel22_tv80s_f3"]

CGND_FLOOR_FF = 1e-3

# Parallel-plate base coefficients (fF/μm²) per metal layer.
# Computed as ε₀ × ε_local / d_to_nearest_power_plane, with:
#   - ε_local = ILD ε ≈ 2.8 (intel22 LK-ILD)
#   - d_to_nearest_power_plane ≈ etch_stop + ILD thickness ≈ 0.07 μm typical
#   - ε₀ = 8.854 × 10⁻³ fF/μm
# These are PRIORS (initial values); NNLS will refine.
PRIOR_C_PER_UM2 = {
    1: 0.30,  # M1 — closest to substrate / contact, high cap density
    2: 0.30,  # M2
    3: 0.30,  # M3
    4: 0.30,  # M4
    5: 0.30,  # M5
    6: 0.20,  # M6
    7: 0.10,  # M7 (thick metal, large pitch)
    8: 0.05,  # M8
    9: 0.03,  # M9
}


def _load(d):
    df = pd.read_parquet(_V3 / "cache" / f"feat_v4_{d}.parquet")
    pins = pd.read_parquet(_V3 / "cache" / f"pins_{d}.parquet")
    v6 = pd.read_parquet(_V3 / "cache" / f"feat_v6_{d}.parquet")
    cgnd = pd.read_parquet(_V3 / "cache" / f"cgnd_{d}.parquet")
    df = df.merge(pins, on="net_name", how="left")
    df = df.merge(v6, on="net_name", how="left")
    df = df.merge(cgnd, on="net_name", how="left").fillna(0.0)
    df = df.dropna(subset=["c_gnd_gold"])
    df = df[df["c_gnd_gold"] > CGND_FLOOR_FF].reset_index(drop=True).copy()
    df["design"] = d
    return df


def compute_wire_area_per_layer(df):
    """Per-net Σ over WIRE segments of (length × width) per metal layer.
    Currently we have nsq_M{i} = Σ L/W. Need Σ L×W (area).
    Approximation: area_M{i} ≈ nsq_M{i} × (typical_W)² since L = nsq × W.
    Better: have rect_area_M{i} which is rect-patch area. We'll use rect_area
    where available + approximate WIRE area from nsq × W².
    """
    typical_W = {1: 0.068, 2: 0.044, 3: 0.044, 4: 0.044, 5: 0.044,
                  6: 0.080, 7: 0.080, 8: 0.160}
    out = {}
    for i in range(1, 10):
        nsq_col = f"nsq_M{i}"
        if nsq_col in df.columns:
            W = typical_W.get(i, 0.044)
            # area = nsq × W² since nsq = L/W and area = L×W
            out[i] = df[nsq_col].values * W * W
        else:
            out[i] = np.zeros(len(df))
    return out


def C_analytic_compute(df, c_per_um2=None):
    """C_analytic = Σ wire_area × C_per_um²[layer] + cell intrinsic proxy."""
    c_per_um2 = c_per_um2 or PRIOR_C_PER_UM2
    wire_area = compute_wire_area_per_layer(df)
    C = np.zeros(len(df))
    for lvl, A in wire_area.items():
        C += c_per_um2[lvl] * A
    # Cell intrinsic Cgg proxy (signal-internal M1 area on cell)
    if "v6_obs_signal_area_M1" in df.columns:
        C += 0.5 * df["v6_obs_signal_area_M1"].values  # initial coefficient
    return C


def select_features(dfs, prefixes):
    cols = set()
    for df in dfs:
        for c in df.columns:
            for p in prefixes:
                if c == p or c.startswith(p):
                    cols.add(c); break
    return sorted(cols)


def design_matrix(df, fcols):
    X = np.zeros((len(df), len(fcols)), dtype=np.float64)
    for j, c in enumerate(fcols):
        if c in df.columns:
            X[:, j] = df[c].values.astype(np.float64)
    return X


class BoundedResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden=64, clamp=np.log(2.0)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # zero-init last layer → day-1 multiplier = exp(0) = 1
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.clamp = clamp

    def forward(self, x):
        logit = self.net(x)
        return torch.clamp(logit, -self.clamp, self.clamp)


def stats_per_design(label, pred, y, design_arr):
    designs = sorted(set(design_arr))
    print(f"\n{label}:", flush=True)
    results = {}
    for des in designs:
        m = np.array([d == des for d in design_arr])
        ape = 100 * np.abs(pred[m] - y[m]) / y[m]
        bias = 100 * (pred[m] - y[m]) / y[m]
        rng = np.random.default_rng(0)
        boots = [ape[rng.integers(0, len(ape), len(ape))].mean() for _ in range(500)]
        ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        results[des] = {"n": int(m.sum()), "mape": float(ape.mean()),
                          "median": float(np.median(ape)),
                          "p90": float(np.percentile(ape, 90)),
                          "bias": float(bias.mean()), "ci": list(ci)}
        print(f"  {des:<40s} n={m.sum():6d}  MAPE={ape.mean():6.3f}%  med={np.median(ape):6.3f}%  P90={np.percentile(ape,90):6.2f}%  bias={bias.mean():+6.3f}%", flush=True)
    ape = 100 * np.abs(pred - y) / y
    bias = 100 * (pred - y) / y
    rng = np.random.default_rng(0)
    boots = [ape[rng.integers(0, len(ape), len(ape))].mean() for _ in range(500)]
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    results["combined"] = {"n": int(len(y)), "mape": float(ape.mean()),
                            "median": float(np.median(ape)),
                            "p90": float(np.percentile(ape, 90)),
                            "bias": float(bias.mean()), "ci": list(ci)}
    print(f"  {'COMBINED':<40s} n={len(y):6d}  MAPE={ape.mean():6.3f}%  med={np.median(ape):6.3f}%  P90={np.percentile(ape,90):6.2f}%  bias={bias.mean():+6.3f}%", flush=True)
    return results


def main():
    print("Loading per-design data ...", flush=True)
    train_dfs = [_load(d) for d in DESIGNS_TRAIN]
    test_dfs  = [_load(d) for d in DESIGNS_TEST]
    test_combined = pd.concat(test_dfs, ignore_index=True)
    test_design_arr = test_combined["design"].values
    print(f"  train: {sum(len(d) for d in train_dfs):,} nets", flush=True)
    print(f"  test:  {len(test_combined):,} nets", flush=True)

    # ---------------- Stage A: Analytic base (with prior coefs) ----------------
    print("\n=== Stage A: Analytic base (parallel-plate prior coefs) ===", flush=True)
    train_C_a = np.concatenate([C_analytic_compute(d) for d in train_dfs])
    test_C_a  = C_analytic_compute(test_combined)
    yt = np.concatenate([d["c_gnd_gold"].values for d in train_dfs])
    ys = test_combined["c_gnd_gold"].values
    stats_per_design("A. analytic prior", test_C_a, ys, test_design_arr)

    # ---------------- Stage B: Calibrate analytic coefs via NNLS-IRLS ----------------
    print("\n=== Stage B: Calibrate analytic coefs (NNLS-IRLS, MAPE-aligned) ===", flush=True)
    # Linear features: wire_area per layer + v6 signal area + intercept
    pref = ["nsq_M", "rsq_M", "v6_obs_signal_area_M", "v6_obs_signal_nsq_M",
             "v6_cell_area", "n_pin_inst", "n_pin_PIN", "one"]
    fcols = select_features(train_dfs + test_dfs, pref)
    Xt = np.vstack([design_matrix(d, fcols) for d in train_dfs])
    Xs = design_matrix(test_combined, fcols)
    print(f"  fcols ({len(fcols)}): {fcols[:5]}...", flush=True)

    # NNLS-IRLS for MAPE-aligned linear fit
    from scipy.optimize import lsq_linear
    w = 1.0 / np.maximum(yt, 1e-3)
    res = lsq_linear(Xt * w[:, None], yt * w, bounds=(0.0, np.inf), method="bvls",
                      max_iter=4000)
    c = res.x
    last_mape = None
    for it in range(20):
        pred = Xt @ c
        rel = np.abs(pred - yt) / np.maximum(yt, 1e-3)
        w = 1.0 / (np.maximum(yt, 1e-3) * np.sqrt(rel + 1e-3))
        res = lsq_linear(Xt * w[:, None], yt * w, bounds=(0.0, np.inf), method="bvls",
                          max_iter=4000)
        c_new = res.x
        mape = float(np.mean(np.abs(Xt @ c_new - yt) / yt) * 100)
        if last_mape is not None and abs(last_mape - mape) < 1e-5:
            c = c_new; break
        last_mape = mape; c = c_new

    pred_lin_train = np.maximum(Xt @ c, 1e-4)
    pred_lin_test  = np.maximum(Xs @ c, 1e-4)
    stats_per_design("B. calibrated analytic (NNLS)", pred_lin_test, ys, test_design_arr)

    # ---------------- Stage C: Bounded MLP residual on top of B ----------------
    print("\n=== Stage C: Bounded MLP residual (multiplicative, day-1 = analytic) ===", flush=True)

    pref_full = ["nsq_M", "rsq_M", "nvian_", "one", "n_pin_PIN", "n_pin_inst",
                  "pin_nsq_M", "v6_", "n_segments", "n_zero_l_wire", "n_pins"]
    fcols_full = select_features(train_dfs + test_dfs, pref_full)
    Xt_full = np.vstack([design_matrix(d, fcols_full) for d in train_dfs])
    Xs_full = design_matrix(test_combined, fcols_full)

    # Add log(C_lin) as feature (so MLP knows the analytic estimate)
    Xt_aug = np.column_stack([Xt_full, np.log(pred_lin_train),
                               np.log(yt) - np.log(pred_lin_train)])
    Xs_aug = np.column_stack([Xs_full, np.log(pred_lin_test),
                               np.log(np.maximum(ys, 1e-4)) - np.log(pred_lin_test)])
    # Strip ground-truth-derived columns (last col is leakage); only use Xt/Xs full + log(C_lin)
    Xt_aug = np.column_stack([Xt_full, np.log(pred_lin_train)])
    Xs_aug = np.column_stack([Xs_full, np.log(pred_lin_test)])
    print(f"  Stage C fcols: {Xt_aug.shape[1]}", flush=True)

    # Standardize features
    feat_mean = Xt_aug.mean(0)
    feat_std = Xt_aug.std(0) + 1e-6
    Xt_aug_n = (Xt_aug - feat_mean) / feat_std
    Xs_aug_n = (Xs_aug - feat_mean) / feat_std

    # Train/val split for early stopping (5% val)
    rng = np.random.default_rng(0)
    n = len(yt)
    val_idx = rng.choice(n, size=int(0.05 * n), replace=False)
    train_mask = np.ones(n, dtype=bool); train_mask[val_idx] = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}", flush=True)
    # Read CLAMP from env, default log(2)
    import os
    clamp_val = float(os.environ.get("RES_CLAMP", np.log(2.0)))
    print(f"  RES_CLAMP = {clamp_val:.3f}  (multiplier range [{np.exp(-clamp_val):.2f}, {np.exp(clamp_val):.2f}])", flush=True)
    model = BoundedResidualMLP(in_dim=Xt_aug.shape[1], hidden=64, clamp=clamp_val).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=3e-5)

    Xt_t = torch.from_numpy(Xt_aug_n).float().to(device)
    Xs_t = torch.from_numpy(Xs_aug_n).float().to(device)
    yt_t = torch.from_numpy(yt).float().to(device)
    pred_lin_train_t = torch.from_numpy(pred_lin_train).float().to(device)
    pred_lin_test_t  = torch.from_numpy(pred_lin_test).float().to(device)

    best_val_mape = float("inf")
    best_state = None
    BATCH = 4096
    n_epochs = 200
    print(f"  training {n_epochs} epochs, batch={BATCH} ...", flush=True)
    train_indices = np.where(train_mask)[0]
    for epoch in range(n_epochs):
        # shuffle and batch
        rng.shuffle(train_indices)
        model.train()
        running_loss = 0.0
        n_batches = 0
        for b0 in range(0, len(train_indices), BATCH):
            idx = train_indices[b0:b0+BATCH]
            x_b = Xt_t[idx]
            y_b = yt_t[idx]
            c_lin_b = pred_lin_train_t[idx]
            logit = model(x_b).squeeze(-1)
            mult = torch.exp(logit)
            pred = c_lin_b * mult
            # MAPE loss
            loss = (torch.abs(pred - y_b) / torch.clamp(y_b, min=1e-3)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()
            n_batches += 1
        sched.step()
        # val
        model.eval()
        with torch.no_grad():
            x_v = Xt_t[val_idx]; y_v = yt_t[val_idx]; c_lin_v = pred_lin_train_t[val_idx]
            logit_v = model(x_v).squeeze(-1)
            pred_v = c_lin_v * torch.exp(logit_v)
            val_mape = float((torch.abs(pred_v - y_v) / torch.clamp(y_v, min=1e-3)).mean()) * 100
        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"    epoch {epoch:3d}  train_loss={running_loss/n_batches:.4f}  val_MAPE={val_mape:.3f}%", flush=True)
        if val_mape < best_val_mape:
            best_val_mape = val_mape
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    print(f"  best val MAPE: {best_val_mape:.4f}%", flush=True)
    model.load_state_dict(best_state)

    # Eval on test
    model.eval()
    with torch.no_grad():
        logit_te = model(Xs_t).squeeze(-1)
        mult_te = torch.exp(logit_te)
        pred_test_t = pred_lin_test_t * mult_te
    pred_test = pred_test_t.cpu().numpy()
    stats_per_design("C. analytic + bounded MLP residual (FINAL)", pred_test, ys, test_design_arr)

    # Save
    out = {"split": "canonical (configs/config.py)",
           "DESIGNS_TRAIN": DESIGNS_TRAIN,
           "DESIGNS_TEST": DESIGNS_TEST,
           "stage_A_prior_coefs": PRIOR_C_PER_UM2,
           "stage_B_NNLS_coefs": {fcols[i]: float(c[i]) for i in range(len(fcols))},
           "stage_B_test_MAPE_combined": float(np.mean(np.abs(pred_lin_test - ys) / ys) * 100),
           "stage_C_test_MAPE_combined": float(np.mean(np.abs(pred_test - ys) / ys) * 100),
           "best_val_MAPE": best_val_mape,
           "n_epochs": n_epochs,
           "feat_dim": Xt_aug.shape[1]}
    with open(_V3 / "outputs" / "cgnd_phase1_hybrid.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved.", flush=True)


if __name__ == "__main__":
    main()
