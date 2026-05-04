"""
Calibration Solver — joint NNLS for ρ_layer (per-layer GND density) and
(s_diag, s_cross) CPL layer-pair scales.

Formulation (see docs/distillation_log.md §2):

    For each train net i:
        Σ_layer  ρ_layer[layer] · A_primary[i, layer]
        + s_diag  · A_power_diag[i]
        + s_cross · A_power_cross[i]
        ≈ golden_gnd[i] − c_vss_pred[i]

    For each (net i, signal aggressor a):
        s_diag · B_diag[i, a] + s_cross · B_cross[i, a]
        ≈ golden_cpl[i, a]

Stack into one big matrix M (K+2 unknowns). Solve via scipy.optimize.nnls.

Output: writes calibration_init.json that the model can consume at init time.

CLI:
    python3 -m src.data.calibration_solver \\
        --phase1 phase1_full.pkl \\
        --phase2 phase2_full.pkl \\
        --output calibration_init.json \\
        --holdout intel22_wb_conmax_top_f3 intel22_ldpc_decoder_802_3an_f3 intel22_vga_enh_top_f3
"""
from __future__ import annotations
import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize


# Numerical helpers --------------------------------------------------------

def softplus_inv(rho: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Stable softplus^{-1}(rho). Floors rho at `floor` to avoid log(0)."""
    rho_safe = np.clip(rho, a_min=floor, a_max=None)
    # log(exp(rho) - 1). For large rho, ≈ rho (avoid overflow); for small rho,
    # ≈ log(rho).
    out = np.where(
        rho_safe > 30.0,
        rho_safe,
        np.log(np.expm1(rho_safe)),
    )
    return out


def softplus(x: np.ndarray) -> np.ndarray:
    """Stable softplus."""
    return np.where(x > 30.0, x, np.log1p(np.exp(np.clip(x, a_min=-50.0, a_max=30.0))))


def make_layer_bucket_map(z_anchors: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Map K z-anchors to physical-metal buckets so NNLS doesn't fight
    collinearity from same-metal top/bottom anchors.

    The buckets mirror flux_head._make_gnd_cap_density_init's z-thresholds.
    Each anchor belongs to exactly one bucket; identical buckets share the
    same fitted ρ. After solve, ρ_anchor[j] = bucket_ρ[bucket_idx[j]].

    Returns:
        bucket_idx: (K,) int — bucket index per anchor.
        bucket_names: list of strings (length = number of buckets).
    """
    bucket_names = [
        'pre_M1',  # z < 0.40
        'M1',      # 0.40 ≤ z < 0.60
        'M2',      # 0.60 ≤ z < 0.75
        'M3',      # 0.75 ≤ z < 0.90
        'M4',      # 0.90 ≤ z < 1.05
        'M5',      # 1.05 ≤ z < 1.20
        'M6',      # 1.20 ≤ z < 1.45
        'upper',   # 1.45 ≤ z < 4.50
        'top',     # 4.50 ≤ z < 6.00
        'others',  # z ≥ 6.00
    ]
    def _b(z):
        if z < 0.40: return 0
        if z < 0.60: return 1
        if z < 0.75: return 2
        if z < 0.90: return 3
        if z < 1.05: return 4
        if z < 1.20: return 5
        if z < 1.45: return 6
        if z < 4.50: return 7
        if z < 6.00: return 8
        return 9
    bucket_idx = np.asarray([_b(float(z)) for z in z_anchors], dtype=np.int64)
    return bucket_idx, bucket_names


# Build NNLS matrix --------------------------------------------------------

def assemble_nnls(
    df1: pd.DataFrame,                # phase 1: A_primary[L*], golden_gnd, golden_cpl
    df2: pd.DataFrame,                # phase 2: A_power_*, B_*, c_vss_pred
    K: int,                           # number of layer anchors
    holdout_designs: set[str],
    bucket_idx: np.ndarray | None = None,    # (K,) int; if given, collapse columns by bucket
    coverage_min: int = 5,            # require ≥ this many supporting nets per BUCKET (or layer)
) -> tuple[np.ndarray, np.ndarray, dict, list[tuple[str, str]]]:
    """Build NNLS matrix M and target y.

    Returns:
        M:  (n_eq, K+2) — column order [ρ_0, ..., ρ_{K-1}, s_diag, s_cross].
        y:  (n_eq,)     — RHS.
        eqlog: dict with row labels and slicing info for diagnostics.
        keys: list of (design, net) for the GND rows (for residual analysis).
    """
    # Filter holdout from training rows.
    df1_train = df1[~df1['design'].isin(holdout_designs)].reset_index(drop=True)
    df2_train = df2[~df2['design'].isin(holdout_designs)].reset_index(drop=True)

    # Index df2 by (design, net) for fast lookup.
    df2_by_key = df2_train.set_index(['design', 'net'])

    A_cols = [f'A_primary_L{j}' for j in range(K)]

    # Determine the GND parameterization (per-anchor or per-bucket).
    if bucket_idx is not None:
        n_buckets = int(bucket_idx.max()) + 1
        n_gnd_unknowns = n_buckets
    else:
        n_buckets = K
        n_gnd_unknowns = K

    M_rows: list[np.ndarray] = []
    y_vals: list[float] = []
    keys: list[tuple[str, str]] = []
    cpl_keys: list[tuple[str, str, str]] = []  # (design, net, aggr)
    n_gnd_rows = 0
    n_cpl_rows = 0
    skipped_phase2_missing = 0

    for _, row in df1_train.iterrows():
        design = row['design']; net = row['net']
        gnd_target = float(row['golden_gnd_fF'])
        if not np.isfinite(gnd_target) or gnd_target <= 0.0:
            continue
        try:
            ph2_row = df2_by_key.loc[(design, net)]
        except KeyError:
            skipped_phase2_missing += 1
            continue
        # Defend against df2_by_key returning a DataFrame (multiple rows) — pick first.
        if isinstance(ph2_row, pd.DataFrame):
            ph2_row = ph2_row.iloc[0]

        # GND row: [ρ coeffs (per bucket if bucketing), s_diag, s_cross]
        gnd_coef = np.zeros(n_gnd_unknowns + 2, dtype=np.float64)
        A_per_anchor = row[A_cols].to_numpy(dtype=np.float64)
        if bucket_idx is not None:
            # Sum A_primary across anchors in same bucket.
            for j in range(K):
                gnd_coef[bucket_idx[j]] += A_per_anchor[j]
        else:
            gnd_coef[:K] = A_per_anchor
        gnd_coef[n_gnd_unknowns]     = float(ph2_row['A_power_diag'])
        gnd_coef[n_gnd_unknowns + 1] = float(ph2_row['A_power_cross'])
        gnd_rhs = gnd_target - float(ph2_row['c_vss_pred_fF'])
        # Skip if all-zero coefficient row (no information about any unknown).
        if not np.any(gnd_coef > 0):
            continue
        # NNLS requires y >= 0 ideally; rhs can be slightly negative if c_vss
        # over-estimates. Clip floor at 0 (rare, treat as "barely enough").
        gnd_rhs = max(gnd_rhs, 0.0)
        M_rows.append(gnd_coef)
        y_vals.append(gnd_rhs)
        keys.append((design, net))
        n_gnd_rows += 1

        # CPL rows: per signal aggressor.
        cpl_per_aggr = pickle.loads(row['cpl_per_aggr_pickled'])
        B_diag_dict  = pickle.loads(ph2_row['B_diag_pickled'])
        B_cross_dict = pickle.loads(ph2_row['B_cross_pickled'])
        all_aggrs = set(cpl_per_aggr) | set(B_diag_dict) | set(B_cross_dict)
        for aggr in all_aggrs:
            golden_a = float(cpl_per_aggr.get(aggr, 0.0))
            bd = float(B_diag_dict.get(aggr, 0.0))
            bc = float(B_cross_dict.get(aggr, 0.0))
            # Drop edges with no physics base for either coefficient AND no golden;
            # nothing to fit.
            if (bd <= 0.0 and bc <= 0.0) and golden_a <= 0.0:
                continue
            # Skip aggressors where golden=0 but B>0 — model says there should
            # be edges but golden has no entry. Could be:
            # (a) StarRC trims very small couplings — fitting these would push
            #     scales DOWN (not what we want).
            # (b) cuboid_net_names mismatch with SPEF aggressor naming.
            # Either way, exclude from solve. The reverse case (golden>0 but
            # B=0) means the model has no physics base for an aggressor that
            # actually couples — this is an architectural miss, not something
            # NNLS can fix; also exclude.
            if (bd <= 0.0 and bc <= 0.0) or golden_a <= 0.0:
                continue

            cpl_coef = np.zeros(n_gnd_unknowns + 2, dtype=np.float64)
            cpl_coef[n_gnd_unknowns]     = bd
            cpl_coef[n_gnd_unknowns + 1] = bc
            M_rows.append(cpl_coef)
            y_vals.append(golden_a)
            cpl_keys.append((design, net, aggr))
            n_cpl_rows += 1

    M = np.asarray(M_rows, dtype=np.float64)
    y = np.asarray(y_vals, dtype=np.float64)

    # Per-bucket (or per-layer) coverage: how many GND rows had non-zero A.
    per_unit_coverage = (M[:n_gnd_rows, :n_gnd_unknowns] > 0).sum(axis=0)
    sparsely_populated = np.where(per_unit_coverage < coverage_min)[0].tolist()

    eqlog = {
        'n_gnd_rows': int(n_gnd_rows),
        'n_cpl_rows': int(n_cpl_rows),
        'n_gnd_unknowns': int(n_gnd_unknowns),
        'n_unknowns': int(n_gnd_unknowns + 2),
        'skipped_phase2_missing': int(skipped_phase2_missing),
        'per_unit_coverage':         per_unit_coverage.tolist(),
        'sparsely_populated_units':  sparsely_populated,
        'cpl_keys_pickled':          cpl_keys,
        'bucket_idx':                bucket_idx.tolist() if bucket_idx is not None else None,
    }
    return M, y, eqlog, keys


def solve_with_fallback(
    M: np.ndarray, y: np.ndarray, K: int, eqlog: dict,
    fallback_rho_per_anchor: np.ndarray,
    bucket_idx: np.ndarray | None,
) -> tuple[np.ndarray, float, float, float]:
    """Solve NNLS. For sparsely-populated buckets/layers, mask their column
    and fill from `fallback_rho_per_anchor` post-hoc (median of fallback values
    of anchors in that bucket).

    Returns (rho_per_anchor (length K), s_diag, s_cross, residual_norm).
    """
    n_gnd = int(eqlog['n_gnd_unknowns'])
    sparse = eqlog['sparsely_populated_units']
    keep_mask = np.ones(n_gnd + 2, dtype=bool)
    keep_mask[sparse] = False
    M_active = M[:, keep_mask]

    print(f"  NNLS: M shape={M.shape}, kept columns={keep_mask.sum()} of {n_gnd+2} "
          f"(dropped sparsely-populated: {sparse})")
    t0 = time.time()
    x_active, residual = scipy.optimize.nnls(M_active, y, maxiter=200_000)
    print(f"  NNLS done in {time.time() - t0:.1f}s, residual={residual:.4f}")

    x_full = np.zeros(n_gnd + 2, dtype=np.float64)
    x_full[keep_mask] = x_active

    # Bucket fallback: pick median fallback value over anchors in that bucket.
    if bucket_idx is not None:
        for b in sparse:
            anchor_ids = np.where(bucket_idx == b)[0]
            if len(anchor_ids) > 0:
                x_full[b] = float(np.median(fallback_rho_per_anchor[anchor_ids]))
            else:
                x_full[b] = 1.0
    else:
        for L in sparse:
            x_full[L] = fallback_rho_per_anchor[L]

    s_diag, s_cross = float(x_full[n_gnd]), float(x_full[n_gnd + 1])

    # Expand bucket ρ → per-anchor ρ.
    if bucket_idx is not None:
        rho_per_anchor = np.asarray([x_full[bucket_idx[j]] for j in range(K)],
                                     dtype=np.float64)
    else:
        rho_per_anchor = x_full[:K]

    return rho_per_anchor, s_diag, s_cross, float(residual)


def evaluate_holdout(
    df1: pd.DataFrame, df2: pd.DataFrame, K: int,
    rho: np.ndarray, s_diag: float, s_cross: float,
    holdout_designs: set[str],
) -> dict:
    """Per-design holdout MAPE for GND and per-aggressor CPL."""
    df2_by_key = df2.set_index(['design', 'net'])
    A_cols = [f'A_primary_L{j}' for j in range(K)]

    out = {}
    for design in holdout_designs:
        sub = df1[df1['design'] == design]
        if len(sub) == 0:
            continue
        gnd_errs, cpl_errs = [], []
        for _, row in sub.iterrows():
            net = row['net']
            try:
                ph2 = df2_by_key.loc[(design, net)]
            except KeyError:
                continue
            if isinstance(ph2, pd.DataFrame):
                ph2 = ph2.iloc[0]

            A = row[A_cols].to_numpy(dtype=np.float64)
            pred_gnd = float(np.sum(A * rho)
                             + s_diag * float(ph2['A_power_diag'])
                             + s_cross * float(ph2['A_power_cross'])
                             + float(ph2['c_vss_pred_fF']))
            gnd_target = float(row['golden_gnd_fF'])
            if gnd_target > 0.005:
                gnd_errs.append(abs(pred_gnd - gnd_target) / gnd_target)

            B_diag_dict  = pickle.loads(ph2['B_diag_pickled'])
            B_cross_dict = pickle.loads(ph2['B_cross_pickled'])
            cpl_per_aggr = pickle.loads(row['cpl_per_aggr_pickled'])
            for aggr, gold in cpl_per_aggr.items():
                if gold <= 0.005: continue
                bd = float(B_diag_dict.get(aggr, 0.0))
                bc = float(B_cross_dict.get(aggr, 0.0))
                pred_a = bd * s_diag + bc * s_cross
                cpl_errs.append(abs(pred_a - gold) / gold)

        out[design] = {
            'n_gnd': len(gnd_errs),
            'gnd_mape': float(np.mean(gnd_errs)) if gnd_errs else float('nan'),
            'gnd_median_mape': float(np.median(gnd_errs)) if gnd_errs else float('nan'),
            'n_cpl': len(cpl_errs),
            'cpl_mape': float(np.mean(cpl_errs)) if cpl_errs else float('nan'),
            'cpl_median_mape': float(np.median(cpl_errs)) if cpl_errs else float('nan'),
        }
    return out


def make_fallback_rho(z_anchors: np.ndarray) -> np.ndarray:
    """Mirror flux_head._make_gnd_cap_density_init's hardcoded values directly
    in ρ-space (fF/μm²)."""
    def density_for_z(z):
        if   z < 0.40: return 2.50
        elif z < 0.60: return 2.50
        elif z < 0.75: return 3.00
        elif z < 0.90: return 3.00
        elif z < 1.05: return 2.75
        elif z < 1.20: return 2.75
        elif z < 1.45: return 2.50
        elif z < 4.50: return 1.50
        elif z < 6.00: return 4.00
        else:           return 2.00
    return np.asarray([density_for_z(float(z)) for z in z_anchors], dtype=np.float64)


# Main ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase1', type=str, required=True)
    ap.add_argument('--phase2', type=str, required=True)
    ap.add_argument('--output', type=str, required=True)
    ap.add_argument('--holdout', nargs='+', default=[
        'intel22_wb_conmax_top_f3',
        'intel22_ldpc_decoder_802_3an_f3',
        'intel22_vga_enh_top_f3',
    ], help='Designs to hold out from NNLS fit (used for cross-validation MAPE).')
    ap.add_argument('--coverage_min', type=int, default=5,
                    help='Minimum supporting nets required per layer; sparser layers fall back to hardcoded.')
    ap.add_argument('--use_buckets', action='store_true', default=True,
                    help='Collapse z anchors into 10 physical-metal buckets to avoid same-metal collinearity.')
    ap.add_argument('--no_buckets', dest='use_buckets', action='store_false',
                    help='Disable bucketing; solve K=29 per-anchor unknowns (will alias on collinear anchors).')
    args = ap.parse_args()

    print(">>> Loading phase 1 + phase 2 outputs...")
    df1 = pd.read_pickle(args.phase1)
    df2 = pd.read_pickle(args.phase2)
    print(f"  phase1: {len(df1)} nets")
    print(f"  phase2: {len(df2)} nets")

    # Sidecar: K, z_anchors
    sidecar1 = Path(args.phase1).parent / (Path(args.phase1).stem + '_meta.json')
    with open(sidecar1) as f:
        meta1 = json.load(f)
    K = int(meta1['K_layers'])
    z_anchors = np.asarray(meta1['z_anchors_um'], dtype=np.float64)

    holdout_set = set(args.holdout)
    print(f">>> Holdout designs: {sorted(holdout_set)}")
    fallback_rho = make_fallback_rho(z_anchors)
    print(f">>> Fallback rho (hardcoded init): {fallback_rho.round(2).tolist()}")

    bucket_idx = None
    bucket_names: list[str] = []
    if args.use_buckets:
        bucket_idx, bucket_names = make_layer_bucket_map(z_anchors)
        print(f">>> Layer buckets: {bucket_names}")
        print(f"    anchor → bucket: {bucket_idx.tolist()}")

    print(">>> Assembling NNLS system...")
    M, y, eqlog, gnd_keys = assemble_nnls(
        df1, df2, K, holdout_set,
        bucket_idx=bucket_idx,
        coverage_min=args.coverage_min,
    )
    print(f"  rows={M.shape[0]}, cols={M.shape[1]}")
    print(f"    GND rows={eqlog['n_gnd_rows']}  CPL rows={eqlog['n_cpl_rows']}")
    print(f"    skipped_phase2_missing={eqlog['skipped_phase2_missing']}")
    print(f"    per_unit_coverage={eqlog['per_unit_coverage']}")
    print(f"    sparsely_populated_units={eqlog['sparsely_populated_units']}")

    rho, s_diag, s_cross, residual = solve_with_fallback(
        M, y, K, eqlog, fallback_rho, bucket_idx=bucket_idx,
    )

    # Diagnostics
    n_gnd = int(eqlog['n_gnd_unknowns'])
    print(f">>> Solved values:")
    if bucket_idx is not None:
        bucket_solved = np.zeros(int(bucket_idx.max()) + 1, dtype=np.float64)
        for j in range(K):
            bucket_solved[bucket_idx[j]] = rho[j]
        for b in range(len(bucket_names)):
            marker = "  *fallback*" if b in eqlog['sparsely_populated_units'] else ""
            print(f"  bucket {b} ({bucket_names[b]}): rho={bucket_solved[b]:.3f} fF/μm²{marker}")
    else:
        for j in range(K):
            marker = "  *fallback*" if j in eqlog['sparsely_populated_units'] else ""
            print(f"  layer {j} (z={z_anchors[j]:.3f}): rho={rho[j]:.3f} fF/μm²{marker}")
    print(f"  s_diag = {s_diag:.3f}")
    print(f"  s_cross = {s_cross:.3f}")
    print(f"  v3 hardcoded reference: s_diag=8.0, s_cross=5.0")

    # Pooled train MAPE — use bucketed coefficients in the M matrix
    # (assemble_nnls already collapsed columns when bucket_idx is set).
    if bucket_idx is not None:
        # Reconstruct ρ in bucket-space for matrix multiply.
        rho_bucket = bucket_solved
    else:
        rho_bucket = rho
    pred = M[:eqlog['n_gnd_rows'], :n_gnd] @ rho_bucket \
         + M[:eqlog['n_gnd_rows'], n_gnd] * s_diag \
         + M[:eqlog['n_gnd_rows'], n_gnd + 1] * s_cross
    target = y[:eqlog['n_gnd_rows']]
    valid = target > 0.005
    pooled_gnd_mape = float(np.mean(np.abs(pred[valid] - target[valid]) / target[valid])) if valid.any() else float('nan')
    print(f"  pooled train GND MAPE: {pooled_gnd_mape:.3f}")

    pred_cpl = M[eqlog['n_gnd_rows']:, n_gnd] * s_diag + M[eqlog['n_gnd_rows']:, n_gnd + 1] * s_cross
    target_cpl = y[eqlog['n_gnd_rows']:]
    valid = target_cpl > 0.005
    pooled_cpl_mape = float(np.mean(np.abs(pred_cpl[valid] - target_cpl[valid]) / target_cpl[valid])) if valid.any() else float('nan')
    print(f"  pooled train CPL per-aggr MAPE: {pooled_cpl_mape:.3f}")

    # Holdout
    print(">>> Holdout evaluation...")
    holdout_metrics = evaluate_holdout(df1, df2, K, rho, s_diag, s_cross, holdout_set)
    for design, m in holdout_metrics.items():
        print(f"  {design}: GND mape={m['gnd_mape']:.3f} "
              f"(median {m['gnd_median_mape']:.3f}, n={m['n_gnd']})  "
              f"CPL mape={m['cpl_mape']:.3f} (median {m['cpl_median_mape']:.3f}, n={m['n_cpl']})")

    # Per-design IQR diagnostic
    df1_train = df1[~df1['design'].isin(holdout_set)]
    A_cols = [f'A_primary_L{j}' for j in range(K)]
    df1_train = df1_train.copy()
    df1_train['_A_dot_rho'] = (df1_train[A_cols].to_numpy(dtype=np.float64) * rho).sum(axis=1)
    df1_train['_pred_ratio'] = df1_train['_A_dot_rho'] / df1_train['golden_gnd_fF'].clip(lower=1e-6)
    iqr = df1_train.groupby('design')['_pred_ratio'].quantile([0.25, 0.75]).unstack()
    iqr['iqr_width'] = iqr[0.75] - iqr[0.25]
    print(">>> Per-design ratio IQR (pred-from-rho only / golden_gnd):")
    print(iqr.round(3).to_string())

    # Output JSON
    rho_softplus_inv = softplus_inv(rho)
    s_diag_softplus_inv  = softplus_inv(np.asarray([s_diag if s_diag > 0 else 1e-3]))[0]
    s_cross_softplus_inv = softplus_inv(np.asarray([s_cross if s_cross > 0 else 1e-3]))[0]

    out_json = {
        'schema_version': '2.0',
        'source': {
            'phase1_pkl': str(args.phase1),
            'phase2_pkl': str(args.phase2),
            'phase1_meta': meta1,
            'designs_train':   sorted(set(df1['design']) - holdout_set),
            'designs_holdout': sorted(holdout_set),
            'n_train_nets':    int(eqlog['n_gnd_rows']),
            'n_train_cpl_pairs': int(eqlog['n_cpl_rows']),
        },
        'metal_z_anchors_um':            z_anchors.tolist(),
        'gnd_density_softplus_inv':      rho_softplus_inv.tolist(),
        'gnd_density_fF_per_um2':        rho.tolist(),
        'gnd_density_fallback_layers':   eqlog['sparsely_populated_units'],
        'cpl_pair_softplus_inv_diag':    float(s_diag_softplus_inv),
        'cpl_pair_softplus_inv_cross':   float(s_cross_softplus_inv),
        'cpl_pair_diag_value':           float(s_diag),
        'cpl_pair_cross_value':          float(s_cross),
        'diagnostics': {
            'pooled_gnd_mape':           pooled_gnd_mape,
            'pooled_cpl_mape':           pooled_cpl_mape,
            'holdout':                   holdout_metrics,
            'per_unit_coverage':         eqlog['per_unit_coverage'],
            'sparsely_populated_units':  eqlog['sparsely_populated_units'],
            'bucketing':                 {
                'enabled':       (bucket_idx is not None),
                'bucket_names':  bucket_names,
                'bucket_idx':    bucket_idx.tolist() if bucket_idx is not None else None,
            },
            'nnls_residual':             residual,
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_json, f, indent=2)

    print(f">>> Wrote calibration_init: {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
