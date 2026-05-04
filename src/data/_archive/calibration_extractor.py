"""
Calibration Extractor — offline data-driven physics-init pipeline.

Phase 1 (this module's main): Pure geometric + SPEF aggregation, no model run.
Builds per-(design, net) tables:
    - golden_gnd[(design, net)]            : SPEF *CAP gnd_caps sum, fF
    - golden_cpl[(design, net)][aggr_net]  : SPEF *CAP cpl_caps sum per aggressor, fF
    - A_primary[(design, net)][layer_idx]  : Σ target cuboids on layer:
                                             (bottom_area + fringe_init[layer]*sidewall) * core_ratio
                                             (μm²; coefficient of softplus(layer_scale_phys_gnd[layer]))

Phase 2 (extract_physics_base, runs model in physics-only mode): per-edge
sparse_cpl coefficients for the (s_diag, s_cross) unknowns. Implemented in
extract_physics_base() — see the function docstring.

Outputs Phase 1 parquet at <PROCESSED_DIR>/calibration_extract/phase1.parquet,
Phase 2 parquet at .../phase2.parquet. The NNLS solver
(src/data/calibration_solver.py) merges both into a calibration_init.json.

Mirrors the model's exact aggregation conventions (datasets.py core_ratio
formula, finetuner.py:486-513 aggregation, flux_head.py:67-77 z-anchor
discretization, flux_head.py:286-291 gnd_area formula). Any deviation here
breaks the NNLS fit, since the model trains against its own conventions.

CLI:
    python3 -m src.data.calibration_extractor phase1 \\
        --output /data/PINNPEX/data/processed/intel22/calibration_extract/phase1.parquet
    python3 -m src.data.calibration_extractor phase2 \\
        --gpu 4 --output .../phase2.parquet
"""
from __future__ import annotations
import argparse
import gzip
import json
import math
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import configs.config as cfg
from src.evaluation.compare_spef import parse_spef_with_coordinates
from src.preprocessing.layer_parser import LayerInfoParser

try:
    import isal.igzip as igzip
    _gzip_open = igzip.open
except ImportError:
    _gzip_open = gzip.open


# ----- Layer anchor + fringe init reproduction -------------------------------

def compute_metal_z_anchors(layer_map: dict) -> np.ndarray:
    """Mirror flux_head.py:67-77 — unique sorted z positions across all layers."""
    if layer_map is None or len(layer_map) == 0:
        return np.linspace(-10.0, 50.0, 61, dtype=np.float32)
    zs = sorted({float(info.get('z_pos', 0.0)) for info in layer_map.values()})
    unique_z: list[float] = []
    for z in zs:
        if not unique_z or abs(z - unique_z[-1]) > 0.05:
            unique_z.append(z)
    return np.asarray(unique_z, dtype=np.float32)


def make_gnd_fringe_scale_init(z_anchors: np.ndarray) -> np.ndarray:
    """Mirror flux_head.py _make_gnd_fringe_scale_init — pre-sigmoid logits per anchor."""
    def init_for_z(z: float) -> float:
        if   z < 0.55: return -2.94   # M1 fringe_frac ≈ 0.05
        elif z < 0.70: return -2.44   # M2 ≈ 0.08
        elif z < 0.84: return -2.20   # M3 ≈ 0.10
        elif z < 0.99: return -1.73   # M4 ≈ 0.15
        elif z < 1.14: return -1.39   # M5 ≈ 0.20
        elif z < 1.45: return -0.85   # M6 ≈ 0.30
        elif z < 4.60: return -1.73   # M7 ≈ 0.15
        else:           return -1.39  # M8 ≈ 0.20
    return np.asarray([init_for_z(float(z)) for z in z_anchors], dtype=np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ----- core_ratio reproduction (mirror datasets.py:84-113) -------------------

def compute_core_ratios(cuboids: np.ndarray, abs_geo: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Per-cuboid core_ratio. Mirrors datasets.py:84-113 in pure numpy.

    Args:
        cuboids: (N, ≥6) tensor — last dims include cuboid (cx, cy, cz, w, h, d) center+size.
        abs_geo: (N, 6)        — absolute world-space (x_min, y_min, z_min, w, h, d).
        origin:  (3,)          — tile center in absolute μm.

    Returns: (N,) float32 core ratios in [0, 1].
    """
    W_X, W_Y = float(cfg.WINDOW_SIZE[0]), float(cfg.WINDOW_SIZE[1])
    O_X, O_Y = float(cfg.TILING_OVERLAP), float(cfg.TILING_OVERLAP)
    CORE_HW_X = (W_X - O_X) / 2.0
    CORE_HW_Y = (W_Y - O_Y) / 2.0

    w = cuboids[:, 3]; h = cuboids[:, 4]; d = cuboids[:, 5]
    # Absolute lower corner shifted to tile origin (matches datasets.py:91-92).
    lx = abs_geo[:, 0] - origin[0]
    ly = abs_geo[:, 1] - origin[1]

    l_max = np.max(cuboids[:, 3:6], axis=1)
    c_len = np.zeros_like(l_max)

    mask_x = (w >= h) & (w >= d)
    mask_y = (h > w) & (h >= d)
    mask_z = ~(mask_x | mask_y)

    c_len[mask_x] = np.clip(
        np.minimum(lx[mask_x] + w[mask_x] / 2.0, CORE_HW_X)
      - np.maximum(lx[mask_x] - w[mask_x] / 2.0, -CORE_HW_X),
        a_min=0.0, a_max=None,
    )
    c_len[mask_y] = np.clip(
        np.minimum(ly[mask_y] + h[mask_y] / 2.0, CORE_HW_Y)
      - np.maximum(ly[mask_y] - h[mask_y] / 2.0, -CORE_HW_Y),
        a_min=0.0, a_max=None,
    )
    via_mask = (np.abs(lx) <= CORE_HW_X) & (np.abs(ly) <= CORE_HW_Y)
    valid_via = mask_z & via_mask
    c_len[valid_via] = d[valid_via]

    safe_l_max = np.clip(l_max, a_min=1e-6, a_max=None)
    core_ratios = np.where(l_max > 0, c_len / safe_l_max, 0.0)
    return core_ratios.astype(np.float32)


# ----- Phase 1: golden + geometry aggregation --------------------------------

def parse_all_train_spefs() -> dict:
    """Parse all SPEFs and return {design_stem_without_starrc: nets_dict}."""
    out = {}
    for spef_path in cfg.TRAIN_SPEFS:
        design = spef_path.stem.replace('_starrc', '')
        t0 = time.time()
        nets = parse_spef_with_coordinates(spef_path)
        out[design] = nets
        print(f"  [SPEF] {design}: {len(nets)} nets, {time.time() - t0:.1f}s")
    return out


def iter_design_tiles(
    design: str,
    manifest: pd.DataFrame,
    keep_nets: set[str] | None = None,
) -> Iterator[tuple[str, dict]]:
    """Yield (sample_filename, loaded pkl dict) for every tile of a design.

    If `keep_nets` is given, only yield tiles whose net_name is in the set
    (used by net-centric sampling so all tiles of selected nets are scanned).
    """
    rows = manifest[manifest['design_name'] == design]
    if keep_nets is not None:
        rows = rows[rows['net_name'].isin(keep_nets)]
    db_root = Path(cfg.PROCESSED_DIR) / design
    for _, row in rows.iterrows():
        fn = row['sample_filename']
        path = db_root / fn
        if not path.exists():
            continue
        try:
            with _gzip_open(path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"    [WARN] Failed to load {fn}: {e}", file=sys.stderr)
            continue
        if not isinstance(data, dict) or 'cuboids' not in data:
            continue
        yield fn, data


def aggregate_phase1(
    train_designs: list[str],
    manifest: pd.DataFrame,
    z_anchors: np.ndarray,
    fringe_logits: np.ndarray,
    max_tiles_per_design: int | None = None,
    max_nets_per_design: int | None = None,
    progress: bool = True,
) -> tuple[dict, dict, dict]:
    """Walk train tiles, build:
        A_primary[(design, net)]       : (K,) per-layer effective gnd_area sum
        n_target_cuboids[(design, net)]: int  (diagnostic)
    Returns (A_primary, n_target_cuboids, n_tiles_per_net).
    """
    K = int(z_anchors.size)
    fringe_frac = sigmoid(fringe_logits)         # (K,)
    A_primary: dict = defaultdict(lambda: np.zeros(K, dtype=np.float64))
    n_target_cuboids: dict = defaultdict(int)
    n_tiles_per_net: dict = defaultdict(int)

    rng = np.random.default_rng(seed=42)
    for design in train_designs:
        t0 = time.time()
        keep_nets: set[str] | None = None
        if max_nets_per_design is not None:
            d_rows = manifest[manifest['design_name'] == design]
            unique_nets = d_rows['net_name'].drop_duplicates().to_numpy()
            if len(unique_nets) > max_nets_per_design:
                chosen = rng.choice(unique_nets, max_nets_per_design, replace=False)
            else:
                chosen = unique_nets
            keep_nets = set(chosen.tolist())

        seen = 0
        for sample_fn, data in iter_design_tiles(design, manifest, keep_nets=keep_nets):
            if max_tiles_per_design is not None and seen >= max_tiles_per_design:
                break
            seen += 1
            cuboids = data['cuboids']                    # (N, C)
            if cuboids.size == 0:
                continue
            abs_geo = data.get('abs_geometries')
            origin  = np.asarray(data.get('origin', np.zeros(3)), dtype=np.float32)
            net_name = data.get('net_name', None)
            cuboid_net_names = data.get('cuboid_net_names', None)
            if abs_geo is None or net_name is None or cuboid_net_names is None:
                continue

            # Target wire cuboids = name match AND logic_flag == 1.0 (channel 7).
            # The model's c_gnd_seg masks by `is_target = cuboids[..., 7] == 1.0`
            # internally (flux_head.py:323), so pin cuboids of the target net
            # (ch7=0, semantic_type=0.5) contribute 0 to predictions even though
            # their name matches. A_primary must mirror this — using name-only
            # mask would inflate gnd_area by ~20% and cause NNLS to fit
            # ρ values too low to compensate.
            cuboid_net_names_arr = np.asarray(cuboid_net_names)
            tgt_mask = (cuboid_net_names_arr == net_name) & (cuboids[:, 7] == 1.0)
            if not tgt_mask.any():
                continue

            target_cubs = cuboids[tgt_mask]                  # (T, C)
            target_abs  = abs_geo[tgt_mask]                  # (T, 6)
            core_ratios_all = compute_core_ratios(cuboids, abs_geo, origin)
            target_core = core_ratios_all[tgt_mask]          # (T,)

            # Per-cuboid layer = closest z anchor to z_abs (mirrors flux_head.py:277-279).
            z_abs = target_cubs[:, 2]                        # (T,)
            layer_idx = np.argmin(np.abs(z_abs[:, None] - z_anchors[None, :]), axis=-1)

            # gnd_area mirrors flux_head.py:286-289:
            #   bottom_area    = max(w*h, 1e-6)
            #   sidewall_area  = 2*(w+h)*d
            #   fringe_frac[layer] applied.
            w = target_cubs[:, 3]; h = target_cubs[:, 4]; d = target_cubs[:, 5]
            bottom_area   = np.clip(w * h, a_min=1e-6, a_max=None)
            sidewall_area = 2.0 * (w + h) * d
            ff_per_cub = fringe_frac[layer_idx]              # (T,)
            gnd_area_eff = (bottom_area + ff_per_cub * sidewall_area) * target_core   # (T,)

            # Accumulate per (design, net, layer).
            key = (design, net_name)
            np.add.at(A_primary[key], layer_idx, gnd_area_eff)
            n_target_cuboids[key] += int(tgt_mask.sum())
            n_tiles_per_net[key] += 1

        if progress:
            print(f"  [GEOM] {design}: {seen} tiles, {time.time() - t0:.1f}s, "
                  f"{sum(1 for k in A_primary if k[0] == design)} unique nets")

    return dict(A_primary), dict(n_target_cuboids), dict(n_tiles_per_net)


def aggregate_golden(
    spefs: dict,
) -> tuple[dict, dict]:
    """From parse_spef_with_coordinates output, build:
        golden_gnd[(design, net)]: float fF
        golden_cpl[(design, net)][aggr_net]: float fF
    """
    golden_gnd: dict = {}
    golden_cpl: dict = defaultdict(lambda: defaultdict(float))

    for design, nets in spefs.items():
        for net_name, n in nets.items():
            gnd_total = float(sum(n['gnd_caps'].values()))
            golden_gnd[(design, net_name)] = gnd_total
            for node_a, aggr_dict in n['cpl_caps'].items():
                for aggr_net, cap in aggr_dict.items():
                    aggr_clean = aggr_net.replace('\\', '')
                    golden_cpl[(design, net_name)][aggr_clean] += float(cap)
    return golden_gnd, dict(golden_cpl)


# ----- Phase 1 main / CLI ----------------------------------------------------

def run_phase1(args) -> None:
    print(">>> Phase 1: Golden + Geometric Aggregation")
    t_start = time.time()

    # 1. Layer anchor / fringe setup (mirror flux_head)
    layer_map = LayerInfoParser(cfg.LAYERS_INFO_PATH).parse()
    z_anchors = compute_metal_z_anchors(layer_map)
    fringe_logits = make_gnd_fringe_scale_init(z_anchors)
    K = int(z_anchors.size)
    print(f"  z_anchors (K={K}): {z_anchors.tolist()}")
    print(f"  fringe_frac (sigmoid): {sigmoid(fringe_logits).round(3).tolist()}")

    # 2. SPEF parse → golden tables.
    print(">>> Parsing TRAIN_SPEFS...")
    spefs = parse_all_train_spefs()
    golden_gnd, golden_cpl = aggregate_golden(spefs)
    print(f"  total nets across {len(spefs)} designs: {len(golden_gnd)}")

    # 3. Manifest scan → A_primary per (design, net, layer).
    manifest = pd.read_csv(Path(cfg.PROCESSED_DIR) / "dataset_manifest.csv")
    train_design_stems = [p.stem for p in cfg.TRAIN_DEFS]
    print(f"  manifest rows: {len(manifest)} (filtering to {len(train_design_stems)} train designs)")
    A_primary, n_target_cuboids, n_tiles_per_net = aggregate_phase1(
        train_design_stems, manifest, z_anchors, fringe_logits,
        max_tiles_per_design=args.max_tiles_per_design,
        max_nets_per_design=args.max_nets_per_design,
    )

    # 4. Materialize a long-format DataFrame: (design, net, [A_primary 0..K-1],
    #    golden_gnd, n_target_cuboids, n_tiles, n_aggressors, n_aggrs_with_cap).
    rows: list[dict] = []
    for (design, net), A_vec in A_primary.items():
        cpl_dict = golden_cpl.get((design, net), {})
        rec = {
            'design': design,
            'net':    net,
            'golden_gnd_fF':            golden_gnd.get((design, net), float('nan')),
            'n_target_cuboids':         int(n_target_cuboids[(design, net)]),
            'n_tiles':                  int(n_tiles_per_net[(design, net)]),
            'n_aggressors':             int(len(cpl_dict)),
            'golden_cpl_total_fF':      float(sum(cpl_dict.values())),
            'cpl_per_aggr_pickled':     pickle.dumps(dict(cpl_dict)),
        }
        for j in range(K):
            rec[f'A_primary_L{j}'] = float(A_vec[j])
        rows.append(rec)
    df = pd.DataFrame(rows)

    # Orphans: SPEF nets with no tiles (filter them out — model can't predict them).
    spef_only = sum(1 for k in golden_gnd
                    if k not in A_primary and golden_gnd[k] > 1e-3)
    geom_only = sum(1 for k in A_primary
                    if k not in golden_gnd)
    print(f"  joined nets in DataFrame: {len(df)}")
    print(f"  SPEF-only (no tiles, dropped): {spef_only}")
    print(f"  geometry-only (no SPEF total, kept with NaN gold): {geom_only}")

    # Drop rows with no golden_gnd (they can't enter the NNLS).
    n_before = len(df)
    df = df.dropna(subset=['golden_gnd_fF']).reset_index(drop=True)
    print(f"  after dropping NaN-golden rows: {len(df)} (dropped {n_before - len(df)})")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # pyarrow/fastparquet not installed in this env — pickle is sufficient
    # (the consumer is also Python). Suffix is .pkl regardless of CLI input.
    if out.suffix != '.pkl':
        out = out.with_suffix('.pkl')
    df.to_pickle(out)

    # Sidecar JSON: K, z_anchors, fringe_logits, design list, run metadata.
    sidecar = out.parent / (out.stem + '_meta.json')
    with open(sidecar, 'w') as f:
        json.dump({
            'phase': 1,
            'K_layers': K,
            'z_anchors_um': z_anchors.tolist(),
            'fringe_init_logits': fringe_logits.tolist(),
            'fringe_init_frac':   sigmoid(fringe_logits).tolist(),
            'train_designs':      train_design_stems,
            'n_train_nets':       int(len(df)),
            'max_tiles_per_design': args.max_tiles_per_design,
            'window_size_um':     list(cfg.WINDOW_SIZE),
            'tiling_overlap_um':  float(cfg.TILING_OVERLAP),
            'extraction_seconds': time.time() - t_start,
        }, f, indent=2)

    # Sanity prints:
    pop_layers = (df[[f'A_primary_L{j}' for j in range(K)]] > 0).sum(axis=0)
    print(">>> Phase 1 done.")
    print(f"  parquet: {out}")
    print(f"  sidecar: {sidecar}")
    print(f"  per-layer net coverage (n nets with A_primary>0): {pop_layers.tolist()}")
    print(f"  total: {time.time() - t_start:.1f}s")


def _setup_physics_only_model(device: str) -> tuple:
    """Instantiate DeepPEX_Model and zero out learned MLPs / layer scales so the
    forward gives clean physics base outputs.

    Mirrors `diag_eval_dump.py:87-111` (--physics_only branch) but does NOT load
    a checkpoint — the learned weights would be zeroed out anyway. Encoder
    weights matter only insofar as they feed gnd_mlp/cpl_mlp Z_prime; since
    those MLPs are zeroed, encoder is irrelevant for the extracted quantities.

    Returns: (model, layer_anchors_tensor)
    """
    import torch
    from src.models.neural_field import DeepPEX_Model

    # Disable DS-PINN macro stream + GINO for simpler setup; doesn't affect
    # sparse_cpl's w_cpl extraction (purely geometric).
    cfg._use_dspinn = False
    cfg._use_gino   = False
    model = DeepPEX_Model(cfg).to(device)

    fr = model.flux_router
    with torch.no_grad():
        # gnd_mlp last linear → 0 → gnd_modifier = exp(0) = 1.0
        fr.gnd_mlp[-1].weight.zero_()
        fr.gnd_mlp[-1].bias.zero_()
        # cpl_mlp last linear → 0; bias=[0, -10] → cpl_modifier=1, cpl_residual≈0
        fr.cpl_mlp[-1].weight.zero_()
        fr.cpl_mlp[-1].bias.copy_(torch.tensor([0.0, -10.0], device=device))
        # Reset per-layer GND density to 0 → softplus(0) = log(2) — clean
        # extractable factor: c_gnd_seg = log(2)*gnd_area + c_vss_contribution.
        fr.layer_scale_phys_gnd.zero_()
        # Reset CPL layer-pair scale to 0 → softplus(0) = log(2). After dividing
        # c_cpl by this we recover w_cpl_base — but we extract w_cpl_base
        # directly from sparse_cpl['w_cpl'], so this normalization is
        # only for c_gnd_seg-side extraction.
        fr.cpl_layer_pair_log_scale.zero_()
        # KEEP vss_gnd_scale at its current (-3.0) init — c_vss contribution
        # is included as a known additive at the current init level.
        # KEEP gnd_fringe_scale at its physics-based init.

    model.eval()
    return model, fr.metal_z_anchors.detach().clone().to(device)


class _PhysicsOnlyDataset:
    """Minimal Dataset wrapping NeuralFieldFinetuneDataset but with synthetic
    labels so the existing robust_collate path runs without errors. Also
    injects 'sample_filename' into meta_dict (not present in the base class).

    We don't need labels — robust_collate just needs them to exist so MAX_AGGR
    and A_tgt are constructed. We synthesize capacitance=1.0 and empty
    coupled_caps; this gives a degenerate batch_aggr_list = [] but the rest
    of the pipeline works.
    """
    def __init__(self, manifest_subset: pd.DataFrame, pad_size: int = 1024):
        from src.data.datasets import NeuralFieldFinetuneDataset
        df = manifest_subset.copy()
        df['capacitance'] = 1.0   # so dropna passes
        df['ground_cap'] = 0.0
        df['coupled_caps'] = [{} for _ in range(len(df))]
        self._inner = NeuralFieldFinetuneDataset(cfg.PROCESSED_DIR, df, pad_size=pad_size)
        # Cache filenames for sample_filename injection
        self._filenames = list(df['sample_filename'])

    def __len__(self): return len(self._inner)
    def __getitem__(self, idx):
        result = self._inner[idx]
        if result is None:
            return None
        tensor, mask, labels_dict, meta_dict = result
        meta_dict = dict(meta_dict)  # copy (it's reference into base)
        meta_dict['sample_filename'] = self._filenames[idx]
        return tensor, mask, labels_dict, meta_dict


def run_phase2(args) -> None:
    """Phase 2: physics-only forward → per-(design, net) and per-(design, net,
    aggressor) coefficients for the (s_diag, s_cross) NNLS unknowns.

    Outputs a pickle with columns:
        design, net,
        A_power_diag, A_power_cross         (scalar per net; sums of w_cpl·core_ratio_eff
                                             over edges to power-net dsts, classified
                                             by same/different src-dst layer)
        c_vss_pred                          (scalar per net; pred GND addend at current
                                             vss_gnd_scale init — subtracted from RHS)
        B_diag_pickled                      (dict {aggr: scalar} for signal-aggr edges)
        B_cross_pickled                     (dict {aggr: scalar} same)
        n_tiles, n_edges_signal, n_edges_power
    """
    import torch
    from torch.utils.data import DataLoader
    from src.data.datasets import robust_collate

    print(">>> Phase 2: Model physics-base aggregation")
    t_start = time.time()
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    model, z_anchors_tensor = _setup_physics_only_model(device)
    K = int(z_anchors_tensor.numel())
    print(f"  model on {device}, K={K} layer anchors")

    # Build manifest subset (train designs only).
    manifest = pd.read_csv(Path(cfg.PROCESSED_DIR) / "dataset_manifest.csv")
    train_design_stems = [p.stem for p in cfg.TRAIN_DEFS]
    sub = manifest[manifest['design_name'].isin(train_design_stems)].reset_index(drop=True)
    if args.max_nets_per_design is not None:
        # Net-centric sampling: pick N nets per design, include ALL tiles of
        # those nets. Ensures per-net A_primary coverage is 100% for the
        # sampled subset (matches the full-coverage assumption phase 1 makes).
        rng = np.random.default_rng(seed=42)
        keep_rows: list[pd.DataFrame] = []
        for d_name, d_sub in sub.groupby('design_name', sort=False):
            unique_nets = d_sub['net_name'].drop_duplicates().to_numpy()
            if len(unique_nets) > args.max_nets_per_design:
                chosen = rng.choice(unique_nets, args.max_nets_per_design, replace=False)
            else:
                chosen = unique_nets
            keep_rows.append(d_sub[d_sub['net_name'].isin(chosen)])
        sub = pd.concat(keep_rows, axis=0).reset_index(drop=True)
        print(f"  net-centric sample: {len(set(zip(sub['design_name'], sub['net_name'])))} unique (design, net)")
    elif args.max_tiles_per_design is not None:
        sub = sub.groupby('design_name', group_keys=False, sort=False).head(
            args.max_tiles_per_design).reset_index(drop=True)
    print(f"  tiles to process: {len(sub)}")

    dataset = _PhysicsOnlyDataset(sub, pad_size=cfg.NF_PAD_TO_CUBOIDS)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=robust_collate, num_workers=4,
                        pin_memory=True)

    POWER_NETS = {'vss', 'vdd', 'vcc', 'gnd', 'vssx', 'vccx'}

    # Aggregation buffers keyed by (design, net):
    A_power_diag:  dict = defaultdict(float)
    A_power_cross: dict = defaultdict(float)
    c_vss_pred:    dict = defaultdict(float)
    B_diag:  dict = defaultdict(lambda: defaultdict(float))
    B_cross: dict = defaultdict(lambda: defaultdict(float))
    n_tiles_per_net:    dict = defaultdict(int)
    n_edges_signal_per_net: dict = defaultdict(int)
    n_edges_power_per_net:  dict = defaultdict(int)

    n_batches = 0
    t_proc = 0.0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            n_batches += 1
            t0 = time.time()
            cuboids, mask, labels_dict, meta_dict = batch
            cuboids = cuboids.to(device, non_blocking=True)
            mask    = mask.to(device, non_blocking=True)
            A_tgt        = labels_dict['A_tgt'].to(device, non_blocking=True)
            core_ratios  = labels_dict['core_ratios'].to(device, non_blocking=True)
            B = cuboids.shape[0]

            # Per-tile design / net.
            sample_filenames = meta_dict.get('sample_filename', None)
            target_net_names = meta_dict.get('target_net_name', None)
            cuboid_name_lists = meta_dict.get('cuboid_net_names', None)
            if sample_filenames is None or target_net_names is None or cuboid_name_lists is None:
                # Fall back to recovering design from sample_filename's df row
                raise RuntimeError("meta_dict missing expected keys; check robust_collate.")
            tile_designs = [fn.split('__')[0] for fn in sample_filenames]

            preds = model(cuboids, mask, compute_coupling=True)
            sparse = preds['sparse_cpl']
            c_gnd_seg = preds['c_gnd_seg'].float()        # (B, N)

            # --- c_vss extraction per tile per net ---
            # In physics-only mode: c_gnd_seg = log(2) * gnd_area + c_vss_contribution.
            # gnd_area = bottom + fringe[layer]*sidewall (mirror flux_head.py:286).
            # Recompute gnd_area on GPU to subtract.
            cubs = cuboids.float()
            w = cubs[..., 3]; h = cubs[..., 4]; d = cubs[..., 5]
            z_abs = cubs[..., 2]
            bottom_area    = torch.clamp(w * h, min=1e-6)
            sidewall_area  = 2.0 * (w + h) * d
            # Layer index = closest z anchor.
            z_idx = torch.argmin(torch.abs(z_abs.unsqueeze(-1) - z_anchors_tensor), dim=-1)
            # fringe_init logits → sigmoid → fringe_frac per anchor
            fringe_logits_np = make_gnd_fringe_scale_init(z_anchors_tensor.cpu().numpy())
            fringe_frac_t = torch.sigmoid(torch.from_numpy(fringe_logits_np)).to(device)
            ff = fringe_frac_t[z_idx]
            gnd_area_t = bottom_area + ff * sidewall_area     # (B, N)
            log2 = float(np.log(2.0))
            c_vss_per_cub = (c_gnd_seg - log2 * gnd_area_t)   # (B, N)
            # Mask by is_target (cuboids[..., 7] == 1.0), NOT A_tgt — the model
            # internally uses is_target to mask c_gnd_seg, so c_vss extraction
            # must use the same mask. A_tgt is name-based and includes pin
            # cuboids of the target net which have ch7=0 and contribute 0 to
            # c_gnd_seg; using A_tgt would leak negative residuals to those.
            is_target_t = (cuboids[..., 7] == 1.0).float() * (~mask).float()
            c_vss_per_cub = c_vss_per_cub * is_target_t
            # core_ratio weighting (mirrors finetuner.py:486)
            c_vss_weighted = c_vss_per_cub * core_ratios       # (B, N)
            c_vss_per_tile = c_vss_weighted.sum(dim=1)         # (B,)

            for b in range(B):
                key = (tile_designs[b], target_net_names[b].replace('\\', ''))
                c_vss_pred[key] += float(c_vss_per_tile[b].item())
                n_tiles_per_net[key] += 1

            # --- per-edge classification ---
            b_idx   = sparse['b_idx'].long()
            src_idx = sparse['src_idx'].long()
            dst_idx = sparse['dst_idx'].long()
            w_cpl   = sparse['w_cpl'].float()                  # raw geometric base (E,)

            if b_idx.numel() == 0:
                t_proc += time.time() - t0
                if n_batches % 50 == 0:
                    print(f"  [{n_batches}] {t_proc:.1f}s elapsed, no edges in this batch")
                continue

            # core_ratio_eff: where A_tgt[b, src]>0, use core_ratios[b, src]; else core_ratios[b, dst]
            cr_src = core_ratios[b_idx, src_idx]
            cr_dst = core_ratios[b_idx, dst_idx]
            tgt_at_src = A_tgt[b_idx, src_idx] > 0
            cr_eff = torch.where(tgt_at_src, cr_src, cr_dst)
            base = w_cpl * cr_eff                                # (E,)

            # src/dst layer
            z_src = cuboids[b_idx, src_idx, 2]
            z_dst = cuboids[b_idx, dst_idx, 2]
            z_idx_src = torch.argmin(torch.abs(z_src.unsqueeze(-1) - z_anchors_tensor), dim=-1)
            z_idx_dst = torch.argmin(torch.abs(z_dst.unsqueeze(-1) - z_anchors_tensor), dim=-1)
            same_layer = (z_idx_src == z_idx_dst)

            # Per-batch power-net mask of dst.
            # cuboid_name_lists: per-tile list of names; resolve dst's net name.
            # We move data to CPU for this Python-loop classification.
            b_idx_cpu   = b_idx.cpu().numpy()
            dst_idx_cpu = dst_idx.cpu().numpy()
            src_idx_cpu = src_idx.cpu().numpy()
            base_cpu    = base.detach().cpu().numpy()
            same_layer_cpu = same_layer.cpu().numpy()

            for e in range(b_idx_cpu.shape[0]):
                b = int(b_idx_cpu[e])
                d_idx = int(dst_idx_cpu[e])
                names = cuboid_name_lists[b]
                if d_idx >= len(names):
                    continue
                dst_net = str(names[d_idx]).replace('\\', '')
                src_net = str(names[int(src_idx_cpu[e])]).replace('\\', '')
                # The "target net" for this tile: tile_designs[b] / target_net_names[b]
                target_net = target_net_names[b].replace('\\', '')
                key = (tile_designs[b], target_net)
                contrib = float(base_cpu[e])
                is_same = bool(same_layer_cpu[e])

                if dst_net.lower() in POWER_NETS:
                    if is_same: A_power_diag[key]  += contrib
                    else:        A_power_cross[key] += contrib
                    n_edges_power_per_net[key] += 1
                else:
                    # Signal aggressor; but only count when src is the target
                    # (consistent with how sparse_cpl edges are built —
                    # is_target on src side, signal_aggr on dst side).
                    if src_net != target_net:
                        # Edge whose src isn't the target's net → skip
                        # (this can happen for tiles where target cuboids span
                        #  multiple instances; mirror the model's own scatter
                        #  which only picks src side).
                        continue
                    if is_same: B_diag[key][dst_net]  += contrib
                    else:        B_cross[key][dst_net] += contrib
                    n_edges_signal_per_net[key] += 1

            t_proc += time.time() - t0
            if n_batches % 50 == 0:
                edges_seen = sum(n_edges_signal_per_net.values()) + sum(n_edges_power_per_net.values())
                print(f"  [{n_batches} batches] {t_proc:.1f}s elapsed, "
                      f"edges aggregated: {edges_seen}, unique nets: {len(n_tiles_per_net)}")

    print(f">>> Phase 2 forward done: {n_batches} batches, {t_proc:.1f}s")

    # Materialize DataFrame
    rows: list[dict] = []
    all_keys = set(n_tiles_per_net) | set(A_power_diag) | set(A_power_cross) | \
               set(B_diag) | set(B_cross) | set(c_vss_pred)
    for key in sorted(all_keys):
        design, net = key
        rec = {
            'design': design,
            'net':    net,
            'c_vss_pred_fF':         float(c_vss_pred.get(key, 0.0)),
            'A_power_diag':          float(A_power_diag.get(key, 0.0)),
            'A_power_cross':         float(A_power_cross.get(key, 0.0)),
            'B_diag_pickled':        pickle.dumps(dict(B_diag.get(key, {}))),
            'B_cross_pickled':       pickle.dumps(dict(B_cross.get(key, {}))),
            'n_tiles':               int(n_tiles_per_net.get(key, 0)),
            'n_edges_signal':        int(n_edges_signal_per_net.get(key, 0)),
            'n_edges_power':         int(n_edges_power_per_net.get(key, 0)),
        }
        rows.append(rec)
    df = pd.DataFrame(rows)

    out = Path(args.output)
    if out.suffix != '.pkl':
        out = out.with_suffix('.pkl')
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out)

    sidecar = out.parent / (out.stem + '_meta.json')
    with open(sidecar, 'w') as f:
        json.dump({
            'phase': 2,
            'K_layers': K,
            'z_anchors_um': z_anchors_tensor.cpu().tolist(),
            'train_designs': train_design_stems,
            'n_nets': int(len(df)),
            'max_tiles_per_design': args.max_tiles_per_design,
            'batch_size': args.batch_size,
            'extraction_seconds': time.time() - t_start,
            'physics_only_settings': {
                'gnd_mlp_zeroed': True,
                'cpl_mlp_zeroed_bias': [0.0, -10.0],
                'layer_scale_phys_gnd': 0.0,
                'cpl_layer_pair_log_scale': 0.0,
                'vss_gnd_scale_kept_at_init': True,
                'gnd_fringe_scale_kept_at_init': True,
            },
        }, f, indent=2)

    print(">>> Phase 2 done.")
    print(f"  pkl: {out}")
    print(f"  sidecar: {sidecar}")
    print(f"  unique nets: {len(df)}")
    print(f"  total: {time.time() - t_start:.1f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibration extractor for data-driven physics init.")
    sub = ap.add_subparsers(dest='phase', required=True)

    p1 = sub.add_parser('phase1', help='Golden + geometric aggregation (no model run).')
    p1.add_argument('--output', type=str, required=True, help='Output parquet path.')
    p1.add_argument('--max_tiles_per_design', type=int, default=None,
                    help='Cap tiles per design for fast smoke runs.')
    p1.add_argument('--max_nets_per_design', type=int, default=None,
                    help='Net-centric sample: pick N nets per design and include ALL their tiles. '
                         'Preferred over --max_tiles_per_design for NNLS validation.')

    p2 = sub.add_parser('phase2', help='Physics base aggregation via model forward.')
    p2.add_argument('--output', type=str, required=True, help='Output parquet path.')
    p2.add_argument('--gpu', type=int, default=4, help='GPU id.')
    p2.add_argument('--max_tiles_per_design', type=int, default=None,
                    help='Cap tiles per design for fast smoke runs (head-N).')
    p2.add_argument('--max_nets_per_design', type=int, default=None,
                    help='Net-centric sample: pick N nets per design and include ALL their tiles. '
                         'Preferred over --max_tiles_per_design for NNLS validation.')
    p2.add_argument('--batch_size', type=int, default=2,
                    help='Batch size for forward pass.')

    args = ap.parse_args()
    if args.phase == 'phase1':
        run_phase1(args)
        return 0
    if args.phase == 'phase2':
        run_phase2(args)
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())
