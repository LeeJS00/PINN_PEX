"""
FNO Feasibility Diagnostic for BEOL PEX
=========================================
Tests three core questions:
  1. Voxelization: what resolution resolves which metal layers?
  2. Spatial utility: does a simple Conv model beat MLP on same data?
  3. FNO proxy: tiny FNO-style model convergence on labeled subset.

Usage:
  python3 scripts/diag_fno_feasibility.py
"""

import sys, gzip, pickle, math, random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
import configs.config as cfg

PROCESSED_DIR = Path(cfg.PROCESSED_DIR)
MANIFEST_PATH = PROCESSED_DIR / "dataset_manifest.csv"

# ──────────────────────────────────────────────
# Section 1: Wire geometry statistics by Z layer
# ──────────────────────────────────────────────

def sample_wire_geometry(n_tiles=500, seed=42):
    """Sample tiles and extract wire dimension distributions per metal layer."""
    random.seed(seed)
    df = pd.read_csv(MANIFEST_PATH)
    sample = df.sample(min(n_tiles, len(df)), random_state=seed)

    # Layer boundaries (z_abs μm): M1~M8 from layer_parser typical Intel22
    LAYER_BOUNDARIES = [0.0, 0.65, 0.85, 1.00, 1.20, 1.40, 1.80, 4.80, 9.0]
    LAYER_NAMES = ['M1','M2','M3','M4','M5','M6','M7','M8']

    def get_layer(z):
        for i, b in enumerate(LAYER_BOUNDARIES[1:]):
            if z < b:
                return i
        return len(LAYER_NAMES) - 1

    layer_stats = defaultdict(lambda: {'widths': [], 'heights': [], 'count': 0})

    for _, row in sample.iterrows():
        pkl_path = PROCESSED_DIR / row['design_name'] / row['sample_filename']
        if not pkl_path.exists():
            continue
        try:
            with gzip.open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            cuboids = np.array(data['cuboids'])  # (N, 10)
        except Exception:
            continue

        for cub in cuboids:
            x, y, z, w, h, d = cub[:6]
            layer_idx = get_layer(z)
            layer_name = LAYER_NAMES[layer_idx]
            # Physical wire width = min(w, h) for wires (excluding span dimension)
            min_dim = min(w, h)
            max_dim = max(w, h)
            if min_dim > 0.001:  # skip sub-nm artifacts
                layer_stats[layer_name]['widths'].append(min_dim)
                layer_stats[layer_name]['count'] += 1

    return layer_stats


def analyze_voxel_resolution(layer_stats, window_um=8.0, grid_sizes=(64, 128, 256)):
    """
    For each (layer, grid_size), compute fraction of wires that are resolved
    (>= 1 pixel in their minimum dimension).
    """
    print("\n" + "="*70)
    print("SECTION 1: Wire Width Distribution & Voxelization Resolution")
    print("="*70)
    print(f"Window size: {window_um}×{window_um} μm\n")

    print(f"{'Layer':<6} {'Count':>7} {'Min W':>8} {'P10':>8} {'P25':>8} {'Median':>8} {'P75':>8}")
    print("-" * 58)
    for layer in ['M1','M2','M3','M4','M5','M6','M7','M8']:
        if layer not in layer_stats:
            continue
        ws = sorted(layer_stats[layer]['widths'])
        if not ws:
            continue
        ws_arr = np.array(ws)
        print(f"{layer:<6} {len(ws):>7,} {np.min(ws_arr):>8.4f} "
              f"{np.percentile(ws_arr,10):>8.4f} {np.percentile(ws_arr,25):>8.4f} "
              f"{np.median(ws_arr):>8.4f} {np.percentile(ws_arr,75):>8.4f} μm")

    print(f"\n{'Layer':<6}", end='')
    for g in grid_sizes:
        px_nm = window_um * 1000 / g
        print(f"  {g}×{g}({px_nm:.0f}nm/px)", end='')
    print()
    print("-" * (6 + len(grid_sizes) * 18))

    for layer in ['M1','M2','M3','M4','M5','M6','M7','M8']:
        if layer not in layer_stats:
            continue
        ws_arr = np.array(layer_stats[layer]['widths'])
        if len(ws_arr) == 0:
            continue
        print(f"{layer:<6}", end='')
        for g in grid_sizes:
            px_um = window_um / g
            resolved_frac = np.mean(ws_arr >= px_um)
            print(f"  {resolved_frac*100:>13.1f}%   ", end='')
        print()

    print("\n  → Fraction of wires resolved (width ≥ 1 pixel) at each grid size")
    print("  → M7/M8 need ~64×64; M1-M4 need ~1024×1024 to resolve properly")


# ──────────────────────────────────────────────
# Section 2: FNO proxy model - convergence test
# ──────────────────────────────────────────────

def build_labeled_dataset(n_tiles=2000, seed=42):
    """
    Build a small dataset of (voxel_grid, gnd_cap) pairs using labeled tiles
    from the AL training cache (predefined subset CSVs).
    Returns numpy arrays.
    """
    import torch

    # Try to load from the predefined AL cache
    cache_paths = [
        Path('cache/predefined_train_subset.csv'),
        Path('/home/jslee/projects/PINNPEX/cache/predefined_train_subset.csv'),
    ]
    cache_df = None
    for cp in cache_paths:
        if cp.exists():
            cache_df = pd.read_csv(cp)
            break

    if cache_df is None:
        print("[WARN] AL cache not found, sampling from manifest directly (no cap labels).")
        return None, None, None

    # Cache has labeled tiles with cap values
    cap_cols = [c for c in cache_df.columns if 'gnd' in c.lower() or 'cap' in c.lower()]
    print(f"  Cache columns: {list(cache_df.columns)}")
    print(f"  Cap-related columns: {cap_cols}")
    print(f"  Cache size: {len(cache_df)} tiles")

    return cache_df, cap_cols, None


def rasterize_cuboids(cuboids, grid_h=64, grid_w=64, window_um=8.0,
                       n_z_layers=8, z_layer_bounds=None):
    """
    Convert (N, 10) cuboid array to (n_z_layers, grid_h, grid_w, 4) voxel tensor.
    Channels: [metal_presence, epsilon, is_target, net_type]
    """
    if z_layer_bounds is None:
        z_layer_bounds = [0.0, 0.65, 0.85, 1.00, 1.20, 1.40, 1.80, 4.80, 9.0]

    grid = np.zeros((n_z_layers, grid_h, grid_w, 4), dtype=np.float32)
    half = window_um / 2.0
    px_um = window_um / grid_w
    py_um = window_um / grid_h

    for cub in cuboids:
        x, y, z, w, h, d = cub[0], cub[1], cub[2], cub[3], cub[4], cub[5]
        is_target = cub[7]
        eps = cub[8]
        net_type = cub[9] if len(cub) > 9 else 0.0

        # Map z → layer index
        layer_idx = n_z_layers - 1
        for li, bnd in enumerate(z_layer_bounds[1:]):
            if z < bnd:
                layer_idx = li
                break

        # Map XY to pixel coords (center-relative)
        x_min_px = int((x - w/2 + half) / px_um)
        x_max_px = int((x + w/2 + half) / px_um)
        y_min_px = int((y - h/2 + half) / py_um)
        y_max_px = int((y + h/2 + half) / py_um)

        # Clamp and ensure at least 1 pixel
        x_min_px = max(0, min(x_min_px, grid_w - 1))
        x_max_px = max(x_min_px + 1, min(x_max_px, grid_w))
        y_min_px = max(0, min(y_min_px, grid_h - 1))
        y_max_px = max(y_min_px + 1, min(y_max_px, grid_h))

        grid[layer_idx, y_min_px:y_max_px, x_min_px:x_max_px, 0] = 1.0     # metal
        grid[layer_idx, y_min_px:y_max_px, x_min_px:x_max_px, 1] = eps
        grid[layer_idx, y_min_px:y_max_px, x_min_px:x_max_px, 2] = is_target
        grid[layer_idx, y_min_px:y_max_px, x_min_px:x_max_px, 3] = net_type

    return grid  # (Z, H, W, C)


def section2_voxel_quality(n_tiles=200, grid_size=128):
    """
    For a sample of tiles, rasterize and measure:
    - Fraction of target wire pixels that are non-zero (coverage)
    - Compare original cuboid count vs voxel non-zero pixel count
    """
    print("\n" + "="*70)
    print(f"SECTION 2: Voxelization Quality at {grid_size}×{grid_size}")
    print("="*70)

    df = pd.read_csv(MANIFEST_PATH)
    sample = df.sample(min(n_tiles, len(df)), random_state=123)

    target_coverages = []
    pixel_counts = []
    cuboid_counts = []

    for _, row in sample.iterrows():
        pkl_path = PROCESSED_DIR / row['design_name'] / row['sample_filename']
        if not pkl_path.exists():
            continue
        try:
            with gzip.open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            cuboids = np.array(data['cuboids'])
        except Exception:
            continue

        grid = rasterize_cuboids(cuboids, grid_h=grid_size, grid_w=grid_size)
        # grid shape: (Z, H, W, 4)
        metal = grid[:, :, :, 0]           # all metal
        target = grid[:, :, :, 2]          # target wire

        # How many cuboids are target?
        is_target_mask = cuboids[:, 7] == 1.0
        n_target_cub = is_target_mask.sum()

        # How many pixels light up for target wire?
        n_target_px = (target > 0).sum()

        # Coverage: fraction of target pixels that exist
        # (can't be 0 if any target wire is big enough to show)
        target_coverages.append(float(n_target_px > 0))
        pixel_counts.append(n_target_px)
        cuboid_counts.append(n_target_cub)

    coverage_rate = np.mean(target_coverages) * 100
    median_px = np.median(pixel_counts)
    median_cub = np.median(cuboid_counts)

    print(f"  Tiles sampled:         {len(pixel_counts)}")
    print(f"  Grid resolution:       {grid_size}×{grid_size} ({8000/grid_size:.1f} nm/pixel)")
    print(f"  Target wire visible:   {coverage_rate:.1f}%  (has ≥1 pixel)")
    print(f"  Median target pixels:  {median_px:.0f}  pixels")
    print(f"  Median target cuboids: {median_cub:.0f}  cuboids")
    print(f"  Pixel/cuboid ratio:    {median_px/max(median_cub,1):.1f}x")

    if coverage_rate < 80:
        print(f"\n  ⚠️  WARN: {100-coverage_rate:.0f}% of tiles lose target wire at this resolution.")
        print(f"     FNO would see no spatial signal for these tiles → poor learning.")
    else:
        print(f"\n  ✓  Most target wires are visible at {grid_size}×{grid_size}.")

    return coverage_rate


def section3_model_comparison(n_train=1000, n_val=200, grid_size=64,
                               epochs=30, device_str='cpu'):
    """
    Compare three models on the same labeled tile subset:
      A) MLP (current approach equivalent) — flat cuboid features
      B) Conv2D (spatial, per-layer pooled)
      C) FNO-2D (spectral convolution, XY plane, per-layer)
    Target: per-tile GND capacitance regression.
    Uses SPEF data from existing AL-labeled tiles in replay buffer.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    print("\n" + "="*70)
    print("SECTION 3: Model Comparison — MLP vs Conv vs FNO-proxy")
    print("="*70)

    # ── Load labeled data from AL replay buffer CSV ──
    replay_path = Path('/home/jslee/projects/PINNPEX/cache/predefined_train_subset.csv')
    if not replay_path.exists():
        print("  [SKIP] AL replay cache not found. Run AL first to generate labeled data.")
        print("         Expected: cache/predefined_train_subset.csv")
        return

    replay_df = pd.read_csv(replay_path)
    print(f"  Replay buffer: {len(replay_df)} labeled tile rows")
    print(f"  Columns: {list(replay_df.columns)}")

    # Check for cap label columns
    cap_col = None
    for c in ['gnd_cap_fF', 'c_gnd', 'gnd_cap', 'target_gnd_cap', 'cap_gnd']:
        if c in replay_df.columns:
            cap_col = c
            break
    if cap_col is None:
        print(f"  [SKIP] No cap label column found in replay CSV.")
        print(f"         Available: {list(replay_df.columns)}")
        return

    print(f"  Using cap label: '{cap_col}'")

    # ── Load and rasterize tiles ──
    total_needed = n_train + n_val
    sample_df = replay_df.dropna(subset=[cap_col]).sample(
        min(total_needed, len(replay_df)), random_state=42)

    voxels, caps, flat_feats = [], [], []
    print(f"  Loading {len(sample_df)} tiles and rasterizing at {grid_size}×{grid_size}...")

    for _, row in sample_df.iterrows():
        pkl_path = PROCESSED_DIR / row.get('design_name', '') / row['sample_filename']
        if not pkl_path.exists():
            continue
        try:
            with gzip.open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            cuboids = np.array(data['cuboids'])
        except Exception:
            continue

        cap_val = float(row[cap_col])
        if cap_val <= 0 or math.isnan(cap_val):
            continue

        # Voxel grid for Conv/FNO
        grid = rasterize_cuboids(cuboids, grid_h=grid_size, grid_w=grid_size)
        # shape (Z=8, H, W, 4) → flatten Z,C → (Z*4, H, W) for CNN input
        grid_ch = grid.transpose(0, 3, 1, 2).reshape(-1, grid_size, grid_size)
        voxels.append(grid_ch)

        # Flat features for MLP: aggregate cuboid stats
        tgt = cuboids[cuboids[:, 7] == 1.0]
        agg = cuboids[cuboids[:, 7] == 0.0]
        feat = np.array([
            tgt[:, 3].mean() if len(tgt) else 0,   # mean target width
            tgt[:, 4].mean() if len(tgt) else 0,   # mean target height
            tgt[:, 5].mean() if len(tgt) else 0,   # mean target thickness
            tgt[:, 2].mean() if len(tgt) else 0,   # mean z_abs
            tgt[:, 8].mean() if len(tgt) else 1,   # mean eps
            len(tgt),                                # n_target cuboids
            len(agg),                                # n_aggressor cuboids
            tgt[:, 3].sum() * tgt[:, 4].sum() if len(tgt) else 0,  # area proxy
            np.log1p(len(tgt)),
        ], dtype=np.float32)
        flat_feats.append(feat)
        caps.append(math.log1p(cap_val))  # predict log-cap

    if len(voxels) < 100:
        print(f"  [SKIP] Only {len(voxels)} valid samples found (need ≥100).")
        return

    N = len(voxels)
    print(f"  Valid samples: {N}")

    # Convert to tensors
    V = torch.tensor(np.array(voxels), dtype=torch.float32)   # (N, Z*4, H, W)
    F_ = torch.tensor(np.array(flat_feats), dtype=torch.float32)  # (N, 9)
    Y = torch.tensor(np.array(caps), dtype=torch.float32).unsqueeze(1)  # (N, 1)

    # Normalize Y
    Y_mean, Y_std = Y.mean(), Y.std()
    Y_norm = (Y - Y_mean) / (Y_std + 1e-8)

    split = min(n_train, N - n_val)
    V_tr, V_val = V[:split], V[split:split+n_val]
    F_tr, F_val = F_[:split], F_[split:split+n_val]
    Y_tr, Y_val = Y_norm[:split], Y_norm[split:split+n_val]

    device = torch.device(device_str)

    # ── Model A: MLP baseline ──
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(9, 64), nn.GELU(),
                nn.Linear(64, 64), nn.GELU(),
                nn.Linear(64, 1))
        def forward(self, v, f): return self.net(f)

    # ── Model B: Conv2D spatial ──
    class ConvNet(nn.Module):
        def __init__(self, in_ch):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, padding=1), nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(),
                nn.AdaptiveAvgPool2d(4))
            self.head = nn.Sequential(
                nn.Linear(64*16, 64), nn.GELU(),
                nn.Linear(64, 1))
        def forward(self, v, f):
            h = self.enc(v).flatten(1)
            return self.head(h)

    # ── Model C: FNO-2D proxy (spectral conv in XY, per-channel) ──
    class SpectralConv2d(nn.Module):
        def __init__(self, in_ch, out_ch, modes):
            super().__init__()
            self.modes = modes
            self.scale = 1 / (in_ch * out_ch)
            self.R = nn.Parameter(self.scale * torch.randn(
                in_ch, out_ch, modes, modes, dtype=torch.cfloat))
        def forward(self, x):
            B, C, H, W = x.shape
            x_ft = torch.fft.rfft2(x)
            m = self.modes
            out_ft = torch.zeros(B, self.R.shape[1], H, W//2+1,
                                  dtype=torch.cfloat, device=x.device)
            out_ft[:, :, :m, :m] = torch.einsum('bixk,ioxk->boxk', x_ft[:,:,:m,:m], self.R)
            return torch.fft.irfft2(out_ft, s=(H, W))

    class FNO2DProxy(nn.Module):
        def __init__(self, in_ch, modes=8, width=32):
            super().__init__()
            self.lift = nn.Conv2d(in_ch, width, 1)
            self.spec1 = SpectralConv2d(width, width, modes)
            self.conv1 = nn.Conv2d(width, width, 1)
            self.spec2 = SpectralConv2d(width, width, modes)
            self.conv2 = nn.Conv2d(width, width, 1)
            self.pool = nn.AdaptiveAvgPool2d(4)
            self.head = nn.Sequential(nn.Linear(width*16, 64), nn.GELU(), nn.Linear(64, 1))
        def forward(self, v, f):
            x = self.lift(v)
            x = F.gelu(self.spec1(x) + self.conv1(x))
            x = F.gelu(self.spec2(x) + self.conv2(x))
            return self.head(self.pool(x).flatten(1))

    in_ch = V.shape[1]  # Z*4 = 32
    models = {
        'MLP':     MLP().to(device),
        'Conv2D':  ConvNet(in_ch).to(device),
        'FNO-2D':  FNO2DProxy(in_ch).to(device),
    }

    results = {}
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        ds_tr = TensorDataset(V_tr.to(device), F_tr.to(device), Y_tr.to(device))
        loader = DataLoader(ds_tr, batch_size=32, shuffle=True)

        train_losses, val_mapes = [], []
        for ep in range(epochs):
            model.train()
            ep_loss = 0
            for vb, fb, yb in loader:
                pred = model(vb, fb)
                loss = F.mse_loss(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item()
            train_losses.append(ep_loss / len(loader))

            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                model.eval()
                with torch.no_grad():
                    p = model(V_val.to(device), F_val.to(device))
                    p_cap = torch.expm1(p * Y_std + Y_mean)
                    t_cap = torch.expm1(Y_val.to(device) * Y_std + Y_mean)
                    mape = (torch.abs(p_cap - t_cap) / (t_cap + 1e-9)).mean().item() * 100
                val_mapes.append((ep+1, mape))

        results[name] = {
            'params': n_params,
            'final_train_loss': train_losses[-1],
            'val_mapes': val_mapes,
        }
        print(f"  {name:8s} ({n_params:,} params): "
              f"final val MAPE = {val_mapes[-1][1]:.1f}%  "
              f"train_loss = {train_losses[-1]:.4f}")

    print("\n  Val MAPE progression:")
    print(f"  {'Epoch':<8}", end='')
    for name in models: print(f"  {name:>12}", end='')
    print()
    n_rows = len(results['MLP']['val_mapes'])
    for i in range(n_rows):
        ep_str = str(results['MLP']['val_mapes'][i][0])
        print(f"  {ep_str:<8}", end='')
        for name in models:
            mape_str = f"{results[name]['val_mapes'][i][1]:.1f}%"
            print(f"  {mape_str:>12}", end='')
        print()

    # Interpret
    final_mapes = {n: results[n]['val_mapes'][-1][1] for n in models}
    best = min(final_mapes, key=final_mapes.get)
    print(f"\n  Best model: {best} ({final_mapes[best]:.1f}% val MAPE)")
    if final_mapes.get('FNO-2D', 999) < final_mapes.get('MLP', 999) * 0.85:
        print("  ✓ FNO-2D outperforms MLP by >15% → spatial features ARE useful")
        print("    → FNO approach is worth pursuing.")
    elif final_mapes.get('Conv2D', 999) < final_mapes.get('MLP', 999) * 0.85:
        print("  ✓ Conv2D outperforms MLP → spatial signal exists but FNO not needed yet")
        print("    → Consider Conv-based approach or larger FNO.")
    else:
        print("  ✗ Spatial models don't significantly beat MLP at this resolution")
        print(f"    → Voxelization at {grid_size}×{grid_size} loses too much wire geometry.")
        print("    → Try higher resolution, or use a non-voxel FNO (e.g., GINO on cuboid graph).")


# ──────────────────────────────────────────────
# Section 4: Resolution recommendation summary
# ──────────────────────────────────────────────

def section4_summary(layer_stats):
    print("\n" + "="*70)
    print("SECTION 4: Summary & FNO Design Recommendations")
    print("="*70)

    # Find min resolution to resolve P25 wire width in each layer
    LAYER_NAMES = ['M1','M2','M3','M4','M5','M6','M7','M8']
    window_um = 8.0
    print(f"\n  Grid size needed to resolve P25-width wires in each layer:")
    print(f"  (window = {window_um}μm, need ≥ 2 pixels for reliable detection)\n")
    print(f"  {'Layer':<6} {'P25 width':>10} {'Min grid':>10}  {'Feasible?':>10}")
    print("  " + "-" * 42)
    for layer in LAYER_NAMES:
        if layer not in layer_stats:
            continue
        ws = np.array(layer_stats[layer]['widths'])
        if len(ws) < 10:
            continue
        p25 = np.percentile(ws, 25)
        min_grid = math.ceil(window_um / (p25 / 2))  # need 2px per wire
        feasible = "✓" if min_grid <= 512 else "✗ too large"
        print(f"  {layer:<6} {p25:>8.4f} μm  {min_grid:>8}px  {feasible:>10}")

    print("""
  Recommendations:
  ─────────────────────────────────────────────────────────────────
  1. VOXEL FNO (standard):
     - Use 256×256 grid → resolves M5+ wires well, M1-M4 partial
     - Suitable for coarse feasibility test; not for publication accuracy

  2. MULTI-SCALE FNO (proposed for paper):
     - Coarse FNO (64×64): long-range field, M7/M8 coupling
     - Fine FNO (256×256, per-layer XY cross-section): M1-M4 detail
     - Hierarchical: coarse output conditions fine prediction

  3. GRAPH-FNO / GINO (best for publication):
     - Use cuboid graph directly (no voxelization needed)
     - FNO-style operator on irregular mesh
     - Avoids resolution problem entirely
     - Reference: GINO (Li et al., NeurIPS 2023)

  4. PRACTICAL FIRST STEP:
     - Test Conv2D at 128×128 on M7/M8 dominant tiles first
     - If MAPE improvement over MLP: validates spatial hypothesis
     - Then scale to full FNO on higher-resolution or graph-based input
""")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_sample', type=int, default=500)
    parser.add_argument('--grid_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip Section 3 model comparison (faster)')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    print("FNO Feasibility Diagnostic")
    print(f"Dataset: {PROCESSED_DIR}")
    print(f"Manifest tiles: checking...")

    # Section 1: Wire geometry
    print("\nSampling wire geometry statistics...")
    layer_stats = sample_wire_geometry(n_tiles=args.n_sample)
    analyze_voxel_resolution(layer_stats, grid_sizes=(64, 128, 256, 512))

    # Section 2: Voxel quality
    section2_voxel_quality(n_tiles=200, grid_size=args.grid_size)

    # Section 3: Model comparison (optional)
    if not args.skip_training:
        section3_model_comparison(
            n_train=800, n_val=200,
            grid_size=args.grid_size,
            epochs=args.epochs,
            device_str=args.device)

    # Section 4: Summary
    section4_summary(layer_stats)

    print("\nDiagnostic complete.")
