"""
FNO Feasibility - Option B: StarRC Golden Labels
=================================================
Matches tiles to *D_NET entries from StarRC SPEF files (real ground-truth).
Per-tile cap = net_total_fF / n_tiles (uniform split across tiles).

Compares: MLP vs Conv2D vs FNO-2D on the same dataset.
Output: convergence curves + final MAPE table.

Usage:
  python3 scripts/diag_fno_option_b.py --gpu 2 --n_tiles 2000 --epochs 80
"""

import sys, gzip, pickle, math, re, random, argparse, time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import configs.config as cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROCESSED_DIR = Path(cfg.PROCESSED_DIR)
MANIFEST_PATH = PROCESSED_DIR / "dataset_manifest.csv"
SPEF_DIR      = Path("/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22")

LAYER_BOUNDARIES = [0.0, 0.65, 0.85, 1.00, 1.20, 1.40, 1.80, 4.80, 9.0]
N_LAYERS = 8


# ─────────────────────────────────────────────
# SPEF parser
# ─────────────────────────────────────────────

def parse_spef_caps(spef_path: Path) -> dict:
    """Return {net_name: total_cap_fF} from a *C_UNIT 1.0 FF SPEF file."""
    caps = {}
    c_unit = 1.0  # fF by default
    with open(spef_path, 'r', errors='ignore') as fp:
        for line in fp:
            line = line.strip()
            if line.startswith('*C_UNIT'):
                parts = line.split()
                # e.g. *C_UNIT 1.0 FF  →  1.0 fF; 1.0 PF → 1e3 fF
                if len(parts) >= 3:
                    val = float(parts[1])
                    unit = parts[2].upper()
                    c_unit = val * {'FF': 1.0, 'PF': 1000.0, 'NF': 1e6}.get(unit, 1.0)
            elif line.startswith('*D_NET'):
                parts = line.split()
                if len(parts) >= 3:
                    net_name = parts[1]
                    try:
                        caps[net_name] = float(parts[2]) * c_unit
                    except ValueError:
                        pass
    return caps


def load_all_spef_caps() -> dict:
    """Return {(design_name, net_name): cap_fF}."""
    result = {}
    for spef_file in SPEF_DIR.glob("*_starrc.spef"):
        design_name = spef_file.stem.replace("_starrc", "")
        caps = parse_spef_caps(spef_file)
        for net_name, cap in caps.items():
            result[(design_name, net_name)] = cap
    print(f"  Loaded SPEF caps for {len(result):,} (design, net) pairs "
          f"from {len(list(SPEF_DIR.glob('*_starrc.spef')))} SPEF files")
    return result


# ─────────────────────────────────────────────
# Voxelizer (identical to Option A)
# ─────────────────────────────────────────────

def rasterize(cuboids: np.ndarray, G: int = 128, window: float = 8.0) -> np.ndarray:
    """(N,10) → (N_layers*4, G, G) float32 voxel grid."""
    grid = np.zeros((N_LAYERS, G, G, 4), dtype=np.float32)
    half = window / 2.0
    px = window / G

    for cub in cuboids:
        x, y, z, w, h, d = cub[0], cub[1], cub[2], cub[3], cub[4], cub[5]
        is_tgt = cub[7]; eps = cub[8]
        net_type = float(cub[9]) if len(cub) > 9 else 0.0

        li = N_LAYERS - 1
        for i, b in enumerate(LAYER_BOUNDARIES[1:]):
            if z < b:
                li = i; break

        c0 = int((x - w/2 + half) / px)
        c1 = max(int((x + w/2 + half) / px), c0 + 1)
        r0 = int((y - h/2 + half) / px)
        r1 = max(int((y + h/2 + half) / px), r0 + 1)
        c0, c1 = max(0, c0), min(G, c1)
        r0, r1 = max(0, r0), min(G, r1)

        grid[li, r0:r1, c0:c1, 0] = 1.0
        grid[li, r0:r1, c0:c1, 1] = eps
        grid[li, r0:r1, c0:c1, 2] = is_tgt
        grid[li, r0:r1, c0:c1, 3] = net_type

    return grid.transpose(0, 3, 1, 2).reshape(-1, G, G)  # (Z*4, G, G)


def flat_features(cuboids: np.ndarray) -> np.ndarray:
    """Aggregate cuboid stats → 12-dim feature for MLP baseline."""
    tgt = cuboids[cuboids[:, 7] == 1.0]
    agg = cuboids[cuboids[:, 7] == 0.0]

    def safe(arr, fn, default=0.0):
        return fn(arr) if len(arr) > 0 else default

    area_arr = tgt[:, 3] * tgt[:, 4] if len(tgt) else np.array([0.0])
    return np.array([
        safe(tgt[:, 3], np.median),
        safe(tgt[:, 4], np.median),
        safe(tgt[:, 5], np.median),
        safe(tgt[:, 2], np.mean),
        safe(tgt[:, 8], np.mean, 1.0),
        math.log1p(len(tgt)),
        math.log1p(len(agg)),
        math.log1p(safe(area_arr, np.sum)),
        safe(tgt[:, 2], np.min),
        safe(tgt[:, 3], np.max),
        math.log1p(safe(tgt[:, 3], np.std)),
        safe(area_arr, np.median),
    ], dtype=np.float32)


# ─────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────

def build_dataset(n_tiles: int, grid_size: int, seed: int = 42):
    random.seed(seed)
    df = pd.read_csv(MANIFEST_PATH)

    print("Loading SPEF caps...")
    spef_caps = load_all_spef_caps()

    # Join manifest with SPEF caps
    df['_key'] = list(zip(df['design_name'], df['net_name']))
    df['cap_fF'] = df['_key'].map(spef_caps)
    df_matched = df.dropna(subset=['cap_fF'])
    match_rate = len(df_matched) / len(df) * 100
    print(f"  Manifest rows: {len(df):,} | Matched: {len(df_matched):,} ({match_rate:.1f}%)")

    if len(df_matched) == 0:
        raise RuntimeError("No SPEF matches found — check SPEF_DIR and design_name alignment.")

    # Per-tile cap = net_cap / n_tiles (n_tiles already in manifest)
    df_matched = df_matched.copy()
    df_matched['tile_cap_fF'] = df_matched['cap_fF'] / df_matched['n_tiles'].clip(lower=1)

    # Filter out zero/very-small caps
    df_matched = df_matched[df_matched['tile_cap_fF'] > 0.01]

    sample = df_matched.sample(min(n_tiles * 3, len(df_matched)), random_state=seed)

    voxels, feats, caps = [], [], []
    t0 = time.time()
    attempted = 0

    for _, row in sample.iterrows():
        if len(caps) >= n_tiles:
            break
        attempted += 1
        pkl_path = PROCESSED_DIR / row['design_name'] / row['sample_filename']
        if not pkl_path.exists():
            continue
        try:
            with gzip.open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            cub = np.array(data['cuboids'], dtype=np.float32)
        except Exception:
            continue

        cap = float(row['tile_cap_fF'])
        if cap < 1e-5:
            continue

        voxels.append(rasterize(cub, G=grid_size))
        feats.append(flat_features(cub))
        caps.append(math.log1p(cap))

        if len(caps) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  Loaded {len(caps)}/{n_tiles} tiles  ({elapsed:.1f}s, "
                  f"{len(caps)/elapsed:.1f} tiles/s)")

    print(f"  Built {len(caps)} samples from {attempted} attempts  "
          f"({time.time()-t0:.1f}s total)")

    V  = torch.tensor(np.array(voxels), dtype=torch.float32)
    F_ = torch.tensor(np.array(feats),  dtype=torch.float32)
    Y  = torch.tensor(np.array(caps),   dtype=torch.float32).unsqueeze(1)
    return V, F_, Y


# ─────────────────────────────────────────────
# Models (identical to Option A)
# ─────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim=12, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1))

    def forward(self, v, f):
        return self.net(f)


class ConvNet(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(4))
        self.head = nn.Sequential(
            nn.Linear(128 * 16, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, v, f):
        return self.head(self.enc(v).flatten(1))


class SpectralConv2d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        self.W = nn.Parameter(
            (1/(width*width)) * torch.randn(width, width, modes, modes,
                                             dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W_sz = x.shape
        m = self.modes
        xf = torch.fft.rfft2(x, norm='ortho')
        out = torch.zeros_like(xf)
        out[:, :, :m, :m] = torch.einsum('bixk,ioxk->boxk', xf[:, :, :m, :m], self.W)
        return torch.fft.irfft2(out, s=(H, W_sz), norm='ortho')


class FNOBlock(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spec = SpectralConv2d(width, modes)
        self.bypass = nn.Conv2d(width, width, 1)
        self.norm = nn.InstanceNorm2d(width)

    def forward(self, x):
        return F.gelu(self.norm(self.spec(x) + self.bypass(x)))


class FNO2D(nn.Module):
    def __init__(self, in_ch, modes=16, width=64, n_blocks=4):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.head = nn.Sequential(
            nn.Linear(width * 16, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, v, f):
        x = self.lift(v)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.pool(x).flatten(1))


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

def smape(pred, target):
    return (2 * (pred - target).abs() / (pred.abs() + target.abs() + 1e-9)).mean().item() * 100


def train_model(name, model, V_tr, F_tr, Y_tr, V_val, F_val, Y_val,
                epochs, batch_size, lr, device, Y_mean, Y_std):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TensorDataset(V_tr, F_tr, Y_tr),
                        batch_size=batch_size, shuffle=True)

    history = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for vb, fb, yb in loader:
            vb, fb, yb = vb.to(device), fb.to(device), yb.to(device)
            loss = F.huber_loss(model(vb, fb), yb, delta=0.5)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item()
        sched.step()

        if ep % 10 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                p_n = model(V_val.to(device), F_val.to(device))
                p_cap = torch.expm1(p_n * Y_std + Y_mean)
                t_cap = torch.expm1(Y_val.to(device) * Y_std + Y_mean)
                val_mape = smape(p_cap, t_cap)
            elapsed = time.time() - t0
            history.append((ep, val_mape, train_loss / len(loader)))
            print(f"  [{name}] ep {ep:3d}/{epochs}  "
                  f"train={train_loss/len(loader):.4f}  "
                  f"val_MAPE={val_mape:.2f}%  ({elapsed:.0f}s)")

    return history


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_tiles', type=int, default=2000)
    parser.add_argument('--grid_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--val_frac', type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== FNO Feasibility - Option B: StarRC Golden Labels ===")
    print(f"Device: {device} | Tiles: {args.n_tiles} | Grid: {args.grid_size}x{args.grid_size}")
    print(f"Epochs: {args.epochs} | LR: {args.lr}")
    print(f"SPEF dir: {SPEF_DIR}\n")

    # Build dataset
    print("Building dataset (StarRC golden labels)...")
    V, F_, Y = build_dataset(args.n_tiles, args.grid_size)

    # Normalize Y
    Y_mean = Y.mean(); Y_std = Y.std()
    Y_norm = (Y - Y_mean) / (Y_std + 1e-8)

    # Split
    N = len(Y)
    n_val = int(N * args.val_frac)
    n_tr  = N - n_val
    perm  = torch.randperm(N)
    tr, vl = perm[:n_tr], perm[n_tr:]
    V_tr, F_tr, Y_tr = V[tr], F_[tr], Y_norm[tr]
    V_val, F_val, Y_val = V[vl], F_[vl], Y_norm[vl]

    print(f"\nDataset: {n_tr} train / {n_val} val")
    print(f"Input shape: voxel={tuple(V.shape[1:])}, flat={tuple(F_.shape[1:])}")
    print(f"Cap range (tile): {torch.expm1(Y).min():.3f} – {torch.expm1(Y).max():.3f} fF")
    print(f"Cap median (tile): {torch.expm1(Y.median()):.3f} fF\n")

    in_ch = V.shape[1]
    models = {
        'MLP':    (MLP(in_dim=F_.shape[1]), 1e-3),
        'Conv2D': (ConvNet(in_ch), args.lr),
        'FNO-2D': (FNO2D(in_ch, modes=16, width=64, n_blocks=4), args.lr),
    }

    results = {}
    for name, (model, lr) in models.items():
        n_p = sum(p.numel() for p in model.parameters())
        print(f"\n{'─'*60}")
        print(f"Training {name}  ({n_p:,} params)")
        model = model.to(device)
        history = train_model(name, model, V_tr, F_tr, Y_tr, V_val, F_val, Y_val,
                              args.epochs, args.batch_size, lr, device, Y_mean, Y_std)
        results[name] = history

    # Final comparison table
    print(f"\n{'='*60}")
    print("OPTION B RESULT — StarRC Golden Labels")
    print(f"{'='*60}")
    print(f"{'Model':<10} {'Params':>10} {'Best ValMAPE':>14} {'Final ValMAPE':>14}")
    print("─" * 52)
    for name, (model, _) in models.items():
        n_p = sum(p.numel() for p in model.parameters())
        mapes = [h[1] for h in results[name]]
        best  = min(mapes)
        final = mapes[-1]
        print(f"{name:<10} {n_p:>10,} {best:>13.2f}% {final:>13.2f}%")

    mlp_best = min(h[1] for h in results['MLP'])
    fno_best = min(h[1] for h in results['FNO-2D'])
    cov_best = min(h[1] for h in results['Conv2D'])

    print(f"\n-> FNO vs MLP improvement: {(mlp_best - fno_best)/mlp_best*100:+.1f}%")
    print(f"-> Conv vs MLP improvement: {(mlp_best - cov_best)/mlp_best*100:+.1f}%")

    if fno_best < mlp_best * 0.80:
        verdict = "SPATIAL FEATURES STRONGLY USEFUL (+20%p gain on real labels)"
    elif fno_best < mlp_best * 0.90:
        verdict = "SPATIAL FEATURES MODERATELY USEFUL (+10%p gain on real labels)"
    else:
        verdict = "SPATIAL FEATURES NOT USEFUL AT THIS RESOLUTION (real labels)"
    print(f"\nVerdict: {verdict}")

    # Save summary
    out = Path('output_intel22') / 'diag_fno_option_b.txt'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as fp:
        fp.write("Option B -- StarRC Golden Labels\n")
        fp.write(f"Grid: {args.grid_size}x{args.grid_size}, Tiles: {N}, Epochs: {args.epochs}\n\n")
        for name in results:
            fp.write(f"{name}: best={min(h[1] for h in results[name]):.2f}%\n")
        fp.write(f"\n{verdict}\n")
        for name in results:
            fp.write(f"\n{name} history:\n")
            for ep, mape, loss in results[name]:
                fp.write(f"  ep{ep:3d}  val_MAPE={mape:.2f}%  train_loss={loss:.4f}\n")
    print(f"\nResult saved -> {out}")


if __name__ == '__main__':
    main()
