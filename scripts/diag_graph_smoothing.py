"""
Graph Smoothing Diagnostic for GINO vs GNN
===========================================
Measures feature over-smoothing across four aggregation strategies:
  0. No aggregation (raw per-cuboid MLP)
  1. 1-hop spatial mean (GNN-like, r=4μm)
  2. 2-hop spatial mean
  3. P2G → G2P (GINO Gaussian scatter/interpolate, no FNO)
  4. P2G → FNO → G2P (full GINO)

Metrics per strategy:
  - Dirichlet Energy (DE):  mean ||f_i - f_j||² over spatial neighbors (r<2μm)
                             HIGH = features vary between neighbors (good)
  - Net Separation Ratio (NSR): mean_dist(cross-net pairs) / mean_dist(same-net pairs)
                                 HIGH = better inter-net discrimination (good)
  - Effective Rank (ER):    exp(H(σ)) / D   (1.0 = full rank, diverse)

All operations are fully vectorised (no Python element loops).
"""

import sys, gzip, pickle, math, argparse, time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import configs.config as cfg

import torch
import torch.nn as nn
import torch.nn.functional as F

PROCESSED_DIR    = Path(cfg.PROCESSED_DIR)
MANIFEST_PATH    = PROCESSED_DIR / "dataset_manifest.csv"
LAYER_BOUNDARIES = [0.0, 0.65, 0.85, 1.00, 1.20, 1.40, 1.80, 4.80, 9.0]
N_LAYERS = 8
MAX_CUBOIDS = 256   # cap per tile to keep runtime feasible


class CuboidEncoder(nn.Module):
    def __init__(self, in_dim=10, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d), nn.LayerNorm(d), nn.GELU(),
            nn.Linear(d, d), nn.LayerNorm(d))

    def forward(self, x):
        out = x.clone()
        out[..., :3]  /= 2.5
        out[..., 3:6]  = torch.log1p(out[..., 3:6].clamp(0)) / 1.5
        out[..., 8:9]  = torch.log(out[..., 8:9].clamp(min=1.0)) / 2.0
        return self.net(out)


class SpectralConv2d(nn.Module):
    def __init__(self, w, m):
        super().__init__()
        self.m = m
        self.W = nn.Parameter((1/(w*w)) * torch.randn(w, w, m, m, dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W = x.shape
        m = self.m
        xf = torch.fft.rfft2(x, norm='ortho')
        out = torch.zeros_like(xf)
        out[:, :, :m, :m] = torch.einsum('bixk,ioxk->boxk', xf[:, :, :m, :m], self.W)
        return torch.fft.irfft2(out, s=(H, W), norm='ortho')


class FNOBlock2D(nn.Module):
    def __init__(self, w, m):
        super().__init__()
        self.spec   = SpectralConv2d(w, m)
        self.bypass = nn.Conv2d(w, w, 1)
        self.norm   = nn.InstanceNorm2d(w)

    def forward(self, x):
        return F.gelu(self.norm(self.spec(x) + self.bypass(x)))


# ─────────────────────────────────────────────
# Vectorised aggregation strategies
# ─────────────────────────────────────────────

def hop_agg_vec(feats: torch.Tensor, coords: torch.Tensor,
                n_hops: int, r: float = 4.0) -> torch.Tensor:
    """
    Vectorised k-hop mean aggregation using torch.cdist.
    feats:  (N, D)
    coords: (N, 3)
    """
    out = feats.clone()
    for _ in range(n_hops):
        dists = torch.cdist(coords, coords)          # (N, N)
        mask  = (dists < r) & (dists > 1e-6)        # exclude self
        # Weighted mean (include self with 0.5 weight)
        mask_f = mask.float()
        nbr_sum = mask_f @ out                       # (N, D)
        nbr_cnt = mask_f.sum(-1, keepdim=True).clamp(min=1)
        out = out * 0.5 + (nbr_sum / nbr_cnt) * 0.5
    return out


def p2g_vec(feats: torch.Tensor, coords: torch.Tensor,
            G: int, window: float = 8.0,
            sigma_xy: float = 0.25, sigma_z: float = 0.15,
            device: torch.device = None) -> torch.Tensor:
    """
    Vectorised Particle-to-Grid.
    feats:  (N, D)
    coords: (N, 3)  [x_rel, y_rel, z_abs]
    Returns: (L, G, G, D)  normalised by total weight
    """
    N, D = feats.shape
    half = window / 2.0

    # Grid cell centres
    xs = torch.linspace(-half + half/G, half - half/G, G, device=device)
    gx, gy = torch.meshgrid(xs, xs, indexing='ij')   # (G, G)
    gx_flat = gx.reshape(-1)    # (G²,)
    gy_flat = gy.reshape(-1)

    z_centres = torch.tensor(
        [(LAYER_BOUNDARIES[i]+LAYER_BOUNDARIES[i+1])/2
         for i in range(N_LAYERS)], device=device)   # (L,)

    # XY weights: (N, G²)
    dx = coords[:, 0:1] - gx_flat.unsqueeze(0)      # (N, G²)
    dy = coords[:, 1:2] - gy_flat.unsqueeze(0)
    w_xy = torch.exp(-(dx**2 + dy**2) / (2*sigma_xy**2))  # (N, G²)

    # Z weights: (N, L)
    dz = coords[:, 2:3] - z_centres.unsqueeze(0)    # (N, L)
    w_z = torch.exp(-dz**2 / (2*sigma_z**2))         # (N, L)

    # Combined weight: (N, L, G²)
    w = w_xy.unsqueeze(1) * w_z.unsqueeze(2)         # (N, L, G²)
    w_sum = w.sum(0)                                  # (L, G²)

    # Scatter: (L, G², D)
    feat_grid = torch.einsum('nlg,nd->lgd', w, feats) / (w_sum.unsqueeze(-1) + 1e-8)

    return feat_grid.reshape(N_LAYERS, G, G, D)


def g2p_vec(grid: torch.Tensor, coords: torch.Tensor,
            window: float = 8.0,
            sigma_xy: float = 0.25, sigma_z: float = 0.15,
            device: torch.device = None) -> torch.Tensor:
    """
    Vectorised Grid-to-Particle (reverse of P2G).
    grid:   (L, G, G, D)
    coords: (N, 3)
    Returns: (N, D)
    """
    L, G, _, D = grid.shape
    N = coords.shape[0]
    half = window / 2.0

    xs = torch.linspace(-half + half/G, half - half/G, G, device=device)
    gx, gy = torch.meshgrid(xs, xs, indexing='ij')
    gx_flat = gx.reshape(-1); gy_flat = gy.reshape(-1)   # (G²,)

    z_centres = torch.tensor(
        [(LAYER_BOUNDARIES[i]+LAYER_BOUNDARIES[i+1])/2
         for i in range(N_LAYERS)], device=device)

    dx = coords[:, 0:1] - gx_flat.unsqueeze(0)      # (N, G²)
    dy = coords[:, 1:2] - gy_flat.unsqueeze(0)
    w_xy = torch.exp(-(dx**2 + dy**2) / (2*sigma_xy**2))

    dz = coords[:, 2:3] - z_centres.unsqueeze(0)    # (N, L)
    w_z = torch.exp(-dz**2 / (2*sigma_z**2))

    w = w_xy.unsqueeze(1) * w_z.unsqueeze(2)         # (N, L, G²)
    w_norm = w / (w.sum(dim=[1,2], keepdim=True) + 1e-8)

    grid_flat = grid.reshape(L, G*G, D)              # (L, G², D)
    # out[n] = Σ_l Σ_g w[n,l,g] * grid[l,g,:]
    out = torch.einsum('nlg,lgd->nd', w_norm, grid_flat)
    return out


def apply_fno_on_grid_vec(grid: torch.Tensor, blocks: nn.ModuleList,
                          lift: nn.Linear, proj: nn.Linear) -> torch.Tensor:
    """
    Apply FNO to all L layers in parallel by stacking as batch dimension.
    grid: (L, G, G, D_in)
    Returns: (L, G, G, D_out)
    """
    L, G, _, D = grid.shape
    x = grid.permute(0, 3, 1, 2)       # (L, D, G, G) — treat L as batch
    x = lift(x.permute(0,2,3,1)).permute(0,3,1,2)  # lift channels
    for blk in blocks:
        x = blk(x)
    x = proj(x.permute(0,2,3,1)).permute(0,3,1,2)  # project back
    return x.permute(0, 2, 3, 1)       # (L, G, G, D_out)


# ─────────────────────────────────────────────
# Metrics (vectorised)
# ─────────────────────────────────────────────

def dirichlet_energy(feats: torch.Tensor, coords: torch.Tensor,
                     r: float = 2.0) -> float:
    """Mean ||f_i - f_j||² over pairs within distance r."""
    dists = torch.cdist(coords, coords)               # (N, N)
    mask  = (dists < r) & (dists > 1e-6)
    if not mask.any():
        return float('nan')
    # Pairwise squared distances in feature space
    diff2 = torch.cdist(feats, feats).pow(2)          # (N, N)
    return (diff2 * mask.float()).sum().item() / mask.float().sum().item()


def net_separation_ratio(feats: torch.Tensor, net_ids: torch.Tensor) -> float:
    """mean_dist(cross-net) / mean_dist(same-net)."""
    diff2 = torch.cdist(feats, feats).pow(2)          # (N, N)
    same  = (net_ids.unsqueeze(1) == net_ids.unsqueeze(0))
    cross = ~same
    diag  = torch.eye(len(net_ids), dtype=torch.bool)
    same  = same & ~diag

    d_same  = diff2[same].mean().item()  if same.any()  else float('nan')
    d_cross = diff2[cross].mean().item() if cross.any() else float('nan')
    return d_cross / (d_same + 1e-9)


def effective_rank(feats: torch.Tensor) -> float:
    if feats.shape[0] < 2:
        return float('nan')
    f = feats - feats.mean(0)
    sv = torch.linalg.svdvals(f)
    sv = sv / (sv.sum() + 1e-9)
    sv = sv[sv > 1e-9]
    return math.exp(-(sv * sv.log()).sum().item()) / feats.shape[1]


# ─────────────────────────────────────────────
# Load & run
# ─────────────────────────────────────────────

def load_tiles(n: int, seed: int = 42) -> list:
    import random; random.seed(seed)
    df = pd.read_csv(MANIFEST_PATH)
    rows = df.sample(min(n * 8, len(df)), random_state=seed)

    tiles = []
    for _, row in rows.iterrows():
        if len(tiles) >= n:
            break
        p = PROCESSED_DIR / row['design_name'] / row['sample_filename']
        if not p.exists():
            continue
        try:
            with gzip.open(p, 'rb') as f:
                data = pickle.load(f)
            cub = np.array(data['cuboids'], dtype=np.float32)
            net_names = data.get('cuboid_net_names', [])
        except Exception:
            continue

        if len(cub) < 8:
            continue
        # Cap to MAX_CUBOIDS (matches model's actual context window)
        if len(cub) > MAX_CUBOIDS:
            cub = cub[:MAX_CUBOIDS]
            if net_names:
                net_names = net_names[:MAX_CUBOIDS]

        if not net_names or len(net_names) < len(cub):
            net_names = [('t' if c[7] == 1 else f'a_{i}') for i, c in enumerate(cub)]

        unique = set(net_names)
        if len(unique) < 2:
            continue
        tiles.append({'cuboids': cub, 'net_names': net_names})

    print(f"  Loaded {len(tiles)} tiles (MAX_CUBOIDS={MAX_CUBOIDS})")
    print(f"  Avg cuboids/tile: {np.mean([len(t['cuboids']) for t in tiles]):.1f}")
    return tiles


def run_all(tiles, device, G=32):
    D = 64
    torch.manual_seed(0)
    encoder = CuboidEncoder(in_dim=10, d=D).to(device).eval()

    # FNO components
    FNO_W = 32
    lift   = nn.Linear(D, FNO_W, bias=False).to(device)
    blocks = nn.ModuleList([FNOBlock2D(FNO_W, 8) for _ in range(2)]).to(device)
    proj   = nn.Linear(FNO_W, D, bias=False).to(device)

    STRATS = ['0-No Agg', '1-Hop GNN', '2-Hop GNN',
              'GINO P2G→G2P', 'GINO P2G→FNO→G2P']
    res = {s: {'DE': [], 'NSR': [], 'ER': []} for s in STRATS}

    for idx, tile in enumerate(tiles):
        cub_np  = tile['cuboids']
        names   = tile['net_names']
        name2id = {n: i for i, n in enumerate(sorted(set(names)))}
        net_ids = torch.tensor([name2id[n] for n in names])

        cub_t   = torch.tensor(cub_np, dtype=torch.float32, device=device)
        coords  = torch.tensor(cub_np[:, :3], dtype=torch.float32, device=device)

        with torch.no_grad():
            # 0. Raw encoder
            f0 = encoder(cub_t)                                           # (N, D)

            # 1-2. GNN hops
            f1 = hop_agg_vec(f0, coords, 1)
            f2 = hop_agg_vec(f0, coords, 2)

            # 3. P2G → G2P
            grid = p2g_vec(f0, coords, G=G, device=device)               # (L,G,G,D)
            f3   = g2p_vec(grid, coords, device=device)                   # (N, D)

            # 4. P2G → FNO → G2P
            grid_fno = apply_fno_on_grid_vec(grid.to(device), blocks, lift, proj)
            f4 = g2p_vec(grid_fno, coords, device=device)

        for s, feats in zip(STRATS, [f0, f1, f2, f3, f4]):
            f = feats.detach().cpu().float()
            c = coords.cpu()
            res[s]['DE'].append(dirichlet_energy(f, c))
            res[s]['NSR'].append(net_separation_ratio(f, net_ids))
            res[s]['ER'].append(effective_rank(f))

        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{len(tiles)} tiles")

    return {s: {m: float(np.nanmean(v)) for m, v in ms.items()}
            for s, ms in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_tiles',   type=int,   default=100)
    ap.add_argument('--gpu',       type=int,   default=5)
    ap.add_argument('--grid_size', type=int,   default=32)
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== Graph Smoothing Diagnostic (vectorised) ===")
    print(f"Device: {device} | Tiles: {args.n_tiles} | G: {args.grid_size} | MAX_N: {MAX_CUBOIDS}")

    tiles = load_tiles(args.n_tiles)
    t0 = time.time()
    results = run_all(tiles, device, G=args.grid_size)
    print(f"\n  Finished in {time.time()-t0:.1f}s")

    # ── Report ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("GRAPH SMOOTHING RESULTS")
    print(f"{'='*70}")
    print(f"\n  DE  (Dirichlet Energy) : HIGH = diverse features (good)")
    print(f"  NSR (Net Sep. Ratio)   : HIGH = can discriminate nets (good)")
    print(f"  ER  (Effective Rank)   : HIGH = rich feature space (good)\n")
    print(f"  {'Strategy':<26} {'DE':>9} {'NSR':>9} {'ER':>9}  Note")
    print("  " + "─" * 66)

    strats = list(results.keys())
    for s in strats:
        m = results[s]
        note = []
        if m['NSR'] < 1.1:  note.append("OVER-SMOOTH")
        elif m['NSR'] < 1.5: note.append("mild smooth")
        else:                note.append("ok")
        if m['ER'] < 0.10:   note.append("rank-collapsed")
        print(f"  {s:<26} {m['DE']:>9.4f} {m['NSR']:>9.4f} {m['ER']:>9.4f}  {', '.join(note)}")

    r = {s: results[s] for s in strats}
    print(f"\n  ── Key comparisons ──────────────────────────────────────")
    for s in strats[1:]:
        de_chg  = (r[s]['DE']  - r[strats[0]]['DE'])  / (r[strats[0]]['DE']  + 1e-9) * 100
        nsr_chg = (r[s]['NSR'] - r[strats[0]]['NSR']) / (r[strats[0]]['NSR'] + 1e-9) * 100
        er_chg  = (r[s]['ER']  - r[strats[0]]['ER'])  / (r[strats[0]]['ER']  + 1e-9) * 100
        print(f"  {s:<26}  DE {de_chg:>+6.1f}%  NSR {nsr_chg:>+6.1f}%  ER {er_chg:>+6.1f}%")

    # ── Save ─────────────────────────────────────────────────
    out = Path('output_intel22/diag_graph_smoothing.txt')
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as fp:
        fp.write("=== Graph Smoothing Diagnostic ===\n")
        fp.write(f"Tiles: {len(tiles)}, G: {args.grid_size}, MAX_N: {MAX_CUBOIDS}\n\n")
        fp.write(f"{'Strategy':<26} {'DE':>9} {'NSR':>9} {'ER':>9}\n")
        fp.write("─" * 55 + "\n")
        for s in strats:
            m = results[s]
            fp.write(f"{s:<26} {m['DE']:>9.4f} {m['NSR']:>9.4f} {m['ER']:>9.4f}\n")
        fp.write("\nDeltas vs no-aggregation baseline:\n")
        for s in strats[1:]:
            de  = (r[s]['DE']  - r[strats[0]]['DE'])  / (r[strats[0]]['DE']  + 1e-9) * 100
            nsr = (r[s]['NSR'] - r[strats[0]]['NSR']) / (r[strats[0]]['NSR'] + 1e-9) * 100
            er  = (r[s]['ER']  - r[strats[0]]['ER'])  / (r[strats[0]]['ER']  + 1e-9) * 100
            fp.write(f"  {s}: DE {de:+.1f}%, NSR {nsr:+.1f}%, ER {er:+.1f}%\n")
    print(f"\n  Saved → {out}")


if __name__ == '__main__':
    main()
