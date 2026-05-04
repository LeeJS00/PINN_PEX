#!/usr/bin/env python3
"""
scripts/diag_eps_channel_utility.py

Validate whether the `eps` channel in MacroDensityFNO's density grid
((B, L, 2, G, G) — channel 0 = vol_frac, channel 1 = eps) carries spatial
information, or is essentially a constant scalar per z-anchor.

Hypothesis: BEOLMaterialStack.get_permittivity_bulk returns eps as a function
of z alone. After soft top-2 z-bucketing into the MacroDensityFNO layer grid,
all cuboids landing in the same anchor share the same eps. The eps grid is
therefore spatially constant per layer — equivalent to a learned per-layer
scalar bias the FNO must work to extract from a high-rank Fourier basis.

Three measurements:
  (1) Per-layer eps std among cuboids assigned to that layer (real tiles).
      Expected: ~0 (eps purely depends on z).
  (2) Spatial std of the eps grid within each layer after P2G — i.e. does the
      Gaussian scatter create xy variation. Expected: 0 if step (1) is 0.
  (3) Effective rank of the per-layer eps grid (svd of (G, G) matrix). Rank=0
      (uniform) means the channel is informationally a layer scalar.

Usage:
  python3 scripts/diag_eps_channel_utility.py --gpu 4 --n_tiles 200
"""
import argparse, gzip, pickle, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import configs.config as cfg
from src.preprocessing.layer_parser import LayerInfoParser
from src.physics.materials import BEOLMaterialStack
from src.models.macro_density_fno import MacroDensityFNO


def load_random_tiles(processed_dir: Path, manifest_path: Path,
                      n_tiles: int = 200, seed: int = 42) -> list[np.ndarray]:
    """Return a list of (N, 10) cuboid tensors sampled from the manifest."""
    import pandas as pd
    df = pd.read_csv(manifest_path)
    sample = df.sample(min(n_tiles, len(df)), random_state=seed)
    tiles = []
    for _, row in sample.iterrows():
        pkl_path = processed_dir / row['design_name'] / row['sample_filename']
        if not pkl_path.exists():
            continue
        try:
            with gzip.open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            cubs = np.asarray(data['cuboids'], dtype=np.float32)
            if cubs.ndim == 2 and cubs.shape[0] > 0:
                tiles.append(cubs)
        except Exception:
            continue
    return tiles


def compute_z_anchors(layer_map):
    """Replicate MacroDensityFNO z-anchor construction (metal + via, deduped)."""
    CONDUCTOR_TYPES = {'C', 'V'}
    zs = sorted({float(info.get('z_pos', 0.0)) for info in layer_map.values()
                 if info.get('type', '').upper() in CONDUCTOR_TYPES})
    if not zs:
        zs = sorted({float(info.get('z_pos', 0.0)) for info in layer_map.values()})
    unique_z = []
    for z in zs:
        if not unique_z or abs(z - unique_z[-1]) > 0.05:
            unique_z.append(z)
    return np.array(unique_z, dtype=np.float32)


def soft_top2_assign(z_abs: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Same soft top-2 logic as MacroDensityFNO.forward — returns (N, L) weights."""
    dist = np.abs(z_abs[:, None] - anchors[None, :])     # (N, L)
    L = anchors.shape[0]
    K = min(2, L)
    # top-K smallest distances per row
    idx_sort = np.argsort(dist, axis=1)[:, :K]            # (N, K)
    top_dist = np.take_along_axis(dist, idx_sort, axis=1) # (N, K)
    sum_d = top_dist.sum(axis=1, keepdims=True).clip(min=1e-6)
    w = (sum_d - top_dist) / sum_d
    w = w / w.sum(axis=1, keepdims=True).clip(min=1e-6)
    z_assign = np.zeros((z_abs.shape[0], L), dtype=np.float32)
    np.put_along_axis(z_assign, idx_sort, w, axis=1)
    return z_assign


# ──────────────────────────────────────────────────────────────────────────────
# Measurement 1: within-layer eps std on real cuboids (bypass P2G aggregation)
# ──────────────────────────────────────────────────────────────────────────────
def measure_within_layer_eps_std(tiles, anchors, threshold_weight: float = 0.5):
    """For each anchor, gather eps values from cuboids assigned to it (top-1
    weight ≥ threshold) and report mean / std / N."""
    per_anchor = defaultdict(list)
    for cubs in tiles:
        if cubs.shape[1] < 9:
            continue
        z = cubs[:, 2]
        eps = cubs[:, 8]
        z_assign = soft_top2_assign(z, anchors)            # (N, L)
        # Hard top-1 assignment (the dominant anchor) for clean per-anchor stats.
        top1 = np.argmax(z_assign, axis=1)
        for n in range(cubs.shape[0]):
            if z_assign[n, top1[n]] >= threshold_weight:
                per_anchor[int(top1[n])].append(float(eps[n]))

    print("\n" + "="*78)
    print("Measurement 1: Within-anchor eps distribution (raw cuboids)")
    print("="*78)
    print(f"{'Anchor':>6}  {'z (μm)':>8}  {'N':>7}  {'mean':>8}  {'std':>10}  {'min':>8}  {'max':>8}")
    print("-" * 78)
    cross_means = []
    for l in range(len(anchors)):
        vals = np.array(per_anchor.get(l, []))
        if len(vals) == 0:
            print(f"{l:>6}  {anchors[l]:>8.3f}  {'—':>7}")
            continue
        mu, sd = vals.mean(), vals.std()
        cross_means.append(mu)
        print(f"{l:>6}  {anchors[l]:>8.3f}  {len(vals):>7,}  {mu:>8.4f}  {sd:>10.4e}  "
              f"{vals.min():>8.4f}  {vals.max():>8.4f}")
    if cross_means:
        cm = np.array(cross_means)
        print(f"\n  Cross-anchor eps range : {cm.min():.3f} → {cm.max():.3f}  (Δ = {cm.max()-cm.min():.3f})")
        print(f"  Cross-anchor eps std   : {cm.std():.4f}")
    return per_anchor


# ──────────────────────────────────────────────────────────────────────────────
# Measurement 2: spatial std of the eps grid after P2G
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def measure_eps_grid_spatial_std(tiles, layer_map, device='cuda', n_samples: int = 50,
                                 grid_size: int = 16, sigma_xy: float = 0.3,
                                 window: float = 4.0):
    """Run MacroDensityFNO's P2G stage only and report per-layer xy std of the
    eps grid. If eps is constant per anchor, the grid for that layer is
    spatially flat → std ≈ 0."""
    mdf = MacroDensityFNO(grid_size=grid_size, d_macro=32, fno_modes=8,
                          n_fno_blocks=2, sigma_xy=sigma_xy, window=window,
                          layer_map=layer_map).to(device)
    mdf.eval()

    # Capture the density_grid before FNO blocks by hooking the first block.
    captured = {}
    def hook(mod, inp, out):
        captured['grid'] = inp[0].detach().clone()
    handle = mdf.fno[0].register_forward_hook(hook)

    L = mdf.n_layers
    G = grid_size
    eps_layer_spatial_std = [[] for _ in range(L)]
    eps_layer_mean        = [[] for _ in range(L)]
    valid_layer_count     = [0] * L

    pad_to = max(c.shape[0] for c in tiles[:n_samples])
    pad_to = min(pad_to, 4096)

    for cubs in tiles[:n_samples]:
        if cubs.shape[0] == 0 or cubs.shape[1] < 9:
            continue
        N = min(cubs.shape[0], pad_to)
        c = np.zeros((1, pad_to, cubs.shape[1]), dtype=np.float32)
        c[0, :N] = cubs[:N]
        pad_mask = np.ones((1, pad_to), dtype=bool)
        pad_mask[0, :N] = False

        cuboids_t = torch.from_numpy(c).to(device)
        mask_t    = torch.from_numpy(pad_mask).to(device)
        _ = mdf(cuboids_t, mask_t)

        grid = captured['grid']                      # (B*L, 2, G, G), B=1
        eps_grid = grid[:, 1].cpu().numpy()          # (L, G, G)
        for l in range(L):
            cell = eps_grid[l]
            mu = cell.mean()
            if mu > 1e-8:                            # layer has at least some mass
                eps_layer_spatial_std[l].append(float(cell.std()))
                eps_layer_mean[l].append(float(mu))
                valid_layer_count[l] += 1

    handle.remove()
    print("\n" + "="*78)
    print(f"Measurement 2: Eps grid xy-spatial std after P2G (N={n_samples} tiles)")
    print("="*78)
    print(f"{'Layer':>6}  {'#tiles':>7}  {'⟨mean⟩':>9}  {'⟨xy-std⟩':>10}  {'xy-std/mean':>12}")
    print("-" * 78)
    overall_ratio = []
    for l in range(L):
        if valid_layer_count[l] == 0:
            print(f"{l:>6}  {'—':>7}")
            continue
        mu  = np.mean(eps_layer_mean[l])
        sd  = np.mean(eps_layer_spatial_std[l])
        rat = sd / mu if mu > 0 else 0.0
        overall_ratio.append(rat)
        print(f"{l:>6}  {valid_layer_count[l]:>7}  {mu:>9.4f}  {sd:>10.4e}  {rat:>11.4%}")
    if overall_ratio:
        print(f"\n  Mean (xy-std / mean) across layers : {np.mean(overall_ratio):.4%}")
        print(f"  Max  (xy-std / mean) across layers : {np.max(overall_ratio):.4%}")


# ──────────────────────────────────────────────────────────────────────────────
# Measurement 3: effective rank of eps grid (singular-value distribution)
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def measure_eps_grid_rank(tiles, layer_map, device='cuda', n_samples: int = 30,
                          grid_size: int = 16, sigma_xy: float = 0.3,
                          window: float = 4.0, rank_eps_ratio: float = 1e-3):
    """Compute SVD of each per-layer eps grid (G, G). Effective rank counts
    singular values above rank_eps_ratio × max(σ). Rank=0 → uniform / zero,
    rank=1 → low-rank (essentially separable layer pattern), rank ≥ 4 →
    spatial structure that the FNO could meaningfully extract."""
    mdf = MacroDensityFNO(grid_size=grid_size, d_macro=32, fno_modes=8,
                          n_fno_blocks=2, sigma_xy=sigma_xy, window=window,
                          layer_map=layer_map).to(device)
    mdf.eval()

    captured = {}
    def hook(mod, inp, out):
        captured['grid'] = inp[0].detach().clone()
    handle = mdf.fno[0].register_forward_hook(hook)

    L = mdf.n_layers
    pad_to = min(max(c.shape[0] for c in tiles[:n_samples]), 4096)
    rank_per_layer_eps = [[] for _ in range(L)]
    rank_per_layer_vol = [[] for _ in range(L)]

    for cubs in tiles[:n_samples]:
        if cubs.shape[0] == 0 or cubs.shape[1] < 9:
            continue
        N = min(cubs.shape[0], pad_to)
        c = np.zeros((1, pad_to, cubs.shape[1]), dtype=np.float32)
        c[0, :N] = cubs[:N]
        pad_mask = np.ones((1, pad_to), dtype=bool)
        pad_mask[0, :N] = False

        cuboids_t = torch.from_numpy(c).to(device)
        mask_t    = torch.from_numpy(pad_mask).to(device)
        _ = mdf(cuboids_t, mask_t)

        grid = captured['grid'].cpu().numpy()        # (L, 2, G, G)
        for l in range(L):
            for ch_idx, store in enumerate([rank_per_layer_vol, rank_per_layer_eps]):
                cell = grid[l, ch_idx]
                if cell.mean() < 1e-8:
                    continue
                sv = np.linalg.svd(cell, compute_uv=False)
                if sv[0] < 1e-12:
                    continue
                eff_rank = int(np.sum(sv > sv[0] * rank_eps_ratio))
                store[l].append(eff_rank)

    handle.remove()
    print("\n" + "="*78)
    print(f"Measurement 3: Effective rank (SVD) of per-layer (G, G) grid")
    print("="*78)
    print(f"  rank_eps_ratio = {rank_eps_ratio} → counts σ_i > σ_max × ratio")
    print(f"\n{'Layer':>6}  {'⟨rank vol⟩':>12}  {'⟨rank eps⟩':>12}  {'σ_max(eps)/σ_max(vol)':>22}")
    print("-" * 78)
    for l in range(L):
        rv = np.mean(rank_per_layer_vol[l]) if rank_per_layer_vol[l] else None
        re = np.mean(rank_per_layer_eps[l]) if rank_per_layer_eps[l] else None
        if rv is None or re is None:
            print(f"{l:>6}  {'—':>12}  {'—':>12}")
        else:
            print(f"{l:>6}  {rv:>12.2f}  {re:>12.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=4)
    ap.add_argument('--n_tiles', type=int, default=200)
    ap.add_argument('--n_p2g_samples', type=int, default=50)
    ap.add_argument('--n_rank_samples', type=int, default=30)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)

    layer_map = LayerInfoParser(cfg.LAYERS_INFO_PATH).parse()
    anchors = compute_z_anchors(layer_map)
    print(f"\nLayer map: {len(layer_map)} entries, {len(anchors)} z-anchors after dedup")
    print(f"Anchor z-positions (μm): {anchors.tolist()}")

    processed = Path(cfg.PROCESSED_DIR)
    manifest  = processed / "dataset_manifest.csv"

    print(f"\nLoading {args.n_tiles} tiles from {manifest}...")
    t0 = time.time()
    tiles = load_random_tiles(processed, manifest, n_tiles=args.n_tiles)
    print(f"Loaded {len(tiles)} tiles in {time.time()-t0:.1f}s")
    if not tiles:
        print("No tiles loaded; aborting.")
        return

    measure_within_layer_eps_std(tiles, anchors)
    measure_eps_grid_spatial_std(tiles, layer_map, device=device,
                                  n_samples=args.n_p2g_samples)
    measure_eps_grid_rank(tiles, layer_map, device=device,
                          n_samples=args.n_rank_samples)

    print("\n" + "="*78)
    print("Interpretation guide")
    print("="*78)
    print("  M1 within-anchor std ≪ cross-anchor Δ → eps purely a function of z.")
    print("  M2 xy-std / mean < 1%                 → eps grid is spatially flat per layer.")
    print("  M3 ⟨rank eps⟩ ≈ 1                     → eps channel = constant on (G,G).")
    print("  All three together → drop eps channel, replace with per-layer scalar bias")
    print("                       (or just a learned (L,) embedding) — saves FFT work.")


if __name__ == '__main__':
    main()
