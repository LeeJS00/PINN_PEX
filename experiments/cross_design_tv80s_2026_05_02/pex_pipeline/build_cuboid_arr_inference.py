"""Build cuboid array npz (target + aggressor + power streams) from pkls.

Inference variant of scripts/precache_cuboid_arrays.py — works without
the global manifest, taking pkl files directly from a directory.
"""
from __future__ import annotations

import argparse
import gzip
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

T_MAX = 128
A_MAX = 256
P_MAX = 128


def _hash_geom(arr: np.ndarray) -> np.ndarray:
    rounded = np.round(arr, 4).astype(np.float32)
    rb = rounded.tobytes()
    rec_len = arr.shape[1] * 4
    out = np.empty(len(arr), dtype=np.int64)
    for i in range(len(arr)):
        out[i] = hash(rb[i * rec_len:(i + 1) * rec_len])
    return out


def _truncate_priority(c_all: np.ndarray, ag_all: np.ndarray, target_xy: np.ndarray, k: int):
    if c_all.shape[0] == 0:
        return c_all, 0
    if c_all.shape[0] <= k:
        return c_all, c_all.shape[0]
    if target_xy.shape[0] > 0:
        cx = target_xy[:, 0].mean(); cy = target_xy[:, 1].mean()
        d2 = (ag_all[:, 0] - cx) ** 2 + (ag_all[:, 1] - cy) ** 2
        idx = np.argsort(d2)[:k]
    else:
        idx = np.arange(k)
    return c_all[idx], k


def _process_net(args):
    pkl_paths, name = args
    records = []
    for p in pkl_paths:
        try:
            with gzip.open(p, "rb") as fh:
                records.append(pickle.load(fh))
        except Exception:
            continue
    if not records:
        return None

    target_rows, agg_rows, pwr_rows = [], [], []
    target_geom, agg_geom, pwr_geom = [], [], []
    for rec in records:
        if not isinstance(rec, dict) or "cuboids" not in rec or "abs_geometries" not in rec:
            continue
        c = rec["cuboids"]; ag = rec["abs_geometries"]
        m_t = c[:, 7] == 1.0
        m_pwr = (c[:, 7] == 0.0) & (c[:, 9] >= 0.6)
        m_agg = (c[:, 7] == 0.0) & (c[:, 9] < 0.6)
        target_rows.append(c[m_t]); target_geom.append(ag[m_t])
        agg_rows.append(c[m_agg]); agg_geom.append(ag[m_agg])
        pwr_rows.append(c[m_pwr]); pwr_geom.append(ag[m_pwr])

    if not target_rows or all(r.shape[0] == 0 for r in target_rows):
        return None

    T = np.concatenate(target_rows, axis=0).astype(np.float32)
    Tg = np.concatenate(target_geom, axis=0).astype(np.float32)
    A = np.concatenate(agg_rows, axis=0).astype(np.float32) if agg_rows else np.zeros((0, 10), dtype=np.float32)
    Ag = np.concatenate(agg_geom, axis=0).astype(np.float32) if agg_geom else np.zeros((0, 6), dtype=np.float32)
    P = np.concatenate(pwr_rows, axis=0).astype(np.float32) if pwr_rows else np.zeros((0, 10), dtype=np.float32)
    Pg = np.concatenate(pwr_geom, axis=0).astype(np.float32) if pwr_geom else np.zeros((0, 6), dtype=np.float32)

    if T.shape[0] > 0:
        h = _hash_geom(Tg)
        _, idx = np.unique(h, return_index=True)
        idx = np.sort(idx)
        T = T[idx]; Tg = Tg[idx]

    T_keep, n_t = _truncate_priority(T, Tg, Tg, T_MAX)
    A_keep, n_a = _truncate_priority(A, Ag, Tg, A_MAX)
    P_keep, n_p = _truncate_priority(P, Pg, Tg, P_MAX)

    T_pad = np.zeros((T_MAX, 10), dtype=np.float32)
    A_pad = np.zeros((A_MAX, 10), dtype=np.float32)
    P_pad = np.zeros((P_MAX, 10), dtype=np.float32)
    T_pad[:n_t] = T_keep
    A_pad[:n_a] = A_keep
    P_pad[:n_p] = P_keep

    return name, T_pad, A_pad, P_pad, n_t, n_a, n_p


def build_cuboid_arr_inference(design_name: str,
                                 cuboid_pkl_dir: Path,
                                 out_path: Path,
                                 manifest_df: pd.DataFrame = None,
                                 n_workers: int = 12) -> Path:
    if manifest_df is None:
        pkl_files = sorted(Path(cuboid_pkl_dir).rglob("*.pkl.gz"))
        if not pkl_files:
            raise RuntimeError(f"No pkl.gz in {cuboid_pkl_dir}")
        net_to_pkls = {}
        for p in pkl_files:
            try:
                with gzip.open(p, "rb") as f:
                    rec = pickle.load(f)
                # Skip non-tile auxiliary pkls (e.g. inst_net_map)
                if not isinstance(rec, dict) or "cuboids" not in rec:
                    continue
                net_name = rec.get("net_name", p.stem)
                net_to_pkls.setdefault(net_name, []).append(p)
            except Exception:
                continue
    else:
        sub = manifest_df[manifest_df["design_name"] == design_name].copy()
        if "abs_path" not in sub.columns and "rel_path" in sub.columns:
            sub["abs_path"] = sub["rel_path"].astype(str)
        net_to_pkls = sub.groupby("net_name")["abs_path"].apply(
            lambda x: [Path(p) for p in x]
        ).to_dict()

    print(f"[{design_name}] {len(net_to_pkls)} nets")
    work = [(pkls, name) for name, pkls in net_to_pkls.items()]

    rows = []
    t0 = time.time()
    with mp.Pool(processes=n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_process_net, work, chunksize=4)):
            if res is None:
                continue
            rows.append(res)
            if (i + 1) % 500 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(work) - i - 1) / max(rate, 1e-3)
                print(f"[{design_name}]  {i+1}/{len(work)}  {rate:.1f}/s eta {eta:.0f}s", flush=True)

    if not rows:
        raise RuntimeError(f"No valid nets")

    rows.sort(key=lambda r: r[0])
    net_names = np.array([r[0] for r in rows])
    T_arr = np.stack([r[1] for r in rows], axis=0)
    A_arr = np.stack([r[2] for r in rows], axis=0)
    P_arr = np.stack([r[3] for r in rows], axis=0)
    n_t = np.array([r[4] for r in rows], dtype=np.int32)
    n_a = np.array([r[5] for r in rows], dtype=np.int32)
    n_p = np.array([r[6] for r in rows], dtype=np.int32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        target=T_arr, aggressor=A_arr, power=P_arr,
        n_target=n_t, n_agg=n_a, n_pwr=n_p,
        net_names=net_names,
    )
    print(f"[{design_name}] → {out_path} ({len(rows)} nets) in {time.time()-t0:.1f}s")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design_name", required=True)
    ap.add_argument("--cuboid_pkl_dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    manifest = pd.read_csv(args.manifest) if args.manifest else None
    build_cuboid_arr_inference(args.design_name, args.cuboid_pkl_dir, args.out,
                                 manifest_df=manifest, n_workers=args.workers)


if __name__ == "__main__":
    main()
