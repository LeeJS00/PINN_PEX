#!/usr/bin/env python3
import argparse
import os
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import configs.config as cfg
from src.data.datasets import NeuralFieldFinetuneDataset, robust_collate
from src.data.replay_buffer import NetGroupedSampler
from src.models.neural_field import DeepPEX_Model


VALID_CSV = REPO_ROOT / "output_intel22" / "active_learning" / "cache" / "predefined_valid_subset.csv"
CKPT_PATH = REPO_ROOT / "output_intel22" / "active_learning" / "v3_netlevel" / "best_model.pth"
OUT_CSV = REPO_ROOT / "output_intel22" / "diag" / "gnd_cpl_decomp.csv"


def build_model(device: torch.device) -> DeepPEX_Model:
    model = DeepPEX_Model(cfg)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]

    clean_ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
    current_state = model.state_dict()
    filtered_ckpt = {
        k: v for k, v in clean_ckpt.items()
        if k in current_state and hasattr(v, "shape") and v.shape == current_state[k].shape
    }
    missing, unexpected = model.load_state_dict(filtered_ckpt, strict=False)
    model.to(device).eval()

    print(f">>> Loaded checkpoint: {CKPT_PATH}")
    print(f"    kept {len(filtered_ckpt)}/{len(clean_ckpt)} tensors | missing={len(missing)} unexpected={len(unexpected)}")
    return model


def attach_raw_gnd_capture(model: DeepPEX_Model) -> None:
    router = model.flux_router
    original_forward = router.forward

    def wrapped_forward(self, features, cuboids, is_target, is_aggr, padding_mask, compute_coupling=True, frw_ratio_matrix=None):
        sizes = cuboids[..., 3:6]
        w = sizes[..., 0]
        h = sizes[..., 1]
        z_abs = cuboids[..., 2]
        valid = (~padding_mask).float()

        z_idx_gnd = torch.argmin(torch.abs(z_abs.unsqueeze(-1) - self.metal_z_anchors), dim=-1)
        layer_cap_density = F.softplus(self.layer_scale_phys_gnd[z_idx_gnd])
        raw_gnd_base = (w * h).clamp(min=1e-6) * layer_cap_density * valid
        self._diag_last_raw_gnd_base = raw_gnd_base.detach()

        return original_forward(
            features,
            cuboids,
            is_target,
            is_aggr,
            padding_mask,
            compute_coupling=compute_coupling,
            frw_ratio_matrix=frw_ratio_matrix,
        )

    router.forward = types.MethodType(wrapped_forward, router)


def build_val_inputs():
    valid_df = pd.read_csv(VALID_CSV)
    dataset = NeuralFieldFinetuneDataset(str(cfg.PROCESSED_DIR), valid_df, pad_size=cfg.NF_PAD_TO_CUBOIDS)
    max_tiles_per_net = int(valid_df.groupby(["design_name", "net_name"]).size().max())
    batch_sampler = NetGroupedSampler(valid_df, batch_nets=1, max_tiles_per_batch=max_tiles_per_net)

    print(f">>> Validation subset: {len(valid_df)} tiles, {valid_df[['design_name', 'net_name']].drop_duplicates().shape[0]} nets")
    print(f"    sampler batch_nets=1, max_tiles_per_batch={max_tiles_per_net}, pad_size={cfg.NF_PAD_TO_CUBOIDS}")
    return valid_df, dataset, batch_sampler


def iter_val_batches(dataset: NeuralFieldFinetuneDataset, batch_sampler: NetGroupedSampler):
    for batch_indices in batch_sampler:
        batch_items = [dataset[i] for i in batch_indices]
        batch = robust_collate(batch_items)
        if batch is None:
            continue
        batch_df = dataset.meta_df.iloc[list(batch_indices)].reset_index(drop=True)
        yield batch, batch_df


def to_device(batch, device: torch.device):
    cuboids, mask, labels_dict, meta_dict = batch
    cuboids = cuboids.to(device)
    mask = mask.to(device)
    if cuboids.shape[-1] > 9:
        cuboids = cuboids[..., :9]

    labels_on_device = {}
    for key, value in labels_dict.items():
        labels_on_device[key] = value.to(device) if torch.is_tensor(value) else value

    frw_ratio_matrix = meta_dict.get("frw_ratio_matrix", None)
    if isinstance(frw_ratio_matrix, torch.Tensor):
        frw_ratio_matrix = frw_ratio_matrix.to(device)

    return cuboids, mask, labels_on_device, meta_dict, frw_ratio_matrix


def format_percent(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value * 100.0:.2f}%"


def print_metric_line(name: str, series: pd.Series) -> None:
    clean = series.dropna()
    if clean.empty:
        print(f"  {name:<12} mean=nan  median=nan")
        return
    print(f"  {name:<12} mean={format_percent(clean.mean())}  median={format_percent(clean.median())}")


def print_bucket_table(df: pd.DataFrame) -> None:
    bins = [-np.inf, 0.2, 0.5, 0.8, np.inf]
    labels = ["<0.2", "0.2-0.5", "0.5-0.8", ">0.8"]
    bucketed = df.copy()
    bucketed["bucket"] = pd.cut(bucketed["cpl_fraction"], bins=bins, labels=labels, right=False)
    grouped = bucketed.groupby("bucket", observed=True)["total_mape"].agg(["count", "mean"])
    for label in labels:
        if label in grouped.index:
            row = grouped.loc[label]
            mean_text = format_percent(float(row["mean"])) if int(row["count"]) > 0 else "nan"
            print(f"  {label:<8} count={int(row['count']):>4}  mean_total_mape={mean_text}")
        else:
            print(f"  {label:<8} count={0:>4}  mean_total_mape=nan")


def print_scatter_table(df: pd.DataFrame, gt_col: str, pred_col: str, mape_col: str, title: str) -> None:
    print(title)
    sample = df.sort_values(gt_col, ascending=False).head(10)
    if sample.empty:
        print("  <no rows>")
        return
    header = f"  {'design_name':<30} {'net_name':<36} {gt_col:>12} {pred_col:>12} {mape_col:>10}"
    print(header)
    for _, row in sample.iterrows():
        print(
            f"  {str(row['design_name'])[:30]:<30} "
            f"{str(row['net_name'])[:36]:<36} "
            f"{row[gt_col]:>12.4f} "
            f"{row[pred_col]:>12.4f} "
            f"{format_percent(row[mape_col]):>10}"
        )


def run_diagnosis(model: DeepPEX_Model, dataset: NeuralFieldFinetuneDataset, batch_sampler: NetGroupedSampler, device: torch.device) -> pd.DataFrame:
    use_amp = device.type == "cuda"
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    records = {}

    with torch.no_grad():
        for batch_idx, (batch, batch_df) in enumerate(iter_val_batches(dataset, batch_sampler), start=1):
            cuboids, mask, labels_dict, meta_dict, frw_ratio_matrix = to_device(batch, device)

            with autocast_ctx:
                preds = model(cuboids, mask, compute_coupling=True, frw_ratio_matrix=frw_ratio_matrix)

            raw_gnd_base = model.flux_router._diag_last_raw_gnd_base.float()
            c_gnd_seg = preds["c_gnd_seg"].float()
            sparse_cpl = preds["sparse_cpl"]

            a_tgt = labels_dict["A_tgt"]
            y_total = labels_dict["Y_total"]
            y_gnd = labels_dict["Y_gnd"]
            core_ratios = labels_dict["core_ratios"]
            batch_net_ids = labels_dict["batch_net_ids"]
            num_nets = int(labels_dict["num_unique_nets"])

            pred_gnd = torch.zeros(num_nets, dtype=torch.float32, device=device).scatter_add_(
                0,
                batch_net_ids,
                torch.sum(c_gnd_seg * a_tgt * core_ratios, dim=1),
            )

            pred_cpl = torch.zeros(num_nets, dtype=torch.float32, device=device)
            if len(sparse_cpl["c_cpl"]) > 0:
                b_idx = sparse_cpl["b_idx"]
                src_idx = sparse_cpl["src_idx"]
                edge_cpl = sparse_cpl["c_cpl"].float()
                edge_is_target = a_tgt[b_idx, src_idx] > 0
                edge_flux = edge_cpl * core_ratios[b_idx, src_idx]
                if edge_is_target.any():
                    pred_cpl.scatter_add_(0, batch_net_ids[b_idx[edge_is_target]], edge_flux[edge_is_target])

            modifier_ratio = c_gnd_seg / raw_gnd_base.clamp(min=1e-12)
            modifier_mask = (a_tgt > 0) & (raw_gnd_base > 0) & (core_ratios > 0)
            modifier_num = torch.zeros(num_nets, dtype=torch.float32, device=device).scatter_add_(
                0,
                batch_net_ids,
                torch.sum(modifier_ratio * modifier_mask.float() * core_ratios, dim=1),
            )
            modifier_den = torch.zeros(num_nets, dtype=torch.float32, device=device).scatter_add_(
                0,
                batch_net_ids,
                torch.sum(modifier_mask.float() * core_ratios, dim=1),
            )

            nid_to_key = {}
            batch_net_ids_cpu = batch_net_ids.cpu().numpy()
            for row_idx, nid in enumerate(batch_net_ids_cpu):
                key = (
                    str(batch_df.iloc[row_idx]["design_name"]),
                    str(batch_df.iloc[row_idx]["net_name"]).replace("\\", ""),
                )
                nid_to_key[int(nid)] = key

            for nid, key in nid_to_key.items():
                design_name, net_name = key
                rec = records.setdefault(
                    key,
                    {
                        "design_name": design_name,
                        "net_name": net_name,
                        "gt_total": float(y_total[nid].item()),
                        "gt_gnd": float(y_gnd[nid].item()),
                        "pred_gnd": 0.0,
                        "pred_cpl": 0.0,
                        "modifier_num": 0.0,
                        "modifier_den": 0.0,
                    },
                )
                rec["pred_gnd"] += float(pred_gnd[nid].item())
                rec["pred_cpl"] += float(pred_cpl[nid].item())
                rec["modifier_num"] += float(modifier_num[nid].item())
                rec["modifier_den"] += float(modifier_den[nid].item())

            if batch_idx % 25 == 0:
                print(f"    processed {batch_idx} net-batches")

    rows = []
    for rec in records.values():
        gt_total = rec["gt_total"]
        gt_gnd = rec["gt_gnd"]
        gt_cpl = gt_total - gt_gnd
        pred_total = rec["pred_gnd"] + rec["pred_cpl"]
        gnd_modifier_mean = rec["modifier_num"] / rec["modifier_den"] if rec["modifier_den"] > 0 else np.nan
        cpl_fraction = gt_cpl / gt_total if gt_total > 0 else np.nan
        cpl_mape = abs(rec["pred_cpl"] - gt_cpl) / (gt_cpl + 1e-6) if gt_cpl > 0.01 else np.nan
        rows.append(
            {
                "net_name": rec["net_name"],
                "design_name": rec["design_name"],
                "gt_gnd": gt_gnd,
                "pred_gnd": rec["pred_gnd"],
                "gt_cpl": gt_cpl,
                "pred_cpl": rec["pred_cpl"],
                "gt_total": gt_total,
                "pred_total": pred_total,
                "gnd_mape": abs(rec["pred_gnd"] - gt_gnd) / (gt_gnd + 1e-6),
                "cpl_mape": cpl_mape,
                "total_mape": abs(pred_total - gt_total) / (gt_total + 1e-6),
                "cpl_fraction": cpl_fraction,
                "gnd_modifier_mean": gnd_modifier_mean,
            }
        )

    return pd.DataFrame(rows).sort_values(["design_name", "net_name"]).reset_index(drop=True)


def print_analysis(df: pd.DataFrame) -> None:
    print("\n[Section 1] Overall")
    print_metric_line("gnd_mape", df["gnd_mape"])
    print_metric_line("cpl_mape", df.loc[df["gt_cpl"] > 0.01, "cpl_mape"])
    print_metric_line("total_mape", df["total_mape"])

    gnd_mean = df["gnd_mape"].dropna().mean()
    cpl_mean = df.loc[df["gt_cpl"] > 0.01, "cpl_mape"].dropna().mean()
    if np.isnan(cpl_mean) or gnd_mean >= cpl_mean:
        print("  Bottleneck: GND is the bottleneck")
    else:
        print("  Bottleneck: CPL is the bottleneck")

    print("\n[Section 2] CPL Fraction Buckets")
    print_bucket_table(df)

    mods = df["gnd_modifier_mean"].dropna()
    p50 = mods.quantile(0.50) if not mods.empty else np.nan
    p90 = mods.quantile(0.90) if not mods.empty else np.nan
    p99 = mods.quantile(0.99) if not mods.empty else np.nan
    print("\n[Section 3] GND Modifier Distribution")
    print(f"  P50={p50:.4f}" if not np.isnan(p50) else "  P50=nan")
    print(f"  P90={p90:.4f}" if not np.isnan(p90) else "  P90=nan")
    print(f"  P99={p99:.4f}" if not np.isnan(p99) else "  P99=nan")
    if not np.isnan(p90) and p90 > 7.0:
        print("  WARNING: GND modifier saturating near 10x cap")

    print("\n[Section 4] Text Scatter Tables")
    print_scatter_table(df, "gt_gnd", "pred_gnd", "gnd_mape", "  pred_gnd vs gt_gnd")
    print_scatter_table(df, "gt_cpl", "pred_cpl", "cpl_mape", "  pred_cpl vs gt_cpl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose total-cap MAPE into GND vs CPL contributions.")
    parser.add_argument("--gpu", type=int, default=cfg.GPU_ID, help="GPU id to use")
    args = parser.parse_args()

    if torch.cuda.is_available():
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    print(f">>> Device: {device}")
    valid_df, dataset, batch_sampler = build_val_inputs()
    _ = valid_df  # kept for explicit construction verification

    model = build_model(device)
    attach_raw_gnd_capture(model)

    df = run_diagnosis(model, dataset, batch_sampler, device)
    print_analysis(df)

    os.makedirs(OUT_CSV.parent, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
