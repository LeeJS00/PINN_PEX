"""
Train DeepSet/MLP cap regression model.

Reads cached parquet features + on-the-fly cuboid sets from v3 pkls. Targets
log(total_cap_fF). Validates on nova, tests on tv80s.

Multi-seed support — pass --seed multiple times via repeated calls or use
the orchestrator script.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS))

from configs import cfg
from src.data_loader import load_split, report_mape
from src.deepset_dataset import CuboidSetDataset, MAX_TARGET, MAX_AGG, MAX_PWR
from src.deepset_model import DeepSetCapModel, hybrid_loss


def set_seeds(s: int):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def make_loaders(splits, manifest, feature_cols, batch_size: int, num_workers: int):
    dsets = {
        k: CuboidSetDataset(df, manifest, feature_cols, train=(k == "train"))
        for k, df in splits.items()
    }
    Xtr = splits["train"][feature_cols].to_numpy(dtype=np.float32)
    mean = np.nanmean(Xtr, axis=0); std = np.nanstd(Xtr, axis=0)
    for d in dsets.values():
        d.set_hand_normalizer(mean, std)
    loaders = {}
    for k, d in dsets.items():
        loaders[k] = DataLoader(
            d,
            batch_size=batch_size,
            shuffle=(k == "train"),
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            drop_last=(k == "train"),
        )
    return loaders, mean, std


def evaluate(model, loader, device, target_lin: bool = True):
    model.eval()
    ys = []
    yhats = []
    designs = []
    nets = []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
            out = model(batch)               # log domain
            yhats.append(torch.exp(out).cpu().numpy())
            ys.append(batch["y"][:, 0].cpu().numpy())
            designs.extend(batch["design"])
            nets.extend(batch["net"])
    y = np.concatenate(ys)
    yhat = np.concatenate(yhats)
    return y, yhat, designs, nets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--ckpt-tag", type=str, default="deepset")
    ap.add_argument("--max-train-batches", type=int, default=-1,
                    help="cap batches per epoch for fast iteration; -1 = full")
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    set_seeds(args.seed)

    splits, feature_cols = load_split()
    manifest = pd.read_csv(cfg.MANIFEST_PATH)

    loaders, hand_mean, hand_std = make_loaders(splits, manifest, feature_cols,
                                                 batch_size=args.batch_size,
                                                 num_workers=args.num_workers)

    model = DeepSetCapModel(hand_dim=len(feature_cols)).to(device)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_root = cfg.OUTPUT_DIR / "deepset" / f"seed{args.seed}"
    out_root.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    history = []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        losses = []
        for bi, batch in enumerate(loaders["train"]):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
            opt.zero_grad()
            pred = model(batch)              # log domain
            y_lin = batch["y"][:, 0]
            loss = hybrid_loss(pred, y_lin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            if args.max_train_batches > 0 and bi >= args.max_train_batches:
                break
        sched.step()
        train_loss = float(np.mean(losses)) if losses else float("nan")

        y_val, yhat_val, _, _ = evaluate(model, loaders["val"], device)
        val_metrics = report_mape(y_val, yhat_val, label=f"epoch{epoch} val")

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "wall": time.time() - t0,
        })
        if val_metrics["mape_median"] < best_val:
            best_val = val_metrics["mape_median"]
            torch.save({"model": model.state_dict(),
                        "feature_cols": feature_cols,
                        "hand_mean": hand_mean,
                        "hand_std": hand_std,
                        "epoch": epoch,
                        "val_metrics": val_metrics},
                       out_root / "best.pt")

    pd.DataFrame(history).to_csv(out_root / "history.csv", index=False)

    # Test evaluation with best ckpt
    state = torch.load(out_root / "best.pt", map_location=device)
    model.load_state_dict(state["model"])
    y_test, yhat_test, designs, nets = evaluate(model, loaders["test"], device)
    test_metrics = report_mape(y_test, yhat_test, label="TEST tv80s")

    # save predictions
    pd.DataFrame({
        "design_name": designs,
        "net_name": nets,
        "y_true_total_cap_fF": y_test,
        "y_pred_total_cap_fF": yhat_test,
    }).to_csv(out_root / "preds_test.csv", index=False)

    with open(out_root / "summary.json", "w") as f:
        json.dump({
            "seed": args.seed,
            "best_val_mape_median": best_val,
            "test": test_metrics,
        }, f, indent=2)


if __name__ == "__main__":
    main()
