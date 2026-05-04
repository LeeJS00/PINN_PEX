#!/usr/bin/env python3
"""
run_ablation_5seed.py — generic 5-seed ablation runner.

Reads `pex_v3/configs/ablation_manifest.yaml`, picks the variant by name,
and executes 5 deterministic training seeds (default 0..4). Each seed
writes:

    pex_v3/output/ablation/<variant>/seed{S}/
        provenance.json        # manifest hash, git SHA, config snapshot, seed
        summary.json           # day-1, final_valid, final_test, best_epoch (Mesh schema)
        history.json           # train_loss + per-epoch valid metrics
        model.pt               # state_dict
        eval_logger_valid.parquet   # per-net pred/gold + stratify covariates
        eval_logger_test.parquet

The runner reuses EXACTLY the same train/valid/test split as the locked
HybridPexV3Mesh 5-seed (split column from H1 manifest) and the same NNLS
per-layer calibration. Only the model class + train hyperparams may
differ between variants; the manifest enforces this.

Usage:
    # Single seed (self-test or quick smoke):
    python3 pex_v3/scripts/run_ablation_5seed.py \\
        --variant HybridPexV3Mesh --seeds 0 --gpu 0

    # Full 5-seed parallel on 5 GPUs:
    python3 pex_v3/scripts/run_ablation_5seed.py \\
        --variant HybridPexV3Mesh --seeds 0 1 2 3 4 --gpus 0 1 2 3 4

    # Inline single-seed (no subprocess):
    python3 pex_v3/scripts/run_ablation_5seed.py \\
        --variant HybridPexV3Mesh --seeds 0 --gpus 0 --inline
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "pex_v3"))

import yaml  # noqa: E402


DEFAULT_MANIFEST = _PROJECT_ROOT / "pex_v3" / "configs" / "ablation_manifest.yaml"


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_variant(manifest: dict, name: str) -> dict:
    """Merge defaults with the named variant; return a fully-populated dict."""
    if name not in manifest["variants"]:
        raise SystemExit(
            f"Variant {name!r} not in manifest. "
            f"Available: {list(manifest['variants'])}"
        )
    v = manifest["variants"][name]
    defaults = manifest.get("defaults", {})

    out = {
        "name": name,
        "description": v.get("description", ""),
        "model": v["model"],
        "data": {**defaults.get("data", {}), **v.get("data", {})},
        "train": {**defaults.get("train", {}), **(v.get("train") or {})},
        "output_root": v.get("output_root", defaults.get("output_root", "pex_v3/output/ablation")),
        "base_variant": v.get("base_variant"),
    }
    return out


# ---------------------------------------------------------------------------
# Inline single-seed execution (deterministic)
# ---------------------------------------------------------------------------


def _run_one_seed_inline(variant: dict, seed: int, output_dir: Path, device: str) -> dict:
    """Execute one seed in this process. Mirrors 19_finetune_hybrid_mesh_smoke."""
    import importlib
    import numpy as np
    import pandas as pd
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from src.models.hybrid_v3 import per_channel_mape_loss
    from src.models.residual_head_v3 import res_clamp_for_epoch
    from src.data.cuboid_set_dataset import (
        PerNetCuboidStore, CuboidAugmentedDataset, collate_cuboid_batch,
    )
    from src.trainers.finetune_hybrid_v3 import (
        split_by_manifest_column, _SELF_FEATURE_COLS, _PAIR_FEATURE_COLS,
    )
    from src.baselines.calibration_v3 import (
        fit_per_layer_calibration, apply_per_layer_calibration, validate_calibration,
        fit_scalar_calibration, apply_scalar_calibration,
    )
    from src.utils.seeds import set_all_seeds, worker_init_fn
    from src.utils.manifest_hash import write_provenance
    from src.utils.eval_logger import collect_per_net_predictions, write_eval_parquet
    from configs import config_v3 as cfg

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- determinism: 4-way seed + cudnn deterministic + CUBLAS workspace
    set_all_seeds(seed, deterministic=True)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    # Disable torch.compile (legacy uses it; v3 disables for true determinism)
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

    # --- heartbeat file
    hb_path = output_dir / "heartbeat.txt"
    def heartbeat(msg: str) -> None:
        hb_path.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")

    heartbeat("loading features")
    df = pd.read_csv(variant["data"]["features_csv"])
    train_df, valid_df, test_df = split_by_manifest_column(df)
    for d in (train_df, valid_df, test_df):
        d.drop(d[(d["c_gnd_fF"] + d["c_cpl_total_fF"]) <= 1e-4].index, inplace=True)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f">>> splits: train={len(train_df):,}  valid={len(valid_df):,}  test={len(test_df):,}",
          flush=True)

    # --- NNLS calibration (per_layer | scalar | none) on TRAIN ONLY
    cal_kind = variant["data"].get("use_calibration", "per_layer")
    cal_summary: dict[str, Any] = {"calibration": cal_kind}
    if cal_kind == "per_layer":
        before_v = validate_calibration(valid_df)
        calib = fit_per_layer_calibration(train_df)
        train_df = apply_per_layer_calibration(train_df, calib)
        valid_df = apply_per_layer_calibration(valid_df, calib)
        test_df  = apply_per_layer_calibration(test_df,  calib)
        after_v = validate_calibration(valid_df)
        cal_summary["before_valid"] = before_v
        cal_summary["after_valid"] = after_v
    elif cal_kind == "scalar":
        calib = fit_scalar_calibration(train_df)
        train_df = apply_scalar_calibration(train_df, calib)
        valid_df = apply_scalar_calibration(valid_df, calib)
        test_df  = apply_scalar_calibration(test_df,  calib)

    heartbeat("loading cuboid store")
    store = PerNetCuboidStore(Path(variant["data"]["cuboid_dir"]))
    print(f">>> cuboid store: {len(store):,}", flush=True)

    train_ds = CuboidAugmentedDataset(train_df, store, _SELF_FEATURE_COLS, _PAIR_FEATURE_COLS)
    valid_ds = CuboidAugmentedDataset(valid_df, store, _SELF_FEATURE_COLS, _PAIR_FEATURE_COLS)
    test_ds  = CuboidAugmentedDataset(test_df,  store, _SELF_FEATURE_COLS, _PAIR_FEATURE_COLS)

    train_loader = DataLoader(
        train_ds, batch_size=variant["train"]["batch_size"], shuffle=True,
        num_workers=variant["train"]["num_workers"], collate_fn=collate_cuboid_batch,
        pin_memory=("cuda" in device), worker_init_fn=worker_init_fn,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=variant["train"]["batch_size"], shuffle=False,
        num_workers=variant["train"]["num_workers"], collate_fn=collate_cuboid_batch,
        pin_memory=("cuda" in device),
    )
    test_loader = DataLoader(
        test_ds, batch_size=variant["train"]["batch_size"], shuffle=False,
        num_workers=variant["train"]["num_workers"], collate_fn=collate_cuboid_batch,
        pin_memory=("cuda" in device),
    )

    # --- model: dynamic import from manifest
    mod = importlib.import_module(variant["model"]["module"])
    cls = getattr(mod, variant["model"]["class"])
    torch.manual_seed(seed)            # ensures init reproducibility
    model = cls(**(variant["model"].get("kwargs") or {})).to(device)
    if hasattr(model, "parameter_count"):
        pc = model.parameter_count()
    else:
        pc = {"total": sum(p.numel() for p in model.parameters())}
    print(f">>> model params: {pc}", flush=True)

    # --- provenance
    snap = cfg.v3_snapshot()
    snap["task"] = f"ablation_{variant['name']}"
    snap["calibration"] = cal_summary["calibration"]
    snap["n_epochs"] = variant["train"]["n_epochs"]
    snap["model_params"] = pc.get("total", 0)
    snap["model_class"] = f"{variant['model']['module']}.{variant['model']['class']}"
    snap["seed"] = seed
    snap["pythonhashseed"] = os.environ.get("PYTHONHASHSEED")
    write_provenance(output_dir, Path(variant["data"]["features_csv"]), snap, seed)

    # --- evaluator (streaming)
    def evaluate_full(loader):
        import torch as _torch
        model.eval()
        gnd_l, cpl_l, tot_l = [], [], []
        with _torch.no_grad():
            for batch in loader:
                ag = batch["analytic_gnd"].to(device)
                ac = batch["analytic_cpl"].to(device)
                sf = batch["self_features"].to(device)
                pf = batch["pair_features"].to(device)
                cb = batch["cuboids"].to(device)
                mk = batch["padding_mask"].to(device)
                gg = batch["golden_gnd"].to(device)
                gc = batch["golden_cpl"].to(device)
                pg = model.predict_gnd(ag, sf, cb, mk)
                pc_ = model.predict_cpl(ac, pf, cb, mk)
                gnd_l.append(((pg - gg).abs() / gg.clamp(min=1e-3)).cpu())
                cpl_l.append(((pc_ - gc).abs() / gc.clamp(min=1e-3)).cpu())
                tot_l.append(((pg + pc_ - gg - gc).abs() / (gg + gc).clamp(min=1e-3)).cpu())
        gnd = _torch.cat(gnd_l); cpl = _torch.cat(cpl_l); tot = _torch.cat(tot_l)
        return {
            "gnd_mape_median": float(gnd.median().item()),
            "gnd_mape_mean":   float(gnd.mean().item()),
            "cpl_mape_median": float(cpl.median().item()),
            "cpl_mape_mean":   float(cpl.mean().item()),
            "total_mape_median": float(tot.median().item()),
            "total_mape_mean":   float(tot.mean().item()),
            "n_nets": int(len(gnd)),
        }

    # --- day-1 eval
    heartbeat("day-1 eval")
    day1 = evaluate_full(valid_loader)
    print(f">>> day-1 valid: gnd={day1['gnd_mape_median']*100:.2f}%  "
          f"cpl={day1['cpl_mape_median']*100:.2f}%  "
          f"total={day1['total_mape_median']*100:.2f}%", flush=True)

    # --- training
    optimizer = optim.Adam(
        model.parameters(),
        lr=variant["train"]["lr"], weight_decay=variant["train"]["weight_decay"],
    )
    history = {
        "train_loss": [], "valid_total_mape": [], "valid_gnd_mape": [],
        "valid_cpl_mape": [], "epoch_complete": [],
    }
    best_epoch, best_total, best_gnd, best_cpl = -1, float("inf"), float("inf"), float("inf")
    epochs_without_improvement = 0
    n_epochs = int(variant["train"]["n_epochs"])
    es_patience = int(variant["train"].get("early_stop_patience", 99999))
    use_curriculum = bool(variant["train"].get("curriculum", True))
    eval_every = int(variant["train"].get("eval_every_n_epochs", 1))

    t0 = time.time()
    for epoch in range(n_epochs):
        if use_curriculum:
            clamp = res_clamp_for_epoch(epoch)
            if hasattr(model, "set_clamp_bounds"):
                model.set_clamp_bounds(clamp)
        model.train()
        running = 0.0
        n_b = 0
        for step, batch in enumerate(train_loader):
            ag = batch["analytic_gnd"].to(device)
            ac = batch["analytic_cpl"].to(device)
            sf = batch["self_features"].to(device)
            pf = batch["pair_features"].to(device)
            cb = batch["cuboids"].to(device)
            mk = batch["padding_mask"].to(device)
            gg = batch["golden_gnd"].to(device)
            gc = batch["golden_cpl"].to(device)
            pg = model.predict_gnd(ag, sf, cb, mk)
            pc_ = model.predict_cpl(ac, pf, cb, mk)
            losses = per_channel_mape_loss(pg, gg, pc_, gc)
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()
            running += float(losses["total_loss"].item())
            n_b += 1
        avg = running / max(1, n_b)
        history["train_loss"].append(avg)

        if (epoch + 1) % eval_every == 0:
            v = evaluate_full(valid_loader)
            history["valid_total_mape"].append(v["total_mape_median"])
            history["valid_gnd_mape"].append(v["gnd_mape_median"])
            history["valid_cpl_mape"].append(v["cpl_mape_median"])
            history["epoch_complete"].append(epoch)
            elapsed = time.time() - t0
            print(
                f"  epoch {epoch}/{n_epochs}: clamp={res_clamp_for_epoch(epoch):.3f}  "
                f"train_loss={avg:.4f}  "
                f"valid mape: gnd={v['gnd_mape_median']*100:.2f}%  "
                f"cpl={v['cpl_mape_median']*100:.2f}%  "
                f"total={v['total_mape_median']*100:.2f}%  ({elapsed:.0f}s)",
                flush=True,
            )
            heartbeat(f"epoch {epoch}/{n_epochs} valid_total={v['total_mape_median']:.4f}")
            if v["total_mape_median"] < best_total:
                best_total = v["total_mape_median"]
                best_gnd = v["gnd_mape_median"]
                best_cpl = v["cpl_mape_median"]
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= es_patience:
                    print(f"  early stop at epoch {epoch}", flush=True)
                    break
    train_elapsed = time.time() - t0

    # --- final eval
    final_valid = evaluate_full(valid_loader)
    final_test = evaluate_full(test_loader)
    print(f">>> final valid: gnd={final_valid['gnd_mape_median']*100:.2f}%  "
          f"cpl={final_valid['cpl_mape_median']*100:.2f}%  "
          f"total={final_valid['total_mape_median']*100:.2f}%", flush=True)
    print(f">>> final test : gnd={final_test['gnd_mape_median']*100:.2f}%  "
          f"cpl={final_test['cpl_mape_median']*100:.2f}%  "
          f"total={final_test['total_mape_median']*100:.2f}%", flush=True)

    # --- per-net eval logger (parquet) for stratifier + paired MWU
    heartbeat("eval_logger valid")
    valid_pred_df = collect_per_net_predictions(model, valid_loader, device, valid_df)
    write_eval_parquet(valid_pred_df, output_dir / "eval_logger_valid.parquet")
    heartbeat("eval_logger test")
    test_pred_df = collect_per_net_predictions(model, test_loader, device, test_df)
    write_eval_parquet(test_pred_df, output_dir / "eval_logger_test.parquet")

    # --- save model + summary + history
    import torch as _torch
    _torch.save(model.state_dict(), output_dir / "model.pt")
    with open(output_dir / "summary.json", "w") as f:
        json.dump({
            "seed": seed,
            "variant": variant["name"],
            "n_epochs": n_epochs,
            "model_params": pc,
            "elapsed_train_sec": train_elapsed,
            **cal_summary,
            "day1_valid": day1,
            "final_valid": final_valid,
            "final_test": final_test,
            "best_epoch": best_epoch,
            "best_valid_total_mape": best_total,
            "best_valid_gnd_mape": best_gnd,
            "best_valid_cpl_mape": best_cpl,
        }, f, indent=2, default=str)
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    heartbeat("done")
    return {"final_valid": final_valid, "final_test": final_test,
            "best_epoch": best_epoch, "best_valid_total": best_total}


# ---------------------------------------------------------------------------
# Multi-seed orchestrator (subprocess fan-out per GPU)
# ---------------------------------------------------------------------------


def _run_one_seed_subprocess(variant_name: str, seed: int, gpu: int,
                             output_dir: Path, manifest_path: Path) -> tuple[int, Path]:
    """Spawn a child process that runs ONE seed inline on `gpu`. Returns (rc, log)."""
    log_path = output_dir.parent / f"runner_seed{seed}_gpu{gpu}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TORCH_COMPILE_DISABLE"] = "1"
    env["PYTHONHASHSEED"] = str(seed)

    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--variant", variant_name,
        "--manifest", str(manifest_path),
        "--seeds", str(seed),
        "--gpus", "0",            # post-CUDA_VISIBLE_DEVICES → device 0
        "--inline",
    ]
    with open(log_path, "w") as logf:
        logf.write(f">>> seed={seed} gpu={gpu} pid={os.getpid()} "
                   f"started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        logf.flush()
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                              cwd=str(_PROJECT_ROOT), env=env)
    return proc.returncode, log_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generic 5-seed ablation runner")
    p.add_argument("--variant", type=str, required=True,
                   help="Variant name from configs/ablation_manifest.yaml")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--gpus", nargs="+", type=int, default=None,
                   help="GPU index per seed (length must match --seeds when subprocess)")
    p.add_argument("--gpu", type=int, default=None,
                   help="Shorthand for --gpus <gpu> when --seeds has length 1")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output-root", type=Path, default=None,
                   help="Override the manifest's output_root")
    p.add_argument("--inline", action="store_true",
                   help="Run sequentially in this process (single-seed self-test).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu is not None and args.gpus is None:
        args.gpus = [args.gpu]
    if args.gpus is None:
        args.gpus = list(range(len(args.seeds)))

    if len(args.gpus) != len(args.seeds):
        raise SystemExit(f"--seeds (n={len(args.seeds)}) and --gpus (n={len(args.gpus)}) lengths must match")

    manifest = load_manifest(args.manifest)
    variant = resolve_variant(manifest, args.variant)
    output_root = args.output_root or (_PROJECT_ROOT / variant["output_root"])
    variant_dir = Path(output_root) / variant["name"]
    variant_dir.mkdir(parents=True, exist_ok=True)

    print(f">>> variant: {variant['name']}  ({variant['description']})")
    print(f">>> model:   {variant['model']['module']}.{variant['model']['class']}")
    print(f">>> output:  {variant_dir}")
    print(f">>> seeds × GPUs:")
    for s, g in zip(args.seeds, args.gpus):
        print(f"      seed {s}  →  GPU {g}")

    # Snapshot the manifest into the variant dir for provenance
    (variant_dir / "ablation_manifest_snapshot.yaml").write_text(
        Path(args.manifest).read_text()
    )

    if args.inline:
        if len(args.seeds) != 1:
            raise SystemExit("--inline supports a single seed at a time; use parallel for >1.")
        seed = args.seeds[0]
        gpu = args.gpus[0]
        # In inline+subprocess case, CUDA_VISIBLE_DEVICES already set; device='cuda:0'
        # In direct inline case, honour --gpu and use cuda:N
        import torch
        if torch.cuda.is_available():
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                device = "cuda:0"
            else:
                device = f"cuda:{gpu}"
        else:
            device = "cpu"
        seed_dir = variant_dir / f"seed{seed}"
        result = _run_one_seed_inline(variant, seed, seed_dir, device)
        print(f">>> inline seed {seed} done. last test total = "
              f"{result['final_test']['total_mape_median']*100:.3f}%")
        return

    # Parallel multi-seed via subprocesses
    procs = []
    for seed, gpu in zip(args.seeds, args.gpus):
        seed_dir = variant_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        log_path = variant_dir / f"runner_seed{seed}_gpu{gpu}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["TORCH_COMPILE_DISABLE"] = "1"
        env["PYTHONHASHSEED"] = str(seed)
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--variant", args.variant,
            "--manifest", str(args.manifest),
            "--seeds", str(seed),
            "--gpus", "0",
            "--inline",
        ]
        if args.output_root is not None:
            cmd.extend(["--output-root", str(args.output_root)])
        logf = open(log_path, "w")
        logf.write(f">>> seed={seed} gpu={gpu} started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        logf.flush()
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                cwd=str(_PROJECT_ROOT), env=env)
        procs.append((seed, gpu, proc, logf, log_path))
        print(f"  launched seed {seed} on GPU {gpu} (pid {proc.pid}) → {log_path}")

    print(f">>> launched {len(procs)} processes; waiting ...")
    t0 = time.time()
    failed = []
    for seed, gpu, proc, logf, log_path in procs:
        rc = proc.wait()
        logf.close()
        elapsed = time.time() - t0
        status = "OK" if rc == 0 else "FAIL"
        print(f"  [{status}] seed {seed} GPU {gpu}  rc={rc}  elapsed={elapsed/60:.1f}min  log={log_path}")
        if rc != 0:
            failed.append((seed, gpu, rc))

    if failed:
        print(f"!! {len(failed)} seed(s) failed.")
        sys.exit(1)
    print(f">>> all {len(procs)} seeds finished. Variant dir: {variant_dir}")


if __name__ == "__main__":
    main()
