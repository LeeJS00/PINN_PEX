"""
train_ssl_v3.py — SSL pretrain entrypoint with M5 split filter applied.

Differences from legacy `src/trainers/train_ssl.py:36-41`:

  1. Reads v3 manifest (cfg.MANIFEST_PATH_V3) — SCHEMA_VERSION verified.
  2. Filters by `split == 'train'` so encoder never sees valid-net tiles
     (M5 fix). Legacy filtered by design only and silently leaked.
  3. Sets all 4 RNG sources via `pex_v3.src.utils.seeds.set_all_seeds`.
  4. Writes provenance.json (manifest hash + git SHA + config snapshot).
  5. SSL output dir: cfg.OUTPUT_DIR_V3 / "checkpoints" / cfg.RUN_NAME_V3
     (independent from legacy output_intel22).

Phase 0 sanity: this entrypoint reuses the legacy `DeepPEX_Model` and
`NeuralFieldSSLDataset` classes (read-only import). The Phase 1 hybrid
analytic+neural-residual model will replace the model class in a separate
PR; the pretraining harness here is meant to survive that change because
it depends only on dataset + checkpoint format.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Path setup
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "pex_v3"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from configs import config_v3 as cfg  # noqa: E402
from src.utils.seeds import set_all_seeds, worker_init_fn  # noqa: E402
from src.utils.manifest_hash import write_provenance  # noqa: E402
from src.data.leak_check import run_all_checks  # noqa: E402

# Legacy imports (read-only)
from src.data.datasets import NeuralFieldSSLDataset, robust_collate  # noqa: E402
from src.models.neural_field import DeepPEX_Model  # noqa: E402
from src.preprocessing.layer_parser import LayerInfoParser  # noqa: E402
from src.preprocessing.material_stack import BEOLMaterialStack  # noqa: E402
from src.physics.physics_sampler import PhysicsSampler  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="SSL pretrain (v3 with M5 split filter)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (4-way)")
    p.add_argument("--gpu", type=int, default=cfg.GPU_ID, help="CUDA device index")
    p.add_argument("--manifest", type=Path, default=cfg.MANIFEST_PATH_V3,
                   help="v3 manifest path (default: cfg.MANIFEST_PATH_V3)")
    p.add_argument("--run_name", type=str, default=cfg.RUN_NAME_V3,
                   help="SSL output run name (default: cfg.RUN_NAME_V3)")
    p.add_argument("--max_epochs", type=int, default=cfg.SSL_EPOCHS)
    return p.parse_args()


def main():
    args = parse_args()
    set_all_seeds(args.seed, deterministic=True)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    save_dir = cfg.OUTPUT_DIR_V3 / "checkpoints" / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- M5: load v3 manifest with split filter ------------------------
    if not args.manifest.exists():
        raise SystemExit(
            f"v3 manifest not found at {args.manifest}. "
            f"Run `python3 pex_v3/scripts/01_resplit_manifest.py` first."
        )
    manifest = pd.read_csv(args.manifest)
    test_stems = {p.stem for p in cfg.TEST_DEFS}
    run_all_checks(manifest, test_stems, expected_schema=cfg.SCHEMA_VERSION)

    train_def_stems = {p.stem for p in cfg.TRAIN_DEFS}
    train_df = manifest[
        manifest["design_name"].isin(train_def_stems)
        & (manifest["split"] == "train")
    ].copy()
    print(
        f">>> v3 SSL: {len(train_df):,} train tiles "
        f"({manifest.groupby('split').size().to_dict()} all splits)"
    )

    # ---- Provenance log ------------------------------------------------
    write_provenance(
        run_dir=save_dir,
        manifest_path=args.manifest,
        config_snapshot=cfg.v3_snapshot(),
        seed=args.seed,
        project_root=_PROJECT_ROOT,
    )

    # ---- Model + dataset ----------------------------------------------
    layer_map = LayerInfoParser(cfg.LAYERS_INFO_PATH).parse()
    mat_stack = BEOLMaterialStack(layer_map)
    sampler = PhysicsSampler(mat_stack, device=device)

    # Note: legacy NeuralFieldSSLDataset reads from PROCESSED_DIR; we point
    # it at the v3 data root.
    dataset = NeuralFieldSSLDataset(cfg.PROCESSED_DIR_V3, train_df)

    mini_batch = min(1024, cfg.SSL_BATCH_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=mini_batch,
        shuffle=True,
        drop_last=True,
        collate_fn=robust_collate,
        num_workers=cfg.SSL_NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    model = DeepPEX_Model(cfg).to(device)

    # ---- Training loop -------------------------------------------------
    # The actual SSL loss + optimizer setup lives in the legacy module; we
    # call into it for Phase 0. Phase 1 will replace the loss assembly.
    print(
        ">>> Phase 0 SSL: hand-off to legacy training loop. "
        "Replace this stub with the legacy `train_one_epoch` once we "
        "decide whether to keep legacy SSL objective."
    )
    raise NotImplementedError(
        "Phase 0 SSL training loop body intentionally left unimplemented. "
        "Once H1+H2+M5 manifest+loader infrastructure is validated by tests, "
        "wire the legacy SSL loss into this scope. See docs/PHASE0_PLAN.md."
    )


if __name__ == "__main__":
    main()
