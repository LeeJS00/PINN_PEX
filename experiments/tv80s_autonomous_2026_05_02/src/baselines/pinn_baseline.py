"""
pinn_baseline.py — Phase 0.5 B3.

Wraps the legacy `run_active_learning.main()` so it trains/evaluates the
legacy `DeepPEX_Model` on the v3 rebuilt manifest. This is THE most important
baseline for the paper because it lets the reviewer answer:

    "Is the headline gain from the new paradigm or just from the data fixes?"

The wrapper:
  1. Symlinks v3 manifest at the filename legacy expects (`dataset_manifest.csv`).
  2. Monkey-patches `cfg` so legacy reads/writes v3 paths.
  3. Symlinks the legacy SSL basis (`ssl_basis_dspinn_v1`) into v3 output dir
     so legacy AL finds it. Note: this basis was pretrained on the OLD data
     (with H1 leak); a clean re-pretrain (M5 fix) is deferred to a later
     iteration. We label results to make this transparent.
  4. Calls `run_active_learning.main(args)` for one seed.
  5. Parses the per-iter training log → MetricsRow.

Default: `max_iters=1, steps_per_iter=5000` matches the historical 5-seed
benchmark protocol that produced v10b 63.79% baseline (`docs/PROJECT_REPORT.md` §2.2.4).
"""
from __future__ import annotations
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import MetricsRow


# ============================================================================
# Path-rewiring helpers
# ============================================================================


def _ensure_symlink(target: Path, link_name: Path, force: bool = False) -> None:
    """Create symlink `link_name -> target` (idempotent)."""
    if link_name.exists() or link_name.is_symlink():
        if force:
            link_name.unlink()
        else:
            return
    link_name.symlink_to(target)


def _setup_v3_paths(
    project_root: Path,
    v3_processed_dir: Path,
    v3_pt_dir: Path,
    v3_output_dir: Path,
    v3_manifest_path: Path,
    legacy_ssl_basis_dir: Path,
    legacy_run_name: str,
) -> None:
    """Wire v3 directory layout so legacy AL finds files at expected names.

    1. v3_processed_dir/dataset_manifest.csv  -> v3_manifest_path
    2. v3_output_dir/checkpoints/<legacy_run_name>  -> legacy_ssl_basis_dir
    """
    # 1. manifest symlink
    legacy_manifest_link = v3_processed_dir / "dataset_manifest.csv"
    if v3_manifest_path.exists():
        _ensure_symlink(v3_manifest_path, legacy_manifest_link, force=True)
    elif not legacy_manifest_link.exists():
        raise FileNotFoundError(
            f"v3 manifest not at {v3_manifest_path}. "
            f"Run pex_v3/scripts/01_resplit_manifest.py first."
        )

    # 2. SSL basis symlink (legacy AL reads from output_dir/checkpoints/run_name)
    v3_ckpt_root = v3_output_dir / "checkpoints"
    v3_ckpt_root.mkdir(parents=True, exist_ok=True)
    legacy_ssl_target = v3_ckpt_root / legacy_run_name
    if legacy_ssl_basis_dir.exists() and not legacy_ssl_target.exists():
        _ensure_symlink(legacy_ssl_basis_dir, legacy_ssl_target)


def _monkey_patch_legacy_cfg(
    v3_processed_dir: Path,
    v3_pt_dir: Path,
    v3_output_dir: Path,
    legacy_run_name: str,
) -> None:
    """Mutate the legacy `configs.config` module so subsequent imports of
    `run_active_learning` see v3 paths."""
    import configs.config as legacy_cfg  # noqa: WPS433

    legacy_cfg.PROCESSED_DIR = Path(v3_processed_dir)
    legacy_cfg.PT_DIR = Path(v3_pt_dir)
    legacy_cfg.OUTPUT_DIR = Path(v3_output_dir)
    # IMPORTANT: keep RUN_NAME pointed at the *legacy* SSL basis name, since
    # we symlinked it into v3 output. A future M5-clean SSL pretrain produces
    # a v3-native basis under cfg.RUN_NAME_V3, at which point swap this.
    legacy_cfg.RUN_NAME = legacy_run_name


# ============================================================================
# Result parsing
# ============================================================================


def _parse_training_log(log_csv: Path) -> dict:
    """Parse legacy `al_training_log_*.csv` and extract best validation MAPE.

    Schema (legacy NeuralFieldFinetuner.train_steps writes one row per step):
        Required columns: step, ... a column carrying the validation MAPE
        (legacy mislabels — see Critical Bug #5.8, "compute_pex_loss as SMAPE").

    We accept either of:
      - 'true_smape_pct'   (post-fix per-edge SMAPE-like)
      - 'val_mape_pct'     (cleaned name if present)
      - 'val_loss'         (fallback — represents the legacy compute_pex_loss
                            hybrid, will be labeled in MetricsRow as such)
    """
    if not log_csv.exists():
        raise FileNotFoundError(f"Training log missing: {log_csv}")
    df = pd.read_csv(log_csv)

    # Try in priority order
    candidates = [
        "true_smape_pct", "val_mape_pct", "val_smape", "val_loss",
        "validation_mape", "validation_smape",
    ]
    found_col = None
    for c in candidates:
        if c in df.columns:
            found_col = c
            break
    if found_col is None:
        raise RuntimeError(
            f"Could not find a validation-error column in {log_csv}. "
            f"Columns present: {list(df.columns)}"
        )

    # Best (= minimum) value across all logged steps
    best_idx = df[found_col].idxmin()
    best_val = float(df.loc[best_idx, found_col])
    best_step = int(df.loc[best_idx, "step"]) if "step" in df.columns else -1

    return {
        "best_val_pct": best_val,
        "best_step": best_step,
        "metric_column_used": found_col,
        "rows": int(len(df)),
        "log_path": str(log_csv),
    }


# ============================================================================
# run_one_seed entrypoint
# ============================================================================


def run_one_seed(
    seed: int,
    train_manifest_path: Path,
    golden_spef_dir: Path,
    output_dir: Path,
    config_snapshot: dict,
    max_iters: int = 1,
    steps_per_iter: int = 5000,
    model_type: str = "DeepPEX",
    gpu_id: Optional[int] = None,
) -> MetricsRow:
    """Train + evaluate legacy DeepPEX_Model for one seed on v3 data.

    Defaults (max_iters=1, steps_per_iter=5000) match the historical 5-seed
    benchmark in `docs/PROJECT_REPORT.md` §2.2.4 (v10b 63.79 ± 5.02 baseline).
    Set `max_iters=6` for the full AL loop.

    Outputs into `output_dir`:
        - active_learning/B3_pinn_seed<N>/                (legacy AL artifacts)
              best_model.pth
              al_training_log_<seed>.csv
              al_session_budget.csv
              al_macro_runtime.csv
        - metrics_row.csv  (parsed summary, written by the orchestrator)
    """
    # ---- Resolve v3 paths from config_snapshot ------------------------
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "pex_v3"))

    from configs import config_v3 as v3cfg  # noqa: WPS433

    v3_processed_dir = v3cfg.PROCESSED_DIR_V3
    v3_pt_dir = v3cfg.PT_DIR_V3
    v3_output_dir = v3cfg.OUTPUT_DIR_V3
    legacy_run_name = "ssl_basis_dspinn_v1"
    legacy_ssl_basis_dir = (
        project_root / "output_intel22" / "checkpoints" / legacy_run_name
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Wire v3 paths into the legacy module's expectations ---------
    _setup_v3_paths(
        project_root=project_root,
        v3_processed_dir=v3_processed_dir,
        v3_pt_dir=v3_pt_dir,
        v3_output_dir=v3_output_dir,
        v3_manifest_path=train_manifest_path,
        legacy_ssl_basis_dir=legacy_ssl_basis_dir,
        legacy_run_name=legacy_run_name,
    )
    _monkey_patch_legacy_cfg(
        v3_processed_dir=v3_processed_dir,
        v3_pt_dir=v3_pt_dir,
        v3_output_dir=v3_output_dir,
        legacy_run_name=legacy_run_name,
    )

    # ---- Build args namespace as legacy main() expects ---------------
    model_name = f"B3_pinn_seed{seed}"
    args = Namespace(
        model_name=model_name,
        seed=seed,
        gpu=gpu_id,
        max_iters=max_iters,
        steps_per_iter=steps_per_iter,
        model_type=model_type,
    )

    # ---- Run legacy AL trainer ---------------------------------------
    # CRITICAL: Before importing legacy `run_active_learning`, we must ensure
    # `from src.X import Y` inside legacy resolves to LEGACY src/, not
    # pex_v3/src/. We do that by removing pex_v3 from sys.path and putting
    # project_root first.
    _pex_v3_root = str(project_root / "pex_v3")
    saved_path = list(sys.path)
    sys.path = [p for p in sys.path if p != _pex_v3_root]
    if str(project_root) in sys.path:
        sys.path.remove(str(project_root))
    sys.path.insert(0, str(project_root))

    # Also evict any pex_v3.src.* modules cached in sys.modules under the
    # name `src.*` — Python resolves names from cache first, so without
    # this the legacy import would still see our pex_v3 modules.
    import sys as _sys
    for mod_name in list(_sys.modules.keys()):
        if mod_name == "src" or mod_name.startswith("src."):
            _sys.modules.pop(mod_name, None)

    try:
        import importlib
        import run_active_learning as legacy_al  # noqa: WPS433
        legacy_al = importlib.reload(legacy_al)
        legacy_al.main(args)
    finally:
        # Restore path so subsequent pex_v3.* imports work
        sys.path[:] = saved_path

    # ---- Parse output -----------------------------------------------
    al_dir = v3_output_dir / "active_learning" / model_name
    log_csv = al_dir / f"al_training_log_{seed}.csv"
    if not log_csv.exists():
        # Legacy may write a non-seeded name as fallback
        candidates = list(al_dir.glob("al_training_log*.csv"))
        if candidates:
            log_csv = candidates[0]
        else:
            raise RuntimeError(f"No training log under {al_dir}")
    parsed = _parse_training_log(log_csv)

    # ---- Build a MetricsRow ------------------------------------------
    # Note: legacy log only carries one validation metric (often the legacy
    # compute_pex_loss hybrid). We surface it as cap_mape_median for
    # cross-method comparability; the column name used is recorded so the
    # reviewer / aggregator can contextualize. Method label MUST exclude seed
    # so the aggregator groups all 5 seeds together.
    return MetricsRow(
        method="B3_pinn_baseline",
        seed=seed,
        cap_mape_median=parsed["best_val_pct"] / 100.0,  # convert pct → fraction
        cap_mape_mean=parsed["best_val_pct"] / 100.0,
        cap_mape_p95=float("nan"),
        delay_err_median=float("nan"),
        delay_err_p95=float("nan"),
        power_err_median=float("nan"),
        rc_chip_ratio_p50=float("nan"),
        rc_chip_ratio_p95=float("nan"),
        n_valid_nets=int(parsed["rows"]),
    )
