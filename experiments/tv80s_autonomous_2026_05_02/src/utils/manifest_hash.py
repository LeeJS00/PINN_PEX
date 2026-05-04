"""
manifest_hash.py — log content hash + git SHA + config snapshot per run.

Every Strategy v3 run dir gets a `provenance.json` containing:

    {
        "run_id":            <UUID>,
        "timestamp":         <ISO 8601>,
        "git_sha":           <commit hash, dirty? flagged>,
        "manifest_path":     <abs path>,
        "manifest_sha256":   <hash of manifest CSV bytes>,
        "config_snapshot":   <dict from config_v3.v3_snapshot()>,
        "seed":              <int>,
        "cuda_env": {
            "torch_version":  <str>,
            "cuda_version":   <str>,
            "device_name":    <str>,
            "device_index":   <int>,
        },
    }

This is the experiment-systems-engineer discipline against the historical
"single-seed BEST is meaningless without manifest fingerprint" failure.
"""
from __future__ import annotations
import hashlib
import json
import os
import socket
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def git_sha(project_root: Path) -> dict:
    """Return dict with git SHA + dirty flag. Falls back gracefully if git unavailable."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        # Detect dirty state
        diff = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=str(project_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        dirty = diff.returncode != 0
        return {"sha": sha, "dirty": bool(dirty)}
    except Exception as e:
        return {"sha": "unknown", "dirty": False, "error": str(e)}


def cuda_env() -> dict:
    """Return torch + CUDA environment summary."""
    if not _HAS_TORCH:
        return {"torch": "not installed"}
    env = {
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        env.update({
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(idx),
            "device_index": int(idx),
        })
    else:
        env["cuda_available"] = False
    return env


def write_provenance(
    run_dir: Path,
    manifest_path: Path,
    config_snapshot: dict,
    seed: int,
    project_root: Optional[Path] = None,
) -> Path:
    """Write provenance.json into `run_dir`. Returns its path.

    Caller must ensure `run_dir` exists.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[3]

    record = {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "user": os.environ.get("USER", "unknown"),
        "git": git_sha(project_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else None,
        "config_snapshot": config_snapshot,
        "seed": int(seed),
        "cuda_env": cuda_env(),
    }
    out_path = run_dir / "provenance.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    return out_path
