"""
test_seed_runner.py — unit tests for 5-seed orchestrator plumbing.

The orchestrator script is exercised end-to-end by the XGBoost smoke test,
but specific failure paths (bad method-spec, missing function, etc.) deserve
fast unit tests so we catch regressions before kicking off a 50-minute run.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path

import pytest


def _load_main_module():
    """Import scripts/05_5seed_runner.py as a module so we can call its helpers."""
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / "pex_v3" / "scripts" / "05_5seed_runner.py"
    spec = importlib.util.spec_from_file_location("seed_runner_under_test", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_load_method_resolves_path_function():
    """`load_method` finds a public function in a Python file by path:name spec."""
    mod = _load_main_module()
    project_root = Path(__file__).resolve().parents[2]
    target_file = project_root / "pex_v3" / "src" / "baselines" / "xgboost_baseline.py"
    fn = mod.load_method(f"{target_file}:run_one_seed")
    assert callable(fn)


def test_load_method_rejects_missing_colon():
    mod = _load_main_module()
    with pytest.raises(SystemExit, match="path:function_name"):
        mod.load_method("just_a_path.py")


def test_load_method_rejects_missing_file():
    mod = _load_main_module()
    with pytest.raises(SystemExit, match="not found"):
        mod.load_method("/nonexistent/path.py:fn")


def test_load_method_rejects_missing_function(tmp_path):
    """If the file exists but the function doesn't, error cleanly."""
    target = tmp_path / "tiny_module.py"
    target.write_text("def other_fn(): pass\n")
    mod = _load_main_module()
    with pytest.raises(SystemExit, match="no function"):
        mod.load_method(f"{target}:does_not_exist")
