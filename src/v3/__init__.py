"""src.v3 — re-export of validated pex_v3 modules for main-folder integration.

This sub-package symlinks `pex_v3/src/{models,baselines,data,trainers,utils}`
so they can be imported as `src.v3.models.*`, etc.

The pex_v3 source code uses `from src.X import ...` (absolute imports) which
were resolved when `pex_v3/` was prepended to sys.path by pex_v3 scripts.
To make those imports work when invoked via `src.v3.*` from the main folder,
we prepend pex_v3 to sys.path on first import below.

Boundary safety: `pex_v3/src/{preprocessing,evaluation}` are essentially
empty sub-packages (just __init__.py), so adding pex_v3 to sys.path does
not shadow main `src.preprocessing.*` or `src.evaluation.*` imports
elsewhere in the same Python session.

Documented in: pex_v3/docs/CROSS_BOUNDARY_v3_merge_to_main.md
"""
from __future__ import annotations
import sys
from pathlib import Path

_PEX_V3 = str(Path(__file__).resolve().parent.parent.parent / "pex_v3")
if _PEX_V3 not in sys.path:
    sys.path.insert(0, _PEX_V3)
