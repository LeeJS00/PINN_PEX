# Cross-Boundary Edit — legacy `src/__init__.py`

_Date: 2026-05-01_
_Authorized: Strategy v3 Phase B real-data experiments (user directive: proceed with experiments after weakness fixes)_

## Why this edit was made

`pex_v3/src/baselines/pinn_baseline.py` wraps the legacy
`run_active_learning.main()` so the PINN baseline can run on v3 data.

When pinn_baseline imports `run_active_learning`, the module loads torch +
many deep models. Torch's library introspection walks call stack frames
trying to find the source file of every module touched. With legacy
`src/` as a NAMESPACE package (no `__init__.py`), torch raises:

```
TypeError: <module 'src' (<NamespaceLoader>)> is a built-in module
```

at `inspect.getfile(src)` because namespace packages have no single source
file.

The fix: add an empty `__init__.py` to legacy `src/`, making it a REGULAR
package. Behavior unchanged for all existing legacy callers (they were
implicitly relying on namespace lookup, which still works for explicit
packages).

## Diff

```diff
+ src/__init__.py    (empty marker file with header comment)
```

## Backward compatibility

Verified by inspection:
- All legacy entrypoints (`scripts/build_dataset.py`, `run_active_learning.py`,
  `src/trainers/train_ssl.py`) import via `from src.X.Y import Z`. This works
  identically for namespace and regular packages.
- No existing test or CI relies on `src` being a namespace package.
- `pex_v3/src/__init__.py` already exists; converting both to regular
  packages eliminates the asymmetry that caused the namespace-collision
  issues earlier in this session.

## Strategy v3 usage

`pex_v3/src/baselines/pinn_baseline.py` runs through the import chain:
1. Removes pex_v3 from sys.path before importing legacy AL
2. Evicts `src.*` from sys.modules cache to avoid pex_v3 leakage
3. Imports `run_active_learning` (which now finds legacy regular `src/`)
4. Calls `run_active_learning.main(args)` for one seed
5. Restores sys.path to pre-call state

## When this edit becomes obsolete

When Phase 1 model code replaces legacy DeepPEX_Model entirely (Phase 1
ships in `pex_v3/src/models/hybrid_v3.py`), `pinn_baseline.py` may no
longer need to import `run_active_learning`. At that point the legacy
`src/__init__.py` can stay (zero-cost) or be removed.

## Discoverability

This file is referenced from:
- the inline comment in `src/__init__.py` itself
- `pex_v3/PHASE_STATUS.md` (work log)
- `pex_v3/docs/CROSS_BOUNDARY_h3_context_margin.md` (companion)
