# Cross-Boundary Edit — H3 context_margin env var

_Date: 2026-05-01_
_Authorized by user: "재구축을 먼저 해야할거같아" (proceed with rebuild)_

## Why this edit was made

`pex_v3/CLAUDE.md` boundary rule says "do not modify any file outside
`pex_v3/`". H3 rebuild requires changing the hardcoded
`context_margin = 2.0` in `scripts/build_dataset.py:528` to 5.0 μm.

Two paths considered:
1. **Port the entire 648-line `scripts/build_dataset.py` into `pex_v3/`** —
   clean boundary, but ~500 LOC duplication that would be deleted in
   Phase 1 when we redo the data pipeline anyway. High risk of porting bugs.
2. **Single-line cross-boundary edit (env var read)** — minimal surgical
   change that defaults to legacy behavior when env var unset. Boundary
   protocol followed (this doc).

Path #2 chosen. Edit is **fully backward compatible** — any caller that
runs `python3 scripts/build_dataset.py ...` without setting
`PEX_CONTEXT_MARGIN` gets the legacy 2.0 μm value.

## Diff

`scripts/build_dataset.py`:

```diff
 import sys
 import argparse
+import os                                    # added at top
 ...

-    context_margin = 2.0
+    # H3 fix (Strategy v3): allow env var override of legacy 2.0 μm hardcode.
+    # Default preserves legacy behavior; pex_v3/scripts/02_rebuild_dataset_h3.py
+    # sets PEX_CONTEXT_MARGIN=5.0 to capture 4 μm cutoff_radius coupling.
+    # See pex_v3/docs/CROSS_BOUNDARY_h3_context_margin.md.
+    context_margin = float(os.environ.get('PEX_CONTEXT_MARGIN', '2.0'))
     context_size = np.array([cfg.WINDOW_SIZE[0]+2*context_margin, ...])
```

## Backward compatibility

Verified by inspection:
- `os.environ.get('PEX_CONTEXT_MARGIN', '2.0')` returns `'2.0'` when unset.
- `float('2.0')` is `2.0`, identical to the legacy literal.
- No other call site or test depends on this value being a literal.

## Strategy v3 usage

`pex_v3/scripts/02_rebuild_dataset_h3.py` sets
`os.environ['PEX_CONTEXT_MARGIN'] = '5.0'` before subprocess-launching
`scripts/build_dataset.py` for each design. Output goes to
`/data/PINNPEX/data/processed_v3/intel22/<design>/`, never to legacy
data paths.

## When this edit becomes obsolete

Phase 1 will replace cuboid-tile representation with conductor-surface mesh.
At that point, `scripts/build_dataset.py` is no longer the data builder
for v3, and this env var becomes vestigial. The legacy edit can be reverted
or left in place (default still 2.0 μm).

## Discoverability

This file is referenced from:
- the inline code comment at `scripts/build_dataset.py:528`
- `pex_v3/PHASE_STATUS.md`
- `pex_v3/docs/H3_REBUILD_SPEC.md`
