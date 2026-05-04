# pex_v3/src/synthetic — Phase 1 pretraining curriculum

5-stage curriculum that teaches the Phase 1 hybrid analytic + neural
residual model proper electrostatics on geometry it has never seen,
**before** finetuning on real BEOL intel22 data.

Owned by `synthetic-data-pipeline-owner` agent. Activated when Phase 1
architecture is in place.

## 5 stages (Codex round 2 design)

```
Stage 1 — Parallel plate (analytic, instantaneous)
Stage 2 — Layered slab + image charges (analytic, Sommerfeld via rational fitting)
Stage 3 — 3D box pairs (Q3D oracle)
Stage 4 — Multi-conductor 3D fringe with ε_above ≠ ε_below (Q3D oracle)
Stage 4.5 — Real-density multi-conductor motif library (Q3D oracle)
Stage 5 — Real BEOL intel22 (StarRC golden, finetune only)
```

## Files

| Stage | File | Status | Cost |
|---|---|---|---|
| 1 | `stage1_parallel_plate.py` | scaffold | ~minutes (closed-form) |
| 2 | `stage2_layered_image.py` | scaffold | ~hours (rational fitting) |
| 3 | `stage3_box_pairs.py` | TBD | ~700 GPU-hours (Q3D) |
| 4 | `stage4_multi_conductor.py` | TBD | ~1500 GPU-hours (Q3D) |
| 4.5 | `stage4_5_real_density.py` | TBD | ~1500 GPU-hours (Q3D) |
| 5 | (real intel22) | uses cfg.MANIFEST_PATH_V3 directly | covered by Phase 0.5 |
| - | `ground_truth.py` | scaffold | analytic verifier |
| - | `transfer_canary.py` | scaffold | gating script |

## Sample count targets

```
Stage 1:   1 M  (instant, no oracle)
Stage 2:   2 M  (rational fitting required for scale)
Stage 3:   500 K  (Q3D ~5 sec/sample)
Stage 4:   200 K  (Q3D ~30 sec/sample)
Stage 4.5: 50 K   (Q3D ~120 sec/sample, hardest)
Stage 5:   1.3 M tiles from intel22  (already exists; v3 manifest)
```

## Hard kill K3

If pretrain → real-data finetune gain < 1 pp on the transfer canary
(500-1000 net validation slice), abort the synthetic strategy. This
prevents committing GPU-months of Q3D compute to a curriculum that
doesn't transfer.

## Validation discipline

- **Every stage** must reproduce its closed-form / analytic limit case
  to 0.1% before being trusted on novel inputs.
- **Stage 3-4** must cross-validate Q3D vs FastCap on 1000 shared samples;
  discrepancies indicate solver bias, exclude or reweight.
- **Stage 4.5** must spot-check vs in-house FRW (highest fidelity) on 100
  samples; if FRW agrees with Q3D within 1%, accept Q3D as oracle.

See `synthetic-data-pipeline-owner.md` agent definition for the full
operating rules.
