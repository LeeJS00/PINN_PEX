# H4 Pairwise CPL Search — Design

_Status: design — implementation deferred to Phase 1_

## Why H4 is deferred

Legacy `src/models/flux_head.py` selects coupling edges via `closest_dist`:

```
edge (target_net, aggressor_net) exists if
  min over (tgt_cuboid, aggr_cuboid) pairs of  surface_dist(tgt, aggr) < cutoff_r
```

After the edge is created, only the *closest* (target_cuboid, aggressor_cuboid)
pair contributes to the geometry features. A 10 μm-long parallel run at 3 μm
distance is collapsed to a single near-point representation. Long-parallel
coupling (the dominant CPL effect on top metal) is structurally lost.

H4 fix: enumerate ALL `(tgt_cuboid, aggr_cuboid)` pairs whose
surface-to-surface distance ≤ `cutoff_r`, and feed each pair as its own
edge to the CPL head. Edge count grows ~2.25× empirically.

## Why deferred to Phase 1

The legacy `flux_head.py` is the ζ-tuned, DS-PINN-aware module from the
prior 4 failed tracks. Strategy v3 plans to replace it with a hybrid
analytic Green's function + bounded neural residual (Phase 1 architecture).
Retrofitting H4 into the legacy module would:

1. Force a model-output regression that we'd have to re-validate with 5-seed
   protocol (3-4 GPU-day per validation cycle).
2. Touch the legacy `_archive/` interface zone where ζ values, layer-pair
   scales, and stale residuals live — high risk of silent drift.
3. Be undone in Phase 1 anyway.

Decision: **design H4 here as a Phase 1 requirement so the new architecture
inherits the correct edge enumeration from day 1.**

## Phase 1 H4 spec

```
Input: target_net cuboid set T, aggressor_net cuboid set A
       cutoff radius r_c (μm)

Edge set E = { (t, a) : t ∈ T, a ∈ A, surface_dist(t, a) ≤ r_c }
```

Per-edge features (Phase 1 hybrid arch will define the exact list):

- `D_surf` — surface-to-surface distance
- `A_xy` — broadside (xy) overlap area
- `A_lateral` — lateral overlap (max of xz, yz)
- `dz_gap` — vertical gap
- `eps_above`, `eps_below`, `eps_pair` — direction-aware permittivity
- `L_par` — **parallel-run length** (NEW — was lost in `closest_dist`)
- `core_ratio_eff` — effective coupling fraction (legacy carries this)
- physics-base features per Phase 1 arch decision

Implementation notes:
- Use spatial hashing (already in `src/preprocessing/tiling.py:SpatialGrid`)
  to avoid O(|T|·|A|) brute force. Hash both T and A by 2× cutoff bins;
  enumerate only pairs in adjacent bins.
- Edge count empirically grows ~2.25× → bump `MAX_AGGR_BUDGET` to 768
  (already set in `cfg.MAX_AGGR_BUDGET_V3`).
- Memory check: `A_aggr (B, MAX_AGGR_BUDGET, PAD) = (2, 768, 1024)` ≈ 6 MB
  per batch — still cheap.
- Maintain backward-compat for the legacy "single closest pair" flag in
  case Phase 1 ablation needs it: `cfg.H4_EDGE_MODE = "pairwise"` |
  `"closest_dist"`.

## Validation

After Phase 1 architecture is up:

- [ ] Unit test: synthetic 2 wires running parallel for 10 μm at 3 μm
      distance — H4 should emit ~7 edges (one per μm of parallel run via
      cuboid discretization), legacy emits 1
- [ ] Per-edge feature `L_par` distribution sanity (median, P95)
- [ ] Memory usage on `ldpc_decoder` (densest design) stays bounded
- [ ] CPL chip ratio improves vs legacy (target: 1.5 → 1.0-1.2 range)

## Cross-references

- `pex_v3/configs/config_v3.py` — `H4_EDGE_MODE`, `MAX_AGGR_BUDGET_V3`
- `docs/PROJECT_REPORT.md` §4.4 — H4 description (legacy)
- Memory `project_data_pipeline_bottlenecks.md` — H4 catalog
