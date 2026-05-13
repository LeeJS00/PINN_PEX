# Feature-extraction speed-up plan

> Next task. Cold-start TOTAL time is >99 % feature extraction for tree
> models (tv80s 161 / 169 s ≈ 95 %, nova 7955 / 8059 s ≈ 99 %).
> Cutting V3 + V4 feature build time is the only meaningful end-to-end
> speed-up.

Date opened: 2026-05-13
Last revised: 2026-05-13 (post Codex+Gemini round 2)
Status: planning
Owner: TBD
Parent doc: `TreePEX/COLD_START_REPORT.md`

---

## 1. Goal

* Cut per-design **shared feature build wall time** (PDK + DEF + V3 + V4)
  by **≥ 5 ×** on nova while keeping cold-start MAPE within +0.2 pp on
  `tot` AND +0.3 pp on per-channel `cpl` of the current numbers (TreePEX
  nova 5.54 % / tv80s 5.10 % tot; gnd / cpl bounds in §6).
* Target wall budgets (16 worker fork-Pool, gpu-8):
  * tv80s: 162 s → **≤ 30 s** (Round 1), **≤ 8 s** (Round 2 GPU)
  * nova:  8,050 s → **≤ 1,800 s** (Round 1, ~25 min), **≤ 200 s** (Round 2 GPU)

## 2. Current bottleneck breakdown (from `COLD_START_REPORT.md` §3.2)

| Stage | tv80s (s) | nova (s) | Notes |
|---|---:|---:|---|
| PDK parse | 0.77 | 0.39 | one-time; negligible. **No work needed.** |
| DEF parse | 3.96 | 93.67 | streaming parser, single-thread. **Maybe.** |
| V3 features (41-D) | **69.79** | **5,607.13** | **bottleneck (nova).** |
| V4 H3 features (26-D, tile cache) | 87.65 | 2,348.97 | **2nd bottleneck.** |
| Inference + SPEF | 1.8 – 187 | 1.0 – 2,277 | model-specific, not in this scope. |

## 3. Root-cause analysis

### 3.1 V3 features (nova: 93 min)

`_v3_per_net(net_name)` (in `pex_cold.py:251`) for each of 118,959 nets:

1. SpatialGrid query → candidate aggressor cuboid set (size *N_c*).
2. Numpy broadcast distance matrix: `(N_t × N_c)` for target with *N_t* cuboids
   (lines 294-303).
3. Per-aggressor closest-pair aggregation (Python dict loop, lines 326-349).
4. `compact_gnd_estimate_fF` — Python `for i in range(n)` loop over the
   full target_arr (lines 436-444), independent of subsampling.
5. Edge stats / VSS shield / compact priors.

**Pathology**: long-tail nets where *N_t* and *N_c* are both large (clock
spines, reset nets, top-level signals in nova). Memory and CPU scale
O(N_t · N_c). A 2k-cuboid clock spine with 20k candidates → 40 M-pair
float matrix per net ≈ 320 MB transient + 40 M sqrt/cmp ops. With 16
workers, peak host memory can spike to ~5 GB. nova has dozens of such nets
and chunksize=64 (line 685) serializes them onto individual workers.

### 3.2 V4 H3 features (nova: 39 min)

`_v4_process_net((net_name, tile_paths))` (line 579):

1. For each of *T_n* tile pkl.gz files for this net, gzip-decompress + unpickle.
2. Concatenate `target_chunks` + `agg_groups`.
3. Compute pairwise top-K stats (full broadcast, lines 526-541).

**Pathology**: nova has 446 K tiles total (mean ~4 tiles / net), stored as
individual `.pkl.gz` files in a flat per-design directory
(`/data/PINNPEX/data/processed_v3/intel22/<design>/*.pkl.gz`). Each pkl.gz
is ~40-100 KB. Even at 10 ms per tile load that's 75 min wall on 1 worker.
At 16 workers it's I/O- and decompress-bound on local NVMe.

> **Plan-vs-disk note**: an earlier draft assumed
> `per_net_cuboids/*.npz` already exists "for mesh-PINN" — disk reality
> shows only `<design>_map.csv` and `<design>_net_mapping.csv` at the
> design level. Any per-net or per-design pre-aggregation is a **new**
> asset to build, not a reuse.

## 4. Proposed approaches (ranked by expected impact / engineering cost)

### 4.1 V3 features

| # | Approach | Expected gain (nova V3) | Effort | Risk |
|---|---|---:|---:|---|
| V3-A | **Target-cuboid sub-sampling for the broadcast only** at *N_t* = 256 (mirrors V4's `MAX_TARGET_CUBS_V4`). **Sum / scalar features (`n_cuboids`, `total_wire_length_um`, `total_metal_area_um2`, bbox stats, `layer_hist`, `density_*`, `compact_gnd_estimate_fF`) MUST be computed on the full target_arr.** Only the `(N_t × N_c)` distance broadcast (pex_cold.py:294-303) and the closest-pair dict loop see the subsampled rows. Subsample with a deterministic per-net seed: `np.random.RandomState(hash(net_name) & 0xFFFFFFFF)`. | **5 – 10 ×** | low | small accuracy drift if very large nets dominate; verify on training designs first. |
| V3-A' | **Vectorize `compact_gnd_estimate_fF`** (pex_cold.py:436-444): replace Python loop with NumPy. Free win, independent of V3-A. | 1.1 – 1.3 × additive | trivial | none — pure refactor, identical output. |
| V3-B | **Aggressor cap per net** *N_c* ≤ 4096 before broadcast, ranking candidates by **bbox-edge distance to target bbox** (NOT centroid: elongated nets such as clock spines have periphery aggressors that centroid ranking would miss). Use `max(0, |cx − tcx| − (cw + tbw)/2)`-style metric per axis. | 2 – 4 × | low | over-cap may drop weak couplings, slightly bias `n_aggressor_nets`. |
| V3-C | **`chunksize=1` for `imap_unordered`** + dispatch-by-size sort (largest nets first) in BOTH `extract_v3_features` (pex_cold.py:685) and `extract_v4_h3_from_tile_cache` (line 715). Eliminates long-tail straggler. | 1.2 – 1.5 × additive | trivial | none. |
| V3-D | **Move broadcast to numba/cython** or use `scipy.spatial.cKDTree.query_ball_tree` for cutoff-bounded pair enumeration. **Caveat**: cKDTree is point-based; the V3 metric is bbox-edge with `w/2, h/2` dilation. Need bounding-radius dilation by `max_w/2, max_h/2`. Plan-original "medium" effort is understated. | 3 – 5 × | medium-high | introduces dep; verify correctness on canonical net. |
| V3-E | **Re-use V4 H3 candidate set** (already computed downstream) instead of running V3 SpatialGrid query separately. Both stages query the same neighborhood. | 1.5 × | medium | pipeline restructuring; absorbed into §4.3 G if Round 2 lands. |
| V3-F | **C++ extension** for `_enumerate_coupling_edges`. | 10 × | high | last-resort. |

### 4.2 V4 H3 features

| # | Approach | Expected gain (nova V4) | Effort | Risk |
|---|---|---:|---:|---|
| V4-A | **Pre-aggregate to per-net npz** (one file per net) so V4 inference becomes one numpy load + per-net loop. Eliminates 446 K gzip+pickle calls. **Schema is new** — existing mesh-PINN per-net assets (if any) contain only `target_cubs[N,10]`; V4 needs `(target_cubs, dict[agg_name → agg_cubs])`. | **10 – 20 ×** | low-medium | new schema; absorbed into §4.3 G if Round 2 lands. |
| V4-B | **mmap'd tile cache** (uncompressed npy) instead of gzip pkl. | 3 – 5 × | medium | requires rebuilding cache; doubles disk usage. |
| V4-C | **Lazy / streaming aggregation** in `pex_cold.py` itself: read tile pkl.gz once, emit V3 + V4 simultaneously. | 1.5 – 2 × | medium | non-trivial refactor; needs tile-cache during V3. |
| V4-D | **Drop V4 H3** entirely for cold-start, retrain TreePEX on 41-D only. | infinite (V4 = 0) | high | model retrain + 5-seed; expected accuracy hit +0.5-1 pp based on B1 result. |

### 4.3 Tensor + GPU axis (added 2026-05-13)

Environment: 8× NVIDIA RTX A6000 (48 GB each, idle), torch 2.4.0+cu121,
xgboost 3.2.0 (supports `device='cuda'` predict). Current CSV/pkl.gz
storage and numpy per-net broadcast leave ~400 GB of GPU RAM unused.

| # | Approach | Expected gain (nova V3+V4 combined) | Effort | Risk |
|---|---|---:|---:|---|
| G | **Per-design tensor asset** (`<design>_cuboids.pt`): single `torch.save` containing `all_cuboids[N,10]` fp32, `owner_id[N]` int32 (integer not string — string is RAM + serialization hostile), and a CSR-style `(net_offsets, net_ids)`. nova size ~ 376 MB cuboid tensor + 38 MB owner + < 100 MB index ≈ **< 1 GB per design vs. 446 K pkl.gz files**. Load with `torch.load(..., mmap=True)` for zero-copy lazy access. Replaces V4-A (and V3-E folds in for free: V3 reads the same asset). | rebuild-once asset; runtime gain comes from B | medium (one build_dataset extension) | new schema; one-time disk rebuild ~tens of minutes per design. |
| B | **Batched torch+GPU broadcast** for V3 (pex_cold.py:294-303) and V4 (pex_cold.py:526-541). Per-net GPU calls have prohibitive PCIe overhead → batch by **pair budget** (e.g. ~500 M pairs ≈ 8-12 GB VRAM with fp32 × 4-6 tensors). Long-tail large nets routed to GPU; small nets stay on CPU (hybrid path, first-cut). Subsequent escalation: global sparse adjacency + `torch.scatter_add` per-net on a single A6000. | **20 – 80 ×** on V3 + V4 combined | medium-high | requires asset G; PCIe overhead; CUDA-fork incompat (see C below). |
| C-arch | **fork-Pool + CUDA are incompatible** (pex_cold.py:1039, `mp.set_start_method("fork")` breaks CUDA context in workers). Three options:<br/>(a) `spawn` — slow init, loses copy-on-write of geo dict.<br/>(b) Single-process, multi-GPU shard over the 8× A6000 via `torch.cuda.set_device`.<br/>(c) Hybrid — keep fork-Pool for small nets on CPU; route long-tail nets to one (or a few) single-GPU subprocesses started with spawn. | enables B | medium | risk of host-RAM duplication when spawn is used. |
| D-pred | **XGBoost GPU predict** (`device='cuda'` on the 5-seed ensemble). nova current inference 2.7 s → ~0.5 s. **Deferred — not bottleneck.** | < 5 s wall | trivial | none. |
| F | **`torch.compile` / Triton kernel** for the bbox-edge distance metric + scatter aggregation. Only revisit if naive torch in B leaves > 50 % headroom unexplored. | 2 – 3 × on top of B | high | premature; profile first. |

### 4.4 DEF parse (nova: 94 s)

Not in top-2 bottleneck but still meaningful for tv80s end-to-end:

| # | Approach | Expected gain | Effort |
|---|---|---:|---:|
| DEF-A | Cache parsed DEF as pkl per-design after first read; skip re-parse on subsequent feature regeneration. | 50 × for warm reruns | trivial |
| DEF-B | Switch to a multi-threaded LEF/DEF parser (e.g., `lefdef` C++ binding). | 5 × | high |

## 5. Recommended sequence

### Round 0 — profiling gate (DONE 2026-05-13)

Script: `TreePEX/scripts/profile_single_net.py`. Outputs at
`TreePEX/outputs/cold_reports/profile_intel22_{tv80s,nova}_f3.json`.

Findings (single-process, no Pool):

| Design | Net | N_t | N_c | wall (s) | broadcast share | pair MB |
|---|---|---:|---:|---:|---:|---:|
| tv80s | CTS_2 | 1,110 | 52,662 | 3.17 | 99 % | 446 |
| tv80s | CTS_5 | 943 | 67,181 | 3.74 | 99 % | 483 |
| nova  | CTS_330 | 1,257 | 119,302 | 10.69 | 100 % | **1,144** |
| nova  | CTS_326 | 1,234 | 113,245 | 8.67 | 99 % | 1,066 |

Decisions locked from Round 0 data:
* **Broadcast dominates absolutely (99-100 %)** — every other V3 stage
  (scalar, grid query, owner filter, dict aggregation) is < 0.1 % combined.
* **V3-A alone is insufficient**: capping N_t to 256 yields ~4-5×
  reduction; N_c is the long-tail driver (nova reaches 119k candidates).
  V3-A + V3-B combined target a ~55-145× pair-count reduction.
* **V3-A' (compact_gnd vectorization) is dead code**: every profiled net
  shows `t_compact_gnd_loop < 1 ms`. Drop from Round 1.
* **V4 tile load + broadcast are co-equal** (50/50 split, concat < 2 %).
  Tile-load reduction (V4-A → Round 2 §4.3 G) and broadcast acceleration
  must both move for V4 to drop substantially.

### Round 1 — CPU/numpy speedup (LOCKED 2026-05-13 on tv80s)

**Final patch contents**:
1. **V3-A** (`MAX_TARGET_CUBS_V3 = 512`) — broadcast-only target-cuboid
   sub-sampling, deterministic per-net seed = `hash(net_name) & 0xFFFFFFFF`.
   Sum / scalar features (n_cuboids, total_wire_length, bbox, layer_hist,
   density_*, eps_*, compact_gnd) stay on full `target_arr`.
2. **V3-C** (size-sorted dispatch + `chunksize=1`) — applied to both
   `extract_v3_features` and `extract_v4_h3_from_tile_cache` Pool loops.

DROPPED (post-validation):
* **V3-A'** (compact_gnd vectorize) — Round 0 measured < 1 ms loop on all
  nets; dead code.
* **V3-B** (aggressor cand cap 4096) — first iteration measured
  `n_aggressor_nets` / `fanout` (cpl XGBoost feature_importance 0.81)
  collapsing to R² = **−5.72**, MAE 138 / mean 733. Count-based candidate
  cap drops entire aggressor net identities at the tail.
* **V3-D'** (cKDTree pre-filter + vectorized refinement) — implemented
  and benchmarked; **made V3 6.5× SLOWER** (45.8 s → 295.5 s on tv80s
  120-net dump). Root cause: per-net `sparse_distance_matrix` returns
  millions of centroid-within-(CUTOFF + max_half_diag)-pairs because of
  large-cuboid dilation, and the lexsort tail dominates the saved FLOPs.
  Saved in plan as a Round-3 footnote for a per-pair-size-bucketed retry.

**Measured outcome (tv80s, dev/validate cycle complete)**:

| Metric | Baseline (plan §2) | Patched | Δ |
|---|---:|---:|---:|
| pipeline wall | 169.47 s | **78.95 s** | **2.15 ×** |
| V3 features | 69.79 s | 16.55 s | 4.22 × |
| V4 H3 features | 87.65 s | 56.27 s | 1.56 × |
| MAPE_tot | 5.105 % | 5.107 % | +0.002 pp ✅ |
| MAPE_gnd | 17.63 % | 17.63 % | 0.00 pp ✅ |
| MAPE_cpl | 13.88 % | 13.88 % | 0.00 pp ✅ |
| R²_tot | 0.992 | 0.992 | = ✅ |

**Per-feature drift** (top:20 + sample:100, deterministic seed 2026):
27/41 V3 features bit-exact (MAE = 0); remaining 14 stochastic features
all R² ≥ 0.977 (most ≥ 0.99). All 26 V4 features bit-exact (kernel
unchanged). Diff report:
`TreePEX/outputs/cold_reports/diff_intel22_tv80s_f3_v3a512.md`.

**Key finding — V3-C is the real hero on tv80s**: with V3-A disabled,
pipeline still drops to 77.8 s — virtually all of the 2.15× gain comes
from size-sorted dispatch + `chunksize=1` killing the straggler tail
across 16 Pool workers. V3-A=512 only meaningfully clips the top ~20
nets and contributes < 1 % to the wall reduction here. **V3-A's value
will only materialize on nova**, where N_t reaches 1257 (vs tv80s 1110)
and a 2-3× per-tail-net reduction matters across more long-tail nets.

**Promotion to nova (next)**:
* Re-run `pex_cold.py --design intel22_nova_f3 --workers 16` with the
  Round 1 patch. Expected: ~2700-3000 s (vs baseline 8059 s = 2.7-3.0 ×).
  Does NOT hit the §1 nova ≤ 1,800 s gate — Round 2 GPU is required for
  that.
* Same MAPE gates: tot ∈ [5.30, 5.75] %, gnd / cpl within ±0.3 pp of
  current.
* If nova MAPE checks out, Round 1 commits and we move to Round 2.

### Round 2 — Tensor asset + GPU broadcast (target: nova ≤ 200 s, tv80s ≤ 8 s)

5. **G** (per-design `<design>_cuboids.pt` asset built by an extension
   of `scripts/build_dataset_multi.py`). Replaces V4-A entirely and
   absorbs V3-E (V3 reads the same asset). One-time disk rebuild per
   design; net-name → integer id mapping persisted.
6. **B** (torch + GPU broadcast for V3 and V4). Hybrid CPU+GPU first-cut
   (small nets stay on CPU multiproc, long-tail routed to a single GPU
   subprocess started with spawn). Pair-budget batching ~500 M pairs per
   GPU call. Escalate to single-process multi-GPU shard if hybrid hits
   PCIe ceiling.
7. **C-arch decision** during Round 2 PR — choose hybrid (a)+(c) vs pure
   single-process (b) based on profiling.

Expected end-to-end: nova 30-200 s, tv80s 5-15 s.

### Round 3 — optional ceiling pushes (only if Round 2 leaves headroom)

8. **D-pred** (XGBoost GPU predict) — quick free win.
9. **F** (torch.compile / Triton custom kernel) — only if naive torch in
   Round 2 leaves > 50 % headroom.
10. Global sparse adjacency + `scatter_add` (the more aggressive
    interpretation of B). Replaces per-net GPU dispatch with one-shot
    design-level kernel.
11. **DEF-A** (cache parsed DEF) for re-run workflow.

## 6. Acceptance criteria

For Round 1 to be considered done:

* `pex_cold.py` produces the same 67-D feature parquet shape on tv80s + nova.
* Re-running `pex_cold_predict.py --model treepex` gives:
  * tv80s MAPE_tot_med ∈ [4.85, 5.30] % (current 5.105)
  * nova  MAPE_tot_med ∈ [5.30, 5.75] % (current 5.538)
  * tv80s **MAPE_cpl_med within +0.3 pp** of current; same for nova
  * tv80s **MAPE_gnd_med within +0.3 pp** of current; same for nova
  * R²_tot ≥ 0.985 on both; R²_cpl ≥ current −0.005 on both
* Wall time (16 worker fork-Pool, gpu-8) on idle host:
  * tv80s ≤ 30 s shared feature build
  * nova ≤ 1,800 s shared feature build
* `summarize_cold_results.py` output committed alongside the change.
* Profiling note from Round 0 attached.

For Round 2 to be considered done:

* Same MAPE bounds as Round 1.
* `<design>_cuboids.pt` asset committed to the dataset-build pipeline;
  build time + disk size documented per design.
* Wall time on a single A6000 (no other GPU process):
  * tv80s ≤ 8 s shared feature build
  * nova ≤ 200 s shared feature build
* End-to-end determinism preserved (per-net stable seed, no
  non-deterministic CUDA ops in the broadcast path).

## 7. Out of scope

* Mesh-PINN inference speed-up (separate task: port to GPU).
* SPEF write speed-up (already < 4 s on nova).
* Multi-design parallel run optimization: cold tv80s + nova ran in
  parallel previously (3.5 h wall vs serial 3.7 h) — diminishing returns
  on a 64-core host already saturated.
* DEF parser C++ binding (DEF-B): low ROI vs Round 2 ceiling.

> Tile-cache build cost (`build_dataset.py`) is **now in scope** as
> Round 2 G is essentially a tile-cache rebuild with a new schema.

## 8. Reference numbers (carry into Round 1 / Round 2 PR descriptions)

| | Current cold-start | Target (Round 1) | Target (Round 2) |
|---|---:|---:|---:|
| tv80s pipeline wall (TreePEX, treepex model) | 169.47 s | ≤ 30 s | ≤ 8 s |
| nova  pipeline wall (TreePEX, treepex model) | 8,059.16 s | ≤ 1,800 s | ≤ 200 s |
| tv80s MAPE_tot | 5.105 % | within ±0.2 pp | within ±0.2 pp |
| nova  MAPE_tot | 5.538 % | within ±0.2 pp | within ±0.2 pp |
| tv80s MAPE_cpl | (current value) | ≤ +0.3 pp | ≤ +0.3 pp |
| nova  MAPE_cpl | (current value) | ≤ +0.3 pp | ≤ +0.3 pp |

Cold-start full-tab data: `TreePEX/outputs/cold_reports/cold_summary.json` and
per-(design, model) JSONs in the same directory.
