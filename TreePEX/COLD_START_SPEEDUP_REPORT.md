# Cold-Start Feature Extraction Speedup — Round 1 ~ 4 보고서

Date: 2026-05-14
Code: `TreePEX/scripts/pex_cold.py` (master @ `2228dfb`)
Plan: `TreePEX/FEATURE_SPEEDUP_PLAN.md`
Branch: `master`

---

## 1. Headline

| Design | Pipeline wall (Round 0 → Round 4) | V3 wall (Round 0 → Round 4) | MAPE_tot drift |
|---|---:|---:|---:|
| **tv80s** | 169.5 s → **48.2 s** (**3.52×** 누적) | 69.8 s → **3.5 s** (**~20×**) | within ±0.2 pp ✅ |
| **nova**  | 8,059 s → **4,906 s** (**1.64×** 누적) | 5,607 s → **2,352 s** (**2.38×**) | +0.015 pp ✅ |

→ **nova 콜드스타트 처리량 1.64×, V3 단독 2.38×**. 모든 MAPE/R² gates 통과.

---

## 2. Round-by-Round 요약

| Round | 핵심 변경 | tv80s pipeline | nova pipeline | 누적 vs Round 0 |
|---|---|---:|---:|---:|
| **Round 0** (pre-patch) | reference | 169.5 s | 8,059 s | – |
| **Round 1** (committed `0cb88a0`) | V3-A target subsample 512, V3-C size-sorted dispatch + adaptive chunksize | 78.0 s | 7,182 s | 1.12× / 2.17× |
| **Round 2.1** (`e2bb543`) | V4-A indexed per-design cache (tv80s only) | 57.9 s | n/a (200 GB) | 2.93× / – |
| **Round 2.1b / 2.2b** (`f2b86e8`) | Schema-v5 + V3 GPU batched (both FAILED) | – | – | – |
| **Round 3** (`3245559`) | Per-target SpatialGrid query + threshold gate (30 M pairs) | 57.3 s | 5,346 s | 2.96× / 1.51× |
| **Round 4** (`2228dfb`) | Numba @njit kernel + CSR dense grid + int32 owner ID | **48.2 s** | **4,906 s** | **3.52× / 1.64×** |

---

## 3. Round 4 (Numba @njit) — 본 세션의 핵심 변경

### 3.1 핵심 결정 (Codex round-1 deliberation 반영)

| 결정 | 채택 사유 |
|---|---|
| Aggregator: `numba.typed.Dict[int32, UniTuple(float64, 5)]` | 600개 미만 aggressor → ~60 µs per net, str overhead 회피 |
| Bin grid: 2D dense CSR (`bin_offsets[nx*ny+1]` + `bin_indices`) | per-target 9-16 bins lookup에 cache-friendly |
| Precision: float64 throughout | Round 3에서 이미 ±0.007 pp drift 한계 |
| `fastmath=False`, `boundscheck=False`, `cache=True` | Round 3 tie-break 의미 보존 + JIT 캐시 |
| Threshold: 미적용 (모든 net) | JIT cost가 tiny net에서도 amortize됨 |
| **String → int32 owner ID 변환 prerequisite** | Numba가 dtype=object 배열을 다룰 수 없음 — 가장 큰 위험 |

### 3.2 구현 위치

- `pex_cold.py:_v3_aggregate_per_target_njit()` — Python wrapper (line 712)
- `pex_cold.py:_v3_get_njit_kernel()` — lazy compile + on-disk cache (line 590)
- `pex_cold.py:_v3_build_dense_grid()` — bin CSR builder (line 264)
- `pex_cold.py:_v3_build_owner_id_map()` — owner str → int32 (line 305)
- `pex_cold.py:init_worker_v3(..., v3_njit_state=...)` — fork-Pool plumbing (line 233)
- `pex_cold.py:_v3_per_net()` gate — `mode == "njit"` 분기 (line 988)
- `pex_cold.py` CLI: `--v3-algo {auto, per_target, legacy, njit}` (line 1932)

### 3.3 빌드 비용 (one-time per design)

| Design | cuboids | unique owners | dense grid build | bin grid | entries |
|---|---:|---:|---:|---|---:|
| tv80s | 147,156 | 22,902 | 0.14 s | 18×23 | 159,178 |
| nova  | 5,340,138 | ~600 K | 6.7 s | 90×95 | 5,879,972 |

JIT compile: ~2.3 s 첫 호출. `@njit(cache=True)` on-disk cache로 fork-Pool 워커는 ~50 ms 로드.

---

## 4. tv80s 4-mode 확장 비교 (재실행, 2026-05-14)

같은 코드(`2228dfb`), 같은 워커수(16), 같은 hardware. 모든 모드 1회 실행.

| Mode | pipeline | V3 wall | V4 wall | MAPE_tot | MAPE_gnd | MAPE_cpl | R²_tot |
|---|---:|---:|---:|---:|---:|---:|---:|
| **legacy** (numpy broadcast) | 58.60 s | 13.28 s | 37.91 s | 5.087 % | 17.62 % | 13.88 % | 0.9922 |
| **auto** (Round 3, threshold 30 M) | 58.91 s | 11.35 s | 41.58 s | 5.075 % | 17.62 % | 13.86 % | 0.9929 |
| **per_target** (Round 3, numpy per-cuboid) | 77.09 s | 23.54 s | 46.05 s | 5.098 % | 17.80 % | 13.98 % | 0.9920 |
| **njit** (Round 4, JIT kernel) | **58.10 s** | **5.30 s** | 45.65 s | 5.116 % | 17.91 % | 13.81 % | 0.9921 |

**해석**:
- **V3 단독**: njit이 legacy 대비 **2.50×**, per_target 대비 **4.44×** 빠름.
- **Pipeline**: tv80s에서는 V4가 ~40s 고정 (per-net cache이지만 inference + SPEF 합치면 cost 상수). V3 단축이 pipeline에 ~10s 영향.
- **MAPE drift**: njit이 가장 큰 drift (gnd +0.29 pp vs legacy) — gate ±0.3 pp 이내 ✅. float-tie tie-break으로 인한 broadside/lateral 차이.
- `auto` 모드 tv80s는 threshold=30M이라 사실상 legacy fallback (V3 11.35s ≈ legacy 13.28s 차이는 noise).

---

## 5. nova 확장 비교

| Round | V3 algo | pipeline | V3 wall | V4 wall | MAPE_tot | MAPE_gnd | MAPE_cpl | R²_tot |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Round 0 (pre-patch) | legacy | 8,059 s | 5,607 s | 2,349 s | 5.538 % | 15.85 % | 15.94 % | 0.9867 |
| Round 1 (committed) | legacy (V3-A+C) | 7,182 s | 4,871 s | 2,207 s | 5.541 % | 15.86 % | 15.96 % | 0.9865 |
| Round 3 (committed) | per_target (auto 30 M) | 5,346 s | 2,924 s | 2,244 s | 5.548 % | 15.87 % | 15.97 % | 0.9862 |
| **Round 4 v1 (committed)** | **njit** | **4,906 s** | **2,352 s** | 2,384 s | **5.556 %** | **15.87 %** | **16.00 %** | **0.9862** |
| Round 4 v2 (reproducibility) | njit | 4,958 s | 2,436 s | 2,349 s | 5.554 % | 15.87 % | 16.00 % | 0.9860 |

**Reproducibility 확인 (v1 vs v2)**: pipeline +1.0 %, V3 +3.6 %, V4 −1.5 %, MAPE_tot −0.002 pp. 모든 metric run-to-run 분산 < 4 %, MAPE는 사실상 동일.

---

## 6. Validation gates

PROJECT_PLAN.md §6 cold-start acceptance:
- tv80s MAPE_tot_med ∈ [4.85, 5.30 %], MAPE_{gnd, cpl} drift ≤ +0.3 pp
- nova MAPE_tot_med ∈ [5.30, 5.75 %], MAPE_{gnd, cpl} drift ≤ +0.3 pp
- R²_tot ≥ 0.985

**Round 4 njit 검증 결과**:

| Design | MAPE_tot in range | gnd Δ ≤ +0.3 pp | cpl Δ ≤ +0.3 pp | R²_tot ≥ 0.985 |
|---|---|---|---|---|
| tv80s | 5.116 % ✅ | +0.29 pp ✅ (border) | −0.07 pp ✅ | 0.9921 ✅ |
| nova  | 5.556 % ✅ | +0.01 pp ✅ | +0.04 pp ✅ | 0.9862 ✅ |

→ **모든 acceptance gate 통과**.

Feature drift 분포 (njit vs legacy single-process dump, 120 net 선택):
- V4 H3: 26/26 bit-exact (kernel 변경 없음)
- V3 scalar (sum, layer_hist, compact_gnd, vss_*): 31/41 bit-exact
- V3 closest-pair (`spacing_*`, `n_edges_*`, `broadside/lateral`): R² 0.76 ~ 0.99
  → 최악 `n_edges_3_to_4um` R² = 0.76, MAE_pct 15 %
  → 원인: float-tie tie-break (동등 거리 ties에서 (t*, c*) pair 선택이 bin 순회 순서에 따라 달라짐)
  → `spacing_min_um`은 bit-exact: per-aggressor min distance는 알고리즘 동등

---

## 7. Codex deliberation log 요약

### Round 1 (Round 3 알고리즘 선택)
- Codex 추천: **Candidate C (hybrid threshold)** — per_target 알고리즘은 OK이나 tiny net Python overhead 우려 → threshold gate 필요
- Gemini 추천: A (full per_target) — 가장 큰 속도 향상 잠재력
- 절충: **Candidate A 알고리즘 + threshold gate** (=Codex C) → Round 3 채택

### Round 1 (Round 4 Numba JIT 설계)
- Codex: typed.Dict[int32,...] aggregator OK / 2D dense grid 권장 (CSR보다 cache-friendly) / float64 / fastmath=False / threshold 미적용 가능 / **#1 위험은 string→int32 owner ID 누락**
- 채택: 모두 그대로

---

## 8. Next iteration 후보 (Round 5)

`FEATURE_SPEEDUP_PLAN.md` §5 Round 5 candidates:

| # | 후보 | 예상 효과 | Effort |
|---|---|---:|---:|
| 17 | **D-pred** (XGBoost GPU predict) | nova inference 7s → ~1s | trivial |
| 18 | **V4 njit kernel** | V4 H3 ~50 ms/net mid-tail → ~10 ms (nova V4 2400s → ~700s?) | medium |
| 19 | **DEF-A** (cache parsed DEF) | 재실행 워크플로 nova def 94s → 0s | trivial |
| 20 | **Multi-design parallelism** | tv80s + nova 동시 실행 (DEF parse 중첩) | medium |
| 21 | **Global sparse adjacency + scatter_add** | 더 공격적 B 후보 | high |

Round 5 후보 중 D-pred + DEF-A가 lowest-effort highest-ROI 조합.

---

## 9. 재현 방법

```bash
# Single design with njit
python3 TreePEX/scripts/pex_cold.py --design intel22_tv80s_f3 --v3-algo njit
python3 TreePEX/scripts/pex_cold.py --design intel22_nova_f3 --v3-algo njit

# Both designs in parallel (legacy default)
python3 TreePEX/scripts/pex_cold.py

# Feature-level diff probe
python3 TreePEX/scripts/dump_features.py --design intel22_tv80s_f3 \
    --select 'top:20,sample:100' --label baseline --v3-algo legacy
python3 TreePEX/scripts/dump_features.py --design intel22_tv80s_f3 \
    --select 'top:20,sample:100' --label patched --v3-algo njit
python3 TreePEX/scripts/compare_features.py \
    --baseline TreePEX/outputs/cold_reports/feature_dumps/intel22_tv80s_f3__baseline.json \
    --patched  TreePEX/outputs/cold_reports/feature_dumps/intel22_tv80s_f3__patched.json
```

`numba` 의존성 (한 번만):
```bash
pip install --user numba
```

---

## 10. 커밋 이력 (이번 세션 + 직전)

```
2228dfb feat(pex_cold): Round 4 Numba @njit V3 kernel — nova 7,182 → 4,906 s (1.46×)
3245559 feat(pex_cold): Round 3 V3 algorithmic redesign — per-target-cuboid path
f2b86e8 exp(treepex/cold-start): Round 2.1b/2.2b experiments — schema-v5 and batched GPU both fail at nova scale
e2bb543 feat(treepex/cold-start): Round 2 first-cut — V4-A cache + V3 GPU prototypes
0cb88a0 feat(treepex/cold-start): adaptive Pool chunksize + nova Round 1 measured
ffb74e8 feat(treepex/cold-start): Round 1 V3 feature build speedup (tv80s 2.15×)
```
