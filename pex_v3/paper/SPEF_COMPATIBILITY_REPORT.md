# StarRC SPEF Compatibility Verification

_Golden_: `/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22/intel22_tv80s_f3_starrc.spef`
_Pred_:   `/tmp/intel22_tv80s_f3_HERO_v2.spef`

## Verdict: ⚠ partial

| Check | Result |
|---|---:|
| headers_compatible | ❌ |
| net_coverage_complete | ✅ |
| structure_present | ❌ |
| node_types_present | ❌ |
| cap_consistency | ✅ |

## Header fields

Matching: 9 fields

- ✅ `bus_delimiter`: `*BUS_DELIMITER []`
- ✅ `c_unit`: `*C_UNIT 1.0 FF`
- ✅ `delimiter`: `*DELIMITER :`
- ✅ `design_flow`: `*DESIGN_FLOW "PIN_CAP NONE" "NAME_SCOPE LOCAL"`
- ✅ `divider`: `*DIVIDER /`
- ✅ `l_unit`: `*L_UNIT 1.0 HENRY`
- ✅ `r_unit`: `*R_UNIT 1.0 OHM`
- ✅ `spef_version`: `*SPEF "IEEE 1481-1999"`
- ✅ `t_unit`: `*T_UNIT 1.0 NS`
- ℹ `date`: golden=`*DATE "Sun Apr 12 00:27:40 2026"` pred=`*DATE "Sun May 03 08:53:42 2026"`
- ❌ `design`: golden=`*DESIGN "tv80s"` pred=`*DESIGN "intel22_tv80s_f3"`
- ℹ `program`: golden=`*PROGRAM "StarRC"` pred=`*PROGRAM "NeuralField_V3"`
- ℹ `vendor`: golden=`*VENDOR "Synopsys Inc."` pred=`*VENDOR "DeepPEX AI"`
- ℹ `version`: golden=`*VERSION "S-2021.06-SP2"` pred=`*VERSION "V3.0"`

## Net coverage

- golden: 3,380 nets
- pred:   3,380 nets
- common: 3,380 (100.00%)
- golden-only: 0 (sample: —)
- pred-only:   0 (sample: —)

## Per-net structure (common 3,380 nets)

| Block | Golden | Pred |
|---|---:|---:|
| `*CONN` | 3,380 | 3,380 |
| `*CAP`  | 3,380 | 3,280 |
| `*RES`  | 3,380 | 3,380 |

## Node types (in *CONN)

| Type | Golden | Pred |
|---|---:|---:|
| `*P` (ports) | 46 | 0 |
| `*I` (instance pins) | 3,380 | 3,380 |
| `*N` (internal nodes) | 3,380 | 3,380 |

## *D_NET ↔ Σ *CAP consistency

| | median | max |
|---|---:|---:|
| golden | 0.0001% | 0.0006% |
| pred   | 0.0000% | 0.0006% |

## Resistance topology

| | golden median | golden mean | golden max | pred median | pred mean | pred max |
|---|---:|---:|---:|---:|---:|---:|
| *RES segments | 27 | 43.7 | 1144 | 24 | 37.3 | 912 |
| *CAP entries  | 82 | 205.2 | 4334 | 164 | 228.6 | 3502 |
| coupling edges | 77 | 195.3 | 4147 | 156 | 218.1 | 3300 |
