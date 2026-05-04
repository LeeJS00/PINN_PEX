# Stratified eval — variant `c1a_modeB` (split=test)

Baseline: n=95,594 nets, gnd median 19.894%, cpl median 15.153%, total median 7.889%.

## Per-design

| design | n_nets | gnd_mape_median | gnd_mape_mean | cpl_mape_median | cpl_mape_mean | total_mape_median | total_mape_mean |
|---|---|---|---|---|---|---|---|
| intel22_nova_f3 | 92425 | 0.1997 | 0.2373 | 0.1519 | 0.1994 | 0.0788 | 0.0936 |
| intel22_tv80s_f3 | 3169 | 0.1770 | 0.2187 | 0.1437 | 0.1713 | 0.0823 | 0.0929 |

## Per-quartile (axis=compact_gnd_estimate_fF)

| quartile | axis | axis_min | axis_max | n_nets | gnd_mape_median | gnd_mape_mean | cpl_mape_median | cpl_mape_mean | total_mape_median |
|---|---|---|---|---|---|---|---|---|---|
| Q1 | compact_gnd_estimate_fF | 0.0287 | 0.0767 | 23944 | 0.2237 | 0.2574 | 0.1832 | 0.2364 | 0.1090 |
| Q2 | compact_gnd_estimate_fF | 0.0767 | 0.1409 | 23853 | 0.2365 | 0.2667 | 0.1731 | 0.2370 | 0.0872 |
| Q3 | compact_gnd_estimate_fF | 0.1409 | 0.4596 | 23898 | 0.1853 | 0.2212 | 0.1432 | 0.1747 | 0.0740 |
| Q4 | compact_gnd_estimate_fF | 0.4596 | 52.4053 | 23899 | 0.1545 | 0.2015 | 0.1180 | 0.1460 | 0.0564 |

## Per-quartile (axis=gnd_rel_err — Mode B giant-CTS surface)

| quartile | axis | axis_min | axis_max | n_nets | gnd_mape_median | gnd_mape_mean | cpl_mape_median | cpl_mape_mean | total_mape_median |
|---|---|---|---|---|---|---|---|---|---|
| Q1 | gnd_rel_err | 0.0000 | 0.0956 | 23899 | 0.0469 | 0.0473 | 0.1060 | 0.1295 | 0.0624 |
| Q2 | gnd_rel_err | 0.0956 | 0.1989 | 23898 | 0.1461 | 0.1464 | 0.1254 | 0.1540 | 0.0678 |
| Q3 | gnd_rel_err | 0.1989 | 0.3279 | 23898 | 0.2585 | 0.2602 | 0.1611 | 0.2079 | 0.0816 |
| Q4 | gnd_rel_err | 0.3279 | 8.2162 | 23899 | 0.4256 | 0.4929 | 0.2369 | 0.3026 | 0.1124 |

## Per-fanout bucket

| fanout_bucket | n_nets | gnd_mape_median | gnd_mape_mean | cpl_mape_median | cpl_mape_mean | total_mape_median |
|---|---|---|---|---|---|---|
| 1 | 44019 | 0.2332 | 0.2649 | 0.1796 | 0.2392 | 0.0981 |
| 2-5 | 312 | 0.1848 | 0.2486 | 0.2145 | 0.2619 | 0.1045 |
| 6-20 | 12037 | 0.1661 | 0.1948 | 0.1621 | 0.2076 | 0.0726 |
| >20 | 39226 | 0.1728 | 0.2178 | 0.1236 | 0.1495 | 0.0641 |

## Per-dominant-layer

| dominant_layer | n_nets | gnd_mape_median | gnd_mape_mean | cpl_mape_median | cpl_mape_mean | total_mape_median |
|---|---|---|---|---|---|---|
| M2 | 73640 | 0.1942 | 0.2325 | 0.1526 | 0.1978 | 0.0789 |
| M3 | 19651 | 0.2174 | 0.2521 | 0.1490 | 0.2047 | 0.0789 |
| M4 | 1708 | 0.2082 | 0.2530 | 0.1369 | 0.1706 | 0.0828 |
| M5 | 595 | 0.1796 | 0.1988 | 0.1336 | 0.1612 | 0.0688 |

