# Stratified eval — variant `c1b_full` (split=test)

Baseline: n=95,594 nets, gnd median 18.684%, cpl median 15.153%, total median 6.582%.

## Per-design

| design | n_nets | gnd_mape_median | gnd_mape_mean | cpl_mape_median | cpl_mape_mean | total_mape_median | total_mape_mean |
|---|---|---|---|---|---|---|---|
| intel22_nova_f3 | 92425 | 0.1869 | 0.2458 | 0.1519 | 0.1994 | 0.0658 | 0.0812 |
| intel22_tv80s_f3 | 3169 | 0.1829 | 0.2382 | 0.1437 | 0.1713 | 0.0652 | 0.0779 |

## Per-quartile (axis=compact_gnd_estimate_fF)

| quartile | axis | axis_min | axis_max | n_nets | gnd_mape_median | gnd_mape_mean | cpl_mape_median | cpl_mape_mean | total_mape_median |
|---|---|---|---|---|---|---|---|---|---|
| Q1 | compact_gnd_estimate_fF | 0.0287 | 0.0767 | 23944 | 0.1976 | 0.2643 | 0.1832 | 0.2364 | 0.0886 |
| Q2 | compact_gnd_estimate_fF | 0.0767 | 0.1409 | 23853 | 0.2135 | 0.2716 | 0.1731 | 0.2370 | 0.0698 |
| Q3 | compact_gnd_estimate_fF | 0.1409 | 0.4596 | 23898 | 0.1808 | 0.2320 | 0.1432 | 0.1747 | 0.0606 |
| Q4 | compact_gnd_estimate_fF | 0.4596 | 52.4053 | 23899 | 0.1553 | 0.2142 | 0.1180 | 0.1460 | 0.0510 |

## Per-quartile (axis=gnd_rel_err — Mode B giant-CTS surface)

| quartile | axis | axis_min | axis_max | n_nets | gnd_mape_median | gnd_mape_mean | cpl_mape_median | cpl_mape_mean | total_mape_median |
|---|---|---|---|---|---|---|---|---|---|
| Q1 | gnd_rel_err | 0.0000 | 0.0871 | 23899 | 0.0433 | 0.0434 | 0.1037 | 0.1351 | 0.0562 |
| Q2 | gnd_rel_err | 0.0871 | 0.1868 | 23898 | 0.1339 | 0.1350 | 0.1248 | 0.1636 | 0.0589 |
| Q3 | gnd_rel_err | 0.1868 | 0.3244 | 23898 | 0.2483 | 0.2505 | 0.1629 | 0.2230 | 0.0698 |
| Q4 | gnd_rel_err | 0.3244 | 9.0888 | 23899 | 0.4556 | 0.5531 | 0.2225 | 0.2724 | 0.0830 |

## Per-fanout bucket

| fanout_bucket | n_nets | gnd_mape_median | gnd_mape_mean | cpl_mape_median | cpl_mape_mean | total_mape_median |
|---|---|---|---|---|---|---|
| 1 | 44019 | 0.2067 | 0.2700 | 0.1796 | 0.2392 | 0.0786 |
| 2-5 | 312 | 0.1786 | 0.2623 | 0.2145 | 0.2619 | 0.0775 |
| 6-20 | 12037 | 0.1510 | 0.2006 | 0.1621 | 0.2076 | 0.0618 |
| >20 | 39226 | 0.1757 | 0.2317 | 0.1236 | 0.1495 | 0.0552 |

## Per-dominant-layer

| dominant_layer | n_nets | gnd_mape_median | gnd_mape_mean | cpl_mape_median | cpl_mape_mean | total_mape_median |
|---|---|---|---|---|---|---|
| M2 | 73640 | 0.1842 | 0.2435 | 0.1526 | 0.1978 | 0.0662 |
| M3 | 19651 | 0.1972 | 0.2551 | 0.1490 | 0.2047 | 0.0642 |
| M4 | 1708 | 0.1913 | 0.2431 | 0.1369 | 0.1706 | 0.0735 |
| M5 | 595 | 0.1561 | 0.1891 | 0.1336 | 0.1612 | 0.0588 |

