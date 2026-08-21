| Method                                    | Success Rate (%)   |   Latency (ms) |   Task 01 (%) |   Task 02 (%) |   Task 03 (%) |   Task 04 (%) |   Task 05 (%) |
|:------------------------------------------|:-------------------|---------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| 1. Single-Intention Baseline              | 37.4 ± 5.7         |           0.95 |            35 |            34 |            70 |            24 |            24 |
| 2. Dijkstra Teacher (high_actor)          | 33.2 ± 5.5         |           0.86 |            41 |            49 |            36 |            15 |            25 |
| 3. Dijkstra + Single WP Translator        | 34.2 ± 6.7         |           0.54 |            28 |            55 |            35 |            18 |            35 |
| 5. Dijkstra + Enhanced Sequence Attention | 40.0 ± 6.4         |           0.93 |            37 |            54 |            47 |            23 |            39 |
| 6. Distilled JAX GatedAttn [O(1)]         | 30.8 ± 4.9         |           0.56 |            30 |            49 |            38 |            11 |            26 |