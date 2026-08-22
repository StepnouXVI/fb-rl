| Method                                    | Success Rate (%)   |   Latency (ms) |   Task 01 (%) |   Task 02 (%) |   Task 03 (%) |   Task 04 (%) |   Task 05 (%) |
|:------------------------------------------|:-------------------|---------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| 1. Single-Intention Baseline              | 49.3 ± 3.6         |           0.95 |          43.5 |          54.5 |          82   |          32.5 |          34   |
| 2. Dijkstra Teacher (high_actor)          | 46.8 ± 4.7         |           0.87 |          48   |          65   |          59.5 |          27.5 |          34   |
| 3. Dijkstra + Single WP Translator        | 48.2 ± 5.0         |           0.54 |          43   |          71   |          53.5 |          31.5 |          42   |
| 5. Dijkstra + Enhanced Sequence Attention | 64.3 ± 4.9         |           0.95 |          56.5 |          75   |          77   |          54   |          59   |
| 6. Distilled JAX GatedAttn [O(1)]         | 47.4 ± 4.2         |           0.57 |          55.5 |          74.5 |          51   |          22.5 |          33.5 |