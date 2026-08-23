| Method                                    | Success Rate (%)   |   Latency (ms) |   Task 01 (%) |   Task 02 (%) |   Task 03 (%) |   Task 04 (%) |   Task 05 (%) |
|:------------------------------------------|:-------------------|---------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| 1. Dijkstra + Enhanced Sequence Attention | 60.9 ± 6.9         |           0.97 |          53   |          66.5 |          72   |          54   |          59   |
| 2. Single-Intention Baseline              | 49.3 ± 5.2         |           0.95 |          44.5 |          54.5 |          79.5 |          36.5 |          31.5 |
| 3. Recursive Bisection Planner            | 46.5 ± 8.0         |           2.36 |          29.5 |          59   |          85.5 |          25.5 |          33   |
| 4. Dijkstra + Single WP Translator        | 57.1 ± 5.6         |           0.54 |          53   |          74.5 |          58   |          44   |          56   |
| 5. Dijkstra Teacher (high_actor)          | 48.7 ± 9.2         |           0.87 |          56   |          65.5 |          55.5 |          32   |          34.5 |
| 6. Distilled JAX GatedAttn [O(1)]         | 45.6 ± 5.8         |           0.57 |          50   |          68   |          55   |          28   |          27   |