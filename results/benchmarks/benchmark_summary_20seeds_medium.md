| Method                                    | Success Rate (%)   |   Latency (ms) |   Task 01 (%) |   Task 02 (%) |   Task 03 (%) |   Task 04 (%) |   Task 05 (%) |
|:------------------------------------------|:-------------------|---------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| 1. Dijkstra + Enhanced Sequence Attention | 83.6 ± 3.8         |           0.95 |          87.5 |          85   |          87.5 |          70   |          88   |
| 2. Single-Intention Baseline              | 81.4 ± 4.5         |           0.95 |          80.5 |          86   |          81   |          69   |          90.5 |
| 3. Recursive Bisection Planner            | 75.9 ± 6.4         |           2.82 |          69.5 |          86   |          70   |          65   |          89   |
| 4. Dijkstra + Single WP Translator        | 83.3 ± 5.0         |           0.53 |          82.5 |          86.5 |          87   |          69   |          91.5 |
| 5. Dijkstra Teacher (high_actor)          | 83.9 ± 3.2         |           0.85 |          72   |          88.5 |          89.5 |          80.5 |          89   |
| 6. Distilled JAX GatedAttn [O(1)]         | 84.3 ± 4.8         |           0.55 |          83   |          95   |          81   |          73   |          89.5 |