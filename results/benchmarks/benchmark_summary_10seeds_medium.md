                                      Method Success Rate (%) Latency (ms) Task 01 (%)
0  1. Dijkstra + Enhanced Sequence Attention      100.0 ± 0.0         3.87       100.0
1               2. Single-Intention Baseline      90.0 ± 30.0         0.99        90.0
2             3. Recursive Bisection Planner      60.0 ± 49.0         8.16        60.0
3         4. Dijkstra + Single WP Translator      90.0 ± 30.0         0.87        90.0
4           5. Dijkstra Teacher (high_actor)      70.0 ± 45.8         0.92        70.0
5          6. Distilled JAX GatedAttn [O(1)]      60.0 ± 49.0         0.81        60.0