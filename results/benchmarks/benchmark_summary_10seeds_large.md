| Method                                    | Success Rate (%) | Latency (ms) | Task 01 (%) | Task 02 (%) | Task 03 (%) | Task 04 (%) | Task 05 (%) |
| ----------------------------------------- | ---------------- | ------------ | ----------- | ----------- | ----------- | ----------- | ----------- |
| 1. Single-Intention Baseline              | 55.0 ± 5.0       | 0.86         | 50.0        | 75.0        | 100.0       | 25.0        | 25.0        |
| 2. Dijkstra Teacher (high_actor)          | 60.0 ± 10.0      | 0.80         | 75.0        | 75.0        | 100.0       | 25.0        | 25.0        |
| 3. Dijkstra + Single WP Translator        | 20.0 ± 0.0       | 0.68         | 0.0         | 50.0        | 0.0         | 25.0        | 25.0        |
| 5. Dijkstra + Enhanced Sequence Attention | 55.0 ± 5.0       | 3.34         | 75.0        | 75.0        | 50.0        | 50.0        | 25.0        |
| 6. Distilled JAX GatedAttn [O(1)]         | 35.0 ± 5.0       | 0.68         | 0.0         | 100.0       | 50.0        | 25.0        | 0.0         |