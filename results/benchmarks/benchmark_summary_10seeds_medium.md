| Method                                      | Success Rate (%)   |   Latency (ms) |   Task 01 (%) |   Task 02 (%) |   Task 03 (%) |   Task 04 (%) |   Task 05 (%) |
|:--------------------------------------------|:-------------------|---------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| 1. Single-Intention Baseline                | 78.8 ± 4.8         |           0.95 |            76 |            80 |            72 |            74 |            92 |
| 2. Dijkstra Teacher (high_actor)            | 85.2 ± 7.7         |           0.85 |            74 |            90 |            84 |            86 |            92 |
| 3. Dijkstra + Single WP Translator          | 85.2 ± 6.0         |           0.53 |            86 |            86 |            90 |            76 |            88 |
| 4. Dijkstra + Sequence Attention Translator | 79.6 ± 4.5         |           0.81 |            80 |            82 |            78 |            66 |            92 |
| 5. Dijkstra + Enhanced Sequence Attention   | 88.0 ± 5.2         |           0.9  |            96 |            92 |            84 |            78 |            90 |
| 6. Distilled JAX GatedAttn [O(1)]           | 85.2 ± 3.0         |           0.56 |            82 |            96 |            80 |            76 |            92 |