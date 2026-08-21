| Method                                      | Success Rate (%)   |   Latency (ms) |   Task 01 (%) |   Task 02 (%) |   Task 03 (%) |   Task 04 (%) |   Task 05 (%) |
|:--------------------------------------------|:-------------------|---------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| 1. Single-Intention Baseline                | 81.8 ± 5.1         |           0.94 |            77 |            85 |            80 |            74 |            93 |
| 2. Dijkstra Teacher (high_actor)            | 83.6 ± 6.7         |           0.84 |            75 |            88 |            84 |            79 |            92 |
| 3. Dijkstra + Single WP Translator          | 82.8 ± 5.3         |           0.53 |            81 |            85 |            85 |            73 |            90 |
| 4. Dijkstra + Sequence Attention Translator | 82.6 ± 4.8         |           0.81 |            83 |            87 |            83 |            68 |            92 |
| 5. Dijkstra + Enhanced Sequence Attention   | 84.6 ± 5.2         |           0.9  |            89 |            84 |            86 |            77 |            87 |
| 6. Distilled JAX GatedAttn [O(1)]           | 85.0 ± 3.5         |           0.56 |            84 |            96 |            79 |            75 |            91 |