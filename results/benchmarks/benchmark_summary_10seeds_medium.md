| Method                                   | Success Rate (%)   |   Latency (ms) |   Task 01 (%) |   Task 02 (%) |   Task 03 (%) |   Task 04 (%) |   Task 05 (%) |
|:-----------------------------------------|:-------------------|---------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| 1. Dijkstra + Sequence Attention         | 78.8 ± 6.1         |           1.43 |            63 |            89 |            79 |            73 |            90 |
| 2. Single-Intention Baseline             | 81.0 ± 4.1         |           0.87 |            82 |            89 |            82 |            61 |            91 |
| 3. Dijkstra + Single Waypoint Translator | 85.0 ± 5.2         |           0.92 |            80 |            91 |            89 |            76 |            89 |
| 4. Dijkstra Teacher (high_actor)         | 81.8 ± 3.2         |           0.95 |            72 |            86 |            85 |            79 |            87 |
| 5. Direct Intention Planner [O(1)]       | 83.6 ± 4.4         |           0.92 |            85 |            97 |            76 |            72 |            88 |