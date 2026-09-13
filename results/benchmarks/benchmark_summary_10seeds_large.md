| Method                                   | Success Rate (%)   |   Latency (ms) |   Task 01 (%) |   Task 02 (%) |   Task 03 (%) |   Task 04 (%) |   Task 05 (%) |
|:-----------------------------------------|:-------------------|---------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| 1. Dijkstra + Sequence Attention         | 62.2 ± 7.1         |           1.45 |            47 |            79 |            75 |            51 |            59 |
| 2. Single-Intention Baseline             | 48.8 ± 4.0         |           0.86 |            41 |            51 |            88 |            30 |            34 |
| 3. Dijkstra + Single Waypoint Translator | 55.8 ± 7.0         |           0.91 |            54 |            73 |            64 |            38 |            50 |
| 4. Dijkstra Teacher (high_actor)         | 51.0 ± 7.5         |           0.95 |            47 |            75 |            58 |            34 |            41 |
| 5. Direct Intention Planner [O(1)]       | 40.0 ± 7.4         |           0.91 |            41 |            67 |            40 |            17 |            35 |