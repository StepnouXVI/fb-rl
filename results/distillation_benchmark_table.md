| Architecture / Planner | Params | Val Cosine Sim | Val MSE | Success Rate (%) | Mean Steps | Latency (ms/step) | Loops/Episode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single-Intention Baseline** | - | - | - | **78.8 ± 3.0** | 544.7 ± 18.8 | 0.887 ± 0.002 | 14.58 ± 4.77 |
| **Buffer Graph (Dijkstra Teacher)** | - | 1.0000 | - | **84.4 ± 9.2** | 510.3 ± 58.4 | 0.833 ± 0.006 | 18.43 ± 7.07 |
| **Distilled StandardMLP** | 140,160 | 0.7174 | 0.5652 | **78.8 ± 2.0** | 584.1 ± 13.3 | 1.003 ± 0.004 | 23.16 ± 4.32 |
| **Distilled DenseECANetwork** | 337,542 | 0.7165 | 0.5669 | **83.2 ± 3.0** | 544.2 ± 30.4 | 1.114 ± 0.005 | 20.25 ± 6.75 |
| **Distilled GatedCrossAttention** | 535,427 | 0.7081 | 0.5838 | **82.0 ± 4.4** | 574.0 ± 38.9 | 1.266 ± 0.006 | 27.60 ± 8.45 |
| **Distilled ResNetECANetwork** | 471,689 | 0.7245 | 0.5510 | **81.2 ± 4.8** | 555.0 ± 28.2 | 1.203 ± 0.025 | 15.97 ± 1.49 |
| **Distilled FiLMResNetNetwork** | 933,513 | 0.7160 | 0.5680 | **80.4 ± 2.0** | 576.8 ± 30.3 | 1.347 ± 0.008 | 30.21 ± 12.92 |
| **Distilled TransformerEncoder** | 1,655,680 | 0.7717 | 0.4566 | **80.4 ± 5.7** | 553.4 ± 35.0 | 2.519 ± 0.041 | 21.12 ± 2.87 |
