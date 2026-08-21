# AntMaze Medium Benchmark Summary

| Method | Overall Success (%) | Mean Steps | Latency (ms/step) | Task 01 (%) | Task 02 (%) | Task 03 (%) | Task 04 (%) | Task 05 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5. Distilled JAX GatedAttn [O(1)]** | **86.0%** | 184.2 | **0.59 ms** | 80.0% | **100.0%** | 80.0% | 80.0% | **90.0%** |
| **2. Dijkstra Teacher (high_actor)** | 78.0% | 210.5 | 0.86 ms | 80.0% | 90.0% | 70.0% | 70.0% | 80.0% |
| **1. Single-Intention Baseline** | 76.0% | 235.1 | 0.95 ms | 70.0% | 80.0% | 60.0% | 90.0% | 80.0% |
| **4. Dijkstra + Sequence Attention Translator** | 66.0% | 248.6 | 0.83 ms | **90.0%** | 80.0% | 80.0% | 30.0% | 50.0% |
| **3. Dijkstra + Single WP Translator** | 18.0% | 412.3 | 0.53 ms | 40.0% | 10.0% | 10.0% | 20.0% | 10.0% |
