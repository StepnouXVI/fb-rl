# Исследовательские статьи по FB-представлениям, Successor Measures и Zero-Shot RL

В этой папке собраны все ключевые статьи, на которые ссылается [Task.md](file:///Users/savvatej/source/fb-rl/Task.md), а также фундаментальные работы по Successor Features и Goal-Conditioned RL.

---

## Каталог Статей

| Файл | Авторы и Год | Название | Тематика / Ключевой вклад |
|---|---|---|---|
| [01_stojanovic2026_switching_successor_measures.pdf](file:///Users/savvatej/source/fb-rl/papers/01_stojanovic2026_switching_successor_measures.pdf) | Stojanovic & Proutiere (2026) | *Switching Successor Measures for Hierarchical Zero-shot Reinforcement Learning* | **Основная статья задания.** Вводит FB $\pi$-Switch: иерархический single-intention контроллер над FB-представлениями. |
| [02_touati2021_learning_one_representation.pdf](file:///Users/savvatej/source/fb-rl/papers/02_touati2021_learning_one_representation.pdf) | Touati & Ollivier (2021) | *Learning One Representation to Optimize All Rewards* | Введение Forward-Backward (FB) представлений, SVD-разложение successor measures $M \approx F^\top B$. |
| [03_touati2023_does_zero_shot_rl_exist.pdf](file:///Users/savvatej/source/fb-rl/papers/03_touati2023_does_zero_shot_rl_exist.pdf) | Touati, Rapin & Ollivier (2023) | *Does Zero-Shot Reinforcement Learning Exist?* | Анализ обобщающей способности zero-shot RL, алгоритмы оценки и регуляризация ковариации. |
| [04_barreto2017_successor_features.pdf](file:///Users/savvatej/source/fb-rl/papers/04_barreto2017_successor_features.pdf) | Barreto et al. (2017) | *Successor Features for Transfer in Reinforcement Learning* | Фундаментальная работа по Successor Features (SF) для переноса между линейными наградами $r = \phi^\top w$. |
| [05_park2024_ogbench.pdf](file:///Users/savvatej/source/fb-rl/papers/05_park2024_ogbench.pdf) | Park et al. (2024) | *OGBench: Benchmarking Offline Goal-Conditioned RL* | Описание бенчмарка OGBench и задач `antmaze-*-navigate-v0`. |
| [06_borsa2018_universal_successor_features.pdf](file:///Users/savvatej/source/fb-rl/papers/06_borsa2018_universal_successor_features.pdf) | Borsa et al. (2018) | *Universal Successor Features Approximators (USFA)* | Обобщение SF на непрерывные пространства политик и задач. |
| [07_blier2021_learning_successor_states_and_goals.pdf](file:///Users/savvatej/source/fb-rl/papers/07_blier2021_learning_successor_states_and_goals.pdf) | Blier et al. (2021) | *Learning Successor States and Goals for Transfer and Generalization* | Обучение латентных пространств целей и successor states для трансферного обучения. |
| [08_eysenbach2022_contrastive_rl.pdf](file:///Users/savvatej/source/fb-rl/papers/08_eysenbach2022_contrastive_rl.pdf) | Eysenbach et al. (2022) | *Contrastive Learning as Goal-Conditioned Reinforcement Learning* | Связь контрастивного обучения представлений и мер занятости в GCRL. |

---

## Рекомендуемый Порядок Чтения:
1. **[01_stojanovic2026]** — понять постановку задачи, архитектуру бейзлайна FB $\pi$-Switch и открытую проблему многоэтапного планирования (multi-subgoal planning).
2. **[02_touati2021]** — разобраться в математике разложения $F(s,a)^\top B(s')$, уравнении Беллмана для мер и TD-loss с ортонормированием.
3. **[05_park2024]** — изучить устройство среды `antmaze-medium-navigate-v0` и метрики OGBench.
4. **[04_barreto2017] & [03_touati2023]** — для глубокого контекста по successor features и оценке zero-shot агентов.
