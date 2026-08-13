# Полный Руководство по Forward-Backward (FB) Представлениям, Уравнениям Беллмана и Мульти-Субгольному Планированию в Zero-Shot RL

---

## Содержание
1. **Введение и Концептуальная Интуиция**
2. **Математический Аппарат: От Оперетора Занятости к Мерам Беллмана**
   - 2.1. Формулировка MDP и Задача Zero-Shot RL
   - 2.2. Successor Representation (SR) и Successor Features (SF)
   - 2.3. Successor Measures (SM) в Непрерывном Пространстве
3. **Forward-Backward (FB) Разложение в Гильбертовом Пространстве**
   - 3.1. Низкоранговое Разложение Ядра Занятости (SVD Интуиция)
   - 3.2. Вывод Q-функции для Произвольной Награды
   - 3.3. Функция Потерь и Функция Скольжения (Loss Functions)
4. **Иерархическая Структура и Ограничение Single-Intention Baseline**
   - 4.1. Двухуровневый Агент $\pi_l$ и $\pi_{\text{high}}$
   - 4.2. Проблема Узких Горлышек (Bottleneck Problem в Лабиринтах)
5. **Мульти-Субгольное Графовое Планирование (Multi-Subgoal Planning)**
   - 5.1. Построение Графа Состояний в Пространстве $B(s)$
   - 5.2. Алгоритм Поиска Пути и Динамического Переключения Интенций
   - 5.3. Полный Псевдокод Решения на Python
6. **Сравнение с Генеративными Моделями Мира (World Models)**
7. **Практические Советы для Сдачи Задания**

---

## 1. Введение и Концептуальная Интуиция

### Аналогия из Физики: Поле Теплопроводности и Источники
Представьте себе физическую среду (например, двумерный металлический лист с препятствиями), в которой распространяется тепло или диффундирует газ:
- Если в точке $s_0$ мы кратковременно включим нагреватель (состояние $s_0$, действие $a_0$) и будем следовать некоторому закону движения частиц (политике $\pi$), то спустя время в других точках $s'$ установится некоторая плотность распределения тепла.
- **Successor Measure $M(s, a, s')$** — это в точности стационарная плотность «теплового следа» (накопленная с дисконтом $\gamma$), которая окажется в точке $s'$, если мы стартовали из $(s, a)$.
- **Forward представление $F(s, a)$** — это характеристики источника (излучателя).
- **Backward представление $B(s')$** — это характеристики сенсора (детектора) в точке $s'$.

Если скалярное произведение $F(s, a)^\top B(s')$ велико, значит из состояния $(s, a)$ динамика системы естественным образом приведёт агента в состояние $s'$.

---

## 2. Математический Аппарат: От Оператора Занятости к Мерам Беллмана

### 2.1. Марковский Процесс Принятия Решений (MDP)
Рассматривается непрерывный MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, \gamma)$, где:
- $\mathcal{S} \subseteq \mathbb{R}^{d_s}$ — пространство состояний;
- $\mathcal{A} \subseteq \mathbb{R}^{d_a}$ — пространство действий;
- $P(s' \mid s, a)$ — вероятность перехода (плотность);
- $\gamma \in (0, 1)$ — коэффициент дисконтирования.

В **Reward-Free / Zero-Shot RL** во время обучения функция награды $r(s, a)$ **неизвестна**. Агенту доступен только оффлайн-датасет $\mathcal{D} = \{(s_t, a_t, s_{t+1})\}_{t=1}^N$.

---

### 2.2. Successor Representations (SR) и Successor Features (SF)

#### 1. Successor Representation (Dayan, 1993)
В дискретном пространстве состояний матрица $M^\pi \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{S}|}$ задается как:
$$M^\pi(s, s') = \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t \mathbb{I}(s_t = s') \middle| s_0 = s \right] = (I - \gamma P^\pi)^{-1}_{s, s'}$$

Если награда задана вектором $r \in \mathbb{R}^{|\mathcal{S}|}$, то ценность состояния $V^\pi(s)$ вычисляется мгновенно линейным произведением:
$$V^\pi(s) = \sum_{s'} M^\pi(s, s') r(s') = M^\pi(s, \cdot)^\top r$$

#### 2. Successor Features (Barreto et al., 2017)
Если состояния непрерывны, но мы имеем фиксированный вектор признаков (features) $\phi(s, a) \in \mathbb{R}^d$, то определяем Successor Features:
$$\psi^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t \phi(s_t, a_t) \middle| s_0 = s, a_0 = a \right]$$

Если награда линейна по признакам $r(s, a) = \phi(s, a)^\top w$, то Q-функция вычисляется как:
$$Q^\pi(s, a) = \psi^\pi(s, a)^\top w$$

---

### 2.3. Successor Measures (SM) в Непрерывном Пространстве (Touati & Ollivier, 2021)

Для произвольного непрерывного пространства без фиксированного базиса $\phi$, определим **Successor Measure** $M^\pi(s, a, \cdot)$ как дисконтированную меру занятости:
$$M^\pi(s, a, A) = \sum_{t=0}^\infty \gamma^t P^\pi(s_t \in A \mid s_0 = s, a_0 = a)$$

В терминах плотности распределения $M^\pi(s, a, s')$ удовлетворяет **рекуррентному уравнению Беллмана для мер**:
$$M^\pi(s, a, s') = \delta(s, s') + \gamma \int_{\mathcal{S}} \int_{\mathcal{A}} P(s'' \mid s, a) \pi(a'' \mid s'') M^\pi(s'', a'', s') \, ds'' \, da''$$
где $\delta(s, s')$ — дельта-функция Дирака.

---

## 3. Forward-Backward (FB) Разложение в Гильбертовом Пространстве

### 3.1. Низкоранговое Разложение Ядра Занятости (SVD Интуиция)

Представим оператор занятости $M^\pi(s, a, s')$ как ядро интегрального оператора в Гильбертовом пространстве $L_2(\mathcal{S})$. Согласно обобщенному спектральному разложению (SVD для интегральных операторов):
$$M^\pi(s, a, s') \approx \sum_{i=1}^d \sigma_i u_i(s, a) v_i(s')$$

Определим:
- **Forward representation:** $F(s, a) = [\sqrt{\sigma_1} u_1(s, a), \dots, \sqrt{\sigma_d} u_d(s, a)]^\top \in \mathbb{R}^d$
- **Backward representation:** $B(s') = [\sqrt{\sigma_1} v_1(s'), \dots, \sqrt{\sigma_d} v_d(s')]^\top \in \mathbb{R}^d$

Тогда плотность меры вычисляется простым скалярным произведением в $d$-мерном латентном пространстве:
$$M^z(s, a, s') = F(s, a)^\top B(s')$$

---

### 3.2. Строгий Вывод Q-функции для Произвольной Награды

Пусть на этапе тестирования нам задали произвольную функцию награды $r(s)$.
Разложим функцию награды по базису $B(s)$:
$$r(s) \approx z_r^\top B(s), \quad \text{где } z_r = \mathbb{E}_{s \sim \mathcal{D}} [r(s) B(s)] \in \mathbb{R}^d$$

Тогда для Q-функции под политикой $\pi_z$:
$$Q^z(s, a) = \mathbb{E}_{\pi_z} \left[ \sum_{t=0}^\infty \gamma^t r(s_t) \middle| s_0 = s, a_0 = a \right] = \mathbb{E}_{\pi_z} \left[ \sum_{t=0}^\infty \gamma^t z_r^\top B(s_t) \right]$$

Вынося постоянный вектор $z_r$ за знак математического ожидания:
$$Q^z(s, a) = z_r^\top \left( \mathbb{E}_{\pi_z} \left[ \sum_{t=0}^\infty \gamma^t B(s_t) \right] \right) = z_r^\top F(s, a) = F(s, a)^\top z_r$$

**Вывод:** Функция $F(s, a)^\top z$ является **точным значением Q-функции** для задачи, задаваемой латентным вектором $z = z_r$!

---

### 3.3. Функция Потерь и Обучение FB-Представлений

Обучение $F_\theta(s, a)$ и $B_\phi(s')$ проводится на оффлайн-датасете $\mathcal{D}$ с помощью TD-обучения с подменяемой наградой (reward substitution).

#### 1. TD Loss для $F_\theta$:
Для случайного латентного вектора $z \sim \mathcal{S}^{d-1}$ (сфера единичного радиуса):
$$\mathcal{L}_{\text{TD}}(\theta) = \mathbb{E}_{(s, a, s') \sim \mathcal{D}, z \sim \mathcal{S}^{d-1}} \left[ \left( F_\theta(s, a)^\top z - \left( B_\phi(s')^\top z + \gamma F_\bar{\theta}(s', \pi_l(s', z))^\top z \right) \right)^2 \right]$$
где $B_\phi(s')^\top z$ выступает в роли мгновенной награды $r(s, a, z)$, а $\bar{\theta}$ — целевые веса (target network).

#### 2. Ограничение Нормировки (Covariance Regularization):
Чтобы предотвратить вырождение эмбеддингов $B(s)$ в ноль, накладывается условие ортонормированности:
$$\mathcal{L}_{\text{cov}}(\phi) = \| \mathbb{E}_{s \sim \mathcal{D}} [B_\phi(s) B_\phi(s)^\top] - I_d \|_F^2$$

#### 3. Низкоуровневая Политика (Low-Level Actor $\pi_l$):
Политика $\pi_l(a \mid s, z)$ обучается максимизировать выученную Q-ценность $F(s, a)^\top z$:
$$\mathcal{L}_{\text{actor}}(\psi) = - \mathbb{E}_{s \sim \mathcal{D}, z \sim \mathcal{S}^{d-1}} \left[ F_\theta(s, \pi_\psi(s, z))^\top z \right]$$

---

## 4. Иерархическая Структура и Ограничение Single-Intention Baseline

### 4.1. Схема Двухуровневого Контроллера
В архитектуре **FB $\pi$-Switch (Stojanovic & Proutiere, 2026)**:
- **Low-Level Policy $\pi_l(a \mid s, z)$:** Исполняет латентное намерение $z$ в течение $H$ шагов среды.
- **High-Level Controller $\pi_{\text{high}}(z \mid s, g)$:** По текущему состоянию $s$ и целевому состоянию $g$ выбирает вектор направления $z \in \mathcal{S}^{d-1}$.

---

### 4.2. Геометрическая Проблема Узких Горлышек (Bottleneck Problem)

В пространственных задачах (например, лабиринт `antmaze-medium-navigate-v0`):

```
       [ Старт s_0 ] --------------> ( Стена ) --------------> [ Цель g ]
             |                                                       ^
             |                                                       |
             +----> [ Коридор 1 ] ---> [ Поворот ] ---> [ Коридор 2 ]+
```

1. **Одноинтенциональный бейзлайн (Single-Intention):**
   Выбирает единственный вектор $z \propto B(g) - B(s_0)$.
   В евклидовом пространстве $B$-эмбеддингов вектор $z$ указывает **прямо сквозь стену** на цель $g$.
   Политика нижнего уровня $\pi_l(a \mid s_0, z)$ пытается двигаться прямо на стену и навсегда застревает в тупике.

2. **Причина:**
   Представление $B(s)$ не является глобально выпуклым в физическом пространстве из-за непроходимых стен. Прямая линия в пространстве $B$ не совпадает с геодезической линией (кратчайшим путем) на многообразии допустимых состояний среды.

---

## 5. Мульти-Субгольное Графовое Планирование (Multi-Subgoal Planning)

Решение заключается в строительстве **дискретного графа связности** поверх выученного $B$-пространства и поиске кратчайшего пути по субголам.

### 5.1. Построение Графа $G = (V, E)$
1. Из оффлайн-датасета $\mathcal{D}$ выбирается $N$ опорных состояний $V = \{s_1, s_2, \dots, s_N\}$ (через равномерное сэмплирование или k-means по $B(s_i)$).
2. Вычисляются латентные векторы $v_i = B(s_i) \in \mathbb{R}^d$.
3. Для каждого узла $s_i$ находятся $k$ ближайших соседей в $B$-пространстве.
4. Вес ребра между $s_i$ и $s_j$:
   $$w(i, j) = \| B(s_i) - B(s_j) \|_2^2$$

---

### 5.2. Пошаговый Алгоритм Мульти-Субгольного Контроллера

```text
               (A* Поиск Пути в B-пространстве)
[ s_0 ] ---> ( Subgoal 1: s_{g1} ) ---> ( Subgoal 2: s_{g2} ) ---> [ Goal g ]
                 |                          |                         |
               z_1 = B(s_{g1})-B(s_0)     z_2 = B(s_{g2})-B(s_{g1})  z_3 = B(g)-B(s_{g2})
                 v                          v                         v
               \pi_l(a|s, z_1)            \pi_l(a|s, z_2)           \pi_l(a|s, z_3)
```

1. **Шаг 1 (Локализация):**
   Находим в графе вершины, ближайшие к текущему $s_0$ и цели $g$:
   $$i_{\text{start}} = \arg\min_{i} \| B(s_0) - B(s_i) \|, \quad i_{\text{goal}} = \arg\min_{i} \| B(g) - B(s_i) \|$$

2. **Шаг 2 (Поиск пути):**
   Запускаем алгоритм A* или Дейкстры на графе $G$:
   $$\text{Path} = (s_{g1}, s_{g2}, \dots, s_{gK} = g)$$

3. **Шаг 3 (Переключение интенций):**
   Для каждой промежуточной подцели $s_{gk}$ формируем латентную интенцию:
   $$z_k = \frac{B(s_{gk}) - B(s_{gk-1})}{\| B(s_{gk}) - B(s_{gk-1}) \|}$$
   Исполняем $\pi_l(a \mid s, z_k)$ в течение $H$ шагов либо пока $\| B(s) - B(s_{gk}) \| < \epsilon$.

---

### 5.3. Полный Псевдокод Решения на Python

```python
import numpy as np
import torch
import networkx as nx
from scipy.spatial import cKDTree

class MultiSubgoalFBPlanner:
    def __init__(self, forward_net, backward_net, actor_net, dataset_states, k_neighbors=15):
        """
        forward_net: F(s, a)
        backward_net: B(s)
        actor_net: pi_l(a | s, z)
        dataset_states: numpy array (N, state_dim)
        """
        self.F = forward_net
        self.B = backward_net
        self.actor = actor_net
        self.dataset_states = dataset_states
        
        # 1. Извлекаем B-эмбеддинги для всех оффлайн-состояний
        print("Вычисление B-эмбеддингов для графа...")
        with torch.no_grad():
            states_t = torch.tensor(dataset_states, dtype=torch.float32)
            self.B_embeds = self.B(states_t).cpu().numpy() # (N, d)
            
        # 2. Строим KDTree для быстрого k-NN поиска
        self.tree = cKDTree(self.B_embeds)
        
        # 3. Строим граф NetworkX
        print("Построение k-NN графа связности...")
        self.graph = nx.Graph()
        N = len(dataset_states)
        for i in range(N):
            distances, indices = self.tree.query(self.B_embeds[i], k=k_neighbors + 1)
            for dist, j in zip(distances[1:], indices[1:]):
                self.graph.add_edge(i, j, weight=dist)
                
    def plan_path(self, start_state, goal_state):
        """Возвращает список интенций z_1, z_2, ..., z_K"""
        with torch.no_grad():
            s_0_t = torch.tensor(start_state, dtype=torch.float32).unsqueeze(0)
            g_t = torch.tensor(goal_state, dtype=torch.float32).unsqueeze(0)
            
            b_s0 = self.B(s_0_t).cpu().numpy()[0]
            b_g = self.B(g_t).cpu().numpy()[0]
            
        # Находим ближайшие узлы в графе
        _, start_idx = self.tree.query(b_s0)
        _, goal_idx = self.tree.query(b_g)
        
        # Поиск кратчайшего пути Дейкстры / A*
        node_path = nx.shortest_path(self.graph, source=start_idx, target=goal_idx, weight='weight')
        
        # Переводим вершины пути в последовательность интенций z
        intentions = []
        for idx in range(1, len(node_path)):
            b_prev = self.B_embeds[node_path[idx-1]]
            b_next = self.B_embeds[node_path[idx]]
            
            diff = b_next - b_prev
            norm = np.linalg.norm(diff)
            z = diff / (norm + 1e-8)
            intentions.append(z)
            
        return intentions, node_path

    def get_action(self, state, current_z):
        """Вызов низкоуровневой политики"""
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            z_t = torch.tensor(current_z, dtype=torch.float32).unsqueeze(0)
            action = self.actor(s_t, z_t).cpu().numpy()[0]
        return action
```

---

## 6. Сравнение с Генеративными Моделями Мира (World Models)

В чем преимущество этого подхода перед генеративными моделями (Diffuser, Trajectory Transformer):

1. **Скорость и Вычислительная Эффективность:**
   - Поиск A* на графе из 10,000 узлов занимает **< 5 миллисекунд**.
   - Генерация траекторий через диффузию (Diffuser) требует сотни шагов денойзинга тяжелой UNet-сетью (**> 500-1000 мс** на шаг).
2. **Гарантия корректности геометрии:**
   - Граф строится только по состоявшимся реальным состояниям оффлайн-датасета $\mathcal{D}$, исключая галлюцинации несуществующих состояний.
3. **Отсутствие дообучения:**
   - Использованы строго замороженные $F, B, \pi_l$ из предоставленных чекпоинтов.

---

## 7. Практический Чек-Лист для Выполнения Задания

1. **Загрузить чекпоинты** из предоставленной ссылки в `Task.md`.
2. **Инициализировать среду** `ogbench.make_env('antmaze-medium-navigate-v0')`.
3. **Извлечь датасет** переходов через `env.get_dataset()`.
4. **Построить $B$-граф** по датасету (подбор $k \in [10, 30]$ соседей и сэмплирование $N \approx 10,000$ состояний).
5. **Запустить оценку (Evaluation Loop)** на 5 случайных сидах для:
   - Single-Intention Baseline
   - Multi-Subgoal Graph Planner (Ours)
6. **Зафиксировать метрики:** Success Rate (%) и Среднее число шагов до цели.
7. **Оформить отчёт в LaTeX:** Скомпилировать через `lualatex main.tex` в папке `report/`.
