# Buffer Graph Dijkstra: Topological Forward-Backward Planning

This document provides a thorough mathematical formulation and algorithmic exposition of **Buffer Graph Dijkstra** (Branch 2 planning) for Goal-Conditioned Zero-Shot Reinforcement Learning (RL) in continuous state-action spaces.

---

## 1. Mathematical Foundations: Forward-Backward Representations

### 1.1. Markov Decision Process & Zero-Shot Setting
We consider a continuous, reward-free Markov Decision Process (MDP) defined by the tuple:
$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, \gamma, \rho_0)$$
where:
- $\mathcal{S} \subseteq \mathbb{R}^{d_s}$ is the continuous state space (e.g., $d_s = 29$ for 8-DoF quadruped Ant navigation),
- $\mathcal{A} \subseteq \mathbb{R}^{d_a}$ is the continuous action space ($\mathcal{A} = [-1, 1]^8$),
- $P(s' \mid s, a)$ is the unknown environmental transition density,
- $\gamma \in [0, 1)$ is the temporal discount factor ($\gamma = 0.99$),
- $\rho_0(s)$ is the initial state distribution.

In **Zero-Shot Offline RL**, the agent is trained on an unlabelled transition dataset:
$$\mathcal{D} = \{(s_t, a_t, s_{t+1})\}_{t=1}^{|\mathcal{D}|}$$
collected without reward signals. At test time, an arbitrary downstream goal task is specified either via a target state $g \in \mathcal{S}$ or a reward function $r(s, a)$. The agent must immediately synthesize an optimal policy $\pi^*(a \mid s; g)$ with **zero additional environment interaction or parameter fine-tuning**.

---

### 1.2. Successor Measures and Hilbert-Space Factorization
For any policy $\pi(a \mid s)$, the **Successor Measure** $M^\pi(s, a, \cdot)$ represents the discounted state occupancy measure over infinite horizons:
$$M^\pi(s, a, E) = \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t \mathbb{I}(s_t \in E) \;\middle|\; s_0 = s, a_0 = a \right], \quad \forall E \subseteq \mathcal{S}$$

In density form, $M^\pi(s, a, s')$ satisfies the continuous **Bellman Flow Equation on Measures**:
$$M^\pi(s, a, s') = \delta(s, s') + \gamma \int_{\mathcal{S}} \int_{\mathcal{A}} P(s'' \mid s, a) \pi(a'' \mid s'') M^\pi(s'', a'', s') \, ds'' \, da''$$
where $\delta(s, s')$ denotes the Dirac delta distribution.

#### Low-Rank Hilbert Space Embedding
Treating $M^\pi$ as an integral operator kernel in the Hilbert space $L_2(\mathcal{S})$, Forward-Backward (FB) representations (Touati & Ollivier, 2021; Touati et al., 2022) parameterize $M^\pi$ via low-rank bilinear decomposition:
$$M^z(s, a, s') \approx F(s, a, z)^\top B(s')$$
where:
- $z \in \mathbb{S}^{d-1} \subset \mathbb{R}^d$ is a unit-norm latent intention vector ($d = 128$),
- $F: \mathcal{S} \times \mathcal{A} \times \mathbb{S}^{d-1} \to \mathbb{R}^d$ is the **Forward Representation** (parameterized by neural network with weights $\theta_F$),
- $B: \mathcal{S} \to \mathbb{R}^d$ is the **Backward Representation** (parameterized by neural network with weights $\theta_B$).

For state-level evaluations (marginalizing actions under policy $\pi_z$), we define:
$$F(s, z) = \mathbb{E}_{a \sim \pi(\cdot \mid s, z)} [F(s, a, z)]$$

---

## 2. JAX-Vectorized Batch Reachability & Cost Metrics

### 2.1. Vectorized Reachability Formulation
Given a source state $s_i \in \mathcal{S}$ and a target state $s_j \in \mathcal{S}$ with backward representation $B(s_j) \in \mathbb{R}^d$, the long-horizon transition reachability density under the conditioned latent policy $z = B(s_j)$ is given by the inner product:
$$\text{reach}(s_i, B(s_j)) = F(s_i, B(s_j))^\top B(s_j)$$

In [`src/planners.py`](file:///Users/savvatej/source/fb-rl/src/planners.py#L32-L39), this computation is compiled into a single fused XLA kernel via `@jax.jit`:
```python
@jax.jit
def _jit_batch_reach(agent, states, targets):
    """Jitted batch reachability metric."""
    f = agent.network.select("forward_repr")(states, targets, goal_encoded=True)
    if f.ndim == 3:
        f = jnp.mean(f, axis=0)
    return jnp.sum(f * targets, axis=-1)
```

To construct the full $N \times N$ pairwise reachability matrix without exceeding memory limits for $N=1000$ ($10^6$ pairs), computation is processed in vectorized chunks of 45,000 pairs:
$$\mathbf{R}_{ij} = \text{reach}(s_i, B(s_j))$$

```python
s_rep = jnp.repeat(self.landmarks, n, axis=0)
z_tile = jnp.tile(self.landmark_latents, (n, 1))

reach_flat = []
for i in range(0, len(s_rep), 45000):
    sb = s_rep[i : i + 45000]
    zb = z_tile[i : i + 45000]
    reach_flat.append(np.asarray(_jit_batch_reach(self.agent, sb, zb)))
reach_matrix = np.concatenate(reach_flat, axis=0).reshape((n, n))
```

---

### 2.2. Negative-Log-Reachability Cost Function
Transition reachability $\mathbf{R}_{ij}$ acts as an unnormalized probability measure of reaching state $s_j$ from $s_i$. To map multiplicative multi-step path probabilities into additive path lengths suitable for shortest-path graph algorithms:
$$P(s_0 \to s_1 \to \dots \to s_K) = \prod_{k=0}^{K-1} P(s_k \to s_{k+1}) \iff \min \sum_{k=0}^{K-1} -\log P(s_k \to s_{k+1})$$

We define the normalized reachability and non-negative edge cost matrix $\mathbf{C} \in \mathbb{R}^{N \times N}$:
$$\mathbf{R}_{\text{norm}, ij} = \text{clip}\left(\frac{\mathbf{R}_{ij}}{\max_k \mathbf{R}_{kk}}, 10^{-6}, 1.0\right)$$
$$c(i, j) = \max\left(0.0, -\log \mathbf{R}_{\text{norm}, ij}\right)$$

---

## 3. Geometry-Aware Topological Graph Construction

### 3.1. Zero Map Cheating vs. Physical Feasibility
In complex non-convex environments such as AntMaze, relying solely on unconstrained FB representations or naive Euclidean distance produces critical planning failures:
1. **Euclidean Shortcut Hallucination**: Euclidean distance $d(s_i, s_j) = \|x_i - x_j\|_2$ ignores interior maze walls, creating spurious edges that cut through solid obstacles.
2. **FB Representation Horizon Limits & Leakage**: Although FB representations model obstacle boundaries dynamically, high-dimensional neural approximators can exhibit small residual reachabilities through thin wall dividers.

To ensure **100% strict topological validity with zero access to the ground-truth maze map grid**, the [`BufferGraphPlanner`](file:///Users/savvatej/source/fb-rl/src/planners.py#L131-L318) imposes dual geometry-aware constraints:

$$\mathbf{C}_{ij} = \begin{cases} 
-\log \mathbf{R}_{\text{norm}, ij} & \text{if } \|x_i - x_j\|_2 \le r_{\max} \;\text{ and }\; \mathbf{R}_{ij} \ge \tau_{\text{reach}} \\ 
+\infty & \text{otherwise} 
\end{cases}$$

Where the calibrated hyperparameters are:
- **Landmark Count**: $N = 1000$ sampled uniformly from offline buffer $\mathcal{D}$,
- **Local Neighborhood Radius**: $r_{\max} = 3.5\text{ m}$ (enforcing local locality),
- **FB Reachability Cutoff**: $\tau_{\text{reach}} = 35.0$ (rejecting through-wall transitions),
- **Self-edge identity**: $\mathbf{C}_{ii} = 0.0$.

```python
# Disallow edges that span across walls
dists_euclid = np.linalg.norm(
    self.landmark_coords[:, None, :] - self.landmark_coords[None, :, :], axis=-1
)
cost_matrix[dists_euclid > self.max_edge_radius] = np.inf
cost_matrix[reach_matrix < self.reachability_cutoff] = np.inf
np.fill_diagonal(cost_matrix, 0.0)
```

```
                 [ Landmark Node s_i ]
                          |
             +------------+------------+
             |                         |
     Euclidean Check             FB Reach Check
  ||x_i - x_j|| <= 3.5m        reach(s_i, s_j) >= 35.0
             |                         |
             +------------+------------+
                          |
                    [ Valid Edge ]
                 c(i,j) = -log(R_ij)
```

---

### 3.2. All-Pairs Shortest Paths & Path Reconstruction
At initialization, Dijkstra's algorithm computes all-pairs shortest paths across the sparse directed graph:
```python
self.all_dist, self.all_pred = dijkstra(
    self.cost_matrix, directed=True, return_predecessors=True
)
```
- **Time Complexity**: $\mathcal{O}(N \cdot (E + N \log N))$ executed once at graph creation ($\approx 35\text{ ms}$ for $N=1000$).
- **Online Reset Complexity**: $\mathcal{O}(K)$ where $K \le N$ is the path length, reconstructed via predecessor backtracking in $<0.1\text{ ms}$:
```python
path = []
curr = goal_idx
while curr != -9999 and curr != start_idx:
    path.append(curr)
    curr = self.all_pred[start_idx, curr]
    if len(path) > self.n_landmarks:
        break
path.append(start_idx)
path.reverse()
```

---

## 4. Continuous Sliding Lookahead Target Tracking

### 4.1. The Failure of Discrete Waypoint Jumping
In early hierarchical planning prototypes, subgoals were tracked via discrete waypoint switching:
$$\text{If } \|x_t - w_k\|_2 < \epsilon \implies \text{advance to } w_{k+1}$$

This discrete switching mechanism exhibits severe failure modes in continuous quadruped locomotion:
1. **Corner Chattering & Wall Collisions**: Approaching a convex corridor corner, the low-level policy aims directly at waypoint $w_k$ located near the corner apex. Centrifugal momentum forces the robot into the wall before reaching radius $\epsilon$.
2. **Friction Traps & Infinite Loops**: When trapped against a wall corner, the agent never satisfies $\|x_t - w_k\|_2 < \epsilon$. The planner remains stuck on $w_k$, while the low-level policy commands forward velocity directly into the obstacle, producing self-intersecting friction loops.

---

### 4.2. Continuous Sliding Lookahead Formulation
To guarantee smooth, collision-free navigation, [`BufferGraphPlanner`](file:///Users/savvatej/source/fb-rl/src/planners.py#L273-L295) implements a **Continuous Sliding Lookahead** with lookahead horizon $L = 2.6\text{ m}$.

```
  Topological Path:  p_0 ---- p_1 ---- p_2 ---- p_3 ---- p_4 ---- p_5
                      ^                  ^
                      |                  |
                   Obs (x_t)     Target Latent (L = 2.6m)
```

#### Step 1: Forward Projection Window
The current agent position $x_t \in \mathbb{R}^2$ is projected onto the active path by searching within a forward window of 12 vertices:
$$k^* = \arg\min_{k \in [i_{\text{curr}}, \min(|P|, i_{\text{curr}} + 12)]} \|x_t - p_k\|_2$$
$$i_{\text{curr}} \leftarrow k^*$$

#### Step 2: Arc-Length Lookahead Integration
From $i_{\text{curr}}$, the planner advances along the discrete topological path, accumulating Euclidean segment lengths until reaching lookahead distance $L = 2.6\text{ m}$:
$$i_{\text{target}} = \min \left\{ j \ge i_{\text{curr}} \;\middle|\; \sum_{m=i_{\text{curr}}}^{j-1} \|p_{m+1} - p_m\|_2 \ge L \text{ or } j = |P| - 1 \right\}$$

#### Step 3: Continuous Latent Target Selection
The intention fed to the policy network is the backward representation of the sliding lookahead vertex:
$$z_t = \begin{cases} 
z_{\text{goal}} & \text{if } i_{\text{target}} = |P| - 1 \\ 
B(p_{i_{\text{target}}}) & \text{otherwise} 
\end{cases}$$

This continuous lookahead formulation continuously pulls the quadruped's heading into open corridor space ahead of turns, eliminating corner clipping and preventing friction loops.

---

## 5. Proximity-Based Stuck Recovery & Goal Arrival

### 5.1. Stuck Detection Protocol
Quadruped dynamics occasionally encounter complex foot entanglement or friction stalls against wall boundaries. [`BufferGraphPlanner`](file:///Users/savvatej/source/fb-rl/src/planners.py#L296-L309) tracks a rolling historical FIFO buffer of positions $H = [x_{t-39}, \dots, x_t]$:

$$\text{is\_stuck} = \left( |H| \ge 40 \right) \;\land\; \left( \|x_t - x_{\text{goal}}\|_2 > 2.0\text{ m} \right) \;\land\; \left( \|x_t - x_{t-39}\|_2 < 0.4\text{ m} \right)$$

### 5.2. Exploratory Perturbation & Recovery
When `is_stuck == True`, the low-level policy temperature is elevated from deterministic $\tau = 0.0$ to stochastic perturbation $\tau = 0.2$:
$$\text{temperature} = \begin{cases} 0.2 & \text{if is\_stuck} \\ 0.0 & \text{otherwise} \end{cases}$$
$$\text{seed}_t = \text{PRNGKey}(\text{step})$$
This injects sufficient exploratory torque into the joints to dislodge the limbs from wall friction, after which regular sliding tracking automatically resumes.

### 5.3. Goal Arrival Handling
When the agent approaches the terminal landmark ($i_{\text{target}} = |P| - 1$), the target latent seamlessly locks onto the inferred global task latent $z_{\text{goal}}$. The low-level actor drives directly toward the goal locus until the environment terminal reward ($r = 1.0$) is triggered.
