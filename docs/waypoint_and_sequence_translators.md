# Waypoint and Sequence-Aware Attention Translators

## 1. Context and Problem Formulation

In goal-conditioned hierarchical reinforcement learning and Forward-Backward (FB) representations:
- **Baseline Controller**: $\pi^h(s_t, B(g))$ generates a single intention $z$ aiming straight at the distant goal $g$. In complex mazes, Euclidean/latent attractions lead directly into dead-ends and walls because a single vector cannot represent the multi-hop sequence of corridors required to navigate the environment.
- **Topological Sequence Planning (Dijkstra)**: Constructing an offline reachability graph $G = (V, E)$ over dataset landmarks enables computing optimal multi-hop paths $\mathcal{P} = [w_1, w_2, \dots, w_K, g]$.
- **The Translator Objective**: Rather than passing raw waypoints through $\pi^h$ (which has a bottleneck and assumes direct Euclidean reachability), we learn high-level neural translators that directly reason over the planned sequence and emit the optimal command intention $z_{\text{cmd}}$ for the frozen low-level policy $\pi^\ell(a \mid s_t, z_{\text{cmd}})$.

---

## 2. Architectures

### A. Single Waypoint Gated Translator (`FlaxSingleWaypointTranslator`)
Maps the current state $s_t \in \mathbb{R}^{29}$ and the immediate topological lookahead waypoint $B(w_1) \in \mathbb{R}^{128}$ to the action intention $z_{\text{cmd}}$:
$$h_s = \text{MLP}(s_t), \quad h_w = \text{MLP}(B(w_1))$$
$$\gamma = \sigma(\text{Linear}([h_s, h_w]))$$
$$h_{\text{fused}} = \gamma \odot h_w + (1 - \gamma) \odot h_s$$
$$z_{\text{cmd}} = \text{ProjectToSphere}(\text{ResBlocks}(h_{\text{fused}}))$$

### B. Sequence-Aware Multi-Head Attention Transformer (`FlaxSequenceWaypointAttentionTranslator`)
Reasons over the entire multi-hop trajectory sequence $[B(w_1), B(w_2), \dots, B(w_K), B(g)]$:
1. **Positional Encoding**: Temporal order is injected via sinusoidal/learned embeddings $P_k \in \mathbb{R}^D$:
   $$E_k = \text{Linear}(B(w_k)) + P_k$$
2. **Bidirectional Trajectory Self-Attention**: Waypoints interact to capture multi-hop geometry (turns, narrow passages):
   $$H_w = \text{TransformerBlocks}(\{E_k\}_{k=1}^K, \text{mask})$$
3. **Cross-Attention with State Query**: The agent state query $Q_s = \text{Linear}(s_t)$ attends over the contextualized sequence:
   $$C_s = \text{MultiHeadCrossAttention}(Q = Q_s, K = H_w, V = H_w)$$
4. **Gated Residual Fusion & Sphere Projection**:
   $$z_{\text{cmd}} = \sqrt{d} \cdot \frac{\text{Head}(\text{Gate}(Q_s, C_s))}{\|\text{Head}(\text{Gate}(Q_s, C_s))\|_2}$$

---

## 3. Multi-Stage Trajectory Sampling Strategy

To eliminate distribution shift and ensure optimal performance from any stage of the episode:
1. **Variable Trajectory Offsets**: For each planned path, we uniformly sample step index $t \in [0, L]$. The network is trained on:
   - Early stages ($K \approx 16$ waypoints remaining)
   - Intermediate corridors ($K \approx 5..10$ waypoints remaining)
   - Terminal zones ($K \approx 1..2$ waypoints remaining)
2. **Explicit Terminal Anchor**: The final goal latent $B(g)$ is always guaranteed to be the final token in the sequence.
3. **Gaussian State Perturbation**: Noise $\xi \sim \mathcal{N}(0, \sigma^2)$ is added to states to train active drift recovery.

---

## 4. Differentiable Physics-Grounded Loss Formulation

The parameters $\theta$ are optimized end-to-end via JAX/XLA backpropagation:
$$\mathcal{L}(\theta) = \mathcal{L}_{\text{BC}}(\hat{z}, z^*) + \lambda_{\text{Action}} \|\pi^\ell(s_t, \hat{z}) - a^*\|^2 - \lambda_{\text{Reach}} (F(s_t, \hat{z})^\top \hat{z}) - \lambda_{\text{Goal}} (\hat{z}^\top B(w_1))$$
where:
- $\pi^\ell$ is the frozen low-level policy
- $F(s, z)$ is the frozen forward reachability representation
- $\nabla_\theta$ flows directly through $\pi^\ell$ and $F$ without approximations.

---

## 5. Robust Online Execution Mechanics (`EnhancedSequenceWaypointAttentionPlanner`)

During closed-loop rollout in complex environments (`antmaze-medium` and `antmaze-large`), the planner incorporates five critical execution mechanisms:
1. **Local Anti-Jump Window Tracking**: The current path index advances within a strictly bounded local neighborhood $[i - 2, i + 5]$, preventing Euclidean shortcut jumps across thin partition walls into parallel corridors.
2. **Dynamic Off-Track Re-Routing**: If external perturbation or collision causes deviation from the planned path ($d(s_t, \mathcal{P}) > 4.8\text{ m}$), the planner resets and computes a fresh Dijkstra trajectory from the current agent location.
3. **Graph Connectivity Safeguard**: Start and goal landmarks are selected from the top-15 Euclidean and top-15 cosine similarity candidates to guarantee a finite Dijkstra path ($\text{dist}(s, g) < \infty$), eliminating straight-line fallbacks through obstacle geometry.
4. **Curvature-Aware Multi-Waypoint Conditioning**: Slices valid lookahead horizons without terminal coordinate duplication, providing clean continuous $\cos \theta_k$ angle features for geometric turn negotiation.
5. **Adaptive Stagnation Breakout**: When displacement is $< 0.40\text{ m}$ over a 40-step window, the low-level actor samples with exploratory temperature $\tau = 0.25$ to overcome wall friction and navigate sharp 90-degree corners.

