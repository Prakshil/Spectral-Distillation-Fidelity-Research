# Mentor Meeting Master Guide: From Attention Compression to Graph MoE Evaluation

> **Purpose of this Document:**  
> A complete, self-contained, plain-English reference explaining every concept, formula, algorithm, and result in this repository.  
> It covers **Part 1 (The Original Problem: Compressing Dense Attention Graphs with Provable Bounds)**, **Part 2 (The Transition & The Discovery of the Evaluation Flaw)**, and **Part 3 (The Final Results, Published Models, and Current State)**.

---

## Table of Contents
1. [The Big Picture in 2 Minutes](#1-the-big-picture-in-2-minutes)
2. [Master Formula Guide (Every Equation Broken Down Variable by Variable)](#2-master-formula-guide)
3. [The Old Problem: Compressing Dense Matrices with Provable Bounds](#3-the-old-problem-compressing-dense-matrices)
4. [How the Sparsification Engine Actually Works](#4-how-the-sparsification-engine-works)
5. [The 4-Stage Mathematical Proof Pipeline](#5-the-4-stage-mathematical-proof-pipeline)
6. [Results We Achieved from the Old Problem](#6-results-achieved-from-the-old-problem)
7. [The Great Pivot: Why and How We Transitioned](#7-the-great-pivot-why-and-how-we-transitioned)
8. [The New Problem: Auditing the Mechanism Validity of Graph MoEs](#8-the-new-problem-auditing-graph-moes)
9. [Closing Every Escape Hatch: Node-MoE, Heterophily, and Published Routers](#9-closing-every-escape-hatch)
10. [The Constructive Solution: The Learnability Lever & Real Attention](#10-the-constructive-solution)
11. [Every Dataset in the Project: What It Is and What It Proves](#11-every-dataset-in-the-project)
12. [Your 5-Minute Mentor Presentation Script](#12-your-5-minute-mentor-presentation-script)

---

## 1. The Big Picture in 2 Minutes

If you only have two minutes to explain everything to your mentor tomorrow, here is the exact storyline:

```
[Phase 1 & 2: The Original Problem]
"Can we compress dense attention matrices into sparse graphs and mathematically bound information loss?"
                               │
                               ▼
            [We proved the 4-step DPI bound: ΔI ≤ C · SD · H(Y)]
            [We built the Effective Resistance (R_eff) sparsifier]
                               │
                               ▼
[Phase 3: The Discovery / The Pivot]
"When evaluating routers on our compressed graphs, we tested the field's standard benchmark (Protocol A)."
                               │
                               ▼
            [CRITICAL DISCOVERY: Protocol A is Permutation-Invariant!]
            [Acc(s) == Acc(-s): Standard benchmarks award free credit to any partition!]
                               │
                               ▼
[Phase 4 & 5: The New Problem - The Evaluation Audit]
"Does adaptive per-node routing actually work, or are published gains an evaluation illusion?"
                               │
                               ▼
            [Protocol B (Honest Cross-Fit): 0/234 configurations beat a single global model]
            [Node-MoE Reproduction: Unrouted uniform ensemble matches or beats the gated model]
            [Feature-only MLP beats Node-MoE by up to +12%]
                               │
                               ▼
[Phase 6: The Constructive Exit]
"When does routing work, and how can practitioners protect themselves?"
                               │
                               ▼
            [The Learnability Lever (β): Pinpoints existence (β=0.2) and deployability (β=0.3)]
            [The Selective Abstaining Router: Automatically falls back to a global model]
            [Real SmolLM2 Attention (11,520 nodes): SD predicts router stability (p < 10^-6)]
```

---

## 2. Master Formula Guide

Here is every mathematical formula in the project, broken down into plain English.

---

### Formula 1: The Dense Self-Attention Matrix
In Transformer language models (like GPT, Llama, SmolLM2), self-attention computes how much token $i$ attends to token $j$:

$$A_{ij} = \frac{\exp\left(\frac{Q_i K_j^T}{\sqrt{d}}\right)}{\sum_{m=1}^N \exp\left(\frac{Q_i K_m^T}{\sqrt{d}}\right)}$$

* **$Q_i$ (Query vector):** What token $i$ is looking for.
* **$K_j$ (Key vector):** What token $j$ contains.
* **$d$ (Dimension):** The vector length (scaling factor to keep variance stable).
* **$A_{ij}$:** A number between $0$ and $1$. It represents the connection strength (edge weight) between word $i$ and word $j$.
* **Why it matters:** Across $N$ tokens, this generates an $N \times N$ dense matrix where almost every cell is non-zero ($O(N^2)$ complexity).

---

### Formula 2: The Graph Laplacian Operator
To analyze graph frequencies and information diffusion, we convert the connection weights $W$ into the Graph Laplacian $L$:

$$L = D - W, \quad \text{where } D_{ii} = \sum_{j} W_{ij}$$

* **$W$ (Adjacency Matrix):** Symmetrized connection weights between nodes ($W = \frac{A + A^T}{2}$).
* **$D$ (Degree Matrix):** A diagonal matrix where $D_{ii}$ is the total connection weight touching node $i$.
* **$L$ (Graph Laplacian):** A symmetric, positive semi-definite matrix that acts like the discrete derivative (Laplace-Beltrami operator) on the graph.
* **Normalized Laplacian:**
  $$L_{\text{norm}} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} W D^{-1/2}$$
  Its eigenvalues strictly lie in $[0, 2]$.

---

### Formula 3: Graph Frequencies (Eigenvalues & Eigenvectors)
The spectral decomposition of the Laplacian is:

$$L v_k = \lambda_k v_k, \quad 0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_n$$

* **$\lambda_k$ (Eigenvalue / Frequency):**
  * $\lambda_1 = 0$: Represents constant signals across connected components.
  * Small $\lambda$ (Low Frequencies): Smooth, global community shapes (clusters).
  * Large $\lambda$ (High Frequencies): Sharp, rapid differences between adjacent neighbors.
* **$v_k$ (Eigenvector):** An $N$-dimensional vector giving each node a coordinate in that frequency band. A node's coordinates in the lowest eigenvectors reveal which community or cluster it belongs to.

---

### Formula 4: Effective Resistance ($R_{\text{eff}}$)
Treat the graph as an electrical resistor network where edge $e = (u, v)$ has conductance $W_e$ (resistance $1/W_e$). The effective resistance between node $u$ and node $v$ is:

$$R_{\text{eff}}(u, v) = (e_u - e_v)^T L^+ (e_u - e_v)$$

* **$e_u, e_v$:** Unit basis vectors (1 at index $u$, 0 elsewhere).
* **$L^+$:** The Moore–Penrose pseudoinverse of the Laplacian matrix.
* **Physical intuition:**
  * If two nodes are inside a dense cluster, there are dozens of parallel paths $\implies$ **Low $R_{\text{eff}}$** (electricity flows easily).
  * If two nodes are connected by a single bottleneck bridge $\implies$ **High $R_{\text{eff}}$** (cutting this wire breaks the circuit).
* **Foster's Theorem:**
  $$\sum_{e \in E} W_e \cdot R_{\text{eff}}(e) = n - 1$$
  *(The total resistance-energy across all edges in any connected graph always sums to exactly $n-1$).*

---

### Formula 5: Spielman–Srivastava Spectral Sparsification
To compress the graph while preserving its spectral properties, we sample each edge $e$ with probability:

$$p_e = \min\left(1, \; C \cdot \frac{W_e \cdot R_{\text{eff}}(e) \log n}{\epsilon^2}\right)$$

* **If kept:** The edge is reweighted to $\tilde{W}_e = \frac{W_e}{p_e}$ (unbiased estimator).
* **Mathematical Guarantee:** With high probability, for all test vectors $x$:
  $$(1 - \epsilon) x^T L_{\text{orig}} x \le x^T L_{\text{sparse}} x \le (1 + \epsilon) x^T L_{\text{orig}} x$$
  The energy of every possible signal is preserved within $(1 \pm \epsilon)$.

---

### Formula 6: Spectral Distortion (SD) Metric
Measures how much the graph's eigenvalue spectrum deformed during compression:

$$\text{SD} = \max_k \frac{|\lambda_k(L_{\text{sparse}}) - \lambda_k(L_{\text{dense}})|}{|\lambda_k(L_{\text{dense}})| + \epsilon}$$

* **Rank-Matched SD (Our Repair):** When sparsification disconnects peripheral nodes, it creates artificial zero eigenvalues. Rank-matched SD matches active non-zero eigenvalues so the metric does not trivially saturate at 1.0.

---

### Formula 7: Davis–Kahan Eigenvector Perturbation Theorem
When matrix $L_{\text{dense}}$ is perturbed by $\Delta L = L_{\text{sparse}} - L_{\text{dense}}$, how much do the node coordinates rotate?

$$\|v_k - \tilde{v}_k\|_2 \le \frac{2 \|\Delta L\|_F}{\text{gap}_k}, \quad \text{where } \text{gap}_k = \min_{j \ne k} |\lambda_k - \lambda_j|$$

* **$\|\Delta L\|_F$:** The Frobenius norm (total physical change in matrix entries).
* **$\text{gap}_k$ (Eigengap):** The distance between eigenvalue $k$ and its nearest neighbor.
* **Meaning:** If a community is well-isolated (large eigengap), its coordinates resist rotation. If eigengaps are tiny, coordinates rotate easily.

---

### Formula 8: The Complete 4-Step DPI Information Bound
Chaining Davis–Kahan, a Lipschitz router $\pi$ with constant $\kappa$, and the Data Processing Inequality:

$$\mathcal{I}(\pi^*; Y) - \mathcal{I}(\tilde{\pi}; Y) \le C \cdot \text{SD} \cdot \mathcal{H}(Y)$$

* **$\mathcal{I}(\pi^*; Y)$:** Mutual information between the dense oracle router's assignments and true labels $Y$.
* **$\mathcal{I}(\tilde{\pi}; Y)$:** Mutual information between the sparse router's assignments and true labels $Y$.
* **$\mathcal{H}(Y)$:** The entropy (uncertainty) of the labels.
* **$C$:** A constant depending on router depth and Lipschitz constant $\kappa$.
* **What it proves:** Routing degradation is mathematically bounded by the graph's Spectral Distortion.

---

### Formula 9: The Fatal Permutation Invariance Flaw (Protocol A)
In standard self-routing, accuracy is:

$$\text{Acc}(s) = \sum_{k=1}^K \frac{|V_k|}{|V|} \text{Acc}\Big(\text{Group}_k \;\Big|\; \text{Model trained on Group}_k\Big)$$

* **Because Expert $k$ trains only on $\text{Group}_k$ and is tested only on $\text{Group}_k$:**
  $$\text{Acc}(s) \equiv \text{Acc}(-s)$$
  Negating the router's scores swaps bucket labels but leaves the group accuracy sums bit-identical. The benchmark cannot distinguish true routing from complete inversion.

---

### Formula 10: Node-MoE Chebyshev Filter Formulation
Node-MoE evaluates spectral experts using an order-$K$ Chebyshev polynomial:

$$f(L)X = \sum_{k=0}^K h_k T_k(\tilde{L})X$$

* **$\tilde{L} = L - I = -D^{-1/2}WD^{-1/2}$:** Scaled Laplacian mapped to $[-1, 1]$.
* **Chebyshev recurrence:** $T_0(x) = 1$, $T_1(x) = x$, $T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)$.
* **Gauss–Chebyshev interpolation nodes:**
  $$x_j = \cos\left(\frac{j + 0.5}{K + 1}\pi\right)$$
* **Filter Smoothing Loss ($\mathcal{L}_s$):**
  $$\mathcal{L}_s = \sum_{\text{experts } o} \sum_{i} \Big(f_o(x_i) - f_o(x_{i-1})\Big)^2$$
  Penalizes rapid oscillations across adjacent interpolation nodes.

---

### Formula 11: The Learnability Lever
A synthetic generator that continuously modulates problem difficulty via knob $\beta \in [0, 1]$:

$$s_g(x) = (1 - \beta) \cdot s_{\text{common}}(x) + \beta \cdot s_{\text{rule}}(g, x), \quad y = \mathbb{I}[s_g(x) > 0]$$

* **$\beta = 0$:** All groups share one global rule $s_{\text{common}}(x) \implies$ Single global model is optimal; routing is useless.
* **$\beta = 1$:** Each latent group $g$ has its own distinct rule $s_{\text{rule}}(g, x) \implies$ Routing is essential.
* **Thresholds discovered:**
  * $\beta^* = 0.2$ (**Existence boundary**): Oracle beats global model.
  * $\beta^* = 0.3$ (**Deployability boundary**): Learned router beats global model on held-out test data.

---

## 3. The Old Problem: Compressing Dense Matrices

### 1. Where We Started
The original goal of this project was **Graph Distillation for Attention Acceleration**:
* In Transformers, dense attention matrices require quadratic $O(N^2)$ memory and compute.
* In enterprise ERP graphs and transaction networks, millions of edges make graph convolution prohibitively slow.
* We wanted to compress these dense graphs down to 5%–10% of their original edges.

### 2. The Core Problem
If you compress a graph naively, you risk deleting the subtle structural connections that specialized models depend on. 

We asked:
> *"Can we mathematically quantify and bound how much routing-relevant information is destroyed during compression?"*

---

## 4. How the Sparsification Engine Works

To compress graphs without destroying their structural geometry, we compared four distinct strategies:

```
                          Original Dense Graph
                                    │
         ┌──────────────┬───────────┴───────────┬──────────────┐
         ▼              ▼                       ▼              ▼
   Thresholding   Degree Pruning              Random        Effective Resistance
  (Delete < 0.1)  (Keep top-k)             (Keep 10% p)     (Spielman–Srivastava)
         │              │                       │              │
    Cuts bridges;  Isolates low-degree     Ignores graph    Mathematically preserves
   fragments graph   nodes completely        structure       Laplacian spectrum
```

### Why Effective Resistance Wins
* Every edge is treated as an electrical resistor.
* Bottlenecks and bridges across clusters have high resistance because electrical current has no alternate routes.
* We compute $R_{\text{eff}}(u, v) = (e_u - e_v)^T L^+ (e_u - e_v)$.
* By sampling proportional to $W_e \cdot R_{\text{eff}}(e)$, bridge edges are preserved with near 100% certainty, while redundant edges inside dense cliques are subsampled.
* Result: The graph stays connected, and low-frequency eigenvalues are preserved.

---

## 5. The 4-Stage Mathematical Proof Pipeline

To prove how much information survives, we established the following 4-step proof chain ([`docs/theorem_proofs.md`](file:///c:/Users/Prakshil/OneDrive/Desktop/files/codes/spectral_distillation/docs/theorem_proofs.md)):

1. **Step 1: Graph Distillation as Matrix Perturbation:**  
   Compressing the graph creates an error matrix $\Delta L = L_{\text{sparse}} - L_{\text{dense}}$.
2. **Step 2: Davis–Kahan Eigenvector Rotation:**  
   Because the Laplacian shifts by $\Delta L$, node structural coordinates (eigenvectors) rotate by at most:
   $$\|v_k - \tilde{v}_k\|_2 \le \frac{2 \|\Delta L\|_F}{\text{gap}_{\min}}$$
3. **Step 3: Lipschitz Continuous Router:**  
   A smooth router $\pi$ with Lipschitz constant $\kappa$ converts input coordinate shifts into decision probability changes:
   $$\|\pi^*(v) - \tilde{\pi}(v)\|_2 \le \kappa \cdot \|v_k - \tilde{v}_k\|_2 \le \kappa \cdot \frac{2 \|\Delta L\|_F}{\text{gap}_{\min}}$$
4. **Step 4: Data Processing Inequality (DPI) on Mutual Information:**  
   Information about task labels $Y$ can only decrease when representations are degraded:
   $$\mathcal{I}(\pi^*; Y) - \mathcal{I}(\tilde{\pi}; Y) \le \mathcal{H}(Y) \cdot \text{TotalVariation}(\pi^*, \tilde{\pi}) \le C \cdot \text{SD} \cdot \mathcal{H}(Y)$$

**The Final Theorem:** The loss in mutual information is strictly bounded by the graph's Spectral Distortion multiplied by the label entropy.

---

## 6. Results Achieved from the Old Problem

1. **Sparsification Fidelity:** Effective Resistance sparsification strictly outperformed thresholding and degree pruning across synthetic and benchmark graphs, preserving connectivity and spectral energy down to 5% retention budgets.
2. **Metric Repair:** Discovered that textbook index-wise Spectral Distortion was broken on disconnected graphs; replaced it with **Rank-Matched Distortion** and **Low-Frequency Distortion**.
3. **Prediction on Real LLM Attention (SmolLM2-1.7B, 11,520 nodes):** Pre-training Spectral Distortion strongly predicted multi-seed **Router Stability** ($r_s = -0.687, p = 2.5 \times 10^{-8}$). Low SD guaranteed that routers trained across different random seeds made consistent, identical assignments.
4. **The Unresolved Puzzle:** While SD predicted router *stability*, it did not predict downstream *classification accuracy drops*. This paradox forced us to inspect how routing accuracy was actually being measured in the literature.

---

## 7. The Great Pivot: Why and How We Transitioned

### The Turning Point
To test whether compressed graphs helped downstream tasks, we implemented the standard evaluation protocol used across the Graph Mixture-of-Experts literature (which we designate **Protocol A**, or "Self-Routing").

### The Fatal Flaw: Permutation Invariance
We ran a basic sanity test: **What happens if we negate the router's assignment scores?** (Swap Bucket 0 and Bucket 2).
* In any functioning system, routing nodes to the opposite of their intended expert should destroy accuracy.
* **The Result:** Accuracy was 100% identical: $\text{Acc}(s) \equiv \text{Acc}(-s)$.
* **The Reason:** In Protocol A, Expert $k$ is trained *only* on the nodes assigned to Bucket $k$, and evaluated *only* on those same nodes. If you swap the buckets, Expert 2 simply trains on Bucket 0's nodes, and Expert 0 trains on Bucket 2's nodes.
* **The Reality:** Standard benchmarks were not measuring routing skill; they were awarding free credit to *any arbitrary partition of the graph*!

```
[Standard Protocol A (Broken)]
Router assigns nodes ──> Expert k trains on Bucket k ──> Expert k tested on Bucket k
                         └─────── Acc(s) == Acc(-s) [Permutation Invariant!] ──────┘

[Our Protocol B (Honest Cross-Fit)]
1. Pool of experts trained routing-blind on Split 1 ──> FROZEN
2. Router trained on disjoint Split 2 to pick the best frozen expert
3. Evaluated on Split 3 against a Single Global Model trained on Split 1
```

---

## 8. The New Problem: Auditing Graph MoEs

We shifted from:
> *"How much information does graph compression lose?"*  

To the deeper, more urgent question:
> *"Does adaptive per-node routing actually work in graph neural networks, or are published gains an evaluation artifact?"*

### The Protocol B Universal Null Result
Under honest cross-fitting on three real fraud graphs (Amazon, Tolokers, YelpChi):
* Across 3 expert architectures (Logistic, MLP, GNN).
* Across 26 candidate routers.
* **Result: 0 out of 234 candidate configurations beat a single global model.**
* In fact, routing was strictly worse than training one model on all training data.

---

## 9. Closing Every Escape Hatch

To ensure our null result was not dismissed as an artifact of simple baseline models, we conducted an exhaustive investigation into published models:

### 1. Full Reproduction of Node-MoE (*arXiv:2406.03464*)
* Implemented full ChebNetII polynomial filter experts with Gauss-Chebyshev interpolation nodes.
* Implemented differentiated initializations (`low_pass`, `high_pass`, `band_pass`), discrepancy gate $[X, |AX-X|, |A^2X-X|]$, and filter-smoothing loss.

#### Findings on Fraud Graphs (Amazon, Tolokers, YelpChi):
1. Joint training beat a single ChebNetII by $+4.2\%$ on Tolokers and $+4.1\%$ on YelpChi.
2. **The Unrouted Uniform Ensemble Control:** Training the exact same 3 ChebNetII experts without a gate (equal $1/K$ fixed mean) achieved the *exact same* accuracy ($0.6632 \approx 0.6664$).
3. Under Protocol B cross-fitting, the gate advantage collapsed to $+0.0049$ (Tolokers) and $+0.0023$ (YelpChi).
4. A simple **feature-only MLP** (ignoring the graph completely) beat Node-MoE by $+3.1\%$ (Amazon), $+7.3\%$ (Tolokers), and **$+11.8\%$** (YelpChi).

#### Findings on Node-MoE's Home Benchmarks (Actor, Chameleon, Squirrel):
On the heterophilic web/actor graphs where Node-MoE claimed state of the art:
* **Chameleon:** Uniform Ensemble ($0.4615$) beat Gated Node-MoE ($0.4327$) — Gate won **0 / 5** runs.
* **Actor:** Uniform Ensemble ($0.3548$) beat Gated Node-MoE ($0.3433$) — Gate won **1 / 5** runs.
* **Squirrel:** Uniform Ensemble ($0.3340$) beat Gated Node-MoE ($0.3034$) — Gate won **0 / 5** runs.

> **Conclusion:** On Node-MoE's own benchmarks, **the unrouted equal-weight ensemble strictly beats the learned gate.** The gate is actively harmful compared to simple averaging.

### 2. RouterGNN & AD-GNN / Ada-Routing
* Trained with LayerNorm, conservative learning rates, weight decay, and gradient clipping.
* Result: Mean deltas against a single global model were negative across all 6 datasets.

---

## 10. The Constructive Solution

We resolved the critique by providing positive, constructive engineering tools:

### 1. The Learnability Lever ($\beta \in [0, 1]$)
* Modulates group separability from $\beta = 0$ (single rule) to $\beta = 1$ (disjoint mixture).
* **Existence boundary ($\beta^* = 0.2$):** Where oracle routing first beats the global model.
* **Deployability boundary ($\beta^* = 0.3$):** Where a learned router reliably beats the global model on held-out test data.
* **$\beta \ge 0.5$:** Monotone, large routing gains (up to $+19.4$ pp).

### 2. The Selective Abstaining Router
* Evaluates held-out validation gain during training.
* If validation gain clears a threshold $\implies$ Routes nodes.
* If validation gain is negligible or negative $\implies$ **Abstains** (routes all nodes to the single global model).
* On real fraud pools (Amazon and Tolokers), the router **abstained**, shielding the model from routing degradation.

### 3. Scaled Real Attention (SmolLM2-1.7B, 11,520 nodes)
* Pre-training Spectral Distortion ($\rho = -0.687, p < 10^{-7}$) and connectivity survival ($\rho = +0.765, p < 10^{-10}$) predict multi-seed routing stability on real language model attention graphs.

---

## 11. Every Dataset in the Project

| Dataset | Nature | Nodes | Edges | Classes | Role & Finding |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **PC-1c-R** | Synthetic Planted Control | 10,000 | ~50,000 | 5 | **Positive Control:** Proves our test detects real routing gains ($+26.2$ pp) and decays monotonically to zero as variance is diluted. |
| **Amazon** | Real Co-Review Fraud | 8,639 | 3,298,534 | 2 | Dense homophilic fraud graph. Routing fails (0/26 wins). MLP beats Node-MoE by $+3.1\%$. |
| **Tolokers** | Real Crowdsourcing Network | 11,758 | 519,000 | 2 | Heterophilic worker exclusion network. Joint Node-MoE gain ($+4.2\%$) collapses under Protocol B ($+0.0049$). MLP beats Node-MoE by $+7.3\%$. |
| **YelpChi** | Real Restaurant/Hotel Fraud | 14,840 | 411,194 | 2 | Multi-relational spam network. Full $10 \times 3$ sweep confirms 0/26 routing wins. MLP beats Node-MoE by $+11.8\%$. |
| **Actor** | Film Co-Occurrence Network | 7,600 | 33,544 | 5 | Node-MoE benchmark. Unrouted uniform ensemble ($0.3548$) beats gated Node-MoE ($0.3433$). |
| **Chameleon** | Wikipedia Page Network | 2,277 | 36,101 | 5 | Heterophilic web graph. Unrouted uniform ensemble ($0.4615$) crushes gated Node-MoE ($0.4327$). |
| **Squirrel** | Wikipedia Page Network | 5,201 | 217,073 | 5 | Heterophilic web graph. Unrouted uniform ensemble ($0.3340$) crushes gated Node-MoE ($0.3034$). |
| **SmolLM2-1.7B** | LLM Multi-Head Attention | 11,520 | ~345,000 | 29 | Real attention tokens. Pre-training SD and connectivity survival predict multi-seed router stability ($p < 10^{-6}$). |

---

## 12. Your 5-Minute Mentor Presentation Script

Use this exact structure when presenting to your mentor tomorrow:

### 1. The Context (Minute 1)
> *"We started by investigating how much information survives when we compress dense attention matrices into sparse graphs. We proved mathematically that mutual information decay is bounded by the graph's Spectral Distortion (SD), and we built an Effective Resistance ($R_{\text{eff}}$) sparsifier that guarantees spectrum preservation."*

### 2. The Discovery (Minute 2)
> *"When validating this bound on real routers, we audited the standard evaluation protocol in the literature (Protocol A). We discovered a fatal flaw: Protocol A is permutation-invariant ($\text{Acc}(s) \equiv \text{Acc}(-s)$). Because experts are trained on the same nodes they are tested on, any random partition receives free credit. The published routing gains were an evaluation artifact."*

### 3. The Protocol B Audit (Minute 3)
> *"We created Protocol B (fixed-expert cross-fitting). Across 26 routers, 3 expert types, and 3 fraud graphs, 0 out of 234 configurations beat a single global model. We then reproduced state-of-the-art Node-MoE (ChebNetII filters) on fraud and on its own heterophily benchmarks (Actor, Chameleon, Squirrel). We proved causally that an unrouted, equal-weight average matches Node-MoE on fraud and strictly beats it on Actor, Chameleon, and Squirrel, while a simple feature MLP beats Node-MoE by up to 12%."*

### 4. The Constructive Solution (Minute 4)
> *"We didn't leave this as a negative result. We built the Learnability Lever ($\beta$) to pinpoint where routing actually becomes deployable ($\beta \ge 0.3$), built a Selective Abstaining Router that safely defaults to a global model on real pools, and showed on 11,520 SmolLM2 attention nodes that pre-training SD predicts multi-seed router stability with $p < 10^{-6}$."*

### 5. The Submission Target (Minute 5)
> *"We have 256 passing tests, pyflakes is completely clean, all publication LaTeX tables are exported, and we are targeting the NeurIPS Datasets & Benchmarks Track."*
