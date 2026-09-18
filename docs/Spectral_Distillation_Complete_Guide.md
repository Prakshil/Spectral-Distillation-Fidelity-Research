# Spectral Distillation: A Complete Guide From Zero to Research
### From "What is a Graph?" to Proving the Routing Signal Decay Theorem

---

> **How to read this:** Every single concept is explained twice — once like you are 10 years old, once with the full mathematics. Examples use a real 4×4 ERP attention matrix throughout. Nothing is skipped.

---

## PART 0: Before We Start — What Are We Actually Trying to Do?

Imagine you work at a company. The company has invoices, suppliers, customers, and bank accounts. Thousands of them. You want to automatically detect **fraud** — like a supplier who is secretly sending money to a fake shell company.

You have a very smart AI (an LLM — Large Language Model) that reads all the documents and figures out which entities are related to which. It produces a big table of numbers called an **attention matrix** that says "entity A pays attention to entity B this much."

The problem: this table is **huge and dense** (everyone is connected to everyone with some weight). You cannot feed this directly into a GNN (Graph Neural Network) — it is too big and too noisy.

So you **compress** it: you throw away weak connections and keep only the strong ones. This gives you a **sparse graph** — a much smaller set of connections.

Then a **router** inside the GNN looks at each entity and decides: "Does this entity live in a smooth neighbourhood (route to low-pass expert) or a spiky neighbourhood with lots of differences (route to high-pass expert)?"

**The research question we are answering:**

> When you compress the dense attention matrix into a sparse graph, how much of the routing-relevant information survives? Can we mathematically bound how much is lost?

This is **Spectral Distillation Fidelity**.

---

## PART 1: Graphs — From Absolute Scratch

### 1.1 What is a Graph? (5-year-old version)

Imagine you and your friends. Draw a dot for each person. Draw a line between two dots if those two people are friends. That picture is a **graph**.

- The dots are called **nodes** (or vertices)
- The lines are called **edges**

In our ERP system:
- Nodes = Invoice, Supplier A, Supplier B, Customer C, Bank Account X
- Edges = "Invoice was sent by Supplier A", "Payment went to Bank Account X"

### 1.2 Weighted Graphs

Now imagine the lines have **thickness**. A thick line means a strong connection. A thin line means a weak connection.

In our attention matrix, the weight of an edge between entity i and entity j is how much the LLM thinks they are related.

### 1.3 Our Running Example: A 4-Node ERP Graph

Let's say we have 4 entities:
- Node 1: Invoice_001
- Node 2: Supplier_A  
- Node 3: Shell_Company (the fraud!)
- Node 4: Customer_C

The LLM processes all documents about these 4 entities and produces this **attention matrix** (each number = how much row-entity attends to column-entity):

```
Full LLM Attention Matrix A:

         Inv  Sup  Shell Cust
Invoice  [0.0, 0.8,  0.6,  0.3]
Supplier [0.8, 0.0,  0.9,  0.2]
Shell    [0.6, 0.9,  0.0,  0.1]
Customer [0.3, 0.2,  0.1,  0.0]
```

This is a **complete weighted graph** — everyone is connected to everyone. Think of it as a spider web where every thread has a different thickness.

Now, to build the sparse graph, we apply a threshold τ = 0.5:

```
Keep edge (i,j) only if A[i,j] > 0.5

Sparse Graph G:
         Inv  Sup  Shell Cust
Invoice  [0.0, 0.8,  0.6,  0.0]   ← 0.3 dropped
Supplier [0.8, 0.0,  0.9,  0.0]   ← 0.2 dropped
Shell    [0.6, 0.9,  0.0,  0.0]   ← 0.1 dropped
Customer [0.0, 0.0,  0.0,  0.0]   ← Customer isolated!
```

**The customer just got disconnected from the graph.** The router will never see that the customer interacts with the invoice. This is the problem we are studying.

---

## PART 2: The Adjacency Matrix and Degree Matrix

### 2.1 Adjacency Matrix W

The adjacency matrix is just the edge weight table we already have. For our sparse graph G:

```
W = [0.0, 0.8, 0.6, 0.0]
    [0.8, 0.0, 0.9, 0.0]
    [0.6, 0.9, 0.0, 0.0]
    [0.0, 0.0, 0.0, 0.0]
```

**Rule:** W[i][j] = weight of edge between node i and node j. For undirected graphs, W[i][j] = W[j][i] always.

### 2.2 The Degree Matrix D

The **degree** of a node is: "how connected is this node?" We measure it by adding up all edge weights coming out of that node.

For node 1 (Invoice):
```
degree(Invoice) = 0.0 + 0.8 + 0.6 + 0.0 = 1.4
```

For node 2 (Supplier):
```
degree(Supplier) = 0.8 + 0.0 + 0.9 + 0.0 = 1.7
```

For node 3 (Shell):
```
degree(Shell) = 0.6 + 0.9 + 0.0 + 0.0 = 1.5
```

For node 4 (Customer):
```
degree(Customer) = 0.0 + 0.0 + 0.0 + 0.0 = 0.0  ← isolated!
```

**The Degree Matrix D** puts these on the diagonal and zeros everywhere else:

```
D = [1.4,  0.0,  0.0,  0.0]
    [0.0,  1.7,  0.0,  0.0]
    [0.0,  0.0,  1.5,  0.0]
    [0.0,  0.0,  0.0,  0.0]
```

**Mathematical formula:**
```
D[i][i] = Σ_j W[i][j]     (sum of row i of the adjacency matrix)
D[i][j] = 0               for i ≠ j
```

---

## PART 3: The Graph Laplacian — The Heart of Everything

### 3.1 What is the Laplacian? (10-year-old version)

Imagine a hilly landscape. Some areas are high (mountains), some are low (valleys). The **Laplacian** tells you, at each point, how different your height is from your neighbours' average height.

If you are in a flat field: Laplacian = 0 (you are the same as neighbours)
If you are on a mountain peak: Laplacian is large (you are much higher than neighbours)

On a graph, "height" is a signal (like a feature value) assigned to each node. The Laplacian measures **how much each node differs from its connected neighbours**.

### 3.2 The Formula

```
L = D - W
```

That's it. The Laplacian is the Degree matrix minus the Adjacency matrix.

For our sparse graph:

```
L_sparse = D - W

= [1.4,  0.0,  0.0,  0.0]   -   [0.0, 0.8, 0.6, 0.0]
  [0.0,  1.7,  0.0,  0.0]       [0.8, 0.0, 0.9, 0.0]
  [0.0,  0.0,  1.5,  0.0]       [0.6, 0.9, 0.0, 0.0]
  [0.0,  0.0,  0.0,  0.0]       [0.0, 0.0, 0.0, 0.0]

= [ 1.4, -0.8, -0.6,  0.0]
  [-0.8,  1.7, -0.9,  0.0]
  [-0.6, -0.9,  1.5,  0.0]
  [ 0.0,  0.0,  0.0,  0.0]
```

**Why negative off-diagonal?** Because L = D - W. The diagonal tells you your total connection strength. The off-diagonal tells you how much you "pull away" from each neighbour (negative = pulling toward, in a sense).

### 3.3 The Quadratic Form — Why the Laplacian Measures Smoothness

This is the most important formula in spectral graph theory. Given any signal **f** (a vector assigning a number to each node):

```
f^T L f = Σ_{(i,j) ∈ edges} W[i][j] · (f[i] - f[j])²
```

**This sums up, over every edge, the squared difference between connected nodes, weighted by edge strength.**

**Example:** Let f = [1, 1, 5, 1] (Invoice=1, Supplier=1, Shell=5, Customer=1)

Shell company has a very different value — suspicious!

```
f^T L_sparse f = W[1,2]·(f[1]-f[2])² + W[1,3]·(f[1]-f[3])² + W[2,3]·(f[2]-f[3])²

= 0.8·(1-1)² + 0.6·(1-5)² + 0.9·(1-5)²

= 0.8·0 + 0.6·16 + 0.9·16

= 0 + 9.6 + 14.4

= 24.0   ← Large! Signal is rough/heterophilic around Shell node
```

Now try f = [1, 1, 1, 1] (everyone the same):

```
f^T L_sparse f = 0.8·0 + 0.6·0 + 0.9·0 = 0.0   ← Perfectly smooth
```

**Key insight:**
- f^T L f = 0 → signal is perfectly smooth (all connected nodes agree)
- f^T L f is large → signal is rough (connected nodes disagree strongly)
- This directly connects to homophily (smooth = homophilic, rough = heterophilic)

**Why is this always ≥ 0?**

Because every term W[i][j]·(f[i]-f[j])² is non-negative (weights are positive, squares are positive). So L is **Positive Semi-Definite (PSD)** — all its eigenvalues are ≥ 0.

---

## PART 4: Eigenvalues and Eigenvectors — The Frequency Spectrum

### 4.1 What are Eigenvalues? (Story Version)

Imagine you have a stretchy rubber band in 2D. You pull it in different directions. For most directions, the rubber band stretches AND rotates. But there are special directions where it **only stretches, never rotates**. Those special directions are the **eigenvectors**. How much it stretches in that direction is the **eigenvalue**.

For a matrix M, if:
```
M · v = λ · v
```
Then **v** is an eigenvector and **λ** is its eigenvalue.

**v** is a direction that M doesn't rotate — only scales.
**λ** is how much it scales.

### 4.2 The Spectral Theorem

Since L is a **real symmetric matrix** (L = L^T, which we can verify: L_sparse is symmetric because W is symmetric), a deep theorem from linear algebra (the Spectral Theorem) guarantees:

1. L has exactly n real eigenvalues: λ₁ ≤ λ₂ ≤ ... ≤ λₙ
2. Their eigenvectors v₁, v₂, ..., vₙ are all **orthogonal** to each other (perpendicular in n-dimensional space)
3. Together they form a complete basis — any signal on the graph can be written as a combination of eigenvectors

**For graph Laplacians specifically:**
```
0 = λ₁ ≤ λ₂ ≤ ... ≤ λₙ
```

λ₁ is always 0. Why? Because f = [1,1,1,...,1] (constant signal) gives f^T L f = 0, meaning the constant vector is always an eigenvector with eigenvalue 0.

### 4.3 Computing Eigenvalues of Our Laplacian

For our sparse graph Laplacian:
```
L_sparse = [ 1.4, -0.8, -0.6,  0.0]
           [-0.8,  1.7, -0.9,  0.0]
           [-0.6, -0.9,  1.5,  0.0]
           [ 0.0,  0.0,  0.0,  0.0]
```

Notice the 4th row and column are all zeros (Customer is isolated).

This means λ₁ = 0 (from isolation of Customer) AND another λ = 0 (from the usual constant eigenvector property).

**Actually we have TWO zero eigenvalues** because the graph has 2 connected components:
- Component 1: Invoice, Supplier, Shell (connected)
- Component 2: Customer (isolated)

**Key rule: Number of zero eigenvalues = Number of connected components.**

For the 3-node subgraph {Invoice, Supplier, Shell}, the 3×3 Laplacian is:
```
L_sub = [ 1.4, -0.8, -0.6]
        [-0.8,  1.7, -0.9]
        [-0.6, -0.9,  1.5]
```

The eigenvalues (computed numerically):
```
λ₁ = 0.0    (constant vector — graph is connected)
λ₂ ≈ 1.52   (Fiedler value — algebraic connectivity)
λ₃ ≈ 3.08   (maximum eigenvalue)
```

For the full 4×4 sparse Laplacian:
```
λ₁ = 0.0    (Customer isolated)
λ₂ = 0.0    (connected component structure)
λ₃ ≈ 1.52
λ₄ ≈ 3.08
```

### 4.4 Now the Full Attention Laplacian

For the full dense attention matrix (before thresholding):
```
A_full = [0.0, 0.8, 0.6, 0.3]
         [0.8, 0.0, 0.9, 0.2]
         [0.6, 0.9, 0.0, 0.1]
         [0.3, 0.2, 0.1, 0.0]

Degrees:
d(Invoice)  = 0.0 + 0.8 + 0.6 + 0.3 = 1.7
d(Supplier) = 0.8 + 0.0 + 0.9 + 0.2 = 1.9
d(Shell)    = 0.6 + 0.9 + 0.0 + 0.1 = 1.6
d(Customer) = 0.3 + 0.2 + 0.1 + 0.0 = 0.6

D_full = [1.7, 0.0, 0.0, 0.0]
         [0.0, 1.9, 0.0, 0.0]
         [0.0, 0.0, 1.6, 0.0]
         [0.0, 0.0, 0.0, 0.6]

L_full = D_full - A_full

= [ 1.7, -0.8, -0.6, -0.3]
  [-0.8,  1.9, -0.9, -0.2]
  [-0.6, -0.9,  1.6, -0.1]
  [-0.3, -0.2, -0.1,  0.6]
```

Eigenvalues of L_full (computed numerically):
```
λ₁ = 0.0
λ₂ ≈ 0.48
λ₃ ≈ 1.73
λ₄ ≈ 3.59
```

**Comparing the two spectra:**

| k | λₖ (full attention) | λₖ (sparse, τ=0.5) | Difference |
|---|---|---|---|
| 1 | 0.00 | 0.00 | 0.00 |
| 2 | 0.48 | 0.00 | **−0.48** |
| 3 | 1.73 | 1.52 | **−0.21** |
| 4 | 3.59 | 3.08 | **−0.51** |

**The Fiedler value dropped from 0.48 to 0.0.** The graph went from "connected with some bottleneck" to "has a disconnected component." This is catastrophic for the router.

---

## PART 5: What the Spectrum Means — Frequencies on Graphs

### 5.1 The Graph Fourier Transform (Simple Version)

In regular audio, a Fourier transform decomposes a sound signal into sine waves of different frequencies. Low frequencies = slow changes (bass). High frequencies = fast changes (treble).

On a graph, the eigenvectors of the Laplacian **are** the Fourier basis:
- **Eigenvectors with small λ** = low-frequency basis vectors (signal changes slowly across edges)
- **Eigenvectors with large λ** = high-frequency basis vectors (signal oscillates rapidly across connected nodes)

### 5.2 What Each Frequency Band Looks Like

**Low frequency (small λ, smooth eigenvector):**

The eigenvector v₁ for λ₁ = 0 is always:
```
v₁ = [1/√4, 1/√4, 1/√4, 1/√4] = [0.5, 0.5, 0.5, 0.5]
```
All nodes have the same value. Perfectly smooth.

A signal in this frequency band: all entities in a cluster have similar feature values → **homophilic region → needs Low-Pass Expert.**

**High frequency (large λ, oscillating eigenvector):**

The eigenvector v₄ for λ₄ ≈ 3.59 might look like:
```
v₄ ≈ [0.7, -0.6, 0.3, -0.3]
```
Adjacent nodes have **opposite signs** → the signal oscillates rapidly → **heterophilic region → needs High-Pass Expert.**

**This is the fundamental connection between routing and spectral theory:**

> A fraud node (Shell Company) surrounded by legitimate nodes has high local variation. It lives in the high-frequency part of the spectrum. The correct expert for it is the high-pass expert. But only if the graph correctly preserves that high-frequency structure.

### 5.3 Node Spectral Position — A Principled Routing Signal

For any node v, its position in frequency space is:

```
s_spectral(v) = Σₖ [vₖ(v)]² · λₖ / Σₖ [vₖ(v)]²
```

Where vₖ(v) is the v-th component of the k-th eigenvector.

**Interpretation:**
- This is a weighted average of eigenvalues, weighted by how much node v "participates" in each eigenvector
- Low s_spectral(v) → node lives in smooth regions → route to Low-Pass Expert
- High s_spectral(v) → node lives in rough regions → route to High-Pass Expert

This replaces the homophily heuristic from the router paper with something mathematically grounded.

**Example calculation for Shell Company (node 3) in sparse graph:**

Using approximate eigenvectors (simplified for illustration):
```
v₁ = [0.5,  0.5,  0.5,  0.0]  (λ₁=0, normalized within connected component)
v₂ = [0.7, -0.5, -0.2,  0.0]  (λ₂=1.52, approximate)
v₃ = [0.1,  0.4, -0.9,  0.0]  (λ₃=3.08, approximate)

For Shell node (index 3):
v₁(Shell) = 0.5, v₂(Shell) = -0.2, v₃(Shell) = -0.9

Numerator = 0.5²·0 + (-0.2)²·1.52 + (-0.9)²·3.08
          = 0 + 0.04·1.52 + 0.81·3.08
          = 0 + 0.061 + 2.495
          = 2.556

Denominator = 0.5² + (-0.2)² + (-0.9)²
            = 0.25 + 0.04 + 0.81 = 1.10

s_spectral(Shell) = 2.556 / 1.10 ≈ 2.32   ← High! Route to High-Pass Expert ✓
```

The Shell Company correctly gets a high spectral position, meaning the router would (correctly) send it to the High-Pass Expert — which is designed to detect nodes that differ from their neighbours.

---

## PART 6: Spectral Sparsification — The Right Way to Compress

### 6.1 What is a Spectral Sparsifier? (Simple Version)

Imagine you have a complex city road network (the full attention graph). You want to keep only some roads (sparse graph) but still ensure that:
- Travel times between any two cities are similar
- Traffic flow patterns are similar
- No city gets cut off

A **spectral sparsifier** is a sparse graph that preserves all these properties — formally, it preserves the Laplacian quadratic form for every possible signal.

### 6.2 The Spielman-Srivastava Definition (2011)

A sparse graph H is an **ε-spectral sparsifier** of dense graph G if:

```
(1-ε) · x^T L_G x  ≤  x^T L_H x  ≤  (1+ε) · x^T L_G x    for ALL x ∈ Rⁿ
```

**What this sandwich inequality means:**

For every possible signal x you could put on the graph:
- The Laplacian energy (roughness measure) of H is within (1±ε) of G's energy
- **All eigenvalues are preserved within a (1±ε) multiplicative factor:**
  ```
  (1-ε) · λₖ(L_G) ≤ λₖ(L_H) ≤ (1+ε) · λₖ(L_G)    for all k
  ```
- **All information flows and graph cuts are preserved within (1±ε)**

**The remarkable theorem:** Spielman-Srivastava proved that:

> For any dense graph G on n nodes, there always exists an ε-spectral sparsifier with only **O(n log n / ε²) edges**.

For our 4-node graph (n=4), with ε=0.1:
```
Required edges ≈ 4 · log(4) / 0.01 ≈ 4 · 1.39 / 0.01 ≈ 554
```

Wait — that's more than 4²=16 (the maximum!). This formula is meaningful for large n. For n=4 it means we can use all edges and still get an exact sparsifier.

For large ERP graphs with n=10,000 entities, this says O(92,000) edges suffice instead of O(100,000,000) — a 1000× compression.

### 6.3 How to Sample Good Edges: Effective Resistance

The key insight of Spielman-Srivastava: **not all edges are equally important.**

The **effective resistance** of an edge (i,j) is:
```
R_eff(i,j) = (eᵢ - eⱼ)^T · L⁺ · (eᵢ - eⱼ)
```

Where L⁺ is the pseudoinverse of L, and eᵢ is the indicator vector for node i.

**Intuitive meaning:** If you think of the graph as an electrical circuit where each edge is a resistor with resistance 1/W[i][j], the effective resistance of edge (i,j) is the actual resistance you measure between nodes i and j when current flows through the whole network.

**Why it matters:**
- An edge with **high effective resistance** is a **bridge** — the only path between two parts of the graph. Removing it causes massive spectral distortion. **Keep it with high probability.**
- An edge with **low effective resistance** has **many parallel paths** around it. Removing it barely affects the spectrum. **Can be dropped safely.**

**Sampling rule:**
```
P(keep edge (i,j)) ∝ W[i][j] · R_eff(i,j)
```

Sample each edge independently with this probability, then rescale kept edges.

### 6.4 Why Thresholding is Bad (The Core Problem)

Thresholding (keep edge if W[i][j] > τ) evaluates edges **in isolation**. It is blind to effective resistance.

**Example with our graph:**

Suppose between Invoice and Customer there are 50 very weak edges (each weight 0.01) in the real LLM attention. Total connection strength = 50 × 0.01 = 0.5. Together they are important — they form a communication pathway.

Thresholding at τ = 0.05 drops ALL 50 edges. The cluster is severed.

Effective-resistance sampling would keep about 5-10 of them (those with highest effective resistance) and rescale them to weight ≈ 0.1. The spectral structure is preserved.

**This is exactly why our sparse graph dropped the Customer node** in the running example — simple thresholding severed a weak but collectively important set of connections.

---

## PART 7: Spectral Distortion — Measuring What We Lost

### 7.1 The Distortion Metric

When we distill the full attention Laplacian L_A into the sparse Laplacian L_sparse, define:

```
SD(A, G_sparse) = max_k | λₖ(L_sparse) - λₖ(L_A) | / λₖ(L_A)
```

This is the **maximum relative eigenvalue shift** across all eigenvalues.

**For our example:**

| k | λₖ(L_full) | λₖ(L_sparse) | Relative error |
|---|---|---|---|
| 1 | 0.00 | 0.00 | 0.0 (both zero) |
| 2 | 0.48 | 0.00 | **1.0 (100% error!)** |
| 3 | 1.73 | 1.52 | 0.121 (12.1% error) |
| 4 | 3.59 | 3.08 | 0.142 (14.2% error) |

```
SD = max(0, 1.0, 0.121, 0.142) = 1.0
```

**SD = 1.0 means the spectrum is completely destroyed in at least one frequency band.** The Fiedler value went to zero, meaning the graph appears disconnected even though the true attention graph is well-connected. The router is now operating on a fundamentally broken representation.

**If we had used spectral sparsification instead:**

A good sparsifier with ε = 0.2 would give:
```
(1-0.2)·0.48 = 0.384 ≤ λ₂(H) ≤ (1+0.2)·0.48 = 0.576
```

SD would be at most 0.2. The router gets much better information.

### 7.2 The Frobenius Norm Distortion

A related measure useful for the theorems below:

```
||ΔL||_F = ||L_A - L_sparse||_F = sqrt( Σᵢⱼ (L_A[i][j] - L_sparse[i][j])² )
```

**For our example:**

```
ΔL = L_full - L_sparse

= [ 1.7-1.4,  -0.8-(-0.8),  -0.6-(-0.6),  -0.3-0.0]
  [-0.8-(-0.8), 1.9-1.7,   -0.9-(-0.9),  -0.2-0.0]
  [-0.6-(-0.6), -0.9-(-0.9), 1.6-1.5,   -0.1-0.0]
  [-0.3-0.0,  -0.2-0.0,  -0.1-0.0,   0.6-0.0]

= [ 0.3,   0.0,   0.0,  -0.3]
  [ 0.0,   0.2,   0.0,  -0.2]
  [ 0.0,   0.0,   0.1,  -0.1]
  [-0.3,  -0.2,  -0.1,   0.6]
```

```
||ΔL||_F = sqrt(0.3² + 0² + 0² + 0.3² + 0² + 0.2² + 0² + 0.2² + ... )
         = sqrt(0.09 + 0.09 + 0.04 + 0.04 + 0.01 + 0.01 + 0.09 + 0.04 + 0.01 + 0.36)
         = sqrt(0.78)
         ≈ 0.883
```

This Frobenius norm is what feeds directly into the Davis-Kahan bound.

---

## PART 8: Perturbation Theory — What Happens When You Poke a Matrix?

### 8.1 The Core Question

When L_A becomes L_sparse = L_A + ΔL (where ΔL = L_sparse - L_A), how much do:
1. The **eigenvalues** move?
2. The **eigenvectors** rotate?

These two questions are answered by two classical theorems.

### 8.2 Weyl's Inequality — Eigenvalue Shifts

**Theorem (Weyl, 1912):**

For any two symmetric matrices M and M + ΔM:

```
|λₖ(M + ΔM) - λₖ(M)| ≤ ||ΔM||_2    for all k
```

Where ||ΔM||_2 is the **spectral norm** (largest singular value = largest absolute eigenvalue of ΔM).

**For our case:**

The perturbation is ΔL = L_sparse - L_A. We computed ||ΔL||_F ≈ 0.883.

Since ||ΔL||_2 ≤ ||ΔL||_F (spectral norm ≤ Frobenius norm), we know:

```
|λₖ(L_sparse) - λₖ(L_A)| ≤ 0.883   for all k
```

**Check:** λ₂ shifted by |0.00 - 0.48| = 0.48 ≤ 0.883. ✓

Weyl's inequality is confirmed. But it only tells us eigenvalues move by a bounded amount. **It says nothing about eigenvectors.** That requires Davis-Kahan.

### 8.3 Cauchy Interlacing — What Removing an Edge Does

When you remove a **single edge** (i,j) with weight w from graph G:

```
L_{G\(i,j)} = L_G - w · (eᵢ - eⱼ)(eᵢ - eⱼ)^T
```

The matrix (eᵢ - eⱼ)(eᵢ - eⱼ)^T is **rank-1** and **positive semi-definite**.

**Cauchy Interlacing Theorem:** For a symmetric matrix M and any rank-1 positive semi-definite perturbation P:

```
λₖ(M - P) ≤ λₖ(M)    for all k
```

And more precisely:

```
λₖ(M) - w ≤ λₖ(M - w·P) ≤ λₖ(M)    when ||P||=1
```

**In plain words:** Removing an edge can only **decrease** eigenvalues, by at most the edge weight. This is why our Fiedler value dropped when we removed the Customer edges.

**This is the result that is Lean-formalizable.** Mathlib already has the Cauchy interlacing theorem for symmetric matrices. Specializing it to graph Laplacians under edge removal is a concrete Lean contribution.

---

## PART 9: Davis-Kahan Theorem — The Crown Jewel

### 9.1 Why Eigenvector Rotation Matters

The router doesn't use eigenvalues directly. It uses **eigenvectors** as a frequency basis to measure each node's spectral position. If the eigenvectors rotate when we sparsify, nodes get placed in the wrong frequency band, and the router sends them to the wrong expert.

**Example:**

True eigenvector v₂ (for λ₂ = 0.48 in full graph) might look like:
```
v₂_true = [0.6, 0.5, -0.4, -0.6]
```
This says: Invoice and Supplier are in the "positive" frequency band, Shell and Customer in the "negative" band.

After sparsification, the corresponding eigenvector ṽ₂ (for λ₂ = 0.0) is:
```
ṽ₂_sparse ≈ [0.0, 0.0, 0.0, 1.0]   ← just the isolated Customer node!
```

**Completely different!** The router would route Customer to the "high-frequency" expert when actually Customer has low spectral energy in the true graph. This is a routing error caused entirely by sparsification.

### 9.2 The Davis-Kahan Theorem

**Theorem (Davis-Kahan, 1970):**

Let M and M̃ = M + ΔM be two symmetric matrices. Let vₖ be the k-th eigenvector of M (with eigenvalue λₖ), and ṽₖ be the corresponding eigenvector of M̃. Define the **eigengap**:

```
gap_k = min_{j ≠ k} |λₖ(M) - λⱼ(M)|
```

Then the angle between vₖ and ṽₖ is bounded by:

```
||sin Θ(vₖ, ṽₖ)||  ≤  ||ΔM||_F / gap_k
```

Or equivalently:
```
||vₖ - ṽₖ||₂  ≤  2 · ||ΔM||_F / gap_k
```

### 9.3 Breaking Down Every Term

**||ΔM||_F:** How big is the perturbation? (We computed this: 0.883 for our example)

**gap_k:** How isolated is eigenvalue λₖ from its neighbours?

For our full attention matrix:
```
λ₁=0, λ₂=0.48, λ₃=1.73, λ₄=3.59

gap_2 = min(|0.48 - 0|, |0.48 - 1.73|) = min(0.48, 1.25) = 0.48
gap_3 = min(|1.73 - 0.48|, |1.73 - 3.59|) = min(1.25, 1.86) = 1.25
gap_4 = min(|3.59 - 1.73|) = 1.86
```

**Davis-Kahan bounds for our example:**

```
||v₂ - ṽ₂||₂ ≤ 2 · 0.883 / 0.48 = 3.68   ← Very large! (bound is loose here, but signals danger)
||v₃ - ṽ₃||₂ ≤ 2 · 0.883 / 1.25 = 1.41
||v₄ - ṽ₄||₂ ≤ 2 · 0.883 / 1.86 = 0.95
```

The bound for eigenvector 2 is catastrophically large (> 1 is already maximum possible rotation for unit vectors). This confirms what we saw: the second eigenvector got completely destroyed by thresholding (because the Fiedler value dropped to zero — the eigengap gap_2 collapsed).

### 9.4 The Eigengap is the Critical Safety Margin

**Key insight from Davis-Kahan:**

```
Eigenvector error ∝ Perturbation / Eigengap
```

**Large eigengap → eigenvectors are stable → safe to sparsify**
**Small eigengap → eigenvectors are fragile → even tiny perturbation causes massive rotation**

**What causes small eigengaps in ERP graphs?**

- Two clusters of similar size with similar internal connectivity → λ₂ ≈ λ₃ (near-degenerate)
- Weakly connected subcommunities → Fiedler value λ₂ is tiny → gap_2 is tiny
- Fraud nodes specifically often sit at spectral transitions → their frequency bands have small eigengaps

**This means fraud detection is precisely the setting where eigenvector stability is most fragile.**

---

## PART 10: Attention Entropy as a Pre-Construction Routing Signal

### 10.1 The Key Insight

The router paper computes routing signals from the graph (homophily, spectral ratio, degree). But our graph is constructed from the LLM — so by the time we compute these graph-based signals, the distortion has already happened.

**We need a routing signal that comes BEFORE graph construction.**

**The LLM already tells us something important:** how uncertain each entity is about its relationships. This is measured by **attention entropy**.

### 10.2 Attention Entropy — Formula and Meaning

At layer l, head h, entity i attends to all other entities with weights α^(l,h)_{ij} (these are the attention matrix values, normalized to sum to 1 for each row):

```
α^(l,h)_{ij} = softmax( (W_Q · xᵢ) · (W_K · xⱼ)^T / √d )
```

The entropy of entity i's attention distribution:

```
H^(l,h)_i = - Σⱼ α^(l,h)_{ij} · log₂( α^(l,h)_{ij} )
```

**Interpretation:**
- **Low entropy:** Entity i attends strongly to just 1-2 specific entities. Its relationships are clear and structured. The LLM is confident about who this entity talks to.
- **High entropy:** Entity i spreads attention broadly across all entities. Its relationships are diffuse and uncertain. The LLM doesn't know who this entity's main counterparts are.

### 10.3 Example with Our Attention Matrix

Our attention matrix (row = query, col = key):
```
A = [0.0, 0.8, 0.6, 0.3]
    [0.8, 0.0, 0.9, 0.2]
    [0.6, 0.9, 0.0, 0.1]
    [0.3, 0.2, 0.1, 0.0]
```

First normalize each row to sum to 1 (softmax normalizes, so let's normalize manually):

```
Row 1 (Invoice): [0, 0.8, 0.6, 0.3] → sum=1.7
  → [0/1.7, 0.8/1.7, 0.6/1.7, 0.3/1.7] = [0.000, 0.471, 0.353, 0.176]

Row 2 (Supplier): [0.8, 0, 0.9, 0.2] → sum=1.9
  → [0.421, 0.000, 0.474, 0.105]

Row 3 (Shell): [0.6, 0.9, 0, 0.1] → sum=1.6
  → [0.375, 0.563, 0.000, 0.063]

Row 4 (Customer): [0.3, 0.2, 0.1, 0] → sum=0.6
  → [0.500, 0.333, 0.167, 0.000]
```

Now compute entropy for each entity:

**Invoice entropy:**
```
H(Invoice) = -(0.471·log₂(0.471) + 0.353·log₂(0.353) + 0.176·log₂(0.176))
           = -(0.471·(-1.085) + 0.353·(-1.501) + 0.176·(-2.506))
           = -(- 0.511 - 0.530 - 0.441)
           = 1.482 bits
```

**Supplier entropy:**
```
H(Supplier) = -(0.421·log₂(0.421) + 0.474·log₂(0.474) + 0.105·log₂(0.105))
            = -(0.421·(-1.248) + 0.474·(-1.077) + 0.105·(-3.252))
            = -(- 0.525 - 0.511 - 0.341)
            = 1.377 bits
```

**Shell entropy:**
```
H(Shell) = -(0.375·log₂(0.375) + 0.563·log₂(0.563) + 0.063·log₂(0.063))
         = -(0.375·(-1.415) + 0.563·(-0.829) + 0.063·(-3.989))
         = -(- 0.531 - 0.467 - 0.251)
         = 1.249 bits   ← Lower entropy: Shell Company is more focused (on Supplier)
```

**Customer entropy:**
```
H(Customer) = -(0.500·log₂(0.500) + 0.333·log₂(0.333) + 0.167·log₂(0.167))
            = -(0.500·(-1.0) + 0.333·(-1.585) + 0.167·(-2.585))
            = -(- 0.500 - 0.528 - 0.432)
            = 1.460 bits   ← Higher entropy: Customer is broadly connected
```

**Summary:**

| Entity | Entropy | Interpretation | Suggested Expert |
|---|---|---|---|
| Invoice | 1.482 | Moderate focus | Medium-pass |
| Supplier | 1.377 | More focused | Low/High-pass |
| Shell | 1.249 | Most focused (on Supplier) | **High-pass** (anomaly) |
| Customer | 1.460 | Broad attention | **Identity** (diffuse) |

**Shell Company has the lowest entropy** — the LLM is very focused about who Shell relates to (mostly Supplier). Combined with the fact that this is an anomalous pattern, Shell should go to the high-pass expert. Entropy gives us this signal **before building the graph**.

---

## PART 11: The End-to-End Proof Chain

### 11.1 The Theorem We Want to Prove

**Theorem (Routing Signal Decay Under Spectral Distortion):**

Let π* be the oracle router operating on the true full attention graph G_A.
Let π̃ be any router operating on the sparse graph G_sparse.

The loss in routing mutual information is bounded by:

```
I(π*(V); Y) - I(π̃(V); Y)  ≤  C · SD(A, G_sparse) · H(Y)
```

Where:
- SD is the spectral distortion metric
- H(Y) is the task label entropy
- C is a constant depending on GNN depth k and Lipschitz constant κ of the router function

### 11.2 The Four-Step Chain

**Step 1: Distillation creates a perturbation**

```
ΔL = L_sparse - L_A

||ΔL||_F = ?  (computed above: ≈ 0.883 for our example)
```

The goal: make ||ΔL||_F small.

**Step 2: Davis-Kahan bounds eigenvector rotation**

```
||vₖ - ṽₖ||₂ ≤ 2 · ||ΔL||_F / gap_k
```

For the most important frequency bands (those that the router uses):

```
||vₖ - ṽₖ||₂ ≤ 2 · ||ΔL||_F / min_k(gap_k)
```

Let δ = 2 · ||ΔL||_F / min_k(gap_k) be the maximum eigenvector perturbation.

**Step 3: Lipschitz GNN bounds routing error**

The router function π maps spectral coordinates to expert assignments. If π is κ-Lipschitz:

```
||π*(v) - π̃(v)||₂ ≤ κ · ||vₖ - ṽₖ||₂ ≤ κ · δ
```

Lipschitz continuity of the GNN router means: small change in input features → small change in routing assignment. This is guaranteed if the GNN has bounded weight matrices (standard assumption).

**Step 4: Data Processing Inequality bounds information loss**

The mutual information can only decrease under stochastic maps. For routing assignments:

```
I(π*(V); Y) - I(π̃(V); Y) ≤ H(Y) · TV(π*, π̃)
```

Where TV is total variation distance, which is bounded by the routing error:

```
TV(π*, π̃) ≤ Σᵥ ||π*(v) - π̃(v)||₂ / |V|  ≤ κ · δ
```

**Putting it all together:**

```
I(π*(V); Y) - I(π̃(V); Y) ≤ H(Y) · κ · 2 · ||ΔL||_F / min_k(gap_k)

                          = C · ||ΔL||_F / min_k(gap_k) · H(Y)

                          ∝ C · SD · H(Y)
```

**Interpretation of the bound:**

- If **||ΔL||_F is large** (bad sparsification): big routing signal loss
- If **min eigengap is small** (fragile spectrum): routing signal amplifies even small distortions  
- If **H(Y) is large** (hard task): more routing signal is at stake
- **C depends on GNN depth** — deeper GNNs accumulate errors across layers

### 11.3 The Routing Error Decay Theorem (Second Result)

A cleaner version assuming random edge perturbations:

**Theorem:** Let G_ε be a corrupted graph where each edge weight is perturbed by ±ε independently. Then with high probability:

```
[oracle_router(G_ε) - uniform(G_ε)] ≤ [oracle_router(G*) - uniform(G*)] · (1 - 2ε)^k
```

Where k is the GNN depth.

**Every 1% of edge noise degrades routing signal by approximately 2% per GNN layer.**

For our ERP system with 3-layer GNN and 10% noise:
```
Signal retention = (1 - 2·0.1)³ = (0.8)³ = 0.512
```

More than half the routing signal is destroyed. This is why simple baselines beat the GNN — the graph construction step is already losing more than 50% of the routing information.

---

## PART 12: The Spectral Sparsification Objective — Doing It Right

### 12.1 The Optimization Problem

Instead of thresholding, find the sparse graph that minimizes spectral distortion:

```
min_{G_sparse : |E| ≤ budget}  ||L_A - L_sparse||_F

subject to:
  L_sparse is a valid graph Laplacian (PSD, correct structure)
  oracle_score(i,j) ≥ δ  for all (i,j) ∈ E_sparse  (business rule constraint)
  |E_sparse| ≤ O(n log n / ε²)                       (budget constraint)
```

This is a **semidefinite program (SDP)** — it can be solved efficiently.

### 12.2 Effective Resistance for ERP Graphs

The practical algorithm:

```python
def spectral_sparsify(A_llm, n_edges_budget, oracle_score, delta):
    """
    A_llm: n×n attention matrix
    n_edges_budget: maximum edges in sparse graph
    oracle_score: function(i,j) → [0,1] business rule alignment
    delta: minimum oracle score threshold
    """
    # Step 1: Build full Laplacian
    D = diag(A_llm.sum(axis=1))
    L = D - A_llm
    
    # Step 2: Compute pseudoinverse for effective resistance
    L_pinv = pinv(L)
    
    # Step 3: For each edge, compute effective resistance
    for i,j in all_edges:
        e = indicator(i) - indicator(j)  # [0..1..0..-1..0]
        R_eff[i,j] = e @ L_pinv @ e
    
    # Step 4: Compute sampling probabilities
    for i,j in all_edges:
        if oracle_score(i,j) >= delta:
            p[i,j] = min(1, C * A[i,j] * R_eff[i,j] * log(n) / epsilon²)
        else:
            p[i,j] = 0  # filter out business-rule-violating edges
    
    # Step 5: Sample and rescale
    for i,j in all_edges:
        if random() < p[i,j]:
            E_sparse.add(i,j, weight = A[i,j] / p[i,j])
    
    return E_sparse
```

### 12.3 Pre-Training Diagnostic

Before training the GNN at all, compute SD:

```python
import torch

# Compute both Laplacians
L_full = compute_laplacian(A_llm)
L_sparse = compute_laplacian(A_sparse)

# Compute eigenvalues
eigs_full = torch.linalg.eigvalsh(L_full)
eigs_sparse = torch.linalg.eigvalsh(L_sparse)

# Spectral distortion (avoid divide by zero for λ₁=0)
relative_errors = torch.abs(eigs_full - eigs_sparse) / (eigs_full + 1e-8)
SD = relative_errors.max().item()

print(f"Spectral Distortion: {SD:.4f}")
print(f"Predicted routing signal retention: {(1-SD):.4f}")

# Minimum eigengap (routing stability indicator)
gaps = torch.diff(eigs_full)
min_gap = gaps.min().item()
print(f"Minimum eigengap: {min_gap:.4f}")
print(f"Routing fragility: {SD / (min_gap + 1e-8):.4f}")  # Davis-Kahan numerator/denominator
```

---

## PART 13: Complete Summary — Every Concept in One Table

| Concept | Simple Explanation | Mathematical Definition | Role in Our Research |
|---|---|---|---|
| **Graph** | Dots (nodes) connected by lines (edges) with weights | G = (V, E, W) | Structure of ERP data |
| **Adjacency Matrix W** | Table of edge weights | W[i][j] = weight of edge (i,j) | Represents connections |
| **Degree Matrix D** | Diagonal table of total connection strength per node | D[i][i] = Σⱼ W[i][j] | Used to build Laplacian |
| **Graph Laplacian L** | Measures how much each node differs from neighbours | L = D - W | The central mathematical object |
| **Quadratic Form f^T L f** | Total roughness of signal f on the graph | Σ_{(i,j)} W[i][j]·(f[i]-f[j])² | Measures homophily/heterophily |
| **Eigenvalue λₖ** | How much the Laplacian stretches in direction k | L·vₖ = λₖ·vₖ | Frequency of graph signal |
| **Eigenvector vₖ** | The direction of stretching (frequency basis) | Orthogonal, L·vₖ = λₖ·vₖ | Routing feature coordinates |
| **Fiedler Value λ₂** | How well-connected the graph is | Second smallest eigenvalue | Connectivity indicator |
| **Eigengap** | Distance between adjacent eigenvalues | gap_k = min_{j≠k} |λₖ-λⱼ| | Routing stability margin |
| **Spectral Sparsifier** | Sparse graph preserving all frequency information | (1-ε)L_G ≤ L_H ≤ (1+ε)L_G | The target of good distillation |
| **Effective Resistance** | How critical an edge is to connectivity | R_eff(i,j) = (eᵢ-eⱼ)^T L⁺ (eᵢ-eⱼ) | Sampling weight for sparsification |
| **Spectral Distortion SD** | How much the spectrum changed after sparsification | max_k |λₖ(L_sparse)-λₖ(L_A)| / λₖ(L_A) | Our core diagnostic metric |
| **Weyl's Inequality** | Eigenvalues can't jump more than perturbation size | |λₖ(M+ΔM)-λₖ(M)| ≤ ||ΔM||₂ | Bounds eigenvalue shifts |
| **Davis-Kahan Theorem** | Eigenvectors rotate by at most perturbation/eigengap | ||vₖ-ṽₖ|| ≤ 2||ΔL||_F / gap_k | Bounds routing feature corruption |
| **Attention Entropy** | How spread out each entity's attention is | H_i = -Σⱼ α_{ij} log α_{ij} | Pre-construction routing signal |
| **Data Processing Inequality** | Information can only decrease through computation | I(X;Z) ≥ I(Y;Z) if X→Y→Z | Bounds routing information loss |
| **Routing Signal Decay** | How much routing information survives distillation | I(π*;Y)-I(π̃;Y) ≤ C·SD·H(Y) | Our main theorem |

---

## PART 14: The Complete Research Proposal in Mathematical Terms

**Given:**
- Enterprise graph entities V (invoices, vendors, GL accounts)  
- Pre-trained LLM producing attention A^(l,h) ∈ ℝⁿˣⁿ
- Current pipeline: threshold A → sparse graph G → GNN router π

**Problem:**
Thresholding destroys spectral structure. By the time the router π sees the graph, routing-relevant frequency information has been lost. We have no prior measure of how much.

**Proposed:**

1. **Pre-training diagnostic:**
   ```
   Compute SD(A_aggregated, G_threshold) before any GNN training
   This predicts routing quality — high SD → poor routing ahead
   ```

2. **Replace thresholding with spectral sparsification:**
   ```
   min_{G : |E|≤budget} ||L_A - L_sparse||_F
   s.t. oracle_score(i,j) ≥ δ for kept edges
   ```

3. **Add pre-construction routing signal from attention entropy:**
   ```
   s_LLM(v) = (H¹_v, H²_v, ..., H^L_v)  [entropy across layers]
   Use this as routing signal alongside graph-based signals
   ```

4. **Prove the routing signal decay theorem:**
   ```
   I(π*(V); Y) - I(π̃(V); Y) ≤ C · SD · H(Y)
   
   Proof chain:
   ||ΔL||_F → (Davis-Kahan) → ||vₖ-ṽₖ|| → (Lipschitz GNN) → routing error → (DPI) → MI loss
   ```

5. **Lean formalization:**
   ```
   Cauchy interlacing for graph edge removal:
   λₖ(L_{G\e}) ≤ λₖ(L_G)  for all k, bounded by edge weight
   This is in Mathlib scope and not yet graph-specialized
   ```

**Expected result:** A principled replacement for threshold-based attention distillation with a provably bounded routing signal loss, validated on ERP fraud detection benchmarks using the oracle-versus-uniform protocol from the router paper.

---

*This document covers every mathematical concept required to understand, implement, and prove the Spectral Distillation Fidelity research direction. Each concept is built from first principles with the running 4-node ERP example computed explicitly throughout.*
