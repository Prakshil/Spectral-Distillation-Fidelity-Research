# Method Section (draft)

Paper-style description of the implemented pipeline. Code references are to
`spectral_distillation/src/`. Results are *not* restated here; see
`docs/fixed_expert_protocol.md`, `docs/real_attention_results.md`,
`docs/sparsify_ladder_results.md`, and `docs/reproducibility.md`.

**Status:** drafted. Scope honest-set: `§8`.

## 1. Overview

The pipeline turns a dense relational graph (LLM attention, or a benchmark fraud
graph) into a sparse graph, with two coupled pieces:

1. a **spectral-faithful sparsification engine** that samples edges by
   effective resistance, optionally filtered by an oracle score, and
2. a **pre-training diagnostic** (spectral distortion, SD) that predicts how much
   per-node routing information the distillation destroyed, together with a
   **fixed-expert evaluation protocol** that measures whether routing helps at
   all.

A third component — **attention entropy** — is a routing feature computed
*before* graph construction, so distillation cannot corrupt it by construction.

## 2. Notation

| Symbol | Meaning | Code |
|--------|---------|------|
| `W` | symmetric weighted adjacency | `build_adjacency` (`laplacian.py`) |
| `D = diag(W·1)` | degree matrix | `compute_degree_matrix` |
| `L = D − W` | graph Laplacian | `compute_laplacian` |
| `L⁺` | pseudoinverse | `pinvh` (`effective_resistance.py:15`) |
| `R_eff(i,j)` | `(eᵢ−eⱼ)ᵀ L⁺ (eᵢ−eⱼ)` | `effective_resistance_exact` (`:13`) |
| `λ_k`, `v_k` | Laplacian eigenvalues / eigenvectors | `compute_eigenvalues` (`spectral.py`) |
| `gap_k` | `min_{j≠k}\|λ_k−λ_j\|` | `min_eigengap` (`spectral.py`) |
| `ΔL` | `L_sparse − L_A` | — |
| `SD` | `max_k \|λ_k(L_s)−λ_k(L_A)\|/(\|λ_k(L_A)\|+ε)` | `compute_spectral_distortion` (`distortion.py:11`) |

## 3. Graph construction

Attention matrices are symmetrized (`(A+Aᵀ)/2`), self-loops handled per caller,
and converted to a Laplacian. For real attention graphs a **per-row top-k** step
is required first: without it the graph is ~99.8% dense and the oracle
degenerates to a single bucket (`docs/real_attention_results.md` §4).

## 4. Pre-construction routing feature: attention entropy

For each layer/head, per-node entropy of the row-stochastic attention
distribution:

```
H_i = − Σ_j P_ij log₂ P_ij,   P_ij = A_ij / Σ_j A_ij
```

Implemented in `node_attention_entropy` / `compute_attention_entropy`
(`attention_entropy.py:14,24`), aggregated across layers/heads by
mean/max/std (`aggregate_entropy_features`, `:41`). Low entropy = a node attends
to few peers (structured, confident); high entropy = diffuse attention. Because
this is read directly from the LLM, it is available before any graph is built and
is invariant to the later distillation step — the design rationale in
`docs/literature_positioning.md` §7.

## 5. Structural routing features

The label-free router operates on a frozen, standardized feature matrix:

| Field | Definition | Code |
|-------|-----------|------|
| `hx` | feature homophily: mean cosine to neighbours | `mixture.py` |
| `spectral_ratio` | normalized node spectral position `s_spectral(v)` | `spectral_position.py:31` |
| `log_degree` | log degree | — |
| `clustering` | local clustering coefficient | — |

Node spectral position (`node_spectral_position`, `spectral_position.py:8`):

```
s_spectral(v) = Σ_k [v_k(v)]² λ_k / Σ_k [v_k(v)]²,
```

a weighted mean eigenvalue that places a node in the low- or high-frequency band
— the bridge between the spectral and routing projects. `high_frequency_share`
(`:23`) is the alternative band-share form.

## 6. Spectral sparsification engine

### 6.1 Effective-resistance sampling

Per Spielman–Srivastava, sample edge `(i,j)` with probability proportional to
`W[i,j]·R_eff(i,j)` and rescale kept edges by `1/p`, yielding an ε-spectral
sparsifier. Exact `R_eff` via `pinvh` for `n ≤ 2000`; top-k eigen approximation
otherwise (`effective_resistance_topk`, `effective_resistance.py:21`).

**Oracle filter.** A business-rule / attention-weight score `oracle_score(i,j)`
gates edges: `p=0` below `δ`, else `p = min(1, C·W·R_eff·log n / ε²)`
(`sparsifier.py`; guide §5.3). The oracle is the only label-aware component and
is reported separately from label-free routers.

### 6.2 Baselines

| Method | Rule | Exposes | Code |
|--------|------|---------|------|
| Thresholding | keep if `W > τ` | severs weak-but-collective paths (Fiedler collapse) | `threshold_sparsify` (`baselines.py`) |
| Random (budget-matched) | uniform sampling | ignores edge importance | `random_sparsify` |
| Degree-based | keep high-degree edges | drops bridges between low-degree clusters | `degree_sparsify` |
| **Spectral (proposed)** | ER sampling + oracle filter | preserves eigenvalues within `(1±ε)` | `spectral_sparsify` (`sparsifier.py`) |

## 7. Spectral distortion diagnostics

Because the legacy scalar `SD` saturates under fragmentation (index-paired
eigenvalues no longer refer to the same mode once the perturbed graph gains zero
eigenvalues), the following graded, component-count-robust metrics are the ones
actually used:

| Metric | What it measures | Code |
|--------|------------------|------|
| `SD` (legacy) | worst relative index-matched eigenvalue error | `distortion.py:11` |
| `rank_matched_distortion` | worst error over the top-k largest modes | `distortion.py:32` |
| `low_frequency_distortion` | error of the smooth global modes (the ones a router consumes), zero modes discarded per side, normalized by `λ_max` | `distortion.py:172` |
| `connectivity_report` | component counts / isolated mass | `distortion.py:142` |
| `routing_survival` | `(1 − low_freq_err)·(1 − isolated/n)` | `distortion.py:253` |
| `fragility` | `SD / gap_min` (the DK ratio) | `distortion.py:279` |

`routing_survival` multiplies a spectral term by a connectivity term rather than
taking a max, so a pristine spectrum cannot excuse total disconnection, and an
orphaned node (unroutable) is charged for. These are computed pre-training, from
the two Laplacians alone.

## 8. Router evaluation protocol

### 8.1 The confounded baseline (rejected)

The initial self-routing protocol (`mixture.py`) fits expert `k` on the nodes
routed to `k` and scores it on those same nodes. This is **permutation-invariant**
(the score is `Σ_g acc(group_g | model_g)`, independent of channel index), so a
routing score and its negation score identically, and it grants **free credit**
to any label-homogeneous partition. It cannot measure routing
(`docs/fixed_expert_protocol.md` §1).

### 8.2 Fixed-expert (cross-fit) protocol

Build the expert pool **first**, routing-blind, by k-means on features; bind
expert `k` permanently to feature region `k`; then vary only the routing. Four
parameter-identical conditions share experts, training, and budget:

1. `label-free` — the router under test (structural score, or a learned gate);
2. `oracle` — label-derived local-regime buckets (`h ≥ 0.6` low-pass,
   `h ≤ 0.4` high-pass, else identity);
3. `random` — usage-matched random assignment;
4. `uniform` — gates frozen at `(1/3,1/3,1/3)`, runtime-audited.

Plus a **single-global** model (one expert on all nodes) as the reference that
isolates routing value: `routing − no-routing < 0` means routing is worse than
not routing. Decision rules (`guide §6.3`):

```
D1 oracle > uniform        (signal exists)
D2 label-free > random     (router finds it)
D3 oracle > random         (not capacity; identical parameter count)
D4 D1 holds on a mechanism-matched positive control (instrument valid)
```

Statistics: 10 splits × 3 seeds, seeds averaged within split, paired exact
Wilcoxon + paired t over splits, Holm–Bonferroni per family, effect sizes `dz`
and 95% CIs (`evaluation.py`).

### 8.3 Controls

- **Positive control** (`positive_control.py`): a non-separable mixture of three
  linear rules where routing is provably useful; the harness must detect the
  oracle win (it does) and must *not* report random or uniform as wins.
- **Ens-Avg**: average the same frozen pool with no routing — tests whether the
  pool is complementary when routing is not.
- **Implicit routing closures**: `x_only` removes all graph terms from the gate;
  `ragged-edges` stops the graph expert aggregating from nodes it does not own.
  Both channels are closed and the null survives both.

### 8.4 Learned router

`learned_router.py` adds a label-trained MLP gate (Node-MoE gate features
`[X, |AX−X|, |A²X−X|]`) over the same frozen pool, with the crucial cross-fit:
experts are fitted on `fit_mask` and the router's supervision targets are
computed on a disjoint `router_mask`, so the target means "which expert
*generalizes* to this node", not "which cluster is this node".

## 9. Code map

```
src/laplacian.py, spectral.py            graph + eigen-decomposition
src/effective_resistance.py, sparsifier.py, baselines.py   sparsification
src/distortion.py                        SD diagnostics (the contribution's metric)
src/spectral_position.py, attention_entropy.py   routing features
src/mixture.py, evaluation.py            protocol engine (fixed-expert)
src/learned_router.py, gnn_router.py, gnn_expert.py   learned arms
src/positive_control.py                  D4 instrument check
experiments/run_fixed_expert_protocol.py, run_sd_vs_routing.py,
            run_llm_protocol_b.py, run_sparsify_ladder.py,
            run_k_confirmation.py                    runners
```

## 10. Scope (honest)

- The method's *supported* claim is the pre-training diagnostic: SD/low-frequency
  distortion + connectivity survival predict routing **stability**
  (Spearman +0.78…+0.99, `docs/real_attention_results.md` §9.1). The accuracy
  claim is a **null**.
- Under the fixed-expert protocol, **no label-free or learned router beats one
  global model** on the fraud graphs (0/234), and the oracle is *negative*
  (9/9). Published routers (RouterGNN, Ada-Routing, Node-MoE experts) are not yet
  reproduced; the negative is against structural-score routers and a
  label-trained MLP gate.
- Protocol-A (self-routing) oracle numbers are retracted; only Protocol-B
  numbers are quoted (`docs/reproducibility.md` banner).
