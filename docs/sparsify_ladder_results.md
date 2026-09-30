# Spectral-Distillation Retention Ladder on Real Fraud Data — Results

Graph: DGL Amazon co-review fraud (8,639 labeled users, 3,298,534 undirected edges,
edge label homophily 0.954; see `docs/real_fraud_results.md`).
Method: `experiments/run_sparsify_ladder.py`; identical D1–D3 experts/training as the base
protocol; 10 splits × 3 seeds per ladder point; p = exact-Wilcoxon (resolution floor 0.001953).

## 1. Question

On the full graph the label-free structural router collapses to random (D2 fails) — the
co-review relations are so dense/homophilic that the structural fields saturate. Does
**spectral (effective-resistance) distillation sharpen the routing signal** — or merely
preserve it — compared to budget-matched *random* and *degree* pruning?

## 2. Headline: spectral distillation **amplifies** the oracle signal ~7×

D1 = oracle > uniform (does a homophily-bucket routing help?); D3 = oracle > random.
Full-graph baseline (committed run): D1 = **+0.0028**, D3 = **+0.0038**.

| retention | edges kept | method | **D1** | p(D1) | **D3** | p(D3) | edge homophily |
|---|---|---|---|---|---|---|---|
| 1.00 | 3,298,534 | *(full)* | +0.0028 | 0.002 | +0.0038 | 0.002 | 0.954 |
| 0.15 | 494,780 | **ER** | **+0.0031** | 0.002 | +0.0039 | 0.002 | 0.899 |
| 0.15 | 494,780 | random | +0.0022 | 0.002 | +0.0029 | 0.002 | 0.886 |
| 0.15 | 494,780 | degree | +0.0002 | 0.020 | +0.0015 | 0.002 | 0.986 |
| 0.08 | 263,883 | **ER** | **+0.0208** | 0.002 | **+0.0212** | 0.002 | 0.906 |
| 0.08 | 263,883 | random | +0.0012 | 0.010 | +0.0027 | 0.002 | 0.888 |
| 0.08 | 263,883 | degree | +0.0002 | 0.020 | +0.0019 | 0.002 | 0.991 |
| 0.06 | 197,912 | **ER** | **+0.0198** | 0.002 | **+0.0204** | 0.002 | 0.904 |
| 0.06 | 197,912 | random | +0.0011 | 0.010 | +0.0016 | 0.002 | 0.890 |
| 0.06 | 197,912 | degree | +0.0001 | 0.131 | +0.0012 | 0.002 | 0.992 |

- **ER at 6–8% retention raises D1/D3 from +0.0028/+0.0038 to ≈ +0.020/+0.021 — a ~7× amplification of the routing signal** (p at the resolution floor), at *zero* extra expert cost.
- **Budget-matched random pruning drops D1 to ~0.4×** the full graph and **degree pruning collapses it to ~0.07×** (fails D1 at 0.06). The effect is specific to *spectral* distillation, not to "keeping fewer edges."

## 3. Mechanism: ER maximizes node-level homophily *contrast*

Degree pruning keeps the *highest*-homophily edges (0.99) yet **destroys** D1 — because a
graph where nearly every node is homophilic makes the 3-way oracle buckets degenerate
(everything → low-pass), collapsing oracle ≈ uniform. ER instead lands at an intermediate
homophily (~0.90) where node homophily **varies most across nodes**, so the adaptive
homophily buckets are maximally informative. Effective-resistance sampling keeps low-resistance
within-class clique edges and suppresses high-resistance cross-community bridges, sharpening
the fraud/benign contrast that the oracle reads. The amplifier helps the *oracle* channel
(D1/D3); it does **not** rescue the label-free router (D2 — see §4).

## 4. What this does NOT fix

- **D2 (label-free router) still fails at essentially every point.** The structural router
  cannot locate the signal the oracle now sees; distillation amplifies the *label-driven*
  signal but does not make it label-free. This is the project’s remaining open problem.
- The `SD` (max relative eigenvalue error) metric saturates at ~1.0 on this 8,639-node graph,
  so the theory↔empirics bridge is carried by the D-metrics + homophily mechanism, not SD.
- Single dataset; the amplification peak (6–8% retention) is hyperparameter-dependent.

## 5. Why this is publishable (and its limits)

**Claim.** On a real fraud graph, effective-resistance spectral distillation — unlike
random/degree distillation at identical edge budgets — *amplifies* the oracle's routing
advantage ~7× by maximizing node-level label-homophily contrast. This is the first real-data
confirmation that the spectral-distillation framework does more than preserve routing signal.

**Caveats / what a reviewer will push on.** (i) Effect measured via the oracle channel; the
label-free router (D2) is unfixed, so the "recover the signal automatically" story is not
yet complete. (ii) One graph, one retention band; needs ≥2–3 more real graphs (YelpChi's
largest component, Tolokers, DGraph-Fin) to claim generality. (iii) SD does not track the
effect at this scale; a scale-appropriate spectral-similarity metric is needed to tie it back
to the theory bound.

## 6. Reproduce

```
python experiments/run_sparsify_ladder.py \
    --budgets 0.15,0.08,0.06 --methods er,random,degree \
    --splits 10 --seeds 3 --skip-sd
```
(Requires `data/benchmarks/amazon/raw/Amazon.mat`; exact effective-resistance `pinvh` on the
8,639² Laplacian costs ~9 min once per run — see §Reproducibility.)
Artifact: `logs/amazon_fraud_ladder/ladder_results.jsonl` + `ladder_summary.json`.
