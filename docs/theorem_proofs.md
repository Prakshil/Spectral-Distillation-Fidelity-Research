# Theorem proofs

Expanded proof chain for the routing-signal-decay theorems, from
`SPECTRAL_DISTILLATION_IMPLEMENTATION_GUIDE.md` Part 7. Every inequality below
is classical; the contribution is the *chaining* and its operational use as a
pre-training diagnostic. Implementations: `src/distortion.py`,
`src/theoretical_bounds.py`, `src/spectral.py`.

**Status:** drafted. Assumptions and looseness are stated explicitly in §7.
The Lean lemma (§6) is a sketch, not a compiled Mathlib contribution.

## 0. Setup and notation

Let `G_A` be the true (dense attention) graph on `n` nodes with Laplacian
`L_A = D_A − W_A`, and `G_sparse` a distilled graph with Laplacian
`L_sparse`. Write the perturbation

```
ΔL = L_sparse − L_A.
```

Both Laplacians are symmetric PSD with real eigenvalues
`0 = λ₁ ≤ λ₂ ≤ … ≤ λₙ`; eigenvectors `v_k` are orthonormal. Define:

```
Frobenius perturbation   ‖ΔL‖_F = ‖L_sparse − L_A‖_F
eigengap                 gap_k  = min_{j ≠ k} |λ_k(L_A) − λ_j(L_A)|
min eigengap             gap_min = min_k gap_k
spectral distortion      SD     = max_k |λ_k(L_sparse) − λ_k(L_A)| / (|λ_k(L_A)| + ε)
fragility                φ      = SD / gap_min
```

`SD`, `‖ΔL‖_F`, `gap_min`, and `φ` are all implemented
(`src/distortion.py:11` `compute_spectral_distortion`,
`src/distortion.py:275` `compute_frobenius_norm`,
`src/distortion.py:279` `compute_fragility`). Note `SD` uses relative
eigenvalue error, while the perturbation theorems below use the absolute
`‖ΔL‖_F`; §1 bridges them.

## 1. Weyl's inequality — eigenvalue shifts

**Theorem (Weyl, 1912).** For symmetric `M` and `M + ΔM`, for every `k`:

```
|λ_k(M + ΔM) − λ_k(M)| ≤ ‖ΔM‖₂ ≤ ‖ΔM‖_F.
```

**Applied.** With `M = L_A`, `ΔM = ΔL`:

```
|λ_k(L_sparse) − λ_k(L_A)| ≤ ‖ΔL‖_F   for all k.
```

So the *absolute* eigenvalue deviation is bounded by `‖ΔL‖_F`. The running
example (guide Part 3.1) gives `‖ΔL‖_F ≈ 0.883` and `|λ₂ shift| = 0.48 ≤ 0.883`.

**Bridge to SD.** `SD` normalizes by `|λ_k(L_A)|`, so `SD` and `‖ΔL‖_F` are
different scales. On graphs where `λ_min^+ = min_{k:λ_k>0} λ_k` is bounded away
from zero, `SD ≤ ‖ΔL‖_F / λ_min^+`; when the spectrum fragments (many zero
eigenvalues), the per-mode normalization blows up and `SD` saturates at 1.0
(see §7 and `low_frequency_distortion`, `src/distortion.py:172`, for the repair).
The theorems are stated in terms of `‖ΔL‖_F` for this reason.

## 2. Cauchy interlacing — edge-removal monotonicity

**Lemma (Cauchy interlacing, rank-1 PSD case).** Removing edge `(i,j)` of weight
`w` gives

```
L_{G\(i,j)} = L_G − w·(eᵢ − eⱼ)(eᵢ − eⱼ)ᵀ,
```

where `(eᵢ − eⱼ)(eᵢ − eⱼ)ᵀ ⪰ 0` is rank-1 PSD. For any rank-1 PSD `P ⪰ 0`:

```
λ_k(M − P) ≤ λ_k(M)   for all k.
```

**Consequence.** Removing an edge can only *decrease* each eigenvalue, by at most
the edge weight. This is the mechanism behind the Fiedler collapse in the running
example: deleting the weak Customer edges dropped `λ₂` from `0.48` to `0.0`.
Formalization target in §6.

## 3. Davis–Kahan — eigenvector rotation

The router reads *eigenvectors* (node spectral position), not eigenvalues, so
Weyl is not enough.

**Theorem (Davis & Kahan, 1970).** For symmetric `M`, `M̃ = M + ΔM`, with
`v_k`, `ṽ_k` the k-th eigenvectors and `gap_k` as above:

```
‖sin Θ(v_k, ṽ_k)‖ ≤ ‖ΔM‖_F / gap_k
‖v_k − ṽ_k‖₂      ≤ 2 ‖ΔM‖_F / gap_k.
```

**Applied.**

```
‖v_k − ṽ_k‖₂ ≤ 2 ‖ΔL‖_F / gap_k   =: δ_k.
```

Eigenvector error is proportional to perturbation over eigengap. Small gaps
(equal-size clusters, near-degenerate communities) make the bound loose and the
eigenvector fragile. Running example: `δ₂ = 2·0.883/0.48 = 3.68` (> 1 = full
rotation), `δ₃ = 1.41`, `δ₄ = 0.95` (guide Part 3.3).

## 4. The four-step proof chain (Theorem 1)

**Theorem 1 (routing signal decay under spectral distortion).** Let `π*` be the
oracle router on `G_A` and `π̃` any router on `G_sparse`. Then

```
I(π*(V); Y) − I(π̃(V); Y) ≤ C · SD(A, G_sparse) · H(Y),
```

with `H(Y)` the label entropy and `C` a constant depending on GNN depth `k` and
the router Lipschitz constant `κ`. The chain is stated in `‖ΔL‖_F`:

**Step 1 — distillation is a perturbation.** Define `ΔL = L_sparse − L_A`; the
goal of spectral sparsification is to make `‖ΔL‖_F` small.

**Step 2 — Davis–Kahan bounds eigenvector rotation.** From §3,

```
‖v_k − ṽ_k‖₂ ≤ 2 ‖ΔL‖_F / gap_min.
```

**Step 3 — Lipschitz router converts representation error into routing error.**
Assume the router `π` (a bounded-depth GNN) is `κ`-Lipschitz in its features:

```
‖π*(v) − π̃(v)‖₂ ≤ κ ‖v_k − ṽ_k‖₂ ≤ κ · 2 ‖ΔL‖_F / gap_min.
```

**Step 4 — data-processing inequality bounds information loss.** Mutual
information cannot increase under a stochastic map, and for discrete
`Y` with total-variation distance `TV`:

```
I(π*(V); Y) − I(π̃(V); Y) ≤ H(Y) · TV(π*, π̃)
TV(π*, π̃) ≤ (1/|V|) Σ_v ‖π*(v) − π̃(v)‖₂ ≤ κ · 2 ‖ΔL‖_F / gap_min.
```

**Chaining Steps 1–4:**

```
I(π*(V); Y) − I(π̃(V); Y)
  ≤ H(Y) · κ · 2 ‖ΔL‖_F / gap_min
  = C · (‖ΔL‖_F / gap_min) · H(Y)
  ∝ C · SD · H(Y)          (under the λ-normalization bridge of §1).
```

**Interpretation.**
- large `‖ΔL‖_F` (bad sparsification) → large routing-signal loss;
- small `gap_min` (fragile spectrum) → the same distortion is amplified;
- large `H(Y)` (hard task) → more routing signal at stake;
- `C` grows with GNN depth (errors accumulate across `k` layers).

The ratio `‖ΔL‖_F / gap_min` is exactly `compute_fragility`'s numerator pattern
(`fragility = SD / gap_min`), which is why the diagnostic uses `SD/gap_min`.

## 5. Theorem 2 — random-perturbation retention

**Theorem.** Let `G_ε` be a graph whose edge weights are each perturbed by `±ε`
independently. Then with high probability

```
[oracle(G_ε) − uniform(G_ε)] ≤ [oracle(G*) − uniform(G*)] · (1 − 2ε)^k,
```

where `k` is the GNN depth. Each 1% of edge noise degrades routing signal by
≈2% *per layer*.

**Running example.** 3-layer GNN, 10% noise:

```
retention = (1 − 2·0.1)³ = 0.8³ = 0.512.
```

Over half the routing signal is destroyed. This is the qualitative statement
behind "a GNN whose graph-construction step already loses ≥50% of routing
information can be beaten by a simpler baseline".

## 6. Second-order result — Lean formalization sketch

**Target lemma (Cauchy interlacing for edge removal, in Mathlib scope):**

```
theorem eval_laplacian_edge_removal_le
  (G : SimpleGraph (Fin n)) (W : Fin n → Fin n → ℝ)
  (i j : Fin n) (h : i ≠ j) (w : ℝ) (k : Fin n) :
  eigenval (laplacian_of W (G \ single_edge i j w)) k
    ≤ eigenval (laplacian_of W G) k
```

**Proof sketch.** Write `L_{G\e} = L_G − w·(eᵢ−eⱼ)(eᵢ−eⱼ)ᵀ`; the subtracted term
is rank-1 PSD; apply Mathlib's existing Cauchy-interlacing theorem for symmetric
matrices under a rank-1 PSD perturbation; verify the Laplacian structure is
preserved. No new foundational mathematics is required. **This is a sketch: it
has not been compiled against Mathlib.**

## 7. Assumptions, looseness, and honest caveats

Theorems 1–2 are *upper bounds*, and loose ones; the empirical claims in the
repo do not depend on them holding tightly.

1. **Lipschitzness of the router (Step 3).** `π` is assumed `κ`-Lipschitz with a
   finite `κ`. Bounded GNN weights give a finite `κ`, but `κ` is typically large
   and unmeasured here; the constant `C` is therefore only a scaling argument,
   not a numeric prediction.
2. **TV ≤ mean-euclidean step (Step 4).** The bound
   `TV(π*,π̃) ≤ (1/|V|)Σ_v‖π*(v)−π̃(v)‖₂` is the Pinsker/`L¹`-vs-`L²` route and
   is itself loose.
3. **Single eigenvector per node.** The chain is written for one `v_k`; a router
   over a `k`-dimensional spectral embedding accumulates the per-mode bounds
   (a `√k` or sum, depending on the norm), which the constant `C` absorbs.
4. **`SD` vs `‖ΔL‖_F` normalization.** The theorem chain is in `‖ΔL‖_F`; `SD` is
   a *relative* per-mode max. The equivalence holds only where `λ_min^+` is
   bounded away from zero. When sparsification fragments the graph, `SD`
   saturates at exactly 1.0 and carries no gradient — observed at all sampled
   points of the real YelpChi retention ladder. The component-robust repairs
   (`rank_matched_distortion`, `src/distortion.py:32`;
   `low_frequency_distortion`, `src/distortion.py:172`;
   `routing_survival`, `src/distortion.py:253`) exist precisely because the
   legacy `SD` violates this assumption under fragmentation.
5. **Theorem 1 is a bound on the *oracle gap*, not on routing's usefulness.**
   It says how much routing information a distortion can destroy; it does not
   say routing was useful to begin with. Empirically it often is not: on the
   fraud graphs, the oracle itself is below no-routing (9/9 Protocol-B cells),
   so there is nothing for distortion to destroy.
6. **Unit-test role.** Weyl, Davis–Kahan, and Cauchy are checked numerically on
   every computed SPD matrix (`src/theoretical_bounds.py`,
   `tests/test_theoretical_bounds.py`); they are properties that must hold, not
   novel theorems.

## 8. What is proven vs measured

| Statement | Status |
|-----------|--------|
| Weyl, Cauchy, Davis–Kahan | classical; verified numerically in tests |
| Four-step chain (Thm 1) | valid chain under §7 assumptions; bound loose |
| Random-retention `(1−2ε)^k` (Thm 2) | heuristic high-probability bound; running-example value exact arithmetic |
| Lean edge-removal lemma | sketch, not compiled |
| SD/fragility predict routing *stability* | **measured** (Spearman +0.78…+0.99, `docs/real_attention_results.md` §9.1) |
| SD predicts routing *accuracy* drop | **not supported** (Spearman +0.231, p=0.448, §9.2) |
