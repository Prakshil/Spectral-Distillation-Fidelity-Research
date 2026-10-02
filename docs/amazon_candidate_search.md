# Amazon D2 candidate search

Broad, predeclared search for a label-free router that closes the Amazon D2 gap.
Runner: `experiments/run_candidate_search.py`. Artifact:
`logs/amazon_candidate_search/candidates.json`.

## What changed versus `run_d2_candidates.py`

The committed screen tested 6–7 candidates and then **evaluated only the top 4 by
`|rho|`** against label homophily — selecting what to test using the same labels
used for the significance test. This run removes both weaknesses:

1. **No screen-then-test.** The pool of 28 candidates is declared in code up
   front and **every** candidate is evaluated on the full 10 splits × 3 seeds
   protocol. Marginal cost is ~0.04 s/cell, so there was never a reason to
   pre-filter.
2. **Holm over a known family.** Correction is applied across all 28 predeclared
   `as_defined` comparisons, not across whichever subset survived screening.

Both orientations are still computed and reported. They turned out to be exactly
equal — see below — so this is recorded as an empirical confirmation of an
invariance, not as a deployability gap.

## Pool (28 candidates, all label-free)

- **Feature space:** `neigh_feat_sim`, `feat_anomaly`, `neigh_feat_var`
- **Degree:** `degree`, `log_degree`, `neigh_degree`, `neigh_deg_dissimilarity`
- **Triangles:** `triangle_count`, `triforce`, `local_clust`
- **Diffusion:** `inv_pagerank`, `inv_walk2_mixing`
- **Spectral:** `eig_nb_sim_k{2,8,32,128}`, `eig_nb_sim_abs_k{2,8,32,128}`,
  `eig_centrality_k{2,8,32,128}`, `fiedler_nb_sim`, `spectral_energy`
- **Resistance:** `inv_mean_res_nb`, `inv_min_res`

All spectral candidates share **one** eigendecomposition, sliced across `k`. The
eigenvector sign is not canonicalized; `_row_normalize` is applied to `|V|` for
the `_abs_` variants precisely because raw `V` has arbitrary global sign.

## Channel-permutation invariance (new finding)

Every anti-correlated candidate returned **identical** D2 in both orientations,
to full float precision. This was verified directly rather than assumed:
`bucket_from_score(s)` and `bucket_from_score(-s)` are a channel 0↔2 swap, and

- `fit_experts` trains expert `k` on exactly the nodes routed to `k`;
- `mixture_accuracy` scores expert `k` on exactly those nodes.

So accuracy is `sum_k acc(group_k trained on group_k)`, which is independent of
the index each group occupies. **D2 is provably invariant to score sign.**

Two consequences:

- The orientation machinery in `run_d2_candidates.py` cannot change any D2
  number. The earlier claim in `d2_router_fix.md` that a reversed candidate
  "can never work", and that Amazon's `eig_centrality` gain depended on a
  sign-flip orientation, are both **wrong** and are corrected in that file.
- A candidate cannot be rescued or broken by flipping it. Only the induced
  3-way partition matters. The search space is "which quantile partition of
  nodes", not "which oriented score".

Pinned by `test_mixture_accuracy_invariant_to_channel_permutation`.

## Result: Amazon D2 remains unfixed

Oracle ceiling: **D2 = +0.00396** (dz 4.20).

| candidate | D2 (deployable) | p | dz | Holm |
|---|---|---|---|---|
| `eig_centrality_k128` | +0.00128 | 0.00195 | +2.20 | no |
| `eig_nb_sim_k2` | +0.00082 | 0.00977 | +1.32 | no |
| `eig_nb_sim_k8` | +0.00066 | 0.01953 | +0.95 | no |
| `eig_nb_sim_abs_k2` | +0.00054 | 0.08398 | +0.74 | no |
| `eig_nb_sim_k32` | +0.00049 | 0.01367 | +1.15 | no |
| `fiedler_nb_sim` | +0.00047 | 0.10547 | +0.66 | no |

**0 of 28** predeclared candidates give a Holm-significant positive D2.

`eig_centrality_k128` has the best raw effect (+0.00128, 32% of the oracle
ceiling, dz 2.20) but its p is already at the two-sided Wilcoxon floor of
`1/512 = 0.001953`, which fails the first Holm threshold `0.05/28 = 0.001786`.
The previously reported Amazon win (`+0.00093`, corrected `p = 0.003906`) was an
artifact of correcting over only 6 candidates.

## `rho` is not a selection criterion

The strongest label-homophily proxy in the entire pool is `fiedler_nb_sim` at
`rho = +0.534` — far above anything in the original screen — and it delivers only
`+0.00047` (p 0.105), the 6th-best D2. `neigh_feat_var` (`rho = -0.490`),
`neigh_degree` (`+0.449`) and `eig_centrality_k2` (`-0.417`) all correlate far
more strongly with homophily than the candidates that actually moved D2.

This is now a consistent pattern across three independent experiments:

- spectral `k` sweep: YelpChi `rho` gets *worse* (`-0.249` → `-0.288`) from
  `k=2` → `k=128` while D2 nearly doubles;
- SD ladder: rank-matched spectrum error is minimised by candidates that destroy
  the graph (97.4% isolated nodes on YelpChi);
- this search: top-`rho` candidate ranks 6th in D2.

**Correlation with label homophily does not identify a useful routing
partition.** Selecting candidates by `rho` — which the original screen did — is
unsound, and the reason it produced a false positive is now understood.

## Reading

Amazon's D2 gap is not caused by a missing feature, a bad `k`, or a sign error.
Within 28 diverse structural and spectral scores, the best achievable D2 is about
a third of the oracle ceiling and none survives multiplicity correction. The
label-free partition simply does not align with homophily well enough on this
graph. Since `rho` fails as a guide and sign is provably irrelevant, further
candidate enumeration on the same signal is unlikely to pay off; the honest next
step is either a held-out confirmation of `eig_centrality_k128` or accepting that
Amazon needs information `A` and `X` do not carry.
