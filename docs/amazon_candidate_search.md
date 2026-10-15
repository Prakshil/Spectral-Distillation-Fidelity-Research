# Amazon D2 candidate search

Broad, predeclared search for a label-free router that closes the Amazon D2 gap.
Runner: `experiments/run_candidate_search.py`. Artifact:
`logs/amazon_candidate_search/candidates.json`.

## What changed versus `run_d2_candidates.py`

The committed screen tested 6-7 candidates and then **evaluated only the top 4 by
`|rho|`** against label homophily — selecting what to test using the same labels
used for the significance test. This run removes both weaknesses:

1. **No screen-then-test.** The pool of 28 candidates is declared in code up
   front and **every** candidate is evaluated on the full 10 splits x 3 seeds
   protocol. Marginal cost is ~0.04 s/cell, so there was never a reason to
   pre-filter.
2. **Holm over a known family.** Correction is applied (step-down) across all 28
   predeclared `as_defined` comparisons, not across whichever subset survived
   screening.
3. **Routing-blind scoring pool.** Candidates are now scored against a frozen
   `feature_kmeans` pool (identical to the Protocol-B pool in
   `run_fixed_expert_protocol.py`) rather than against experts trained on each
   candidate's own assignment. A candidate is therefore judged on whether it
   routes nodes to a *better fixed model*, not on whether a partition can support
   a specialist trained on the very nodes it selected.

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

## Sign orientation is no longer inert (corrected)

An earlier version of this doc reported that D2 was **identical** in both
orientations (`bucket_from_score(s)` vs `bucket_from_score(-s)`) and concluded
that "D2 is provably invariant to score sign". That argument assumed experts are
trained on exactly the nodes the router sends them:

- `fit_experts` trains expert `k` on the nodes routed to `k`;
- `mixture_accuracy` scores expert `k` on exactly those nodes;

so accuracy is `sum_k acc(group_k trained on group_k)`, independent of the index
each group occupies. It does **not** survive the switch to a routing-blind frozen
pool: the pool ties each channel index to a **fixed** expert, and flipping the
score is a 0<->2 channel swap, so the two orientations route the high- and
low-score nodes to different fixed experts and give different accuracy.

The artifact records both. For `eig_centrality_k8`, as-defined D2 is `+0.02694`
and label-oriented D2 is `-0.00701`; the two agree only for the small family of
candidates whose fixed pool happens to be symmetric in channels 0 and 2. Only the
**as-defined** orientation enters the predeclared 28-way family; the
label-oriented column is reported for reference.

The `test_mixture_accuracy_invariant_to_channel_permutation` test still pins the
invariance of `mixture_accuracy` under self-trained experts; it is simply not the
operative scoring rule here.

## Result: Amazon D2 remains unfixed

Oracle ceiling on the frozen pool: **D2 = -0.00464** (dz -0.45) — the
label-informed 3-bucket oracle partition does **not** beat random routing on top
of the pool.

| candidate | D2 (as-defined) | p | dz | Holm adj. | label-oriented |
|---|---|---|---|---|---|
| `eig_centrality_k8` | +0.02694 | 0.00195 | +3.49 | 0.0547 | -0.00701 |
| `neigh_feat_var` | +0.02579 | 0.00195 | +3.43 | 0.0547 | -0.00489 |
| `eig_centrality_k2` | +0.02548 | 0.00195 | +3.00 | 0.0547 | -0.00724 |
| `eig_centrality_k128` | +0.02292 | 0.00195 | +2.80 | 0.0547 | -0.00948 |
| `eig_centrality_k32` | +0.02243 | 0.00195 | +2.80 | 0.0547 | -0.00951 |
| `local_clust` | +0.02108 | 0.00195 | +3.01 | 0.0547 | -0.01186 |

**0 of 28** predeclared candidates give a Holm-significant positive D2.

`eig_centrality_k8` has the best raw effect (+0.0269, dz 3.49), but its p sits on
the two-sided Wilcoxon floor `1/512 = 0.001953`, above the first step-down
threshold `0.05/28 = 0.001786`; the adjusted p is `0.0547` and the whole family
fails. The earlier reported Amazon win (`+0.00093`, "corrected" `p = 0.003906`)
was an artifact of correcting over only 6 candidates.

Because the frozen pool is a fixed baseline, the raw as-defined effects are
larger than under the previous self-specialized scoring (which produced
`+0.001`-scale numbers); the predeclared multiplicity test is what decides the
claim, and it does not pass, nor does the label-informed oracle beat random on the
same pool. On this evidence Amazon's D2 gap is unfixed.

## `rho` is not a selection criterion

The strongest label-homophily proxy in the entire pool is `fiedler_nb_sim` at
`rho = +0.534` — far above anything in the original screen — and it delivers only
`-0.00094` (rank 15 of 28). `neigh_degree` (`rho = +0.449`) gives `-0.00107` and
`eig_nb_sim_k128` (`+0.346`) gives `-0.00347`. The candidates that actually move
D2 are the `eig_centrality` family, sitting at `|rho| ~= 0.4` — mid-pack.
Correlation with label homophily does not line up with deployable D2.

This is now a consistent pattern across three independent experiments:

- spectral `k` sweep: YelpChi `rho` gets *worse* (`-0.249` -> `-0.288`) from
  `k=2` -> `k=128` while D2 rises from `+0.0043` to `+0.0065`;
- SD ladder: rank-matched spectrum error is minimised by candidates that destroy
  the graph (97.4% isolated nodes on YelpChi);
- this search: the top-`rho` candidate (`fiedler_nb_sim`) ranks 15th in D2 and is
  negative, while the top D2 candidates sit at `|rho| ~= 0.4`.

**Correlation with label homophily does not identify a useful routing
partition.** Selecting candidates by `rho` — which the original screen did — is
unsound, and the reason it produced a false positive is now understood.

## Reading

Amazon's D2 gap is not caused by a missing feature, a bad `k`, or a sign error.
Within 28 diverse structural and spectral scores, no candidate survives
multiplicity correction, and the label-informed oracle ceiling on the same frozen
pool is non-positive. The label-free partition simply does not align with
homophily well enough on this graph. Since `rho` fails as a guide, further
candidate enumeration on the same signal is unlikely to pay off; the honest next
step is either a held-out confirmation of `eig_centrality_k8` or accepting that
Amazon needs information `A` and `X` do not carry.
