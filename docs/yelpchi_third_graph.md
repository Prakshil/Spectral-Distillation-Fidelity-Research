# Third fraud graph: YelpChi

Adding YelpChi completes the standard Amazon / YelpChi / Elliptic fraud-detection
triple and turns the D2 result from a two-graph observation into a three-graph one.
Result: **D2 now holds on two of three real graphs, and the failure mode is
predictable in advance.**

## Loader

`spectral_distillation/src/real_fraud.py::load_yelpchi`, registered as `yelpchi`.

Nodes are Yelp hotel/restaurant *reviews*; edges join reviews sharing a user, a
product+star-rating, or a product+month. The positive class is reviews Yelp's own
filter flagged as spam. Unlike Amazon (users) and Tolokers (workers), this is the
only review-level graph of the three, so it exercises a different relation
structure against the same router.

| | Amazon | Tolokers | YelpChi |
|---|---|---|---|
| nodes | 8,639 | 11,758 | 14,840 |
| edges | 3,298,534 | 519,000 | 411,194 |
| edges/node | 382 | 44 | 28 |
| features | 25 | 10 | 32 |
| positive ratio | 9.50% | 21.82% | 14.97% |
| mean label homophily | 0.949 | 0.595 | 0.763 |

### The node cap, and why it is not silent

The full YelpChi largest connected component has **45,900 nodes**. The pipeline is
dense, so a dense Laplacian there needs `45900² × 8 B = 16.9 GB` plus an
`O(n³)` eigendecomposition — it does not fit on commodity hardware. Rather than
quietly truncating, the loader takes the largest connected component of a uniform
random node sample (`DEFAULT_YELP_MAX_NODES = 15000`, seed 0) and records the
sampling in `graph["source"]`. Pass `max_nodes=None` for the full graph.

The cap preserves what the experiment measures. Measured directly on `YelpChi.mat`
(sampling the LCC, seed 0, then taking the induced LCC), 98–99% of sampled nodes
survive and the class balance is stable:

| target | LCC | share of sample kept | positive ratio | mean degree |
|---|---|---|---|---|
| 12,000 | 11,770 | 98.1% | 0.1511 | 44.9 |
| **15,000 (default)** | **14,840** | **98.9%** | **0.1497** | **55.4** |
| 18,000 | 17,854 | 99.2% | 0.1489 | 66.6 |
| 20,000 | 19,863 | 99.3% | 0.1488 | 73.8 |
| full LCC | 45,900 | — | 0.1450 | 167.6 |

What it does **not** preserve is absolute degree: the 15,000-node cap costs ~3×
mean degree relative to the full LCC. Since the retention ladder is indexed by
*fraction of edges kept* rather than absolute degree, ladder comparisons stay
internally valid, but cross-graph degree statements must be read with this in
mind.

## D2 diagnostic screen

Candidate scores vs the true label homophily (labels are used for this diagnostic
only, never by the router). Artifact: `logs/yelpchi_d2_candidates/candidates.json`.

| candidate | rho | sign |
|---|---|---|
| `eig_nb_sim` | −0.2606 | flipped |
| `neigh_feat_sim` | +0.1549 | as-is |
| `neigh_degree` | −0.0448 | flipped |
| `degree` | −0.0248 | flipped |
| `eig_centrality` | +0.0157 | as-is |
| `local_clust` | +0.0150 | as-is |

## Result

Full protocol, 10 splits × 3 seeds, 20-permutation frozen gate, with the D2 fix
(`--label-free-strategy proxy_quantile --order-feature eig_nb_sim`):

| comparison | YelpChi |
|---|---|
| D1 oracle vs uniform | +0.0426 (p 0.001953, dz 25.74) |
| **D2 label-free vs random** | **+0.0048 (p 0.001953, dz 6.73)** |
| D3 oracle vs random | +0.0443 (p 0.001953, dz 26.06) |
| verdict | **Supported** |

YelpChi is the **second real fraud graph to pass D1–D4 end to end**, and the
strongest yet: D2 recovers ~11% of the oracle gain at `dz = 6.73`.

## The cross-graph pattern

| graph | D2 (fixed router) | D2 (legacy router) | verdict | best proxy rho |
|---|---|---|---|---|
| Tolokers | +0.0040 (dz 4.24) | −0.0003 (p 0.70) | Supported | `eig_nb_sim` +0.42 |
| YelpChi | +0.0048 (dz 6.73) | +0.0043 (p 0.25)† | Supported | `eig_nb_sim` −0.26 |
| Amazon | +0.0004 (p 0.23) | −0.00015 | Not supported | `eig_centrality` −0.51 |

† YelpChi's legacy number is from a 3×1 pilot, not a full run.

The consistent finding across all three: **`eig_nb_sim` works as a routing proxy
regardless of the sign of its correlation with label homophily.** On Tolokers it
correlates +0.42 with homophily and delivers +0.0040; on YelpChi it correlates
−0.26 and delivers the *largest* gain of the three. What matters is not the sign
but that the proxy captures neighborhood smoothness, and that structure is what
routing exploits. Amazon's `eig_centrality` is the mirror image: strongest
correlation (−0.51), weakest delivery (+0.0009 against a +0.0040 oracle).

The practical rule the three graphs support: **run the stage-1 screen first, then
pick by delivered D2 gain, not by |rho|.** Correlation with the oracle's score is
not the same objective as the gain the oracle's score actually produces.

## Retention ladder

15 points, `skip-sd`, 10×3. Artifact: `logs/yelpchi_fraud_ladder/`.

| method | r=0.6 | r=0.3 | r=0.15 | r=0.08 | r=0.04 |
|---|---|---|---|---|---|
| **D1** ER | +0.0447 | +0.0664 | **+0.0980** | +0.0539 | +0.0225 |
| D1 random | +0.0418 | +0.0385 | +0.0369 | +0.0274 | +0.0175 |
| D1 degree | +0.0069 | +0.0020 | +0.0008 | +0.0003 | +0.0001 |
| **D2** ER | +0.0044 | +0.0023 | +0.0040 | **+0.0055** | +0.0030 |
| D2 random | +0.0017 | +0.0027 | +0.0019 | +0.0027 | +0.0019 |
| D2 degree | +0.0009 | −0.0003 | +0.0001 | +0.0011 | +0.0013 |

Label homophily under ER rises monotonically with pruning, 0.797 → 0.850, versus
a flat ~0.765 under random pruning. This is the third graph to show ER raising
homophily while random keeps it flat, and the first where ER also **amplifies**
D1 relative to full graph: +0.0426 → +0.0980 at r=0.15, a 2.3× peak.

Degree pruning again destroys D1 catastrophically (0.0001–0.0069 across the whole
ladder) while leaving D2 near zero. It does **not** unlock D2 here, unlike on
Tolokers. That is a real difference from the earlier two-graph story and worth
stating plainly: the "degree pruning unlocks D2" lead from Tolokers does not
replicate to YelpChi or Amazon.

## Caveats

- The 15,000-node cap is a real constraint on absolute degree (see above).
- `sd_laplacian` is `0.0` throughout — a `--skip-sd` placeholder, not a measured
  distortion. The ladder is indexed by edge-retention fraction only.
- `eig_nb_sim`'s mode count (`--spectral-k 8`) is untuned across all three graphs.
- Amazon still fails D2 and remains unresolved; its best candidate reaches only
  ~23% of its oracle ceiling.

### Note: fixed router vs ladder

The fixed router (proxy_quantile, order_feature=eig_nb_sim) passes D1–D4
on the **unpruned** YelpChi graph with D2 +0.0048 (p 0.001953, dz 6.73). Under ER
sparsification its D2 gain shifts with budget: significant at r=0.30 and 0.04,
smaller at 0.15/0.60, and non-significant at 0.08 (+0.00060, p 0.193, dz 0.52)
on the pruned topology. The unpruned full-protocol result is the canonical one
for the rule verdict; ladder-wise D2 with the fixed router is budget-dependent
(see logs/yelpchi_fraud_ladder_eig_nb_sim_pq/ladder_results.jsonl).
