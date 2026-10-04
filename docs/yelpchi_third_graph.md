> **RETRACTED (2026-10-04).** The D1/D3 oracle rows below (D1 = +0.0426 dz 25.74,
> D3 = +0.0443 dz 26.06) are self-routing artifacts and are void; the same
> partition scores -0.0101 against random and -0.0172 against no-routing under
> the fixed-expert protocol. The "Supported" verdict in the table below rests on
> the D2 comparison, which was against random routing rather than no-routing, and
> it does not survive protocol B. See `docs/fixed_expert_protocol.md` and
> `docs/d2_router_fix.md`.

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

YelpChi is the **second real fraud graph to pass D1–D4 end to end**. Note this
result holds for the legacy router too (see the cross-graph table below), so it is
evidence that *YelpChi has a recoverable label-free signal*, not evidence that the
D2 fix is what unlocked it.

## The cross-graph pattern

Full 10×3 runs of both routers on every graph (no pilots):

| graph | D2 legacy (kmeans+hx) | D2 fixed (pq+eig_nb_sim) | which wins | verdict |
|---|---|---|---|---|
| Tolokers | −0.0003 (p 0.70) | **+0.0040 (dz 4.24)** | fixed, decisively | Supported |
| YelpChi | **+0.0058 (dz 4.33)** | +0.0048 (dz 6.73) | legacy on mean, fixed on dz | Supported |
| Amazon | −0.00015 | +0.0004 (p 0.23) | neither | Not supported |

**YelpChi is not a win for the fix — the legacy router already passes D2 there.**
The full legacy run gives +0.00575 (CI [+0.00474, +0.00675], dz 4.33) versus the
fix's +0.00481 (CI [+0.00427, +0.00535], dz 6.73). The CIs overlap, so on mean
difference the two routers are not separated, but the legacy router's mean is
higher while the fix's effect size is higher — the fix produces a more
*consistent* signal across splits rather than a larger one. An earlier draft of
this document claimed the fix was required to unlock YelpChi; that came from a
3×1 pilot whose p=0.25 is the *resolution floor* for n=3, not evidence of failure.
The full run refutes it. Artifact:
`logs/yelpchi_fraud_protocol/protocol_results.json`.

The honest reading across the three graphs:

- **The fix matters on one graph, not two.** Tolokers is the only case where the
  legacy router genuinely fails (p=0.70) and the fix succeeds. Reporting YelpChi
  as a second fix success would have been wrong.
- **D2 is router-agnostic where the graph supports it.** Two of three graphs have
  a recoverable label-free signal; *how* you bucket it barely matters. That is a
  weaker and more believable claim than "our router is what unlocks D2."
- **`eig_nb_sim` correlates with homophily in the wrong direction on YelpChi**
  (−0.26) yet both routers still deliver. So the routing signal is *not* the
  homophily proxy being correlated — it is neighborhood structure that both
  orderings recover. This weakens the "spectral-smoothness proxy" mechanism
  story and should be described as an open question, not a settled mechanism.
- **Amazon still fails under both.** Its best candidate reaches ~23% of its
  oracle ceiling.

The practical rule the three graphs support: **screen candidates for correlation,
but decide by delivered D2 gain, and run both routers before claiming a fix
helps.** The p-value on a small-n pilot is not a failure signal.

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

### Note: fixed router under the ladder

The D2 fix was re-run under ER sparsification
(`logs/yelpchi_fraud_ladder_eig_nb_sim_pq/`). D1 is bit-identical to the legacy
ladder at every point, confirming D1 is router-independent. Fixed-router D2 under
ER is budget-sensitive: significant at r=0.30 (+0.0027, dz 3.20) and r=0.04
(+0.0031, dz 4.28), smaller at 0.15/0.60, and **non-significant at r=0.08**
(+0.00060, p 0.193, dz 0.52) where the legacy router is significant (+0.0055,
dz 7.21). Under ER pruning the fix is therefore not uniformly better and loses
at r=0.08.
