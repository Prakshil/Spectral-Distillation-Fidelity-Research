# Fixed-expert (cross-fit) router protocol

Runner: `experiments/run_fixed_expert_protocol.py`. Artifacts:
`logs/{amazon,tolokers,yelpchi}_fixed_expert/fixed_expert.json`.

## Why this protocol exists

The existing self-routing protocol in `src/mixture.py` does two things that make
it unable to measure routing:

1. **Permutation invariance.** `fit_experts` trains expert `k` on exactly the
   nodes routed to `k`, and `mixture_accuracy` scores expert `k` on exactly those
   nodes. Accuracy is therefore `sum_g acc(group_g | model trained on group_g)`,
   which does not depend on which channel index a group occupies. Negating a
   score — swapping which quantile bucket is labelled 0 vs 2 — cannot change
   accuracy. The field's "orient the score by its label correlation" step is a
   no-op, and D2 cannot distinguish a score from its negation.
2. **Free credit for any partition.** Because every partition is scored by experts
   fitted to that same partition, an arbitrary split still collects partial
   credit. D2 partly measures "do subgroups support their own specialist", not
   "does this routing send nodes to a better model".

## The fix

Build the expert pool **first**, from a routing-blind but structurally
meaningful basis (k-means on features), then vary only the routing. Expert `k` is
permanently bound to feature region `k`, so:

- channel index becomes identifiable again, so score sign has real content;
- routing is the only free variable, so differences are honestly due to routing;
- feature-k-means enters as a label-free baseline the old protocol never had.

A single-global model (one expert on all nodes) is included as the reference that
isolates the value of routing: `d2_vs_no_routing > 0` means routing actively
helps; `< 0` means it is worse than no routing at all.

## Result 1: the permutation invariance is real and the fix removes it

| graph | self-routing sign-invariant | fixed-expert sign-invariant |
|---|---|---|
| Amazon | 26/26 | 0/26 |
| Tolokers | 26/26 | 0/26 |
| YelpChi | 26/26 | 0/26 |

Every candidate has `D2(s) == D2(-s)` to machine precision under self-routing, and
none does once the expert pool is fixed — exactly as the theory predicts. Pinned by
`test_mixture_accuracy_invariant_to_channel_permutation`.

## Result 2: the reported gains were self-routing artifacts

The D2 **rank** correlation between the two protocols is near zero: Amazon +0.28,
Tolokers **−0.73**, YelpChi +0.08. The old protocol was not merely noisy; it was
ranking candidates essentially independently of the honest one.

Under self-routing, the earlier "wins" appear (Tolokers oracle D2 +0.038 dz 13.4;
YelpChi oracle D2 +0.044 dz 31.7; many candidates Holm-significant). Under the
honest fixed-expert protocol those collapses:

| graph | self-routing Holm-sig | self-routing beats no-routing | fixed-expert Holm-sig | fixed-expert beats no-routing |
|---|---|---|---|---|
| Amazon | 0/26 | 1/26 | 11/26 | **0/26** |
| Tolokers | 9/26 | 16/26 | 0/26 | **2/26** |
| YelpChi | 15/26 | 19/26 | 0/26 | **0/26** |

The one large positive, `eig_nb_sim_k128`, does not survive: on YelpChi under
fixed experts it is the best candidate but still **negative** against no-routing
(−0.0029). The Tolokers fixed-expert D2 magnitudes collapse to ~1e-5, and the
feature-k-means and oracle references are indistinguishable from zero.

## Result 3: with stronger (MLP) experts the negative holds

To rule out "the linear head is simply too weak to specialise," Amazon was rerun
with a 64-unit ReLU MLP expert (`--expert mlp`, artifact
`logs/amazon_fixed_expert_mlp/`). The result is unchanged:

- single-global (no routing) D2 = **+0.0434** (dz 21.5)
- oracle D2 = **−0.0233** (dz −13.6; routing by label homophily is actively harmful)
- feature-kmeans D2 = +0.0442
- 11/26 candidates beat a *random* routing, but **beats no-routing: 0/26** —
  every single candidate is worse than one global MLP

So the negative is not an artefact of linear experts: giving each expert genuine
non-linear capacity does not make routing pay for itself.

## Reading

On these fraud graphs, **no label-free router beats simply training one model on
everything**, under an honest protocol, with either linear or non-linear experts.
The earlier Tolokers/YelpChi gains (dz 7.8/6.6) were measurement artifacts of the
self-routing protocol.

Two independent metrics now agree:
1. Correlation with label homophily does not identify a useful routing partition
   (three separate experiments).
2. The standard self-routing benchmark is confounded by the permutation
   invariance and the free-credit-for-any-partition effect.

Caveats kept honest:
- Specialists here are label-free and routing-independent; the natural routing
  (feature-kmeans) does beat random, but no *label-free structural* score matches
  it.
- Feature-k-means routing wins because it matches the pool it is scored against;
  that is expected and is why it is a reference, not a win.
- GNN experts (as in the routing literature) are not yet tested; MLP is a proxy
  for non-linear capacity, not a graph model.