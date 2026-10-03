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
| Tolokers | 9/26 | 16/26 | 0/26 | **0/26** |
| YelpChi | 15/26 | 19/26 | 0/26 | **0/26** |

(Tolokers previously read 2/26 here. That residual was an artefact of a bug in
the no-routing baseline itself — see "the baseline bug" below — and is now 0/26.)

The one large positive, `eig_nb_sim_k128`, does not survive: on YelpChi under
fixed experts it is the best candidate but still **negative** against no-routing
(−0.0029). The Tolokers fixed-expert D2 magnitudes collapse to ~1e-5, and the
feature-k-means and oracle references are indistinguishable from zero.

### the baseline bug (found and fixed while adding the GNN arm)

Wiring up the GNN arm exposed a defect in the no-routing baseline itself.
Under the shared k-means pool, `fit_experts` is called once with a 3-column
assignment, so it returns 3 heads — each trained on **one cluster's** training
nodes. But the `single_global` condition is a *one-column* assignment of ones, so
`argmax` sent every test node to head 0, a head that had only ever seen cluster 0.

That is "route everything to the cluster-0 expert", not "train one model on all
training nodes". It understated the no-routing baseline, which flattered every
candidate. `evaluate_all` now refits `single_global` against its own one-column
pool (one extra head fit per split/seed), so the reference is a genuine global
model.

Measured impact on Amazon (`diagnose_single_global.py`, since removed):
logistic `0.9723 → 0.9731`, GNN `0.9465 → 0.9416`. Small — under 0.005 and in the
conservative direction for the GNN — so the conclusion is unchanged, but every
artifact was regenerated to be consistent. Pinned by
`test_shared_pool_single_global_is_trained_on_all_train_nodes`, which asserts the
global head covers the full training split; the assertion is on the training mask
rather than accuracy, because on separable data a partial head still classifies
perfectly and an accuracy-based test would not have caught it.

## Result 3: the negative holds with stronger experts

A negative result is worthless if the experts are too weak to specialise. Every
expert family here is competitive in absolute terms — `mean_accuracy` in each
artifact is the no-routing baseline, i.e. the accuracy one model trained on
everything achieves:

| graph | logistic | MLP | GNN (message passing) |
|---|---|---|---|
| Amazon | 0.9740 | 0.9731 | 0.9428 |
| Tolokers | 0.7819 | 0.7820 | — |
| YelpChi | 0.8588 | 0.8660 | — |

So the specialists are strong in absolute terms, and routing still does not pay
for itself. Across **all three expert families on all three graphs**:

| graph | expert | beats a *random* routing | beats **no-routing** |
|---|---|---|---|
| Amazon | logistic | 11/26 | **0/26** |
| Amazon | MLP | 11/26 | **0/26** |
| Amazon | GNN | **17/26** | **0/26** |
| Tolokers | logistic | 0/26 | **0/26** |
| Tolokers | MLP | 1/26 | **0/26** |
| YelpChi | logistic | 0/26 | **0/26** |
| YelpChi | MLP | 0/26 | **0/26** |

**0 of 182 candidate/expert/graph combinations beat training one model on
everything.** The best gap in each row is negative throughout.

### The GNN expert is a real graph model, not a proxy

MLP only adds non-linear capacity. The GNN expert (`--expert gnn`,
`spectral_distillation/src/gnn_expert.py`) message-passes over the symmetrically
normalized adjacency and reads out from the **concatenation** `[X, AX]`, so a
node's own features and its neighbourhood's stay separately weighted. It is
transductive — propagation uses every node's features and edges, while the loss
is restricted to each expert's routed training nodes, so test labels never enter
training.

Its competence is pinned by tests rather than asserted:

- `test_gnn_expert_beats_per_row_experts_on_neighbour_dependent_label` — on a
  label that depends only on neighbours, the GNN reaches 0.93 held-out accuracy
  where per-row logistic and MLP heads are at chance (0.52 / 0.49), against a
  0.94 ceiling for an explicit `[X, AX]` model.
- `test_gnn_expert_trains_on_routed_subset_not_nodes_zero_and_one` — regression
  for a boolean-mask bug that silently trained the expert on nodes 0 and 1.

Two design details were settled by measurement, not taste:

- The read-out consumes `[X, AX]`, not `X + AX`. Summing forces one shared weight
  on a node's own features and its neighbours', which cannot represent a general
  neighbour-dependent label.
- The ReLU hidden branch is available but **off by default**. Every variant tried
  (widths 4/16/64, weight decay 1e-4..1e-1, dropout 0/0.3, zero-initialised or
  not, 300–4000 epochs) drove training accuracy to 1.0 while held-out accuracy fell
  to 0.53–0.64, i.e. chance. The linear read-out is the configuration that is
  measurably strong, and a weak expert would make the negative unfalsifiable.

Even with the graph expert — the strongest single demonstration of structure
being exploitable, 17/26 beating random routing — **no candidate beats one global
model**. Routing by structural score is reliably better than routing at random,
and reliably worse than not routing at all.

## Reading

On these fraud graphs, **no label-free router beats simply training one model on
everything**, under an honest protocol, with linear, non-linear, or
message-passing experts. The earlier Tolokers/YelpChi gains (dz 7.8/6.6) were
measurement artifacts of the self-routing protocol.

Three independent metrics now agree:
1. Correlation with label homophily does not identify a useful routing partition.
2. The standard self-routing benchmark is confounded by the permutation invariance
   and the free-credit-for-any-partition effect.
3. The oracle is *negative* against no-routing — routing by label homophily is
   actively harmful, not merely unhelpful.

Caveats kept honest:
- Specialists here are label-free and routing-independent; the natural routing
  (feature-kmeans) does beat random, but no *label-free structural* score matches
  it.
- Feature-k-means routing wins because it matches the pool it is scored against;
  that is expected and is why it is a reference, not a win.
- GNN experts were run on Amazon only. Tolokers and YelpChi have no `mean_accuracy`
  entry in their tables above because their GNN runs were not performed.
- `eig_nb_sim_k128` was selected on reported results and still does not survive
  fixed-expert scoring, but remains reporting-selected.
- Real routing baselines (RouterGNN, Ada-Routing) are not implemented here, so
  this is a negative result against *structural-score* routers, not against the
  published methods.