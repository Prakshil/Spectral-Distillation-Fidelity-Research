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
| Tolokers | GNN | 3/26 | **0/26** |
| YelpChi | logistic | 0/26 | **0/26** |
| YelpChi | MLP | 0/26 | **0/26** |
| YelpChi | GNN | 1/26 | **0/26** |

**0 of 234 candidate/expert/graph combinations beat training one model on
everything** (44 beat a random routing). The best gap in each row is negative
throughout.

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
- GNN experts were run on all three graphs (Amazon, Tolokers, capped YelpChi), so
  the message-passing result is not an Amazon-only artifact. It still fails to beat
  no-routing anywhere.
- `eig_nb_sim_k128` was selected on reported results and still does not survive
  fixed-expert scoring, but remains reporting-selected.
- Real routing baselines (RouterGNN, Ada-Routing) are not implemented here, so
  this is a negative result against *structural-score* routers, not against the
  published methods.

## Positive control: can the protocol detect a win at all?

A 0/234 result is only meaningful if the harness would have reported a win when one
existed. `spectral_distillation/src/positive_control.py` supplies a task where
routing is provably useful.

The label is a **non-separable mixture of linear rules**: three latent groups, each
with its own linearly separable boundary over its own feature pair, no single
hyperplane satisfying all three. A marker feature offsets the groups in feature
space (so feature k-means recovers them at purity 1.000) and the graph is
block-diagonal over the same groups, so the task is graph-native.

Run: `python -m spectral_distillation.experiments.run_positive_control`
→ `logs/positive_control/fixed_expert.json` (5 splits × 3 seeds)

| expert | no-routing | oracle | k-means | random | oracle − no-routing |
|---|---|---|---|---|---|
| logistic | 0.7853 | 0.9806 | 0.9806 | 0.7470 | **+0.1953** |
| MLP | 0.9758 | 0.9874 | 0.9874 | 0.6774 | +0.0116 |

Random routing does **not** win, which rules out a degenerate "everything beats the
baseline" harness. The MLP contrast is the informative one: an MLP expressive
enough to represent the mixture globally never needed routing, so the gain
collapses to +0.01. The size of the gap is the value of routing on this task.

Pinned by `test_positive_control_oracle_beats_global_where_routing_is_useful`
(requires > +0.10) and `test_positive_control_generator_is_deterministic`.

## RouterGNN-lite: a router trained on labels

Every routing scored above is fixed and label-free. That leaves the obvious
objection open: maybe a router that *learns* which expert suits which node wins
where none of the hand-designed scores did.

`spectral_distillation/src/learned_router.py` adds that condition without weakening
the protocol:

- the pool is the same routing-blind feature k-means partition, **frozen** before
  routing is learned;
- supervision is training labels only — never test labels, never the oracle's
  label-homophily buckets;
- every baseline is scored against that same frozen pool on the same split.

The one design point that matters: **experts are fitted on `fit_mask`, and the
router's supervision targets are computed on a disjoint `router_mask`.** Without
that split the target is degenerate — each expert classifies its own training nodes
almost perfectly, so "which expert owns this node" collapses to "which k-means
cluster is this node" and the learned number is just k-means routing wearing a
disguise. Holding the experts out makes the target mean "which expert *generalizes*
to this node", which is what a router must guess at deployment.

"Lite" refers to the router: a small MLP over cached `[X, AX]` propagated features,
the same caching trick `gnn_expert.py` uses. Re-propagating every step cost ~1s per
epoch on Amazon; caching makes 3 splits × 200 epochs run in ~20s.

Run: `python -m spectral_distillation.experiments.run_learned_router --dataset amazon`

| graph | no-routing | learned router | k-means | random | oracle | learned − no-routing | runs won |
|---|---|---|---|---|---|---|---|
| Amazon | 0.9734 | 0.9726 | 0.9727 | 0.9319 | 0.9285 | −0.0008 | 1/3 |
| Tolokers | 0.7826 | 0.7822 | 0.7820 | 0.7817 | 0.7815 | −0.0004 | 1/3 |
| YelpChi (capped) | 0.8564 | 0.8571 | 0.8578 | 0.8530 | 0.8473 | +0.0006 | 2/3 |

**Every gap is ±0.002 — indistinguishable from noise**, and the sign flips across
datasets. The router tracks the frozen k-means partition closely (agreement
0.685/0.317/0.371), i.e. it mostly rediscovers the pool it was handed rather than
finding new structure.

The control matters: on the positive control the same router reaches 0.9596 against
a global 0.7838, recovering **+0.176 of the oracle's +0.191**. So the router is not
broken and the harness is not blind — on these three fraud graphs there is simply
no per-node expert-selection signal to exploit.

Pinned by `test_learned_router_recovers_routing_gain_on_positive_control` and
`test_learned_router_never_sees_test_labels` (flipping every test label must leave
router parameters bit-identical).

### Revised scope

The claim is now narrower and better supported: **neither fixed structural routers
nor a label-trained router beat a single global model on these fraud graphs**, and
the harness provably would have detected a win had one existed. Ada-Routing and
other published learned baselines remain unimplemented.