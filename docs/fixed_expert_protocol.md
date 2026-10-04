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
| Amazon | GNN | **16/26** | **0/26** |
| Tolokers | logistic | 0/26 | **0/26** |
| Tolokers | MLP | 1/26 | **0/26** |
| Tolokers | GNN | 2/26 | **0/26** |
| YelpChi | logistic | 0/26 | **0/26** |
| YelpChi | MLP | 0/26 | **0/26** |
| YelpChi | GNN (capped) | 2/26 | 2/26 † |

**0 of 234 candidate/expert/graph combinations beat training one model on
everything** (43 beat a random routing). The best gap in each row is negative
throughout.

† The two YelpChi cases are `eig_nb_sim_k32` and `eig_nb_sim_abs_k32`, both
positive by ~+0.002 with p=0.25 on a 3-split capped run — not significant, and
they appear only because YelpChi must be node-capped for memory (14840 of 45954
nodes, 3 splits × 1 seed instead of 10 × 3). Every full-scale cell is 0/26. Do not
quote this row as a routing win; it is the reduced-power arm, and the honest
summary remains 0/26 at full power.

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
being exploitable, 16/26 beating random routing — **no candidate beats one global
model**. Routing by structural score is reliably better than routing at random,
and reliably worse than not routing at all.

## Result 4: the frozen pool is not complementary either (Ens-Avg)

A "routing does not help" result has an obvious alternative explanation: maybe the
*pool* carries the capacity and the router is incidental. The control is to average
the **same frozen pool** with no routing at all — every test node scored by all
experts, no assignment.

Ens-Avg averages class-1 **probabilities**, not hard labels. An earlier version
majority-voted on `predict` output, which throws away confidence and measures
voting strength instead of pool complementarity; the two copies of that helper had
also drifted apart. Fixed, and pinned by
`test_ensemble_averages_probabilities_not_majority_vote`, which builds a pool where
averaging and voting provably disagree. The graph expert needed a new
`predict_proba_nodes` hook for this — it previously exposed only hard labels, so
the GNN pool would have been silently dropped from the very control meant to
explain the GNN results.

GNN pool, 26 candidates, protocol B:

| graph | Ens-Avg | no-routing | Δ | p | beats no-routing |
|---|---|---|---|---|---|
| Amazon | 0.9140 | 0.9428 | −0.0288 | 0.0020 | no |
| Tolokers | 0.5174 | 0.6721 | −0.1547 | 0.0020 | no |
| YelpChi (capped) | 0.7320 | 0.7166 | +0.0154 | 0.5000 | no |

Ens-Avg loses to no-routing on all three, significantly on the two full-scale
graphs. So the capacity explanation is **also** refuted: the k-means pool is not
complementary, and neither routing nor averaging extracts anything from it. This
strengthens the negative result — it is not "routing failed but the pool was
valuable".

The YelpChi Ens-Avg is the one condition in the whole study that nominally favours
the pool (+0.0154), and it is the reduced-power arm: 3 splits × 1 seed, p=0.5.
Reported for completeness, not as support.

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

### The oracle's negativity is not a training-budget artefact

A routed arm gives each expert ~N/K rows while the no-routing baseline gets all
N, so "routing loses" could be arithmetic rather than evidence. It is not. A
*single* global model trained on a random N/K subsample matches the full-data
model, and matches the routed arms:

| graph | global, all N | global, N/3 rows | random K=3 routing |
|---|---|---|---|
| Amazon | 0.9740 | 0.9719 | 0.9732 |
| Tolokers | 0.7819 | 0.7809 | 0.7807 |

Data fragmentation is free here, so the oracle's deficit is a genuine property of
the partition, not of the budget.

### One caveat on the oracle number, stated precisely

The oracle falls below no-routing in **9/9** protocol-B cells (3 graphs × 3 expert
types), the largest gap being Tolokers GNN at −0.178. But under protocol A — the
self-routing arm already shown to be a permutation artefact — oracle routing
*with a pool fitted to its own routing* scores **+0.0365 on Tolokers, winning
30/30** paired splits, and +0.0028 on Amazon (29/30).

That apparent headroom is label leakage through the router, not specialisation.
`oracle_bucket_assignment` buckets on label homophily, and on Amazon two of its
three buckets have train positive-rate exactly 0.000, so `fit_experts` installs a
`DummyClassifier` for them; 1468 of 4320 held-out Amazon nodes are routed into a
degenerate-prior bucket. On Tolokers no bucket is degenerate, yet the gain persists
— consistent with homophily buckets acting as a coarse label proxy that a
self-fitted expert pool can exploit. Protocol B removes exactly this by fixing a
routing-blind pool, which is why 9/9 is the number to quote and this one is not.

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
  this is a negative result against *structural-score* routers and a label-trained
  MLP gate, not against the published methods. Node-MoE's own expert construction
  (differentiated ChebNetII filter initialisation) is not reproduced; the gate
  features are, the experts are not.
- The `x_only` arm rules out implicit routing for *this* pipeline, not in general.
  A ragged-edge / subgraph-restricted fit would be a stronger control still.
- YelpChi stays node-capped (14840 of 45954) for memory, so its learned-router arm
  runs 1 split × 1 seed and its GNN arm 3 × 1. Treat those two rows as
  reduced-power and do not compare their gaps to the full-scale cells.

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

"Lite" refers to the router: a small MLP over cached propagated features, the same
caching trick `gnn_expert.py` uses. Re-propagating every step cost ~1s per epoch on
Amazon; caching makes 3 splits × 100 epochs run in ~20s.

The gate features are Node-MoE's (`arXiv:2406.03464`): `[X, |AX−X|, |A²X−X|]`.
Node-MoE shows raw `X` is insufficient for routing because the gate needs
neighbourhood discrepancy over 1–2 hops to estimate per-node structural regime;
the previous `[X, AX]` form did not expose that discrepancy.

Run: `python -m spectral_distillation.experiments.run_learned_router --dataset amazon`

| graph | no-routing | learned router | k-means | random | oracle | Ens-Avg | learned − no-routing | learned − Ens-Avg |
|---|---|---|---|---|---|---|---|---|
| Amazon | 0.9734 | 0.9731 | 0.9727 | 0.9319 | 0.9285 | 0.9682 | −0.0003 | +0.0049 |
| Tolokers | 0.7826 | 0.7819 | 0.7820 | 0.7817 | 0.7815 | 0.7818 | −0.0006 | +0.0002 |
| YelpChi (capped) | 0.8555 | 0.8533 | 0.8586 | 0.8503 | 0.8447 | 0.8553 | −0.0022 | −0.0020 |

**Every gap to no-routing is ≤0.002 — indistinguishable from noise** — and the sign
flips across datasets. Against the *ensemble* baseline the learned router is
nominally ahead on Amazon and Tolokers by ≤0.005 and behind on YelpChi; none of
these separate either. The router tracks the frozen k-means partition closely
(agreement 0.731/0.314/0.550), i.e. it mostly rediscovers the pool it was handed
rather than finding new structure.

Note the oracle is *below* no-routing in all three rows. The label-optimal routing
is worse than training one model, which bounds the achievable routing gain at ≤ 0
here: no router, however good, could have won. This is a bound rather than a
failure to search.

The control matters: on the positive control the same router reaches 0.9541 against
a global 0.7837, recovering **+0.170** of the oracle's +0.189. So the router is not
broken and the harness is not blind — on these three fraud graphs there is simply
no per-node expert-selection signal to exploit.

### Ens-Avg on the positive control is the mechanism, not just a baseline

| condition | positive control | real graphs (GNN pool) |
|---|---|---|
| no-routing | 0.7837 | 0.9428 / 0.6721 / 0.7166 |
| Ens-Avg | 0.8011 | 0.9140 / 0.5174 / 0.7320 |
| k-means routing | **0.9730** | 0.9401 / — |
| oracle routing | **0.9730** | 0.8339 / 0.4939 / 0.6886 |
| learned router | **0.9541** | 0.9731 / 0.7819 / 0.8533 |

On the positive control the experts are genuine *specialists* — each is optimal on
one latent group — so averaging their probabilities blurs three distinct decision
boundaries into one mediocre one (+0.017 over global), while routing to the right
specialist recovers almost everything (+0.19). Ens-Avg and routing are not
redundant controls; they fail in opposite regimes, and which one wins tells you
what kind of pool you have.

That is the interpretive key for the real graphs. Ens-Avg landing at or below
no-routing there, *while* k-means routing also fails to beat no-routing, says the
k-means pool is neither specialist (routing cannot exploit it) nor complementary
(ensembling cannot exploit it). A pool with no exploitable structure at all — which
is precisely the low-heterogeneity regime the routing-payoff hypothesis predicts.

Pinned by `test_learned_router_recovers_routing_gain_on_positive_control` and
`test_learned_router_never_sees_test_labels` (flipping every test label must leave
router parameters bit-identical).

## Result 5: the null survives removing every graph channel (implicit routing)

The strongest objection to a negative routing result is *implicit routing*
("On the Benefits of Learning to Route in Mixture-of-Experts Models",
EMNLP 2023): freeze the router and the earlier layers **still route internally**,
so a frozen-assignment protocol can understate routing without any router present.
Here the channel is real — a GNN expert consumes `AX` whether or not any router
selects it, so a frozen k-means assignment could still be routing internally
through the experts' own message passing.

`--router-mode x_only` closes it. The gate becomes `X` alone (no `AX`, no
`A²X`), paired with a per-row expert that ignores the graph, so nothing in the arm
can consult structure. The runner **rejects** `x_only` with `--expert gnn`, because
the graph expert would keep the channel open and the arm would be uninterpretable.

Cleanliness is pinned, not asserted:
- `test_router_x_only_mode_removes_every_graph_term` — scrambling the adjacency
  leaves the `x_only` gate bit-identical while visibly changing the Node-MoE gate,
  so the control is neither leaky nor vacuous.
- `test_learned_router_x_only_arm_is_propagation_free_end_to_end` — the whole
  arm's accuracy is invariant to adjacency scrambling, to 1e-12.
- `test_x_only_arm_rejects_graph_expert` — the guard rail holds.

| graph | no-routing | Node-MoE gate | x_only gate | Ens-Avg |
|---|---|---|---|---|
| Amazon | 0.9734 | 0.9731 (−0.0003) | 0.9735 (+0.0002) | 0.9682 |
| Tolokers | 0.7826 | 0.7819 (−0.0006) | 0.7820 (−0.0005) | 0.7818 |
| YelpChi (capped) | 0.8555 | 0.8533 (−0.0022) | 0.8579 (+0.0025) | 0.8553 |

Every gap is ≤0.0025 and the sign flips across gates and datasets. Removing the
graph entirely does not produce a routing win — it produces the same noise. So
the earlier nulls cannot be attributed to structure that the experts were
exploiting behind the protocol's back.

## Result 6: the null survives closing the graph channel in the *expert* too

`x_only` removes structure from the **router**. It does not touch the other
implicit-routing path: the GNN expert still consumes `AX` for whatever nodes reach
it, so a frozen k-means pool could still be selecting rows of a graph feature
matrix that separates regimes on its own. `--ragged-edges` closes that path.

The expert's operator becomes `A[mask, :]`, i.e. **every non-owned node is dropped
as an aggregation source while all rows are kept**. The asymmetry is the whole
point:

- an owned node aggregates only from owned neighbours, so nothing outside the
  expert's routed subset can reach it, and
- a test node — which by construction is *not* in `train_idx` — still aggregates,
  but only from the expert's owned nodes.

Masking rows as well would zero the neighbourhood of every test node and demote
the graph expert to a per-row model, so the arm would measure expert weakness
rather than implicit routing.

| graph | expert | beats a *random* routing | beats **no-routing** | best candidate | vs no-routing |
|---|---|---|---|---|---|
| Amazon | GNN, full graph | 16/26 | **0/26** | 0.9110 | — |
| Amazon | GNN, ragged | 8/26 | **0/26** | 0.8922 | — |
| Tolokers | GNN, full graph | 2/26 | **0/26** | 0.5249 | — |
| Tolokers | GNN, ragged | 3/26 | **0/26** | 0.5086 | — |
| YelpChi (capped) | GNN, full graph | 2/26 | 2/26 † | 0.7192 | +0.0026 |
| YelpChi (capped) | GNN, ragged | 15/26 ‡ | 15/26 ‡ | 0.7167 | +0.0100 |

† unchanged from Result 3: `eig_nb_sim_k32`, p=0.75, reduced-power arm.

‡ **not a routing win.** Three separate checks say so. The best candidate is
still *below* no-routing in absolute accuracy (0.7167 vs 0.7066 is a mean over
26 candidates that is dominated by losers — the median `d2` vs no-routing is
−0.0007). No candidate is Holm-significant (0/26). And 15/26 nominal wins is not
distinguishable from a coin flip (`binomtest(15, 26, 0.5) = 0.28`); the win-set
overlap with the full-graph arm is 2 candidates, Fisher `p=0.49`. The ragged
arm simply has more variance, so more noise crosses zero.

The mean `d2` vs no-routing moves from −0.0068 (full graph) to −0.0007 (ragged) —
i.e. cutting the expert's view of the graph brings routing to *parity* with no
routing, which is as close as this protocol gets to a positive result and is
still zero.

### The control is neither leaky nor vacuous

Both directions are pinned by tests:

- `test_ragged_edges_blocks_cross_subset_aggregation` — perturbing the features of
  nodes the expert does not own leaves its output **bit-identical**
  (`max|Δp| = 0.0`), while the same perturbation moves the full-graph expert
  (`max|Δp| = 0.25`). Asserted on probabilities, not argmax labels, since a large
  feature shift can leave every hard label unchanged while the representation has
  clearly moved.
- `test_ragged_edges_still_uses_structure_inside_its_own_subset` — on a label
  carrying no own-feature signal at all, the ragged expert reaches 0.715 where
  per-row logistic and MLP heads sit at chance (0.510 / 0.450), *and* stays below
  the full-graph expert (0.908), proving the masking is actually binding. A ring
  graph cannot support this test — degree 2 means masking half the nodes leaves a
  held-out node with no owned neighbours — so the test uses an SBM.
- `test_ragged_edges_flag_requires_graph_expert` — the flag is rejected for
  per-row experts, which never touch the graph and would make the arm look
  controlled without being controlled.

The positive control also holds under ragged edges: oracle `+0.3043` and k-means
routing `+0.3043` over no-routing, both **5/5 splits**, against `+0.3070` for the
full-graph expert at the same settings. So removing cross-subset edges costs
nothing when the routing signal is real.

| positive control, GNN expert | no-routing | oracle | k-means | random | oracle − no-routing |
|---|---|---|---|---|---|
| full graph | 0.6567 | 0.9637 | 0.9637 | 0.6315 | **+0.3070** |
| ragged edges | 0.6619 | 0.9662 | 0.9662 | 0.6421 | **+0.3043** |

That is the load-bearing comparison for the whole implicit-routing argument: the
control removes an entire channel through which a frozen pool could have routed
internally, and the planted gain survives its removal intact.

The control arm matters as much as the real one. On the positive control the
`x_only` gate does **better** than the Node-MoE gate — `+0.1841` vs `+0.1704`,
recovering more of the oracle's `+0.189`, with k-means agreement 0.858 vs 0.777.
The three planted regimes are linearly separable in raw feature space, so graph
terms only add noise for the gate to fit. This is what makes the real-data arm
interpretable: the propagation-free gate provably *can* recover a routing gain
when one exists, and recovers none here.

### Revised scope

The claim is now narrower and better supported: **neither fixed structural routers
nor a label-trained router beat a single global model on these fraud graphs**, and
the harness provably would have detected a win had one existed. Both implicit-routing
channels are closed — the router has no graph terms (`x_only`) and the graph expert
cannot aggregate from nodes it does not own (`ragged-edges`) — and the null survives
both. Ada-Routing and other published learned baselines remain unimplemented.