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
| Amazon | 0/26 | 1/26 | 0/26 | **0/26** |
| Tolokers | 9/26 | 16/26 | 0/26 | **0/26** |
| YelpChi | 15/26 | 19/26 | 0/26 | **0/26** |

(Tolokers previously read 2/26 here. That residual was an artefact of a bug in
the no-routing baseline itself — see "the baseline bug" below — and is now 0/26.)

The one large positive, `eig_nb_sim_k128`, does not survive: on YelpChi under
fixed experts it is still **negative** against no-routing (−0.0052). The Tolokers
fixed-expert D2 magnitudes collapse to ~1e-5, and the feature-k-means and oracle
references are indistinguishable from zero.

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
| Tolokers | 0.7819 | 0.7820 | 0.6721 |
| YelpChi (capped) | 0.8588 | 0.8660 | 0.7154 |

So the specialists are strong in absolute terms, and routing still does not pay
for itself. Across **all three expert families on all three graphs**:

| graph | expert | beats a *random* routing | beats **no-routing** |
|---|---|---|---|
| Amazon | logistic | 0/26 | **0/26** |
| Amazon | MLP | 0/26 | **0/26** |
| Amazon | GNN | 0/26 | **0/26** |
| Tolokers | logistic | 0/26 | **0/26** |
| Tolokers | MLP | 0/26 | **0/26** |
| Tolokers | GNN | 0/26 | **0/26** |
| YelpChi | logistic | 0/26 | **0/26** |
| YelpChi | MLP | 0/26 | **0/26** |
| YelpChi | GNN (capped) | 0/26 | **0/26** |

**0 of 234 candidate/expert/graph combinations beat training one model on
everything**, and — once the multiplicity correction is applied — none beats a
random routing either. The best gap in each row is negative throughout.

† **A note on the counts.** "beats a random routing" is the number of candidates
still significant after a Holm step-down across the 26 candidates in that row;
"beats no-routing" is the number whose mean paired difference against a single
global model is positive. Earlier revisions of this table reported 41 and 2
respectively. Those came from (i) a per-rank significance rule that was never
multiplicity-corrected and (ii) a reduced-power YelpChi GNN arm (3 splits × 1
seed). Both were fixed (see `docs/reproducibility.md`), and every one of the 234
cells is now 0/26 on both criteria. The negative result is unchanged and, if
anything, cleaner.

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
being exploitable — **no candidate beats one global model**. Routing by structural
score is not distinguishable from routing at random once multiplicity is
corrected, and is reliably worse than not routing at all.

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
| YelpChi (capped) | 0.7271 | 0.7154 | +0.0117 | 0.0195 | yes |

Ens-Avg loses to no-routing on both full-scale graphs, significantly. So there the
capacity explanation is refuted: the k-means pool is not complementary, and
neither routing nor averaging extracts anything from it. This strengthens the
negative result on those graphs — it is not "routing failed but the pool was
valuable".

The YelpChi Ens-Avg is the one condition in the whole study that **beats** training
one model (Δ +0.0117, p=0.02). It is a *pool* effect, not a routing effect — no
router is involved — and it appears only on the node-capped graph (14,840 of
45,900 nodes). We report it as the single positive the protocol can produce, and
note that the cap inflates mean degree (55 at 15k vs 168 at full scale), so it may
not survive at full scale.

## Reading

On these fraud graphs, **no label-free router beats simply training one model on
everything**, under an honest protocol, with linear, non-linear, or
message-passing experts. The earlier headline effect sizes (dz 7.8/6.6) came from
two different mistakes, and it is worth separating them because they do not carry
the same weight:

- The **oracle** effect sizes were measurement artifacts of the self-routing
  protocol. The self-routing credit is granted to label-homogeneous partitions,
  and the oracle partition is maximally label-homogeneous by construction, so it
  collected +0.038 (Tolokers) and +0.044 (YelpChi) of pure artifact. Those
  numbers are void.
- The **label-free candidate** effect sizes were *not* artifacts of the protocol.
  They were genuine gains over *random routing* on a self-fitted pool. What they
  did not survive is the stronger comparison against a single global model, which
  is what protocol B measures — and there they go to 0/26 on Amazon and Tolokers.

Three independent metrics now agree:
1. Correlation with label homophily does not identify a useful routing partition.
2. The standard self-routing benchmark is confounded by the permutation invariance
   and by label-homogeneous subgroups scoring well against themselves.
3. The oracle is *negative* against no-routing — routing by label homophily is
   actively harmful, not merely unhelpful.

### The self-routing credit is label-specific, not universal

It would be wrong to say self-routing inflates every arm. Measured against
no-routing *within* protocol A:

| graph | random routing − no-routing | oracle routing − no-routing |
|---|---|---|
| Amazon | −0.0005 | **+0.0031** |
| Tolokers | −0.0014 | **+0.0382** |
| YelpChi | −0.0014 | **+0.0439** |

Random partitions collect no credit at all; label-homogeneous ones collect a lot.
The mechanism is that a subgroup only has to predict inside its own subpopulation
while one global model must learn the marginal, so the gain scales with how
label-aligned the partition is. Protocol B removes both the credit and the
double-dipping (the oracle buckets are themselves fitted on all of `y`, test
included).

Two corollaries for how far to retract:

- **Oracle numbers under protocol A are invalid.** On Amazon two of three oracle
  buckets have train positive-rate exactly 0.000, so `fit_experts` installs a
  `DummyClassifier`; 1468 of 4320 held-out nodes route into a degenerate-prior
  bucket. Any doc quoting protocol-A oracle D1/D3 must be marked superseded.
- **Label-free candidate numbers are *not* invalid.** On the Amazon full declared
  family (n=26), protocol A mean D2 is +0.00031 against +0.00450 for protocol B,
  and Spearman(|rho|, D2) is ≈0 in both protocols (p=0.26, p=0.20). Protocol A is
  if anything *harsher*. So the earlier candidate gains were not artifacts of the
  protocol; they were real gains against *random routing* that simply do not beat
  a single global model — which is exactly what protocol B shows (0/26 candidates
  beat no-routing on Amazon and Tolokers).

This narrows the retraction in the *Reading* section below: the self-routing
artifact claim is about the oracle and the earlier headline effect sizes, not
about every protocol-A number ever recorded.

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
- `eig_nb_sim_k128` is confirmed *stable* out-of-fold (selected on 10/10 held-out folds by
  `experiments/run_k_confirmation.py`) yet still does not survive fixed-expert scoring against a
  genuine no-routing model — the choice of `k` is not the artifact, the win is. See
  `docs/reproducibility.md`, "Resolved: spectral-k sweep".
- The published routers **RouterGNN** (message-passing, label-supervised) and
  **Ada-Routing/AD-GNN** (per-node propagation depth via a signal-preservation
  factor) are reproduced in `src/published_routers.py` and scored under this very
  protocol by `experiments/run_published_routers.py`. Both also fail to beat
  no-routing on all six graphs; see "## Published routers under this protocol"
  below. Node-MoE's own expert construction (differentiated ChebNetII filter
  initialisation) and its low-pass GNN gate *are* reproduced in `src/node_moe.py`
  and scored under this protocol by `experiments/run_node_moe.py`; see
  "## Node-MoE under this protocol" below. The gate does not win.
- The `x_only` arm rules out implicit routing for *this* pipeline, not in general.
  A ragged-edge / subgraph-restricted fit would be a stronger control still.
- YelpChi stays node-capped (14840 of 45954) for memory; its absolute accuracies
  are therefore not comparable to the uncapped graph. Its arms now use the same
  splits × seeds as the other graphs (learned router 3 × 1, GNN 10 × 3), so the
  earlier reduced-power caveat no longer applies.

## Node-MoE under this protocol

The strongest published node-wise graph MoE is **Node-MoE** (`arXiv:2406.03464`).
Its distinguishing choice is the *expert construction*: each expert is a
**ChebNetII** spectral filter, and the experts are initialised as deliberately
*different* filter shapes (low-pass, high-pass, band-pass) before training; the
gate is a small low-pass GNN over the neighbourhood-discrepancy descriptor
`[X, |AX-X|, |A^2X-X|]`. Earlier versions of this doc quoted the gate features
only and left the experts unreproduced. That gap is now closed.

`src/node_moe.py` reproduces both: `ChebNetIIFilter` (learnable Chebyshev
coefficients evaluated by Gauss-Chebyshev interpolation, so `gamma = g(nodes)`
is the exact interpolant of a target shape), `NodeMoEGate`, and the joint
`NodeMoEHead` trained end-to-end with Node-MoE's filter-smoothing loss
`L = L_task + gamma * sum_o sum_i (f_o(x_i) - f_o(x_{i-1}))^2`.
`experiments/run_node_moe.py` scores it two ways.

**Faithful replication** (gate + experts trained jointly on all training nodes,
the paper's own protocol) versus a single global ChebNetII and a single global
MLP, mean accuracy over 5 splits x 1 seed:

| graph | Node-MoE (joint) | global ChebNetII | global MLP (features only) |
|---|---|---|---|
| Amazon (n=8639) | 0.9418 | 0.9406 | **0.9727** |
| Tolokers (n=11758) | 0.7089 | 0.6664 | **0.7818** |
| YelpChi (n=14840) | 0.7491 | 0.7079 | **0.8669** |

The joint model does beat a single global ChebNetII (+0.0012 Amazon, +0.0426
Tolokers, +0.0412 YelpChi; 3/5, 5/5, 4/5 runs). But a *plain feature-only MLP*
that never touches the graph beats Node-MoE by 0.03-0.12 on every graph. So the
routing gain is real only relative to a weak spectral base; it does not make
Node-MoE competitive with a non-graph model on these features.

**Honest fixed-expert Protocol B** (routing-blind ChebNetII pool frozen on the
fit half, gate cross-fit on a disjoint half, no-routing = one global model on the
same fit half):

| graph | Node-MoE gate | no-routing | gate - no-routing | gate wins | oracle | Ens-Avg |
|---|---|---|---|---|---|---|
| Amazon | 0.9401 | 0.9399 | **+0.0003** | 2/5 | 0.9393 | 0.9407 |
| Tolokers | 0.6721 | 0.6672 | **+0.0049** | 5/5 | 0.6746 | 0.6682 |
| YelpChi | 0.7100 | 0.7078 | **+0.0023** | 2/5 | 0.7096 | 0.7075 |

Under the honest protocol the gate's edge is at most +0.005 and is within noise on
Amazon and YelpChi (2/5 runs). The pattern is the paper's central claim: a
positive result under faithful *joint* training (+0.04) almost entirely collapses
under routing-blind fixed-expert scoring (+0.005).

**Unrouted uniform-ensemble control.** The obvious objection to the "capacity"
reading is that *any* pool of K experts would help, gate or not. The control
removes the gate entirely: `NodeMoEHead(..., routing="uniform")` trains the same
differentiated experts with the same loss, epochs and optimiser, but combines them
by a fixed equal-weight mean `1/K`. Because only the combination rule differs,
`joint - uniform` is the causal contribution of the gate itself.

| graph | joint (gate) | uniform ensemble | gate - uniform | gate wins | uniform - global Cheb |
|---|---|---|---|---|---|
| Amazon | 0.9418 | 0.9404 | +0.0014 (p=.63) | 3/5 | -0.0002 |
| Tolokers | 0.7089 | 0.6632 | **+0.0457 (p=.06)** | 5/5 | -0.0032 |
| YelpChi | 0.7491 | 0.7089 | +0.0401 (p=.31) | 4/5 | +0.0011 |

Two facts follow. First, the unrouted ensemble is statistically indistinguishable
from a *single* global ChebNetII on every graph (rightmost column, within +-0.003):
K spectral experts with no router add nothing over one spectral expert, so the
joint gain is **not** ensemble capacity. Second, the gate does carry a real
joint-training gain on Tolokers (+0.046, 5/5, dz=1.9) and YelpChi (+0.040, 4/5) but
none on Amazon (+0.001). Because that gain appears only when gate and experts are
co-trained and vanishes under the routing-blind Protocol B (+0.005), it is a
train-time co-adaptation effect, not a per-node routing signal that survives
honest evaluation. The earlier "capacity" wording is corrected here: the control
refutes capacity and localises the residual to gate-expert co-adaptation.

Caveats: the global-MLP baseline is unusually strong (0.97 on Amazon), which is
the repo's existing split/feature pipeline, not new here; it is reported so it
cannot be mistaken for a tuning choice. YelpChi is node-capped at 14840. Protocol
B uses 3 experts and K=2.

### On Node-MoE's own benchmarks (heterophily)

The fraud graphs above are the critique's original test-bed, and they invite one
escape hatch: fraud topologies are noisy and weakly heterophilic, so a routing
gain might exist on the *heterophilic web/actor* graphs Node-MoE actually reports
SOTA on. To close it we run the identical harness on **Node-MoE's own headline
suite -- Actor, Chameleon, Squirrel** (5-way; loaded by
`src/heterophily.py`). Nothing else changes: same 3 differentiated ChebNetII
experts, same `K=2`, same loss, same 5 splits x 1 seed, same Protocol B.

Node-MoE is competitive *within noise* on its own turf even under its favourable
joint protocol. **Faithful replication** (mean accuracy):

| graph | Node-MoE (joint) | global ChebNetII | global MLP (features only) |
|---|---|---|---|
| Chameleon (n=2277) | 0.4327 | 0.4153 | **0.4520** |
| Actor (n=7600) | 0.3433 | **0.3446** | 0.3240 |
| Squirrel (n=5201) | 0.3034 | 0.2999 | **0.3103** |

Joint - global ChebNetII is +0.0174 (4/5, p=.31), **-0.0013** (3/5, p=1.0) and
+0.0035 (2/5, p=.81) -- i.e. Node-MoE does not reliably beat a single global
ChebNetII on the datasets it was designed for. A graph-free MLP matches or beats
it on two of the three.

**Unrouted uniform-ensemble control** -- the decisive result, and it is *stronger*
here than on the fraud graphs:

| graph | joint (gate) | uniform ensemble | gate - uniform | gate wins | uniform - global Cheb |
|---|---|---|---|---|---|
| Chameleon | 0.4327 | **0.4615** | -0.0288 (p=.13, dz=-1.2) | 0/5 | +0.0462 (p=.06, dz=3.9) |
| Actor | 0.3433 | **0.3548** | -0.0115 (p=.13, dz=-1.5) | 1/5 | +0.0102 (p=.06, dz=2.8) |
| Squirrel | 0.3034 | **0.3340** | -0.0306 (p=.06, dz=-3.9) | 0/5 | +0.0341 (p=.06, dz=3.6) |

The unrouted equal-weight ensemble beats the gated joint model on **every** graph
(-0.011 to -0.031; the gate wins 0/5, 1/5, 0/5). So on Node-MoE's home benchmarks
the learned gate is actively *worse* than simply averaging the same experts. The
ensemble does beat a single global ChebNetII here (+0.010 to +0.046, all 5/5 with
dz~3) -- so there *is* a small real benefit to having K spectral experts on these
graphs, but it comes from the pooling, not from any routing decision.

**Honest fixed-expert Protocol B** (frozen pool, cross-fit gate):

| graph | Node-MoE gate | no-routing | gate - no-routing | gate wins | oracle | Ens-Avg |
|---|---|---|---|---|---|---|
| Chameleon | 0.4114 | 0.3902 | +0.0212 (p=.06) | 5/5 | 0.4121 | 0.3998 |
| Actor | 0.3388 | 0.3358 | +0.0030 (p=.06) | 5/5 | 0.3438 | 0.3400 |
| Squirrel | 0.2953 | 0.2896 | +0.0057 (p=.31) | 3/5 | 0.2986 | 0.2925 |

Same pattern as the fraud graphs: the cross-fit gate's edge over a single global
model is at most +0.02 (Chameleon) and <= +0.006 elsewhere, while the oracle that
*knows the labels* buys almost nothing over the gate -- evidence that the
homophily buckets carry little routing signal on these graphs too. Combined with
the uniform-ensemble result, the honest reading is: the joint-training "gain" is
co-adaptation between the gate and the experts, and it does not survive as a
per-node routing signal.

Caveats: these are the **legacy** Chameleon/Squirrel graphs (duplicate-node
leakage under the published Geom-GCN splits, Platonov et al., ICLR 2023); we avoid
that by using the repo's own class-stratified 50/50 split for every dataset, so
the absolute accuracies are lower than published and are **not** comparable to the
literature -- only the within-run joint/uniform/global/Protocol-B contrasts are.
The 5-way switch is handled by `n_classes` in `src/node_moe.py`, which leaves the
binary path byte-identical.

## Positive control: can the protocol detect a win at all?

A 0/234 result is only meaningful if the harness would have reported a win when one
existed. `spectral_distillation/src/positive_control.py` supplies a task where
routing is provably useful.

The label is a **non-separable mixture of linear rules**: three latent groups, each
with its own linearly separable boundary over its own feature pair, no single
hyperplane satisfying all three. A marker feature offsets the groups in feature
space (so feature k-means recovers them at purity 1.000) and the graph is
block-diagonal over the same groups, so the task is graph-native.

Run: `python -m spectral_distillation.experiments.run_positive_control --experts
logistic mlp gnn --splits 5 --seeds 3 --n 1800 --out logs/positive_control_gnn`
→ `logs/positive_control_gnn/fixed_expert.json` (5 splits × 3 seeds)

| expert | no-routing | oracle | k-means | random | oracle − no-routing |
|---|---|---|---|---|---|
| logistic | 0.6518 | 0.9885 | 0.9885 | 0.6274 | **+0.3367** |
| MLP | 0.9810 | 0.9926 | 0.9926 | 0.6164 | +0.0116 |
| GNN | 0.6503 | 0.9893 | 0.9893 | 0.6224 | **+0.3390** |

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
| YelpChi (capped) | 0.8586 | 0.8562 | 0.8579 | 0.8520 | 0.8493 | 0.8587 | −0.0024 | −0.0025 |

**Every gap to no-routing is ≤0.0025 — indistinguishable from noise** — and the
sign flips across datasets. Against the *ensemble* baseline the learned router is
nominally ahead on Amazon and Tolokers by ≤0.005 and behind on YelpChi; none of
these separate either. The router tracks the frozen k-means partition closely
(agreement 0.731/0.314/0.360), i.e. it mostly rediscovers the pool it was handed
rather than finding new structure.

Note the oracle is *below* no-routing in all three rows. The label-optimal routing
is worse than training one model, which bounds the achievable routing gain at ≤ 0
here: no router, however good, could have won. This is a bound rather than a
failure to search.

The control matters: on the positive control the same router reaches 0.9613 against
a global 0.6534, recovering **+0.308** of the oracle's +0.330. So the router is not
broken and the harness is not blind — on these three fraud graphs there is simply
no per-node expert-selection signal to exploit.

### Ens-Avg on the positive control is the mechanism, not just a baseline

| condition | positive control | real graphs (GNN pool) |
|---|---|---|
| no-routing | 0.6534 | 0.9428 / 0.6721 / 0.7154 |
| Ens-Avg | 0.6946 | 0.9140 / 0.5174 / 0.7271 |
| k-means routing | **0.9835** | 0.9401 / — |
| oracle routing | **0.9835** | 0.8339 / 0.4939 / 0.6986 |
| learned router | **0.9613** | 0.9731 / 0.7819 / 0.8562 |

On the positive control the experts are genuine *specialists* — each is optimal on
one latent group — so averaging their probabilities blurs three distinct decision
boundaries into one mediocre one (+0.041 over global), while routing to the right
specialist recovers almost everything (+0.33). Ens-Avg and routing are not
redundant controls; they fail in opposite regimes, and which one wins tells you
what kind of pool you have.

That is the interpretive key for the real graphs. Ens-Avg landing at or below
no-routing on the two full-scale graphs, *while* k-means routing also fails to beat
no-routing there, says that k-means pool is neither specialist (routing cannot
exploit it) nor complementary (ensembling cannot exploit it). A pool with no
exploitable structure at all — which is precisely the low-heterogeneity regime the
routing-payoff hypothesis predicts. On the capped YelpChi graph, by contrast,
Ens-Avg *does* beat no-routing (+0.0117, p=0.02): on that one graph averaging
recovers complementarity the router cannot find, so the pool is complementary but
still not specialist.

Pinned by `test_learned_router_recovers_routing_gain_on_positive_control` and
`test_learned_router_never_sees_test_labels` (flipping every test label must leave
router parameters bit-identical).

## Published routers under this protocol

The two routers the gap analysis names are now reproduced and scored on the
*same* frozen k-means pool, the *same* splits, and with the *same* disjoint
fit/router cross-fit as RouterGNN-lite (see `## RouterGNN-lite` above). No test
label ever enters a router; the MLP gate column below is literally the committed
RouterGNN-lite run, re-emitted by the new runner:

* **RouterGNN** (`src/published_routers.py`): the repo's message-passing
  GraphSAGE-style router, now supervised by training labels instead of imitating a
  clustering. Gate features are held fixed at Node-MoE's `[X, |AX-X|, |A2X-X|]`,
  so the only difference from the MLP gate is the *architecture* — each layer
  mixes a node's representation with its neighbours'. It is trained with
  hidden-layer LayerNorm and a conservative optimiser (lr 1e-3, weight-decay 1e-4,
  grad-clip 1.0) so the comparison is a fair *architecture* contest rather than a
  casualty of an unstable optimiser (see note (4)).
* **AD-GNN / Ada-Routing**: per-node propagation depth via a *signal-preservation
  factor* (label-free): the cosine between a node's features and its 1- and 2-hop
  aggregated features, minimised over depth. Deep-preservation nodes route to the
  low-pass channel, low-preservation nodes to identity. This is the
  mechanism-mismatched control of the implementation guide (guide 11.5).

Run: `python -m spectral_distillation.experiments.run_published_routers --dataset amazon`

| graph | n | C | no-routing | RouterGNN | AD-GNN | MLP gate | Ens-Avg | k-means | oracle |
|---|---|---|---|---|---|---|---|---|---|
| Amazon | 8639 | 2 | 0.9734 | 0.9723 (−0.0011) | 0.9184 (−0.0549) | 0.9731 (−0.0002) | 0.9682 | 0.9727 | 0.9285 |
| Tolokers | 11758 | 2 | 0.7826 | 0.7821 (−0.0005) | 0.7818 (−0.0008) | 0.7819 (−0.0006) | 0.7818 | 0.7820 | 0.7815 |
| YelpChi (capped) | 14840 | 2 | 0.8586 | 0.8601 (+0.0015) | 0.8508 (−0.0078) | 0.8543 (−0.0044) | 0.8587 | 0.8579 | 0.8493 |
| Actor | 7600 | 5 | 0.3492 | 0.3000 (−0.0492) | 0.2922 (−0.0570) | 0.3067 (−0.0425) | 0.3228 | 0.3431 | 0.2948 |
| Chameleon | 2277 | 5 | 0.4407 | 0.3687 (−0.0720) | 0.3860 (−0.0547) | 0.4226 (−0.0181) | 0.4261 | 0.4404 | 0.3477 |
| Squirrel | 5201 | 5 | 0.3219 | 0.2809 (−0.0410) | 0.2217 (−0.1003) | 0.3073 (−0.0146) | 0.2955 | 0.3356 | 0.2510 |

Deltas are against no-routing. Every mean is negative on every graph **except
RouterGNN on the capped YelpChi graph**, whose +0.0015 mean is not significant
(paired-Wilcoxon p=0.5, n=3) and rests on 2 of 3 splits. Across the 18 individual
runs RouterGNN wins 4 (Amazon 1, Tolokers 1, YelpChi 2), AD-GNN wins 1 (a Tolokers
split), and the MLP gate wins 2 (an Amazon and a Tolokers split) — every win is at
the ±0.002 noise scale, and none is significant once per-graph splitting is
accounted for. This matches the 0/234 structural-score result: no router, learned
or fixed, reliably separates from a single global model on any of these pools.

Four notes. **(1) A fairly trained message-passing router buys nothing.** On the
fraud graphs RouterGNN is statistically indistinguishable from both the per-row
MLP gate and from no-routing (Amazon −0.0011 vs −0.0002; Tolokers −0.0005 vs
−0.0006; YelpChi +0.0015 vs −0.0044, where the GNN is actually the *better* of the
two). On the heterophily graphs both routings are solidly negative and the GNN is
the worse of the two (−0.049/−0.072/−0.041 vs the MLP's −0.043/−0.018/−0.015), so
neighbour mixing does not recover the small gate-expert alignment the per-row gate
retains. The GNN router does not reliably rediscover the pool it is scored against
(agreement with the frozen k-means partition 0.246–0.421 across graphs). **(2)
AD-GNN is a mechanism mismatch**, exactly as the guide predicted: its depth
quantiles map onto a feature-k-means pool with agreement only 0.078–0.365, and it
is worst on the graphs with the most heterophilic structure (Squirrel −0.100,
Amazon −0.055). **(3) The oracle stays below no-routing on every row**, so all
three learned routers are bounded: no router could have won on these pools. **(4)
The first pass exposed an optimiser artefact, not a modelling result.** The guide's
default optimiser settings (lr 1e-2, no normalisation) *diverged* on the larger
graphs: on Amazon the training loss rose (11.5 → 12.4, versus a ≈1.1 uniform
baseline), the softmax saturated, and the router collapsed to a single channel —
its agreement with k-means was exactly the pool's dominant channel fraction
(0.185). Training with the LayerNorm/regularised settings above makes every run
converge (loss falls, >1 channel used; logged as `gnn_fit_accuracy` /
`gnn_channels_used`) and leaves the conclusion unchanged, which is the point: the
null is not an artefact of an under-trained router. The committed Protocol-A
router (`router_protocol.py`, label-free KL) keeps the un-normalised default
bit-identically — LayerNorm is opt-in.

Pinned by `test_message_passing_router_recovers_gain_on_positive_control`,
`test_message_passing_router_never_sees_test_labels`, and
`test_router_gnn_forward_depends_on_adjacency` (the message-passing router must
actually consult the adjacency, so this is not the MLP gate in disguise).

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
| YelpChi (capped) | 0.8586 | 0.8562 (−0.0024) | 0.8568 (−0.0018) | 0.8587 |

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
| Amazon | GNN, full graph | 0/26 | **0/26** | 0.9110 | — |
| Amazon | GNN, ragged | 0/26 | **0/26** | 0.8922 | — |
| Tolokers | GNN, full graph | 0/26 | **0/26** | 0.5249 | — |
| Tolokers | GNN, ragged | 0/26 | **0/26** | 0.5086 | — |
| YelpChi (capped) | GNN, full graph | 0/26 | **0/26** | 0.7080 | — |
| YelpChi (capped) | GNN, ragged | 0/26 | **0/26** | 0.7030 | — |

† Not a single candidate — full graph or ragged, any graph — clears zero on either
criterion. An earlier revision reported 2/26 and 15/26 nominal wins for the two
YelpChi arms; both came from the reduced-power GNN run (3 splits × 1 seed) and
vanish at 10 × 3. See the note under Result 3.

The mean `d2` vs no-routing moves from −0.0162 (full graph) to −0.0138 (ragged) —
cutting the expert's view of the graph brings routing closer to *parity* with no
routing, but the gap stays negative on every candidate (best −0.0074 full,
−0.0023 ragged), which is as close as this protocol gets to a positive result and
is still zero.

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

The positive control also holds under ragged edges: oracle `+0.3396` and k-means
routing `+0.3396` over no-routing, both **5/5 splits**, against `+0.3390` for the
full-graph expert at the same settings. So removing cross-subset edges costs
nothing when the routing signal is real.

| positive control, GNN expert | no-routing | oracle | k-means | random | oracle − no-routing |
|---|---|---|---|---|---|
| full graph | 0.6503 | 0.9893 | 0.9893 | 0.6224 | **+0.3390** |
| ragged edges | 0.6505 | 0.9901 | 0.9901 | 0.6248 | **+0.3396** |

That is the load-bearing comparison for the whole implicit-routing argument: the
control removes an entire channel through which a frozen pool could have routed
internally, and the planted gain survives its removal intact.

The control arm matters as much as the real one. On the positive control the
`x_only` gate does **better** than the Node-MoE gate — `+0.3236` vs `+0.3079`,
recovering more of the oracle's `+0.330`, with k-means agreement 0.960 vs 0.847.
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
both. The published learned routers are now reproduced too (`src/published_routers.py`):
RouterGNN (message-passing) and Ada-Routing/AD-GNN (per-node propagation depth) have
negative mean deltas on every graph except YelpChi (RouterGNN +0.0015, not significant) and win **4/18** and **1/18** runs respectively
(all at the +0.0005 noise scale; `docs/fixed_expert_protocol.md`, "## Published
routers under this protocol").