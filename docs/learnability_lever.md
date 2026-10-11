# Phase B — the learnability lever

Protocol B reports a null on every real pool: no routing beats one global model
(0 wins on the fair six-dataset rerun; the union of winning runs is
{amazon #0, amazon #1, tolokers #2} at the ±0.0005 noise scale). A null is only
informative if routing *could* have won. The positive control
(`src/positive_control.py`) shows it could, but only at one discrete endpoint.
This phase supplies the missing continuum and the constructive half: a selective
router that measurably declines to route when routing is not learnable.

Artifacts: `src/learnability_lever.py`, `experiments/run_learnability_lever.py`,
`tests/test_learnability_lever.py`, `logs/learnability_lever/lever.json`.

## 1. The lever

One knob, `separability` β ∈ [0, 1], interpolates the label function between two
regimes over the *same* marginal feature distribution and the *same* graph:

- **β = 0** — every latent group shares one global rule, so a single hyperplane
  is optimal and routing is worth nothing.
- **β = 1** — each group uses its own rule (exactly the positive control), so no
  global hyperplane fits and routing is worth a lot.

With `s_common(x) = x0 + x1` and the per-group rules of
`positive_control._group_rules`,

```
s_g(x) = (1 − β) · s_common(x) + β · s_rule(g, x),      label = [s_g(x) > 0].
```

At β = 1 the lever reproduces the positive control's labels **bit-identically**,
including that function's `int64` accumulator truncation (see the module
docstring). The lever only interpolates; it invents no structure.

The graph stays block-diagonal over the latent groups (`edge_p = 0.02`), so
feature k-means recovers the groups and the protocol's routing-blind pool aligns
with the true structure. A secondary homophily knob `cross_edge_p` dilutes the
block structure at fixed β.

## 2. Results — the separability sweep

n = 3000, 3 groups, d = 8, logistic experts, 5 splits × 1 seed, 200 epochs,
`epochs` for the MLP gate. `gain` is accuracy minus the single global model.
`cond` is the conditioned router of §3.

| β | global | oracle gain | k-means gain | learned gain | cond | cond gain |
|----|--------|-------------|--------------|--------------|------|-----------|
| 0.0 | 0.9916 | −0.0100 | −0.0100 | −0.0065 | **abstain** | 0.0000 |
| 0.1 | 0.9833 | −0.0036 | −0.0036 | −0.0083 | **abstain** | −0.0021 |
| 0.2 | 0.9657 | +0.0091 | +0.0091 | −0.0055 | **abstain** | −0.0008 |
| 0.3 | 0.9405 | +0.0257 | +0.0257 | +0.0031 | route (4/5) | +0.0031 |
| 0.4 | 0.9061 | +0.0492 | +0.0492 | +0.0128 | abstain (1/5) | +0.0032 |
| 0.5 | 0.8720 | +0.0712 | +0.0712 | +0.0283 | route (5/5) | +0.0120 |
| 0.6 | 0.8320 | +0.0979 | +0.0979 | +0.0521 | route (4/5) | +0.0245 |
| 0.7 | 0.7971 | +0.1253 | +0.1253 | +0.0831 | route (4/5) | +0.0521 |
| 0.8 | 0.7911 | +0.1315 | +0.1315 | +0.0927 | route (5/5) | +0.0583 |
| 0.9 | 0.7924 | +0.1261 | +0.1261 | +0.0916 | route (5/5) | +0.0511 |
| 1.0 | 0.7792 | +0.1944 | +0.1944 | +0.1584 | route (5/5) | +0.1295 |

Two boundaries, which need not coincide (`find_thresholds`):

- **Existence β\* = 0.2** — smallest β at which routing gain exists at all (the
  label-informed oracle beats the global model). This is a property of the task.
- **Deployability β\* = 0.3** — smallest β at which the selective router both
  decided to route and won on held-out test.

The k-means gain equals the oracle gain exactly: with a block-diagonal graph and
feature-separated groups the routing-blind pool is already aligned, so a
label-free router captures the entire available gain. This is the strongest
possible version of the lever — it cannot be dismissed as "the oracle was the
only thing that could route."

Above β ≈ 0.5 the result is monotone and robust (5/5 splits route, positive gain
at every setting). The mid-range β ∈ [0.3, 0.4] is genuinely noisy (route rate
4/5 then 1/5), which is itself the honest finding: near the boundary the selective
router's held-out signal is weak.

## 3. The learnability-conditioned router

`learnability_conditioned_router` (`src/learnability_lever.py`) answers the
*deployable* question — not "does routing win?" but "can we tell, from held-out
supervision of the kind the MLP gate already gets?" — with an internal validation
split:

1. carve training nodes into expert-fit and router-supervision halves (cross-fit);
2. carve the router half again into `router_fit` / `router_val`;
3. fit the frozen pool experts and the global expert on the fit half;
4. train the MLP gate on `router_fit`; score routed vs global on the disjoint
   `router_val` nodes;
5. **route** iff that held-out gain exceeds `margin`; else **abstain** (send every
   node to the global head).

No test label ever touches the decision. On the lever it routes exactly where
routing is deployable (§2) and abstains below.

## 4. Results — the real pools abstain

Same conditioned router, same arms, applied to the two binary real fraud graphs
that Protocol B already covers:

| dataset | n | global | oracle gain | k-means gain | learned gain | cond | cond val gain |
|---------|----|--------|-------------|--------------|--------------|------|---------------|
| amazon | 8639 | 0.9732 | −0.0339 | −0.0004 | −0.0004 | **abstain** (2/5) | +0.0007 |
| tolokers | 11758 | 0.7820 | −0.0003 | −0.0001 | −0.0002 | **abstain** (1/5) | −0.0017 |

The conditioned router's held-out signal is essentially zero (+0.0007 and
−0.0017), so it correctly declines on both. The Amazon oracle is *negative*
(−0.0339) — the documented oracle confound (`docs/fixed_expert_protocol.md`): the
adaptive homophily buckets install degenerate-prior heads, so the cheating router
is worse than the honest global model. Even ignoring that, no arm beats global.

This is the Phase B headline: **the real pools lie below the lever's deployability
threshold (β \$<\$ 0.3), and a router that can measure learnability says so. The
null is a statement about the tasks, not a failure of the method.**

## 5. The homophily lever (secondary)

At fixed β = 1, adding cross-group edges (`cross_edge_p`) does **not** reduce
routing gain; it rises slightly (oracle +0.1944 constant; learned +0.158 → +0.172;
cond +0.129 → +0.161 as `cross_edge_p` goes 0 → 0.02).

Interpretation: in this construction the routability is carried by the
**label/feature** structure, not by edge homophily. The expert pool (feature
k-means) and oracle (latent groups) ignore edges, and the per-row logistic experts
never see the graph, so diluting block structure costs the graph-aware gate
nothing. This is the honest, slightly surprising reading: graph homophily is
*sufficient but not necessary* for routing to be learnable here. It also explains
why heterophily real graphs (chameleon, squirrel, actor) showing no routing gain
is consistent — the lever says heterophily per se is not the blocker; the blocker
is low label separability.

## 6. How to run

```powershell
python -m spectral_distillation.experiments.run_learnability_lever `
    --separabilities 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 `
    --n 3000 --splits 5 --epochs 200 --expert logistic `
    --real amazon tolokers --out logs
```

Writes `logs/learnability_lever/lever.json` (sweep + homophily sweep + real
probes + thresholds). ~2.5 min on CPU. `--skip-sweep` / `--skip-real` isolate
halves; `--cross-edge-p` sets the primary sweep's homophily.

## 7. Honest limitations

- The lever is synthetic. It bounds *what* the harness can detect and where the
  boundary is; it does not claim any real graph sits at a specific β.
- The existence/deployability thresholds (0.2 / 0.3) are resolution-limited by
  the β grid (step 0.1) and by the 5-split noise floor; treat them as "between
  0.2 and 0.3" and "≈ 0.3–0.5", not exact.
- The homophily sweep shows the lever's signal is feature-carried *in this
  construction*; a different generator that put the signal in the edges could show
  the opposite, and is a fair target for a follow-up.
- Only logistic experts were swept; the MLP-expert lever (where per-row capacity
  already closes the gap, as in the positive-control note) is not reported here.
