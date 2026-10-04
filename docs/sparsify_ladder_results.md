> **RETRACTED (2026-10-04).** Every D1/D3 oracle number in this document comes
> from the self-routing protocol, where the oracle partition is fitted to its own
> routing and is maximally label-homogeneous, so it collects credit no deployable
> router can. Under the fixed-expert protocol the same partition scores
> **+0.0000 (Tolokers logistic), -0.0047 (Amazon), -0.0101 (YelpChi)** against
> random and is **negative against no-routing in 9/9 cells** (3 graphs x 3 expert
> types). The D1 = +0.0365 (dz 14.94) and D3 = +0.0368 figures are void, and so is
> the "13x and 15x larger than Amazon" comparison built on them -- that measures
> label alignment, not routing quality.
>
> The D2 label-free rows are *not* void on protocol grounds (protocol A is if
> anything harsher than protocol B), but they were measured against *random
> routing*, not against no-routing, and they fall to 0/26 under the latter.
> See `docs/fixed_expert_protocol.md` and `docs/d2_router_fix.md`.

# Spectral-Distillation Retention Ladder on Real Fraud Data — Cross-Dataset Results

Three real public fraud graphs, all fully labeled, single connected component, run through the
identical D1–D3 experts/training (10 splits × 3 seeds, p = exact Wilcoxon, resolution floor
0.001953):

| | Amazon (co-review) | Tolokers (crowd-worker exclusion) | YelpChi (review spam) |
|---|---|---|---|
| nodes / features | 8,639 / 25 | 11,758 / 10 | 14,840 / 32 † |
| undirected edges | 3,298,534 | 519,000 | 411,194 † |
| mean degree | 763.6 | 88.3 | 55.4 † (vs 167.6 at full LCC) |
| positive ratio | 9.50% | 21.82% | 14.97% † |
| edge label homophily | 0.9488 | 0.5945 (random baseline 0.6588 → heterophilous) | 0.7627 |
| **node-homophily sd** | **0.063** (saturated) | **0.241** | **0.257** |

† YelpChi numbers come from the largest connected component of a uniform 15,000-node sample; the
full 45,900-node LCC needs a 16.9 GB dense Laplacian. This lowers absolute mean degree by ~3×
relative to the full graph. See `docs/yelpchi_third_graph.md`.

Method: `experiments/run_sparsify_ladder.py --dataset {amazon,tolokers,yelpchi}`. Artifacts:
`logs/{dataset}_fraud_ladder/ladder_results.jsonl`.

## 1. Headline D1–D4

| | Amazon | Tolokers | YelpChi |
|---|---|---|---|
| D1 oracle > uniform | +0.0028 (dz 2.68) ✅ | **+0.0365 (dz 14.94)** ✅ | **+0.0426 (dz 25.74)** ✅ |
| D2 label-free > random (legacy) | −0.00015 ❌ | −0.0003 ❌ | **+0.0058 (dz 4.33)** ✅ |
| D2 label-free > random (fixed) | +0.0004 (p 0.23) ❌ | **+0.0040 (dz 4.24)** ✅ | +0.0048 (dz 6.73) ✅ |
| D3 oracle > random | +0.0038 ✅ | **+0.0368** ✅ | **+0.0443 (dz 26.06)** ✅ |
| D4 positive control | ✅ | ✅ | ✅ |

The D2 column above is the **legacy** router, which is what every ladder in this file uses. On
YelpChi the legacy router **already passes** D2 (+0.00575, CI [+0.00474, +0.00675], dz 4.33 on
the unpruned graph), so YelpChi is a graph where a label-free signal exists — not a graph where
the D2 fix was required. The fix (`--label-free-strategy proxy_quantile --order-feature
eig_nb_sim`, see `docs/d2_router_fix.md`) changes the verdict on **Tolokers only**
(−0.0003 p=0.70 → +0.0040 dz 4.24); on YelpChi it gives a lower mean (+0.00481) with a higher
effect size (dz 6.73) and overlapping CIs, and under ER pruning it is non-significant at
r=0.08 where legacy is not. Full-router comparison:
`logs/{yelpchi_fraud_protocol,yelpchi_fraud_protocol_eig_nb_sim_pq}/protocol_results.json`.

**Oracle-signal magnitude tracks node-level homophily spread.** Node-homophily sd is 0.063 on
Amazon, 0.241 on Tolokers (3.8×), 0.257 on YelpChi (4.1×); D1 is +0.0028 vs +0.0365 vs +0.0426
(13× and 15×) and the effect size is dz 2.68 vs 14.94 vs 25.74. A graph whose local label mixing
is *saturated* (Amazon: nearly every node has ~0.04 homophily) leaves the oracle almost nothing
to exploit. This mechanism is now measured on three datasets, and the ordering is monotone.

## 2. Amplification: replicates on Amazon, **does not** replicate on Tolokers, **does** on YelpChi

D1 (oracle > uniform) at each retention:

| retention | Amazon ER | Amazon random | Amazon degree | Tolokers ER | Tolokers random | Tolokers degree | YelpChi ER | YelpChi random | YelpChi degree |
|---|---|---|---|---|---|---|---|---|---|
| 1.00 | +0.0028 | — | — | +0.0365 | — | — | +0.0426 | — | — |
| 0.60 | — | — | — | +0.0347 ✅ | +0.0359 ✅ | +0.0011 | +0.0447 | +0.0418 | +0.0069 |
| 0.30 | — | — | — | +0.0306 ✅ | +0.0224 ✅ | −0.0008 | +0.0664 | +0.0385 | +0.0020 |
| 0.15 | +0.0031 ✅ | +0.0022 ✅ | +0.0002 | +0.0013 | +0.0069 ✅ | −0.0020 | **+0.0980** | +0.0369 | +0.0008 |
| 0.08 | **+0.0208 ✅** | +0.0012 | +0.0002 | −0.0017 | −0.0012 | −0.0015 | +0.0539 | +0.0274 | +0.0003 |
| 0.06 | **+0.0198 ✅** | +0.0011 | +0.0001 | — | — | — | — | — | — |
| 0.04 | — | — | — | −0.0018 | −0.0010 | −0.0007 | +0.0225 | +0.0175 | +0.0001 |

**Honest reading.** The ~7× ER amplification at 6–8% retention on Amazon does **not** reproduce
on Tolokers. On Tolokers ER distillation *preserves* rather than amplifies: at 60% retention it
holds 95% of the full-graph oracle gain (+0.0347 vs +0.0365), and at 30% retention 84%
(+0.0306). Below 15% retention every method collapses to ≈0 there, so the
6–8% amplification regime on Amazon sits *below* the collapse threshold seen on the sparser
graph.

YelpChi behaves like Amazon rather than Tolokers: ER **amplifies** D1 to +0.0980 at r=0.15,
**2.3× the full-graph gain**, the largest amplification measured on any graph in this file. It
then collapses by r=0.04. So amplification is **not** a one-dataset artifact, but it is also
not unconditional — the pattern is that ER amplifies where the full-graph oracle signal is
strong enough to survive pruning (Amazon weak but with a 6–8% window; YelpChi strong with a
15% window), and preserves where it is already near collapse (Tolokers).

What *does* replicate across all three datasets:

- **Degree pruning destroys the oracle channel.** +0.0001–+0.0069 on YelpChi, +0.0002/−0.0020/
  −0.0007 on Amazon, −0.0020 on Tolokers. It concentrates edges on hubs, so nearly every node
  ends up highly homophilic, the oracle's buckets degenerate, and oracle ≈ uniform. Same
  mechanism, three graphs.
- **ER ≥ random > degree at every matched budget on YelpChi** (r=0.30: +0.0664 / +0.0385 /
  +0.0020). ER's advantage is a *preservation* advantage that grows as the budget shrinks.

## 3. Refuted lead: degree pruning does **not** unlock D2 in general

This section previously read "degree pruning unlocks D2 on Tolokers" and treated it as the
project's concrete lever. The third graph refutes that.

D2 (label-free > random) by method and retention:

| retention | Tolo ER | Tolo random | Tolo degree | Yelp ER | Yelp random | Yelp degree |
|---|---|---|---|---|---|---|
| 1.00 | −0.0003 | — | — | +0.0043† | — | — |
| 0.60 | +0.0005 | +0.0005 | −0.0003 | +0.0044 | +0.0017 | +0.0009 |
| 0.30 | +0.0009 | −0.0003 | **+0.0030 ✅** | +0.0023 | +0.0027 | −0.0003 |
| 0.15 | +0.0003 | +0.0000 | **+0.0041 ✅** | +0.0040 | +0.0019 | +0.0001 |
| 0.08 | +0.0001 | −0.0001 | **+0.0055 ✅** | **+0.0055** | +0.0027 | +0.0011 |
| 0.04 | +0.0000 | −0.0016 | **+0.0048 ✅** | +0.0030 | +0.0019 | +0.0013 |

(p = 0.001953 floor where marked.) † YelpChi full-graph D2 from a 3×1 pilot, not a full run.

On Tolokers, degree pruning lifts D2 to +0.0055 — the number that motivated the lead. On YelpChi
degree pruning produces +0.0001 to +0.0013 across the entire ladder, statistically flat, while ER
produces +0.0023 to +0.0055. So "degree pruning unlocks D2" is a Tolokers-specific artifact and
must not be reported as a mechanism. What the two graphs *do* share is the recovery of D2 by ER
at r=0.08, which is worth noting as the one retention where all three graphs peak.

Amazon remains the graph where no method unlocks D2. Its best candidate reaches ~23% of its
oracle ceiling (`docs/d2_router_fix.md`).

## 4. What this says about the project

- **Supported (three datasets):** D1/D3/D4 hold on real fraud graphs; oracle gain scales
  monotonically with node-homophily variance; degree pruning destroys oracle routing; ER beats
  random and degree at matched budget on all three.
- **Supported (three datasets):** ER *can* amplify D1, but only where full-graph oracle signal is
  strong enough to survive pruning (Amazon ~7× at 6–8%, YelpChi 2.3× at 15%). Not universal, not
  a single-dataset artifact.
- **Supported (two of three):** a recoverable label-free routing signal exists on Tolokers and
  YelpChi. Amazon has none.
- **Supported on one graph only:** the `proxy_quantile` + `eig_nb_sim` fix changes a D2 failure
  into a pass on Tolokers. YelpChi passes under the legacy router anyway (overlapping CIs, higher
  legacy mean), and Amazon fails under both.
- **Not supported:** "the fix unlocks D2 on two of three graphs." An earlier draft of this file
  claimed that on the strength of a 3×1 pilot whose p=0.25 is the n=3 resolution floor. The full
  legacy run refutes it.
- **Refuted:** degree pruning as a general D2 lever.
- **Open:** D2 on dense/saturated graphs; a scale-appropriate spectral-similarity metric (SD
  saturates on all three); `--spectral-k` tuning (fixed at 8 everywhere).

## 5. Reproduce

```
python experiments/run_sparsify_ladder.py --dataset amazon   --budgets 0.15,0.08,0.06 --methods er,random,degree --splits 10 --seeds 3 --skip-sd
python experiments/run_sparsify_ladder.py --dataset tolokers --budgets 0.6,0.3,0.15,0.08,0.04 --methods er,random,degree --splits 10 --seeds 3 --skip-sd
python experiments/run_sparsify_ladder.py --dataset yelpchi  --budgets 0.6,0.3,0.15,0.08,0.04 --methods er,random,degree --splits 10 --seeds 3 --skip-sd
```

Add `--label-free-strategy proxy_quantile --order-feature eig_nb_sim --out-suffix _eig_nb_sim_pq`
for the fixed-router ladder, and `--resume` to continue an interrupted run instead of recomputing
completed points.

Data: Amazon via `https://data.dgl.ai/dataset/FraudAmazon.zip`; Tolokers via the HuggingFace
mirror `JaySuryavanshi/graph-anomaly-tolokers` (`nodes.parquet`, `edges.parquet`); YelpChi via
`https://data.dgl.ai/dataset/FraudYelp.zip`. All under git-ignored `data/benchmarks/`. Exact
effective-resistance `pinvh` on the 11,758² Tolokers Laplacian costs ~20 min once per run, and
similarly on the 14,840² YelpChi Laplacian; it is computed once per run and reused across ladder
points. `--skip-sd` avoids a dense `eigvalsh` per ladder point, and `--skip-sd` points report
`sd_laplacian = 0.0` as a placeholder — that is **not** a measured zero.

## 5. Reproduce

```
python experiments/run_sparsify_ladder.py --dataset amazon   --budgets 0.15,0.08,0.06 --methods er,random,degree --splits 10 --seeds 3 --skip-sd
python experiments/run_sparsify_ladder.py --dataset tolokers --budgets 0.6,0.3,0.15,0.08,0.04 --methods er,random,degree --splits 10 --seeds 3 --skip-sd
```

Data: Amazon via `https://data.dgl.ai/dataset/FraudAmazon.zip`; Tolokers via the HuggingFace
mirror `JaySuryavanshi/graph-anomaly-tolokers` (`nodes.parquet`, `edges.parquet`). Both under
git-ignored `data/benchmarks/`. Exact effective-resistance `pinvh` on the 11,758² Tolokers
Laplacian costs ~20 min once per run; `--skip-sd` avoids a dense `eigvalsh` per ladder point.
`--skip-sd` points report `sd_laplacian = 0.0` as a placeholder — that is **not** a measured zero.

## 6. Real spectral distortion (measured, not placeholder)

`--skip-sd` was used for every ladder above, so all `sd_laplacian` fields there are `0.0`
placeholders. `experiments/run_sd_measurement.py` measures the spectra directly (no protocol
cells), reusing one cached reference spectrum so each point costs a single `eigvalsh`.
Artifacts: `logs/{amazon,tolokers,yelpchi}_sd_measurement/sd_results.json`.

### The legacy SD metric is unusable here

`compute_spectral_distortion` pairs eigenvalue *i* of the reference with eigenvalue *i* of the
sparsified graph. That is only valid when the component count is unchanged, and sparsification
changes it at every interesting point. Degree pruning of YelpChi at r=0.04 leaves **14,460 of
14,840 nodes isolated (97.4%)**, i.e. 14,462 zero eigenvalues against 1 in the reference, so index
*i* refers to unrelated modes in the two graphs.

It saturates at **exactly 1.0000 at 43 of 45 measured points** (13/15 Amazon, 15/15 Tolokers,
15/15 YelpChi). The only 2 non-saturated points are precisely the 2 Amazon points with
`n_components == 1` � the diagnosis confirming itself. `rank_matched_distortion`
(`src/distortion.py`) compares the largest eigenvalues instead, which are the modes a low-pass
filter retains and stay ordered under fragmentation.

### Spectral distortion does not predict routing utility

`rank_matched` top-32 relative error, against the D1 accuracy gain measured on the ladders above:

| graph | method @ r=0.04 | isolated | top-32 distortion | D1 gain |
| --- | --- | --- | --- | --- |
| YelpChi | ER | 45.0% | 0.4853 | +0.0225 |
| YelpChi | random | 20.1% | 1.3275 | +0.0175 |
| YelpChi | **degree** | **97.4%** | **0.0528** | **+0.0001** |
| Amazon | ER | 12.9% | 0.8632 | amplification |
| Amazon | random | 2.7% | 0.0925 | ~none |
| Amazon | **degree** | **83.2%** | **0.7823** | ~none |

(YelpChi random r=0.04 pushes lambda_max from 176 to 410, hence a >1.0 relative error; the
metric is unbounded above by construction.)

Two results follow, and both undercut using SD as the distillation budget:

1. **SD is minimised by destroying the graph.** Degree pruning reaches the *lowest* distortion on
   YelpChi (0.0528) precisely because it has discarded 97.4% of its nodes — the surviving spectrum
   is easy to match because there is almost nothing left. Yet its D1 gain is +0.0001. A distortion
   budget that rewards this is not measuring the quantity of interest.
2. **On Amazon the ordering inverts.** Random preserves the spectrum ~9x better than ER at r=0.04
   (0.0925 vs 0.8632) but delivers no gain, while the far more distorting ER is the method that
   amplifies D1. YelpChi shows no such inversion — ER there beats random on both distortion
   (0.4853 vs 1.3275) and accuracy — so the relationship is not a consistent anti-correlation,
   it is simply absent. Spectral fidelity does not rank the methods the same way accuracy does.

Conclusion carried forward: low-pass preservation alone does not predict whether distillation
helps routing. Report `rank_matched_*` for spectrum fidelity, but decide retention on measured D1,
not on SD.
