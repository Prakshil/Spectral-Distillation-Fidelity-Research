# Spectral-Distillation Retention Ladder on Real Fraud Data — Cross-Dataset Results

Two real public fraud graphs, both fully labeled, single connected component, run through the
identical D1–D3 experts/training (10 splits × 3 seeds, p = exact Wilcoxon, resolution floor
0.001953):

| | Amazon (co-review) | Tolokers (crowd-worker exclusion) |
|---|---|---|
| nodes / features | 8,639 / 25 | 11,758 / 10 |
| undirected edges | 3,298,534 | 519,000 |
| mean degree | 763.6 | 88.3 |
| positive ratio | 9.50% | 21.82% |
| edge label homophily | 0.9488 | 0.5945 (random baseline 0.6588 → heterophilous) |
| **node-homophily sd** | **0.063** (saturated) | **0.241** |

Method: `experiments/run_sparsify_ladder.py --dataset {amazon,tolokers}`. Artifacts:
`logs/amazon_fraud_ladder/ladder_results.jsonl`, `logs/tolokers_fraud_ladder/ladder_results.jsonl`.

## 1. Headline D1–D4: only D2 fails, on both graphs

| | Amazon | Tolokers |
|---|---|---|
| D1 oracle > uniform | +0.0028 (dz 2.68) ✅ | **+0.0365 (dz 14.94)** ✅ |
| D2 label-free > random | −0.00015 ❌ | −0.0003 ❌ |
| D3 oracle > random | +0.0038 ✅ | **+0.0368** ✅ |
| D4 positive control | ✅ | ✅ |

**Oracle-signal magnitude tracks node-level homophily spread.** Node-homophily sd is 0.063 on
Amazon and 0.241 on Tolokers (3.8×); D1 is +0.0028 vs +0.0365 (13×) and the effect size is
dz 2.68 vs 14.94 (5.6×). A graph whose local label mixing is *saturated* (Amazon: nearly every
node has ~0.04 homophily) leaves the oracle almost nothing to exploit. This is the mechanism
behind the oracle channel, and it is now measured on two datasets rather than one.

## 2. Amplification: replicates on Amazon, **does not** replicate on Tolokers

D1 (oracle > uniform) at each retention:

| retention | Amazon ER | Amazon random | Amazon degree | Tolokers ER | Tolokers random | Tolokers degree |
|---|---|---|---|---|---|---|
| 1.00 | +0.0028 | — | — | +0.0365 | — | — |
| 0.60 | — | — | — | +0.0347 ✅ | +0.0359 ✅ | +0.0011 |
| 0.30 | — | — | — | +0.0306 ✅ | +0.0224 ✅ | −0.0008 |
| 0.15 | +0.0031 ✅ | +0.0022 ✅ | +0.0002 | +0.0013 | +0.0069 ✅ | −0.0020 |
| 0.08 | **+0.0208 ✅** | +0.0012 | +0.0002 | −0.0017 | −0.0012 | −0.0015 |
| 0.06 | **+0.0198 ✅** | +0.0011 | +0.0001 | — | — | — |
| 0.04 | — | — | — | −0.0018 | −0.0010 | −0.0007 |

**Honest reading.** The ~7× ER amplification at 6–8% retention on Amazon does **not** reproduce
on Tolokers. On Tolokers ER distillation *preserves* rather than amplifies: at 60% retention it
holds 95% of the full-graph oracle gain (+0.0347 vs +0.0365), and at 30% retention 84%
(+0.0306). Below 15% retention every method collapses to ≈0 on both graphs, so the
6–8% amplification regime on Amazon sits *below* the collapse threshold seen on the sparser
graph. **Amplification is a property of the dense, homophilically-saturated Amazon regime, not a
general law.** A paper may not claim it generally on this evidence.

What *does* replicate across both datasets:

- **Degree pruning destroys the oracle channel.** +0.0002/−0.0020/−0.0007 at matched budgets.
  It concentrates edges on hubs, so nearly every node ends up highly homophilic, the oracle's
  buckets degenerate, and oracle ≈ uniform. Same mechanism, both graphs.
- **ER ≥ random > degree at moderate budgets** (Tolokers r=0.30: +0.0306 / +0.0224 / −0.0008).
  ER retains ~84% of the signal where random retains ~61%. This is a real but *modest*
  preservation advantage, not an amplifier.

## 3. Unexpected positive lead: degree pruning **unlocks D2 on Tolokers**

D2 (label-free > random) on Tolokers, by method and retention:

| retention | ER | random | degree |
|---|---|---|---|
| 1.00 | −0.0003 | — | — |
| 0.60 | +0.0005 | +0.0005 | −0.0003 |
| 0.30 | +0.0009 | −0.0003 | **+0.0030 ✅** |
| 0.15 | +0.0003 | +0.0000 | **+0.0041 ✅** |
| 0.08 | +0.0001 | −0.0001 | **+0.0055 ✅** |
| 0.04 | +0.0000 | −0.0016 | **+0.0048 ✅** |

(p = 0.001953 floor at 0.30/0.08/0.04.) On the *sparser, heterophilous* graph the label-free
router **does** beat random — but only after degree-based sparsification, and it is the exact
method that destroys D1. This is the first real-data evidence that the label-free structural
router is not hopeless: its signal is recoverable once node-local structure is sharpened.
On Amazon (dense/saturated) no method unlocks D2.

## 4. What this says about the project

- **Supported (two datasets):** D1/D3/D4 hold on real fraud graphs; oracle gain scales with
  node-homophily variance; degree pruning destroys oracle routing; ER preserves better than
  random at matched budget.
- **Not supported:** general spectral amplification (single-dataset artifact); automatic
  label-free recovery on dense graphs.
- **Open and now tractable:** D2 on heterophilous graphs. The degree-pruning result identifies a
  concrete lever (node-local structural sharpening) and a concrete tension (the lever that helps
  D2 hurts D1), which is the mechanism to engineer against.

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
