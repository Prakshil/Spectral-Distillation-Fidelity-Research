# Real Fraud-Graph Router Verification (DGL Amazon) — Results

Graph source: DGL Amazon review-correlation fraud graph (`data/benchmarks/amazon/raw/Amazon.mat`,
downloaded from `https://data.dgl.ai/dataset/FraudAmazon.zip`)
Loader: `src/real_fraud.py::load_amazon_fraud` (labeled block rows 3305+, induced subgraph)
Protocol: identical D1–D4 machinery as the PC-1c-R / real-attention / ERP-fraud runs
(`experiments/run_real_fraud.py`)

## 1. Graph structure

- 11,944 users total in the raw mat; DGL documents rows 0–3304 as unlabeled. The loader
  returns the **labeled-induced subgraph on rows 3305+**: 8,639 users, single connected
  component, 25 review-statistic features.
- Labels are real benign/fraud user labels from the CARE-GNN / DGL release: 7,818 benign,
  **821 fraudulent (9.5%)** — the exact positive ratio GADBench reports for Amazon.
- Adjacency is the union (`homo`) of the three review-correlation relations (U-P-U, U-S-U,
  U-V-U), symmetrized and diagonal-free: 3,298,534 undirected edges, mean degree ≈ 764.
- Edge label homophily = 0.954 (strongly homophilic in the benign majority).

## 2. Headline result

| Rule | Meaning | value (n=8,639, 10×3) | verdict |
|---|---|---|---|
| **D1** | signal exists (oracle > uniform) | +0.0028, d_z 2.68, p 0.001953 | pass |
| **D2** | router finds it (label-free > random) | −0.0002, d_z −0.17, p 0.846 | **fail** |
| **D3** | not capacity (oracle > random) | +0.0038, d_z 3.42, p 0.001953 | pass |
| **D4** | instrument valid (frozen gate + PC control) | PC oracle gain +0.253 (p 0.031); perm-std 0.0011 | pass |

**Verdict: D1/D3/D4 supported, D2 not supported on the real Amazon fraud graph** — the oracle
that reads the real labels beats uniform and usage-matched random, but the label-free
structural router cannot beat random. This mirrors the real-attention result, not the
synthetic ERP-fraud result.

## 3. Why D2 fails here (and why that is the informative outcome)

| condition | usage (n=8,639) |
|---|---|
| oracle (adaptive homophily buckets) | [2881, 2878, 2880] — balanced |
| label_free (structural KMeans) | [3673, 400, 4566] — collapses into 2 dominated channels |
| random (matched to oracle) | [2926, 2783, 2930] |
| uniform | [8639, 0, 0] (argmax-convention display of equal soft rows) |

The three review-correlation relations are all *between users who interacted with the same
product / star / text*, so every node is structurally similar (near-uniform high homophily,
0.954 edge homophily). The structural signal fields saturate on this graph and the KMeans
channels simply do not track the label-homophily buckets — so label-free ≈ random, and D2's
mean difference is statistically indistinguishable from zero. That is a real finding: on
genuine co-review fraud networks there is **no routing-recoverable structural signal beyond
the labels**; the +0.0038 oracle gain is the true ceiling tied to the label signal.

The D4 instrument is honest here: the PC-1c-R positive control inside the same run shows the
machinery *can* recover structure when it exists (oracle gain +0.253, all three control
comparisons significant), so the D2 failure is graph-specific, not a broken test.

## 4. Caveats

- Real but domain-constrained: the Amazon graph is dense, strongly homophilic co-review data —
  a deliberate contrast to the low-homophily ERP-fraud target domain, where the synthetic run
  (D2 +0.289) shows recoverable structural signal.
- 3,305 raw users dropped as DGL's documented unlabeled block; the run is on the 8,639 labeled
  users, matching the GADBench evaluation convention (9.5% positive).
- Oracle uses adaptive (quantile) homophily thresholds; fixed 0.4/0.6 thresholds collapse to
  [7816, 30, 793] and would make even D1 degenerate.
- p-values are exact-Wilcoxon resolution floors of the 10×3 protocol.

## 5. Reproduce

```
python spectral_distillation/experiments/run_real_fraud.py --splits 10 --seeds 3 \
    --router-seed 0 --random-seed 1 --frozen-perms 20 --out logs
```
Artifact: `logs/amazon_fraud_protocol/protocol_results.json`
(Requires `data/benchmarks/amazon/raw/Amazon.mat`, git-ignored under `data/benchmarks/`.)