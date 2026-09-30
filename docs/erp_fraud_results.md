# ERP-Fraud Router Verification — Results

Graph source: scalable synthetic ERP-fraud graph (`src/synthetic.py::generate_erp_fraud_graph`)
Protocol: identical D1–D4 machinery as the PC-1c-R and real-attention runs (`run_protocol.py --graph erp_fraud`)

## 1. Headline result

| Rule | Meaning | n=3000 | verdict |
|---|---|---|---|
| **D1** | signal exists (oracle > uniform) | +0.2763, d_z 46.0 | pass |
| **D2** | router finds it (label-free > random) | +0.2894, d_z 53.2 | pass |
| **D3** | not capacity (oracle > random) | +0.2894, d_z 53.2 | pass |
| **D4** | instrument valid (frozen gate) | learned = oracle_onehot = 0.855, perm-std 0.007 | pass |

**Verdict: Supported —** on the fraud-like ERP graph the structural router genuinely recovers the
planted regime structure (label-free ↔ oracle regime agreement = 1.0 after channel relabel), and the
frozen-gate intervention confirms the label-free gain is real routing, not capacity.

## 2. Graph structure

- 3,000 nodes: 1,650 low-pass (suppliers+invoices), 150 high-pass (shell/fraud), 1,200 noisy (customers).
- Fraud planted as 150 shell nodes = dense hub edges (p≈0.8) to the supplier/invoice backbone, so shells
  carry the highest degree (mean 1,540 vs suppliers+invoices 605 vs customers 226).
- Class labels shared across regimes, projected through regime-specific orthogonal bases (PC-1c-R
  deconfounding), so a linear expert trained in one regime transfers poorly to another.

## 3. Why this is a clean transfer result (vs the real-attention failure)

The real-attention graph showed D2 as a usage-imbalance artifact because its structural features
saturate (hx ≈ 0.997 flat, router collapses to [88, 11408, 24]). The ERP-fraud graph instead has
well-separated degree/spectral structure per regime, so the KMeans assignment does not collapse:

| condition | usage |
|---|---|
| oracle | [1650, 150, 1200] |
| label_free | [1650, 1200, 150] (channels permuted) |
| random (matched to oracle) | [1686, 157, 1157] |
| uniform | [3000, 0, 0] |

Label-free usage tracks oracle within a channel permutation (regime agreement 1.0); this is why
D2 mean_diff equals D3 exactly (+0.2894). The frozen gate lays the equal/global_mean baselines at
0.578 and gives node-shuffled mean 0.564 ± 0.007, so the +0.289 gain is not a capacity artifact.

## 4. Caveats

- Synthetic, not real ERP data: no real ERP/fraud graph files exist in the repo (`data/benchmarks/` is
  empty), so this is the fraud-like domain emulation, not a real-world validation.
- Uniform usage [3000,0,0] is the argmax-bincount convention (equal soft rows), as in the other runs.
- p-values all 0.001953 (exact Wilcoxon, 10×3 protocol).

## 5. Reproduce

```
python -m spectral_distillation.experiments.run_protocol --graph erp_fraud \
    --n 3000 --n-patches 50 --splits 10 --seeds 3 --router-epochs 200 --device cuda --out logs
```
Artifact: `logs/erp_fraud_protocol/protocol_results.json`