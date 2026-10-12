# ERP-fraud protocol results

Full-size run driven by `configs/erp_fraud.yaml` (n=10,000), with the
previously committed n=3,000 run retained as a scale-consistency check.
The graph is synthetic (`generate_erp_fraud_graph` in `src/synthetic.py`); it is
a fraud-*like* domain emulation, **not** a real-world ERP dataset (see caveats).
Protocol and oracle are identical to the PC-1c-R and real-attention runs
(`run_protocol.py --graph erp_fraud`).

## 1. Headline result

| Rule | Meaning | n=10000 (config) | n=3000 (prior) | verdict |
|---|---|---|---|---|
| **D1** | signal exists (oracle > uniform) | +0.2054, d_z 88.10 | +0.2763, d_z 46.0 | pass |
| **D2** | router finds it (label-free > random) | +0.2118, d_z 71.79 | +0.2894, d_z 53.2 | pass |
| **D3** | not capacity (oracle > random) | +0.2118, d_z 71.79 | +0.2894, d_z 53.2 | pass |
| **D4** | instrument valid (frozen gate) | learned = oracle_onehot = 0.8806; node-shuffled 0.6657 ± 0.0025 | learned = oracle = 0.855; perm-std 0.007 | pass |

**Verdict (both sizes): Supported** — on the fraud-like ERP graph the structural
router genuinely recovers the planted regime structure (label-free ↔ oracle
regime agreement = 1.0 after channel relabel), and the frozen-gate intervention
confirms the label-free gain is real routing, not capacity. `D2 == D3` exactly
because the label-free usage equals the oracle usage up to a channel permutation.

All primary p-values = 0.001953 (exact Wilcoxon, 10×3 protocol).

## 2. Graph structure (measured, seed 0)

Regimes: supplier/invoice → L(0), shell/fraud → H(1), customer → M(2).

| size | regime | nodes | mean degree |
|---|---|---|---|
| n=10000 | L suppliers/invoices | 5,500 | 1,512.6 |
| n=10000 | H shell/fraud | 500 | 3,845.8 |
| n=10000 | M customers | 4,000 | 565.5 |
| n=3000 | L suppliers/invoices | 1,650 | 453.3 |
| n=3000 | H shell/fraud | 150 | 1,152.2 |
| n=3000 | M customers | 1,200 | 170.0 |

Fraud is planted as dense shell hubs (p≈0.8) to the supplier/invoice backbone, so
shells carry the highest degree (3,846 vs 1,513 vs 566 at n=10,000). Class labels
are shared across regimes and projected through regime-specific orthogonal bases
(PC-1c-R deconfounding), so a linear expert trained in one regime transfers
poorly to another — the signal the protocol must recover.

## 3. Condition usage (n=10,000)

| condition | usage (L, H, M) |
|---|---|
| oracle | [5500, 500, 4000] |
| label_free | [5500, 4000, 500] (channels 1/2 permuted) |
| random (matched to oracle) | [5557, 526, 3917] |
| uniform | [10000, 0, 0] |

Label-free usage tracks the oracle within a channel permutation (regime
agreement 1.0). The frozen gate lays the equal/global_mean baselines at 0.6682
and the node-shuffled mean at 0.6657 ± 0.0025, so the +0.21 gain is not a
capacity artifact.

## 4. Scale note

The gain *shrinks* with size (D1 +0.2763 → +0.2054; D3 +0.2894 → +0.2118 from
n=3,000 → 10,000) while staying large and at the p floor. The regime proportions
are identical across sizes (0.55 / 0.05 / 0.40), so this is a finite-size effect,
not a change of task.

## 5. Caveats

- Synthetic, not real ERP data: no real ERP/fraud graph files exist in the repo
  (`data/benchmarks/` is empty for ERP), so this is a domain emulation, not
  real-world validation. The real-data counterparts are the DGL Amazon /
  Tolokers / YelpChi fraud graphs (`docs/real_fraud_results.md`,
  `docs/sparsify_ladder_results.md`) and the SmolLM2 attention graph
  (`docs/real_attention_results.md`).
- Uniform usage [10000, 0, 0] is the argmax-bincount convention (equal soft
  rows), as in the other runs.
- `configs/erp_fraud.yaml`'s `sparsification` / `oracle_filter` sections are not
  consumed by `run_protocol.py` (it reads only the planted-control size and the
  protocol block); they apply to the sparsification ladders.

## 6. Reproduce

```
python -m spectral_distillation.experiments.run_protocol --graph erp_fraud \
    --config configs/erp_fraud.yaml --out logs
```
Artifacts: `logs/erp_fraud_protocol/protocol_results.json` (n=10,000, canonical)
and `logs/erp_fraud_protocol/protocol_results_n3000.json` (n=3,000). Runtime
~3.5 min on the RTX 4060 Laptop GPU.
