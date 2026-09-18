# Spectral Distillation Fidelity

Bounding and controlling routing-relevant information loss when compressing LLM attention matrices into sparse graphs, with applications to enterprise fraud detection.

## What this project does

1. **Pre-training diagnostic** — Spectral Distortion (SD) predicts how much routing signal will survive graph distillation *before* any GNN training.
2. **Spectral sparsification** — replaces naive thresholding with effective-resistance sampling (+ business-rule oracle filter) so all eigenvalues survive within `(1±ε)`.
3. **Attention-entropy routing signal** — a pre-construction signal computed from the LLM that graph construction cannot corrupt.
4. **Provable bounds** — `I(π*;Y) − I(π̃;Y) ≤ C·SD·H(Y)` via Davis-Kahan, Lipschitz stability, and the Data Processing Inequality.
5. **Rigorous evaluation** — the oracle / uniform / random / label-free protocol with decision rules D1–D4, planted-regime controls (PC-1c-R), and frozen-model interventions.

## Quick start

```bash
make install
make test-theory      # validates Weyl / Davis-Kahan / Cauchy on the 4-node example
make diagnostic       # SD preview on synthetic graphs
make protocol         # full D1-D4 findings matrix (heavy; use Kaggle/Colab)
make figures          # all paper figures
```

## Source documents

- `docs/Spectral_Distillation_Complete_Guide.md` — the mathematical foundations
- `docs/does-your-router-actually-route 4.pdf` — the router evaluation protocol
- `SPECTRAL_DISTILLATION_IMPLEMENTATION_GUIDE.md` — this repo's implementation spec

## Hardware

- Local: RTX 4060 Ti (16 GB) — development, small-scale ablations
- Heavy protocol runs: Kaggle / Colab GPUs

## Status

Scaffolding only; source modules are empty placeholders ready for phase-wise implementation.

## Target venue

Tier-1 ML conference (NeurIPS / ICML / ICLR).