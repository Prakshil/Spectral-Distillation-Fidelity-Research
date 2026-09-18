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

**Phase 1 (diagnostics + theory), Phase 2 (sparsifiers + comparison), and Phase 3 (router protocol: label-free GNN router, D1-D4 decision rules, PC-1c-R planted control, dilution ladder, ablations, figures) are implemented and tested — 83 tests pass.**

## System requirements

- Python >= 3.10 (developed and tested on 3.12.1, Windows / Linux)
- Packages (`pip install -e ".[dev]"`): `numpy>=1.23`, `scipy>=1.10`, `scikit-learn>=1.2`, `networkx>=2.8`, `pyyaml>=6.0`, `torch>=2.0` (only for CUDA-aware seeding; diagnostics run without it), `pytest`, `matplotlib` (optional; SD-vs-budget plots)
- Hardware: any CPU for Phase 1-2 (`n <= 512` sweeps run in ~2 minutes); the LLM-export / GNN phases later expect the RTX 4060 Ti (16 GB) or a GPU runtime

## Run and test (Phase 1-3)

```bash
pip install -e ".[dev]"          # or: make install
python -m pytest                 # 83 tests: theory bounds, sparsifiers, router protocol, statistics

# Phase 1-2
python -m spectral_distillation.experiments.run_diagnostic     # golden 4-node + ERP SD/fragility preview
python -m spectral_distillation.experiments.compare_methods     # SD vs budget, threshold/random/degree/spectral

# Phase 3 (router protocol)
python -m spectral_distillation.experiments.run_protocol        # D1-D4 findings matrix + frozen-gate interventions
python -m spectral_distillation.experiments.dilution_curve      # label-free/oracle gain vs dilution ladder
python -m spectral_distillation.experiments.ablation            # oracle-filter threshold + router-feature ablation
python -m spectral_distillation.experiments.reproduce_figures   # render all Phase 3 figures from logs
```

- `make test-theory` runs the theory-bound test file only.
- Runners accept `--n`, `--n-patches`, `--splits`, `--seeds`, `--router-epochs`, `--config`, `--out` (see each module's argparse help). Use small `--n` (600-2000) for CPU smoke runs; the defaults target the full 10k-node PC-1c-R control on a GPU.
- Outputs: `logs/protocol/protocol_results.json`, `logs/dilution/dilution_curve.{csv,json}`, `logs/ablation/ablation.{csv,json}`, plus PNGs under `logs/protocol/`, `logs/dilution/`, `logs/ablation/`.
- Config defaults live in `configs/default.yaml` (`router_protocol:`, `planted_control:`, `experiments:` sections).

### Verified findings so far

- Golden 4-node ERP example matches the guide: SD(τ=0.5) = 1.0 (Customer isolated), `‖ΔL‖_F = 0.8832`, fragility ≈ 2.06, `Σ w_e·R_eff = n−1` exactly.
- Spectral sparsification is strictly monotone in budget and beats thresholding whenever the graph stays connected; thresholding drops low-weight *bridge* edges and collapses entirely at `n ≥ 256` on ERP graphs, while spectral search-space selection keeps Fiedler alive at the same budget.
- ERP attention graphs are ultra-fragile (pendant-dominated), `min_eigengap ≈ 4e-3`; any method that drops a unique bridge pushes SD to 1.0.
- Caution: top-k effective resistance is a *lower bound* on true R and only accurate near full rank — the dispatcher uses exact R for `n <= 2000`.

### Phase 3 findings

- Planted PC-1c-R control with 3 orthogonal feature bases creates a genuine expert-transfer gap: single global LR ≈ 0.555, random routing ≈ 0.514, oracle ≈ 0.865.
- Label-free KMeans on {hx, spectral_ratio, log_degree, clustering} reproduces true regimes at ~89.5% contingency; a RouterGNN (KL-divergence imitation) also trains but the crisp KMeans assignment is the actual gate used.
- Decision rules D1–D4 all **Supported** on 10k nodes (10 splits × 3 seeds): label-free gain ≈ 0.20-0.24, oracle gain ≈ 0.24-0.30, Wilcoxon p < 0.01 after Holm correction.
- Frozen-gate intervention: learned router > node-shuffled permutations > uniform/global_mean; oracle one-hot ≈ upper bound.
- Dilution ladder: label-free gain decays monotonically as regime nodes are rewritten to homogeneous low-pass wiring; oracle gain (true regime id) stays constant — consistent with the claim that dilution erases the structural signal the label-free router reads.
- Oracle-bucket threshold sweep shows a stable plateau for δ ∈ [0.1, 0.5] on the homophily-range SBM, degrading at extreme widths.
- Router-feature ablation: all four fields contribute; hx and spectral_ratio are the strongest individual features; leave-one-out gains near the full-set gain confirm redundancy.

## Target venue

Tier-1 ML conference (NeurIPS / ICML / ICLR).