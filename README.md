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
make protocol         # full D1-D4 findings matrix (~2.5 min on an RTX 4060)
make figures          # all paper figures
```

## Source documents

- `docs/Spectral_Distillation_Complete_Guide.md` — the mathematical foundations
- `docs/does-your-router-actually-route 4.pdf` — the router evaluation protocol
- `SPECTRAL_DISTILLATION_IMPLEMENTATION_GUIDE.md` — this repo's implementation spec
- `docs/Insights.md` — plain-English walkthrough of the model, the results, and next steps (no background needed)
- `docs/reproducibility.md` — implementation status, verified results, your next steps, further phases
- `docs/method_section.md` — paper-style method draft (Phase 5)
- `docs/theorem_proofs.md` — full proof chain + honest caveats draft (Phase 5)
- `docs/literature_positioning.md` — positioning vs MoG / GOKU / MORGAN and the router-evaluation lineage

## Hardware

- Local: NVIDIA RTX 4060 Laptop (8 GB), CUDA 13.x. The Phase-3 10k pipeline (`run_protocol`+`dilution`+`ablation`+`figures`) was measured at ~9 minutes with `--device cuda` (`docs/reproducibility.md` §4.2). That figure **excludes** `scalability_reeff` (Table 3), whose n=10,000 top-k grid takes ~25 minutes on its own.
- Committed logs are the **CPU** reference (`--device cpu`) — CUDA and CPU are not bit-identical, so reproduce with the default `device: cpu` before comparing numbers.
- Heavy protocol runs: Kaggle / Colab GPUs

## Status

**Phase 1 (diagnostics + theory), Phase 2 (sparsifiers + comparison), Phase 3 (router protocol: label-free GNN router, D1-D4 decision rules, PC-1c-R planted control, dilution ladder, ablations, figures), the fair published-router audit (`RouterGNN` / `AD-GNN` / MLP gate under Protocol B), and the learnability lever (`β`-sweep + abstaining selective router) are implemented and tested — 250 tests pass.**

> **Headline result (honest, Protocol B).** On the planted **PC-1c-R control** the D1–D4 instrument is valid and routing helps (see "Phase 3 findings"). On **real graphs** (fraud benchmarks and real LLM attention) under the fixed-expert protocol, **no label-free or learned router beats training one global model** (2/234 candidate/expert/graph cells — both reduced-power YelpChi GNN cells at ~+0.002, p>=0.75; every full-scale cell is 0/26), and the oracle is *negative* (9/9). The one supported positive is that connectivity **survival** predicts routing *stability* (Spearman +0.78..+0.99); low-frequency spectral distortion is a collinear proxy that loses significance once the isolated-node fraction is partialled out (pooled partial rho = -0.20, p = 0.16, vs survival +0.83, p = 1e-13; `docs/real_attention_results.md` §9.3). Protocol-A self-routing oracle numbers are **retracted**. See `docs/fixed_expert_protocol.md` and `docs/real_attention_results.md`.

## System requirements

- Python >= 3.10 (developed and tested on 3.12.1, Windows / Linux)
- Packages (`pip install -e ".[dev]"`): `numpy>=1.23`, `scipy>=1.10`, `scikit-learn>=1.2`, `networkx>=2.8`, `pyyaml>=6.0`, `torch>=2.0` (only for CUDA-aware seeding; diagnostics run without it), `pytest`, `matplotlib` (optional; SD-vs-budget plots)
- Hardware: any CPU for Phase 1-2 (`n <= 512` sweeps run in ~2 minutes); the GNN / LLM-export phases expect a CUDA GPU (developed on an NVIDIA RTX 4060 Laptop, 8 GB) or a larger GPU runtime

## Run and test (Phase 1-3)

```bash
pip install -e ".[dev]"          # or: make install
python -m pytest                 # 250 tests: theory bounds, sparsifiers, router protocol, statistics

# Phase 1-2
python -m spectral_distillation.experiments.run_diagnostic     # golden 4-node + ERP SD/fragility preview
python -m spectral_distillation.experiments.compare_methods     # SD bands vs budget/epsilon, threshold/random/degree/spectral
python -m spectral_distillation.experiments.scalability_reeff   # WS2: exact vs top-k R_eff at n=100/1k/10k (Table 3)

# Phase 3 (router protocol)
python -m spectral_distillation.experiments.run_protocol        # D1-D4 findings matrix + frozen-gate interventions
python -m spectral_distillation.experiments.frozen_interventions --graph pc   # WS3: expert-architecture/capacity ablations
python -m spectral_distillation.experiments.dilution_curve      # label-free/oracle gain vs dilution ladder
python -m spectral_distillation.experiments.ablation            # oracle-filter threshold + router-feature ablation
python -m spectral_distillation.experiments.reproduce_figures   # render all Phase 3 figures from logs
```

- `make test-theory` runs the theory-bound test file only. The Makefile targets run on `cuda` by default (runners auto-fallback to `cpu` when CUDA is absent).- Runners accept `--n`, `--n-patches`, `--splits`, `--seeds`, `--router-epochs`, `--device`, `--config`, `--out` (see each module's argparse help). Use small `--n` (600-2000) for CPU smoke runs; the defaults target the full 10k-node PC-1c-R control on a GPU.
- Outputs: `logs/protocol/protocol_results.json`, `logs/dilution/dilution_curve.{csv,json}`, `logs/ablation/ablation.{csv,json}`, `logs/compare_methods/{results,bands}.{csv,json}` + `bands_vs_*_n*.png`, `logs/frozen_interventions/{pc,erp_fraud}.json`, `logs/scalability_reeff/table3.{csv,json}`, plus Phase 3 figures under `logs/protocol/`, `logs/dilution/`, `logs/ablation/`. `reproduce_figures.py` also exports Tables 1-3 as CSV + LaTeX (`table1_*`, `table2_*`, `table3.tex`; `make tables`).
- Config defaults live in `configs/default.yaml` (`router_protocol:`, `planted_control:`, `experiments:` sections).

### Verified findings so far

- Golden 4-node ERP example matches the guide: SD(τ=0.5) = 1.0 (Customer isolated), `‖ΔL‖_F = 0.8832`, fragility ≈ 2.06, `Σ w_e·R_eff = n−1` exactly.
- On equal-edge-mass budget/ε bands (120 grid points), spectral reliably beats *thresholding* (74.2%) but does **not** dominate all heuristics: raw random/degree stay competitive and spectral is the single best method at only 23.3% of points. An earlier point-estimate claim ("beats threshold, collapses at `n≥256`") was confounded by spectral's inverse-propensity weight rescale; see `docs/robustness_scaling.md`.
- ERP attention graphs are ultra-fragile (pendant-dominated), `min_eigengap ≈ 4e-3`; any method that drops a unique bridge pushes SD to 1.0.
- Caution: top-k effective resistance is a *lower bound* on true R and only accurate near full rank — the dispatcher uses exact R for `n <= 2000`. Quantified in Phase 6 / WS2 (`docs/robustness_scaling.md` Part 2): absolute edge error stays high (rel-Frob ≈ 0.87 at n=1000, k=128) but the *ranking* is faithful (Spearman ρ → 0.91) and the induced low-frequency distortion is within 0.26–1.2% of exact, so `k = 64–128` suffices; exact R at n=10000 is skipped (O(n²) memory).

### Phase 3 findings

> **Scope.** Everything in this subsection is the **synthetic PC-1c-R planted
> control** scored under the (later-retracted) self-routing Protocol A. It is the
> **D4 instrument check** — proof that the harness detects a planted routing
> signal — *not* a real-data routing win. Protocol-A oracle effect sizes are
> void on real graphs; see `docs/fixed_expert_protocol.md` for the honest
> fixed-expert (Protocol B) numbers.

- Planted PC-1c-R control with 3 orthogonal feature bases creates a genuine expert-transfer gap: single global LR ≈ 0.555, random routing ≈ 0.514, oracle ≈ 0.865.
- Label-free KMeans on {hx, spectral_ratio, log_degree, clustering} reproduces true regimes at ~89.5% contingency; a RouterGNN (KL-divergence imitation) also trains but the crisp KMeans assignment is the actual gate used.
- Decision rules D1–D4 all **Supported** on 10k nodes (10 splits × 3 seeds): label-free gain ≈ 0.20-0.24, oracle gain ≈ 0.24-0.30, Wilcoxon p < 0.01 after Holm correction.
- Frozen-gate intervention: learned router > node-shuffled permutations > uniform/global_mean; oracle one-hot ≈ upper bound.
- Frozen-gate ablations (Phase 6 / WS3): the ordering holds for **all** expert architectures — logistic, mlp{8,32,64,128}, and heterogeneous dendrite — on both pc and erp_fraud (n = 2000); the within-regime permutation is near-identity (gap ≤ 0.002), confirming the crisp router is a regime detector. See `docs/robustness_scaling.md` Part 3.
- Dilution ladder: label-free gain decays monotonically as regime nodes are rewritten to homogeneous low-pass wiring; oracle gain (true regime id) stays constant — consistent with the claim that dilution erases the structural signal the label-free router reads.
- Oracle-bucket threshold sweep shows a stable plateau for δ ∈ [0.1, 0.5] on the homophily-range SBM, degrading at extreme widths.
- Router-feature ablation: all four fields contribute; hx and spectral_ratio are the strongest individual features; leave-one-out gains near the full-set gain confirm redundancy.

## Target venue

Tier-1 ML conference (NeurIPS / ICML / ICLR).