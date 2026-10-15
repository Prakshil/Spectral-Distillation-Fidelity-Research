> **RETRACTED (2026-10-04).** This is the historical record and predates the
> fixed-expert protocol. The D1/D3 oracle effect sizes it quotes (dz 14.94,
> dz 25.74, dz 26.06) are self-routing artifacts and are void, as are the
> "13x larger than Amazon" comparison and the label-free candidate wins they are
> used to support. For current numbers see `docs/fixed_expert_protocol.md`, which
> supersedes this document for every result claim.

# Spectral Distillation — Implementation Status, Results & Next Steps

This document is the living record of what has been built, what has been verified,
how to reproduce every result, and what remains.

Companion documents:
- `SPECTRAL_DISTILLATION_IMPLEMENTATION_GUIDE.md` — the full engineering spec (Parts 1–11 + appendices).
- `docs/Spectral_Distillation_Complete_Guide.md` — the mathematical foundations.
- `docs/does-your-router-actually-route 4.pdf` — the router-evaluation protocol we implement.
- `docs/theorem_proofs.md` and `docs/method_section.md` — publication drafts (see "Further phases").

---

## 1. Status overview

| Phase | Scope | Status |
|-------|-------|--------|
| Phase 1 | Diagnostics + theory (SD metric, fragility, entropy; provable bounds) | **Implemented & tested** |
| Phase 2 | Spectral sparsifiers + method comparison (threshold / random / degree / spectral) | **Implemented & tested** |
| Phase 3 | Router evaluation protocol (label-free GNN router, D1–D4, PC-1c-R control, dilution ladder, ablations, figures) | **Implemented & tested** |
| Phase 4 | Real ERP attention data + LLM attention export (SmolLM2-1.7B) feeding the pipeline | **Done** (synthetic ERP-fraud, real LLM attention, and three real fraud graphs Amazon/Tolokers/YelpChi; see `docs/real_fraud_results.md`) |
| Phase 5 | Publication materials from verified results | Partially drafted (`docs/*`) |
| Phase 6 | Robustness and scaling (budget/epsilon bands, top-k R_eff, extended frozen interventions) | **Implemented & tested** (`docs/robustness_scaling.md`) |

**Tests:** 257 pass (`python -m pytest` on CPU, cosmetic warnings only).

**Lint:** `pyflakes` clean on `src/`, `experiments/`, and `tests/`. (`ruff`/`ty` are not installed locally; the Makefile targets remain for CI.)

---

## 2. What is implemented, step by step

### 2.1 Module layout (`spectral_distillation/src/`)

| Module | Responsibility |
|--------|----------------|
| `laplacian.py` | Ordered Laplacian, `normalize_adjacency`, spectral utilities |
| `distortion.py` | Spectral Distortion (SD) metric + Weyl / Davis-Kahan / Cauchy helpers |
| `theoretical_bounds.py` | The provable routing-signal-decay bounds (Lipschitz + DPI chain) |
| `effective_resistance.py` | Exact and top-k effective resistance (the sparsification sampler) |
| `sparsifier.py` | Effective-resistance spectral sparsifier (+ business-rule oracle filter) |
| `attention_entropy.py` | Pre-construction entropy routing signal from LLM attention |
| `spectral_position.py` | Node-level spectral footprint features (ratio, position, fragility) |
| `synthetic.py` | ERP-style graph + label generators used by all experiments |
| `real_fraud.py` | Loaders for real public fraud graphs (DGL Amazon co-review, Tolokers crowd-workers, YelpChi reviews) into the protocol schema |
| `baselines.py` | Dummy / uniform / random routing baselines |
| `mixture.py` | Mixture-of-experts classifier (`low_pass` / `high_pass` / `identity`) |
| `gnn_router.py` | RouterGNN (label-free learned router, KL-divergence training) |
| `gnn_expert.py` | Message-passing expert head over cached `[X, AX]`; `gnn_head_factory` |
| `learned_router.py` | RouterGNN-lite: MLP router over cached `[X, AX]`, supervised by training labels on a held-out router split; frozen routing-blind expert pool |
| `published_routers.py` | Published learned routers under Protocol B: RouterGNN (message-passing, label-supervised) + AD-GNN (per-node propagation depth via a label-free signal-preservation factor) |
| `positive_control.py` | Positive control: non-separable mixture of linear rules, oracle routing from the latent partition |
| `learnability_lever.py` | One knob interpolates the label function from globally-separable (β=0) to the positive control (β=1); + `learnability_conditioned_router` (route iff held-out router-val gain > margin) |
| `planted_control.py` | PC-1c-R positive control: 3 orthogonal feature bases, dilution ladder |
| `router_protocol.py` | The oracle / label-free / random / uniform protocol, D1–D4 decision rules, frozen-model intervention, statistics |
| `evaluation.py` | Splits, seeds, paired statistical tests (Wilcoxon exact, paired-t, Holm, CI, effect size) |

### 2.2 Experiments (`spectral_distillation/experiments/`)

| Script | What it does | Output |
|--------|--------------|--------|
| `run_diagnostic.py` | Golden 4-node ERP verification + SD/fragility preview | `logs/diagnostic/diagnostic.json` |
| `compare_methods.py` | Budget/ε **bands** for threshold / random / degree / spectral (SD + low-freq distortion, optional `--no-mass-match`) | `logs/compare_methods/results.{csv,json}`, `bands.{csv,json}` + `bands_vs_*.png` |
| `run_protocol.py` | D1–D4 findings matrix + frozen-gate interventions (`--graph pc` default; `--graph erp_fraud` for the fraud-like graph) | `logs/protocol/protocol_results.json` (PC) / `logs/erp_fraud_protocol/protocol_results.json` (ERP-fraud) + PNG |
| `frozen_interventions.py` | WS3: expert-architecture (logistic / mlp{w} / dendrite) and per-expert-capacity ablations of the frozen-gate protocol, with a within-regime-permutation null | `logs/frozen_interventions/{pc,erp_fraud}.json` |
| `scalability_reeff.py` | WS2: top-k vs exact effective resistance at n = 100 / 1,000 / 10,000 (edge-error, rank fidelity, induced distortion, timing) | `logs/scalability_reeff/table3.{csv,json}` |
| `run_real_fraud.py` | D1–D4 protocol on a **real public fraud graph** (`--dataset amazon\|tolokers\|yelpchi`) | `logs/{dataset}_fraud_protocol*/protocol_results.json` |
| `run_sparsify_ladder.py` | Retention-ladder on any real graph: ER (effective-resistance) vs budget-matched random/degree distillation, D1–D3 vs retention. `--resume` keeps completed points; `--label-free-strategy/--order-feature` pick the router | `logs/{dataset}_fraud_ladder*/ladder_results.jsonl` + `ladder_summary.json` (see `docs/sparsify_ladder_results.md`, `docs/yelpchi_third_graph.md`) |
| `dilution_curve.py` | Label-free vs oracle gain across the dilution ladder | `logs/dilution/dilution_curve.{csv,json}` + PNG |
| `run_spectral_k_sweep.py` | Sweeps `--spectral-k` for the D2 label-free router on all three real graphs; one shared eigendecomposition per graph | `logs/{dataset}_spectral_k_sweep/sweep.json` |
| `run_k_confirmation.py` | Out-of-fold `k` selection: picks `k` on 9/10 splits, scores the winner on the held-out split, under both self-routing and Protocol B | `logs/{dataset}_k_confirmation/confirmation.json` |
| `run_sd_measurement.py` | Real spectral distortion across the retention ladder, spectra only (no protocol cells). Reports both the legacy index-wise metric and `rank_matched_distortion` | `logs/{dataset}_sd_measurement/sd_results.json` |
| `ablation.py` | Oracle-bucket δ-sweep + router-feature ablation (single / leave-one-out) | `logs/ablation/ablation.{csv,json}` + PNG |
| `reproduce_figures.py` | Renders all Phase 3 figures from cached logs | PNGs under `logs/{protocol,dilution,ablation}/` |
| `run_fixed_expert_protocol.py` | Protocol B: one frozen routing-blind expert pool, 26 routings, logistic/MLP/GNN heads (`--expert`) | `logs/{dataset}_fixed_expert*/fixed_expert.json` |
| `run_positive_control.py` | Positive control: same protocol on a non-separable mixture of linear rules, where routing provably helps; also scores RouterGNN-lite | `logs/positive_control/fixed_expert.json` |
| `run_learnability_lever.py` | Sweeps the planted `separability` (and homophily) lever to locate where routing becomes possible/deployable; probes real pools with the abstaining selective router | `logs/learnability_lever/lever.json` |
| `run_learned_router.py` | RouterGNN-lite vs no-routing / k-means / random / oracle on a frozen pool, real graphs (`--dataset amazon\|tolokers\|yelpchi`) | `logs/{dataset}_learned_router*/fixed_expert.json` |
| `run_published_routers.py` | RouterGNN + AD-GNN vs the same frozen pool/baselines, on fraud and heterophily graphs (`--dataset amazon\|tolokers\|yelpchi\|actor\|chameleon\|squirrel`) | `logs/{dataset}_published_routers/fixed_expert.json` |
| `run_sd_vs_routing.py` | Spectral distortion vs label-free routing stability across a retention ladder on fraud **and** real LLM attention (`--dataset llm`); same-seed multi-router stability, collapse detection | `logs/sd_vs_routing/{dataset}/sd_vs_routing.json` |
| `analyze_sd_vs_routing.py` | Pooled/per-graph Spearman of low-frequency error and connectivity survival vs stability and oracle agreement, plus partial Spearman controlling for isolated-fraction, bootstrap CIs, and leave-one-graph-out pooling | `logs/sd_vs_routing/summary.json`, `logs/sd_vs_routing/partial_correlations.json` |
| `run_llm_protocol_b.py` | Protocol B on real LLM attention: frozen routing-blind k-means pool, only the routing varies with sparsification. `--feature-mode per_layer` (24 features) | `logs/llm_protocol_b/llm_protocol_b.json` |

### Two environment landmines (cost real debugging time)

1. **Import order.** Importing `transformers` *before* `spectral_distillation.src.llm_export`
   hard-segfaults with `0xC0000005` on this machine; the reverse order is stable.
   `run_real_attention.py` only avoids it by importing `transformers` lazily inside
   `main()`. Both new LLM scripts import `llm_export` first, with a comment.
2. **`--top-k` must match.** `run_sd_vs_routing.py` reads `top_k: 32` from
   `configs/default.yaml`; a runner that defaults to `None` keeps the fully dense
   attention matrix and produces a ~8x denser graph (2,759,040 vs 345,216 edges),
   which silently makes SD numbers incomparable. `run_llm_protocol_b.py` now
   defaults to 32 for this reason.

### Feature modes

`build_attention_graph(..., feature_mode=...)`: `agg` (default, legacy) gives 3
columns (mean/max/std of per-layer entropy) and keeps the committed artifacts
reproducible; `per_layer` gives one column per transformer layer and is required
for any downstream accuracy arm. A 29-way task from 3 features is near-chance
regardless of routing.

### 2.3 Tests (`spectral_distillation/tests/`, 257 tests)

| File | Coverage |
|------|----------|
| `test_theoretical_bounds.py` | Weyl, Davis-Kahan, Cauchy interlacing on golden 4-node example |
| `test_distortion.py`, `test_spectral.py`, `test_laplacian.py` | SD metric, spectrum, Laplacian correctness |
| `test_effective_resistance.py` | Exact vs top-k R_eff, Σ w·R_eff = n−1 identity |
| `test_sparsifier.py` | Sparsifiers stay within (1±ε); connectivity invariants |
| `test_spectral_position.py` | Node spectral footprint features |
| `test_baselines.py` | Uniform / random / usage-matched assignment floors |
| `test_protocol.py` | Homophily, assignments, RouterGNN, dilution-ladder invariants, mixture router, statistics, decision rules, frozen intervention |
| `test_real_fraud.py` | Amazon + Tolokers + YelpChi loader schema/labels, adjacency contract, single-component invariant, node-cap and sampling-determinism checks, `--max-nodes` rejection on uncapped datasets, regime-divergence test (density + homophily spread), registry dispatch, oracle buckets, condition shapes |
| `test_sparsify_ladder.py` | Resistance-injection path, exact-budget ER sparsification, resistance-energy direction check, ladder registry + homophily/serialization (6 tests) |

---

## 3. Verified results

All numbers below come from committed outputs in `logs/` (`protocol_results.json`,
`dilution_curve.json`, `ablation.json`) or from the README's Phase 3 findings.
CPU-smoke runs use `n=800`; full-resolution runs target the 10k-node PC-1c-R control.

### 3.1 Phase 1 — theory and diagnostics

- Golden 4-node ERP example matches the guide: `SD(τ=0.5) = 1.0` (Customer isolated),
  `‖ΔL‖_F = 0.8832`, fragility ≈ 2.06, and the identity `Σ w_e·R_eff = n−1` holds exactly.
- ERP attention graphs are ultra-fragile (pendant-dominated), `min_eigengap ≈ 4e-3`; dropping a
  single unique bridge pushes SD to 1.0.
- Caution (from the guide): top-k effective resistance is a lower bound on true R and is only
  accurate near full rank — the code uses exact R for `n ≤ 2000`.

### 3.2 Phase 2 — sparsifier comparison

- On an equal-edge-mass budget/ε grid (120 points) spectral reliably beats **thresholding**
  (74.2% of points) but does not dominate: raw random/degree stay competitive and spectral is
  the single best method at 23.3% of points. Ranking is insensitive to ε. This supersedes the
  earlier point estimate (`docs/robustness_scaling.md`).

### 3.3 Phase 3 — router protocol (D1–D4)

Full run (`n=10000`, 50 patches, 10 splits × 3 seeds, `logs/protocol/protocol_results.json`):

| Comparison | mean diff | Wilcoxon p | d_z |
|------------|-----------|-----------|-----|
| oracle vs uniform | 0.2591 | 0.001953 | 108.3 |
| label-free vs random | 0.2308 | 0.001953 | 101.8 |
| oracle vs random | 0.2663 | 0.001953 | 116.1 |

- Verdict: **D1–D4 all Supported** — *"signal exists, router finds it, not capacity, instrument valid."*
- Frozen-gate intervention (20 perms): learned `0.852` > node-shuffled `0.624`; equal / global-mean
  `0.631`; oracle one-hot `0.885` ≈ upper bound. → the learned router's value is genuine, not an
  artifact of the frozen gate settings.

### 3.4 Phase 3 — dilution ladder (`logs/dilution/dilution_curve.json`)

| dilution λ | oracle gain | label-free gain (mean ± sd over 5 conversion seeds) |
|-----------|-------------|-----------------|
| 0.00 | 0.2591 | 0.2308 ± 0.0000 |
| 0.25 | 0.2591 | 0.1641 ± 0.0010 |
| 0.50 | 0.2591 | 0.0507 ± 0.0017 |
| 0.75 | 0.2591 | 0.0294 ± 0.0018 |
| 1.00 | 0.2591 | 0.0395 ± 0.0246 |

Each rung is averaged over 5 conversion seeds (`--dilution-seeds 0 1 2 3 5`); the wiring is
drawn per node from `(seed, node)` so the rungs are strictly nested and a rung does not depend
on the ladder's composition. Label-free gain falls ~8× (0.231 → 0.029) up to λ=0.75; at full
dilution the mean (0.040) carries large across-seed scatter (sd 0.025, per-seed range 0.012–0.066)
and is not distinguishable from the λ=0.75 value, so it is substantially reduced rather than
collapsing to ~0. This reflects that full homophilization still preserves class-based structure
the label-free router can exploit. Oracle gain (true regime id) stays flat.

### 3.5 Phase 3 — ablations (`logs/ablation/ablation.json`)

Feature sweep (label-free gain vs random, KMeans gate on the listed fields, 10k nodes):

| Variant | Gain |
|---------|------|
| all four features | 0.2318 |
| single: hx | 0.1505 |
| single: spectral_ratio | 0.1137 |
| single: clustering | 0.1050 |
| single: log_degree | 0.1050 |
| loo: clustering | 0.2318 |
| loo: log_degree | 0.2314 |
| loo: spectral_ratio | 0.2314 |
| loo: hx | 0.1051 |

All four fields contribute; `hx` (feature homophily) is the strongest single feature; the
leave-one-out gains sit near the full-set gain (redundancy). Removing `hx` is the only LOO that
hurts materially.

Oracle-bucket δ-sweep (on the homophily-range 3-block SBM, block homophily 0.80 / 0.60 / 0.26):
- Gain peaks mid-range (δ 0.3 → 0.081, δ 0.1 → 0.041, δ 0.5 → 0.061) and collapses to ~0 at the
  extreme widths δ ∈ {0.7, 0.9}; i.e., the protocol result is *not* a knife-edge artifact of the
  chosen homophily thresholds.

### 3.6 Known caveats

- The dilution ladder runs at the protocol resolution (10 splits × 3 seeds) with each rung
  averaged over 5 conversion seeds; the ablation sweep still uses 4 splits × 2 seeds (config
  default), so its rep-wise p-values saturate at the Wilcoxon resolution floor and its point
  estimates are the noisier of the two.
- The `uniform` arm is a degenerate gate (all nodes to one expert) in the current ladder — by
  construction `uniform_gates = (1/3, 1/3, 1/3)` is only realized for the oracle/conditional
  comparisons; see `configs/router_protocol.yaml` and the `test_protocol.py` uniform-baseline tests.
- GPU/CPU differences on the RouterGNN can shift epoch-level behavior; the crisp KMeans gate is
  deterministic and is the operational gate.

### 3.7 Phase B — learnability lever (`logs/learnability_lever/lever.json`)

A planted `separability` knob interpolates the label function from "one rule fits
every group" (routing worth nothing) to the positive control (routing worth a
lot), on a block-diagonal graph with feature-separated groups
(`src/learnability_lever.py`, `experiments/run_learnability_lever.py`,
`docs/learnability_lever.md`). n=3000, 5 splits, logistic experts:

- routing gain **exists** (label-informed oracle beats global) from **β ≈ 0.2**;
- routing is **deployable** (a held-out selective router decides to route and
  wins) from **β ≈ 0.3**, and monotone/robust above **β ≈ 0.5** (the 0.3–0.4 band
  is noisy: route rates 4/5 then 1/5);
- on the real pools the selective router **abstains** — held-out gain +0.0007
  (Amazon, n=8639) and −0.0017 (Tolokers, n=11758) — matching the Protocol-B null
  as a *decision*, not a failure to find a win.

The homophily knob (`cross_edge_p`) does not reduce routing gain in this
construction: the signal is carried by the label/feature structure, not the
edges. The lever is synthetic and locates the boundary; it does not place any
real graph at a specific β.

---

## 4. How to reproduce everything

```bash
pip install -e ".[dev]"          # or: make dev-install

python -m pytest                 # 257 tests
python -m spectral_distillation.experiments.run_protocol        # D1-D4 + frozen gates
python -m spectral_distillation.experiments.dilution_curve      # dilution ladder
python -m spectral_distillation.experiments.ablation            # oracle δ + feature ablation
python -m spectral_distillation.experiments.reproduce_figures   # render PNGs from logs
```

CPU smoke run that reproduces the numbers in §3.3–3.5 (~minutes):

```bash
python -m spectral_distillation.experiments.run_protocol --n 800 --n-patches 5 --splits 8 --seeds 2 --router-epochs 50 --frozen-perms 2
python -m spectral_distillation.experiments.dilution_curve --n 800
python -m spectral_distillation.experiments.ablation --n 800
python -m spectral_distillation.experiments.reproduce_figures
```

### 4.1 Config knobs

- `configs/default.yaml` — global defaults: seed, device (`cuda`/`cpu`), sparsification ε and
  budget, oracle filter δ and score type, protocol thresholds, PC-1c-R shape, GNN router, LLM
  export model id, experiment sweeps.
- `configs/router_protocol.yaml` — the four router conditions, oracle bucket cutoffs
  (`h_low=0.4`, `h_high=0.6`), test configuration (10 splits × 3 seeds, Wilcoxon exact + paired-t,
  Holm), D1–D4 decision rules, frozen-gate settings.
- `configs/planted_control.yaml` — the PC-1c-R positive control: 10k nodes, 5 classes, 50 patches,
  regime ratio (0.4 / 0.4 / 0.2), dilution ladder `[0, 0.25, 0.5, 0.75, 1.0]`, and the four
  deconfounding invariants (feature-noise constant, nested converted sets, unconverted retain
  wiring, **no arm saturates** ≈ single-M accuracy ~0.80).
- Runners accept `--n`, `--n-patches`, `--splits`, `--seeds`, `--router-epochs`, `--device`,
  `--config`, `--out` (see each module's argparse help). `--device` defaults to `cpu` —
  the **committed logs are the CPU reference**, and CUDA and CPU are not bit-identical, so
  reproduce with the default before comparing numbers. Pass `--device cuda` (or
  `--device cuda` on a machine with CUDA) for a faster GPU run; it auto-falls back to
  `cpu` when CUDA is unavailable.

**Provenance.** Every committed log under `logs/` records the device it was produced on —
either at the JSON root (`"device": "cpu"`) or inside its embedded run `config`
(`config.device`). The `sd_vs_routing` derived views (`*/points.json`, `summary.json`) also
carry it at the root, so "CPU reference vs CUDA run" is legible from the artifact alone,
without reading the code.

#### 4.2 Measured runtime (RTX 4060 Laptop GPU, n=10,000)

Measured on the local dev machine (RTX 4060 Laptop, Python 3.12, torch 2.6.0+cu124),
default configs, `--device cuda`:

| Experiment | Runtime |
|------------|---------|
| `run_protocol` (n=10000, 10 splits × 3 seeds, 200 router epochs, 20 frozen perms) | **2.5 min** |
| `dilution_curve` (n=10000, 5 ladder rungs × 5 conversion seeds) | **~70 min** |
| `ablation` feature sweep (n=10000, 9 variants) | **35 s** |
| `ablation` oracle δ-sweep (n=10000, 6 deltas) | **10 s** |
| `reproduce_figures` | **< 1 s** |
| **Full pipeline** (`run_protocol`+`dilution`+`ablation×2`+`figures`) | **~75 min** |

Two optimizations made this possible (commit in this session):

- **GPU eigendecomposition.** `np.linalg.eigh` on the 10000×10000 Laplacian is
  ~102 s on CPU; `torch.linalg.eigh` (fp64, CUDA) is ~32 s with bit-identical
  low eigenvalues (max abs diff 2e-14). New `laplacian.eigh_symmetric(A, device)`
  transparently picks CUDA above n=1000 and falls back to numpy/CPU otherwise.
- **Single eigh per feature sweep.** The ablation previously re-ran the full
  eigendecomposition for each of the 9 feature variants (9 × 102 s CPU);
  `_feature_cache` computes the four structural fields once and reuses them.

### 4.3 Reproducibility checklist (guide §9.3) — enforced

| Requirement | How enforced in code |
|-------------|----------------------|
| Fixed seeds | `utils.set_seed` at every entry point (numpy, torch, random) |
| Config-driven | Every run records the exact YAML + graph shape + flags in `protocol_results.json` |
| Paired statistics | splits × seeds; seeds averaged within split; Wilcoxon exact + paired-t + Holm |
| Theoretical bounds hold | `test_theoretical_bounds.py` asserts Weyl / Davis-Kahan / Cauchy on the 4-node example |
| Protocol identity | `test_protocol.py` asserts usage-matched random and uniform baselines |
| A/A negative control | random/self comparisons in the findings matrix |
| Device provenance | every committed log records `device` (root or `config.device`) |

---

## 5. Confirm-with-source summary

Every claim in §3 maps to a committed artifact:

| Claim | Artifact |
|-------|----------|
| SD = 1.0, ‖ΔL‖_F = 0.8832, fragility 2.06, Σ w·R_eff = n−1 | `logs/diagnostic/diagnostic.json`, `test_theoretical_bounds.py` |
| Equal-edge-mass budget/epsilon bands: spectral beats threshold (74.2%) but no method dominates | `logs/compare_methods/bands.json`, `docs/robustness_scaling.md` |
| Frozen-gate ablations: ordering holds across logistic / mlp{w} / dendrite at n=2000 (within-regime null near-identity) | `logs/frozen_interventions/{pc,erp_fraud}.json`, `docs/robustness_scaling.md` Part 3 |
| Top-k R_eff: rank-faithful (ρ→0.91) though absolute edge error high; induced distortion ≤1.2% of exact | `logs/scalability_reeff/table3.json`, `docs/robustness_scaling.md` Part 2 |
| D1–D4 supported, frozen-gate ordering, stats | `logs/protocol/protocol_results.json` |
| Dilution decay (0.231 → 0.029 at λ=0.75; 0.040 ± 0.025 at λ=1.0) vs flat oracle (0.259), 10k, 10 splits × 3 seeds × 5 conversion seeds | `logs/dilution/dilution_curve.json` |
| Feature ablation ordering + oracle δ plateau | `logs/ablation/ablation.json` |
| 255-test pass | `python -m pytest` |

---

## 6. Next steps (what you need to do)

1. **Full-resolution run (DONE locally, §4.2).** The default 10k-pipeline was executed and
   verified on the RTX 4060 at ~9 min total; results are committed as logs + figures. If you
   re-run elsewhere (Kaggle / Colab), archive the JSON/PNG outputs:
   `make full-pipeline` (or `make protocol dilution ablation figures`).
2. **Raise statistical resolution.** For the paper, `--splits 10 --seeds 3` (defaults) gives
   Wilcoxon p-values of 0.001953 (resolution floor already reached); bump to 12–15 splits only if
   you want tighter CIs.
3. **Final consistency check.** Re-run `python -m pytest` and `reproduce_figures` against the
   final logs so README's stated numbers still match the committed artifacts.
4. **Version control.** The repo is a git repository with verified-phase commits. Keep each new
   verified run (results JSON + docs + tests) as its own commit:
   ```bash
   git add <changed files>
   git commit -m "feat: <what the run/writeup adds>"
   ```
5. **Read the Phase-3 summary in README** ("Phase 3 findings") — it is kept in sync with this doc;
   if you change numbers up-stream, update both.

---

## 7. Further implementation phases

### Phase 4 — Real ERP data + LLM attention export

Goal: move off the synthetic ERP generator onto real attention matrices.

- **DONE (fraud-like synthetic): `generate_erp_fraud_graph` in `src/synthetic.py`** — a scalable,
  vectorized stochastic-block ERP-fraud graph (suppliers/invoices/backbone, shell-fraud hubs,
  weak customers) returning the protocol contract (`W`, `features`, `y`, `regimes`). Run via
  `run_protocol.py --graph erp_fraud`; all four D1–D4 rules hold and the label-free router
  recovers the regime structure exactly (see `docs/erp_fraud_results.md`).
- **DONE (real-attention): LLM attention export** on `HuggingFaceTB/SmolLM2-1.7B` over a
  public-domain corpus; D1–D4 all hold at top-k ≥ 32 (D2 verified against a random
  baseline matched to the router's own usage; the only failure is D1 at top-k=16; see
  `docs/real_attention_results.md`).
- **DONE (real public graph): DGL Amazon co-review fraud** — real benign/fraud labels,
  D1/D3/D4 hold, D2 does not (structural signal saturates on the strongly-homophilic
  co-review relations; see `docs/real_fraud_results.md`). Data lives under git-ignored
  `data/benchmarks/amazon/raw/Amazon.mat`; the loader is `src/real_fraud.py`.
- **DONE (second real graph): Tolokers crowd-worker exclusion** — 11,758 nodes, 519k edges,
  21.8% banned, genuinely heterophilous (edge homophily 0.595 vs 0.659 baseline). Only D2 fails;
  D1/D3 are **13x larger than Amazon** (dz 14.94), confirming oracle gain scales with node-level
  homophily variance (sd 0.241 vs 0.063). See `docs/sparsify_ladder_results.md`.
- **DONE (cross-dataset retention ladder):** on Amazon, ER spectral distillation *amplified* the
  oracle signal ~7x at 6-8% retention; **this does not replicate on Tolokers**, where ER only
  *preserves* it (~84% of full-graph gain at r=0.30). Replication across both datasets: degree
  pruning destroys the oracle channel, and ER >= random > degree at matched budget.
- ~~**NEW lead (D2):** on Tolokers, degree pruning *significantly improves* D2 (+0.0055 at
  r=0.08, p at floor)~~ — **superseded.** The lead does not replicate: degree pruning leaves D2
  near zero on YelpChi (+0.0001..+0.0013) and Amazon. Do not treat "degree pruning unlocks D2"
   as a general mechanism. What *is* supported is the D2 **router** fix
   (`--label-free-strategy proxy_quantile --order-feature eig_nb_sim`, `docs/d2_router_fix.md`),
   which converts D2 from fail to pass on **Tolokers only**.
- **DONE (third real graph): YelpChi review fraud** — 14,840 nodes (capped sample, see
  `docs/yelpchi_third_graph.md`), 411k edges, 15.0% spam, 32 features. Full protocol **passes
  D1-D4** under the legacy router (D2 +0.00575, dz 4.33) and under the fix (+0.00481, dz 6.73,
  overlapping CIs). Establishes the 3-dataset generality claim. It does **not** add a second
  fix success: the label-free signal is present either way, and `eig_nb_sim` correlates
  -0.26 with label homophily here while still delivering — so the homophily-proxy mechanism is
  an open question, not a settled one.
- **Outstanding:** D2 on dense/homophilic real graphs (no method unlocks it on Amazon, whose
  best candidate reaches ~23% of its oracle ceiling). Two items previously listed here are now
  resolved — see "Spectral-k sweep" and "Real spectral distortion" below.

### Resolved: spectral-k sweep

Every committed D2 result used `--spectral-k 8`, which was never tuned. `experiments/run_spectral_k_sweep.py`
sweeps k over `{2,4,8,16,32,64,128}` on all three graphs (10 splits x 3 seeds). The Laplacian
eigendecomposition is k-independent, so it is computed once and reused; artifacts are in
`logs/{amazon,tolokers,yelpchi}_spectral_k_sweep/sweep.json`.

**k=8 is optimal on none of the three graphs.**

| k | Amazon D2 | Tolokers D2 | YelpChi D2 |
| --- | --- | --- | --- |
| 2 | +0.00063 (p=0.049) | +0.00263 | +0.00432 |
| 4 | +0.00018 | +0.00484 | +0.00419 |
| **8 (default)** | +0.00048 | +0.00404 | +0.00481 |
| 16 | +0.00022 | +0.00526 | +0.00353 |
| 32 | +0.00031 | +0.00432 | +0.00407 |
| 64 | -0.00020 | +0.00231 | +0.00497 |
| 128 | +0.00005 | **+0.00863** | **+0.00649** |

- **Tolokers and YelpChi both prefer k=128**, roughly doubling the D2 gain (Tolokers
  +0.00404 -> +0.00863, dz 7.82; YelpChi +0.00481 -> +0.00649, dz 6.64). Both stay at the
  p floor of 0.001953.
- **Amazon stays null at every k.** Best is +0.00063 at k=2 (p=0.049 raw); across 7 k values a
  Holm correction removes it entirely. Amazon's D2 failure is therefore **not** a bandwidth or
  mode-count problem, which is a negative result worth stating: tuning `k` cannot fix Amazon.
- **The label-homophily proxy is anti-correlated with delivered D2 on YelpChi.** Spearman rho
  against true label homophily moves from -0.249 at k=2 to -0.288 at k=128, i.e. the *more*
  anti-correlated the score, the *better* the routing outcome. On Amazon rho rises with k
  (+0.346 at k=128) while D2 stays flat at zero. Selecting k by maximising rho would pick the
  worst k on YelpChi. The proxy must not be used to tune `k`.
- **Confirmed out-of-fold (no longer reporting-selected).** `experiments/run_k_confirmation.py`
  picks `k` on nine of ten splits and scores the winner on the held-out split
  (`logs/{amazon,tolokers,yelpchi}_k_confirmation/confirmation.json`). The selector recovers
  `k=128` on **10/10** folds for both Tolokers and YelpChi under self-routing (held-out D2
  +0.00863 and +0.00650, matching the in-sample numbers exactly) and on **10/10** YelpChi folds
  even under the honest fixed-expert protocol. The *choice* of `k=128` is therefore stable, not a
  selection artifact. On Amazon the out-of-fold winner is `k=2` (10/10, +0.00057) and `k=128` is
  null, as in-sample — the k-sweep never claimed a 128 win there.
- **But the win is still a protocol artifact.** A stable winner is not a win: under the honest
  fixed-expert protocol scored against a *genuine* no-routing global model, every `k` has
  `D2_vs_no_routing < 0` on all three graphs (YelpChi k=128: +0.00159 vs random but −0.00519 vs
  no-routing; Amazon best k=4: +0.00966 vs random but −0.04197 vs no-routing). The no-routing
  global itself scores +0.05163 / +0.00024 / +0.00678 over random on Amazon / Tolokers / YelpChi,
  so "routing beats random" is satisfied by routing nothing. The corrected caveat therefore
  *strengthens* the null: `k=128` is a stable choice whose gain does not survive an honest
  reference.

### Resolved: real spectral distortion

All committed ladders were produced with `--skip-sd`, so every `sd_laplacian` field is the
placeholder `0.0`. `experiments/run_sd_measurement.py` measures the spectra directly (no protocol
cells), computing the reference spectrum once so each point costs one `eigvalsh` instead of two.

**The legacy SD metric is not usable on these graphs.** `compute_spectral_distortion` pairs
eigenvalue *i* of the reference with eigenvalue *i* of the sparsified graph. That pairing is only
meaningful when the component count is unchanged — and sparsification changes it at every
interesting point. Concretely, degree pruning of YelpChi at r=0.04 leaves **14,460 of 14,840
nodes isolated (97.4%)**, giving 14,462 zero eigenvalues against 1 in the reference. After
sorting, index *i* refers to an unrelated mode in each graph.

The consequence is visible in the artifact: the legacy scalar returns **exactly 1.0000 at 13 of
15 Amazon points and 15 of 15 YelpChi points** — no information at all. The only points where it
does not saturate are exactly the two Amazon points with `n_components == 1` (random r=0.60 ->
0.3394, r=0.30 ->0.5771), which is the diagnosis confirming itself.

`rank_matched_distortion` (in `src/distortion.py`) compares the *largest* eigenvalues instead.
These are the modes a low-pass filter actually retains, they stay ordered under fragmentation,
and they do not depend on component count. It separates the methods cleanly (Amazon r=0.15):

| method | isolated | components | legacy SD | lambda_max rel err | top-8 |
| --- | --- | --- | --- | --- | --- |
| ER | 1.3% | 116 | 1.0000 | **0.0440** | 0.1214 |
| degree | 71.8% | 6201 | 1.0000 | **0.5654** | 0.5907 |

So **quote `rank_matched_*`, not the legacy `SD`**, and treat every previously committed
`sd_laplacian` as a placeholder that was never a measurement.

### Bug found and fixed: pyarrow/torch DLL load order

The Tolokers loader (`pd.read_parquet` -> `pyarrow.dataset`) crashed the interpreter with a
Windows access violation (0xC0000005, exit code -1073741819) and **no traceback**, whenever
torch had already imported its OpenMP/MKL DLLs. `read_parquet` imports `pyarrow.dataset` lazily,
so import order decided whether the loader ran at all. Fixed by preloading it in
`spectral_distillation/src/__init__.py`, the earliest hook in the package. Regression tests are in
`tests/test_real_fraud.py`.

- **DONE (config-sized ERP-fraud run):** `run_protocol.py --graph erp_fraud
  --config configs/erp_fraud.yaml` at n=10,000 (10×3) gives D1/D2/D3 = +0.2054 /
  +0.2118 / +0.2118, D4 frozen gate 0.8806 vs shuffled 0.6657±0.0025 (all
  Supported); the n=3,000 run is retained for scale comparison. See
  `docs/erp_fraud_results.md`, `logs/erp_fraud_protocol/`. There is no real ERP
  dataset in the repo, so "real labels" are covered by the fraud benchmarks
  (homophily-bucket oracle, already implemented — same path used for benchmarks).

### Phase 5 — Publication materials

- `docs/method_section.md` (drafted): expand Parts 5–8 of the guide into the
  method with the verified numbers from §3 embedded.
- `docs/theorem_proofs.md` (drafted): full proof chain for
  Theorem 1 (routing-signal decay, `I(π*;Y) − I(π̃;Y) ≤ C·SD·H(Y)`), Theorem 2 ((1−2ε)^k
  routing-error decay), and Corollary 3 (edge-removal monotonicity, Lean-formalization sketch in
  guide Appendix B).
- `docs/literature_positioning.md` (drafted): related work and the honest differences vs MoG / GOKU / MORGAN.
- Target the Figures/Tables spec (guide §10.2): Fig 3–6 and Tables 1–3 come from
  `compare_methods.py`, `scalability_reeff.py`, `run_protocol.py`, `frozen_interventions.py`, `dilution_curve.py`, `ablation.py` outputs.
- Done: `reproduce_figures.py` exports Tables 1–3 as CSV + LaTeX next to the logs
  (`--tables`, on by default; `--no-tables` to skip).

### Phase 6 — Robustness and scaling

- **Done (WS1):** budget/ε sensitivity bands on an equal-edge-mass basis around Table 2
  (`compare_methods.py`, `docs/robustness_scaling.md`): spectral beats threshold at 74.2% of
  120 grid points but does not dominate — random/degree stay competitive; ranking is ε-flat.
- **Done (WS2):** top-k R_eff accuracy study vs n (guide §9.4 note on lower bounds) — `scalability_reeff.py`: absolute edge error stays high but rank fidelity (ρ→0.91) keeps induced distortion within 0.26–1.2% of exact; exact skipped at n=10,000. See `docs/robustness_scaling.md` Part 2.
- **Done (WS3):** extended frozen interventions — logistic / mlp{w} / dendrite experts and per-expert capacity, plus a within-regime-permutation null; the ordering holds for all 12 architecture/graph configs at n = 2000 (`frozen_interventions.py`, `docs/robustness_scaling.md` Part 3).

---

**Status:** living document — update §3 and §6 whenever a new verified run lands.
