# Real-Attention Router Verification — Results

Model: `HuggingFaceTB/SmolLM2-1.7B` (24 layers, 32 heads) · Torch fp16 · RTX 4060 Laptop
Corpus: public-domain "The Time Machine" (Gutenberg), scaled to 24 passages (~11.5k tokens)
Graph: mean-pooled attention over tokens, per-row top-k, block-diagonal across passages
Node features: attention entropy · Labels: real sentence membership from tokenizer offsets

Protocol identical to the synthetic PC-1c-R D1–D4 pipeline (see `docs/does-your-router-actually-route 4.pdf`).

## 1. Headline result

| Rule | Meaning | k=32 | verdict |
|---|---|---|---|
| **D1** | signal exists (oracle > uniform) | +0.0063, d_z 3.50 | pass |
| **D2** | router finds it (label-free > random) | +0.0138, d_z 6.33 | pass |
| **D3** | not capacity (oracle > random) | +0.0208, d_z 11.32 | pass |
| **D4** | instrument valid (positive control) | +0.2534, p=0.031 | pass |

**Verdict: Supported — real LLM attention encodes a sentence-routing signal that a label-free
structural router recovers from attention-entropy features and graph structure alone.**

All primary p-values = 0.001953 (exact Wilcoxon, 10×3 protocol). Router usage at k=32:
oracle [2046, 3840, 5634], label-free [88, 11408, 24], random [2081, 3713, 5726], uniform [11520, 0, 0].

## 2. Robustness sweep

All runs on the same 11,520-node, 24-passage graph (10×3 protocol, dilution ladder skipped except k=32 seed 0).

| Config | D1 mean (sig) | D2 mean (sig) | D3 mean (sig) | Verdict |
|---|---|---|---|---|
| top-k=32, seed 0 | +0.0063 ✓ | +0.0138 ✓ | +0.0208 ✓ | **Supported** |
| top-k=32, seed 7 | +0.0063 ✓ | +0.0140 ✓ | +0.0210 ✓ | **Supported** (seed-stable) |
| top-k=64 | +0.0128 ✓ | +0.0072 ✓ | +0.0207 ✓ | **Supported** |
| top-k=16 (fixed 0.4/0.6) | **−0.0046 ✗** | +0.0132 ✓ | +0.0093 ✓ | **Not supported** (D1 only) |
| top-k=16 (adaptive tercile) | **−0.0030 ✗** | +0.0160 ✓ | +0.0137 ✓ | **Not supported** (D1 only) |

**Interpretation.** The label-free router (D2) and the upper-bound signal (D3) hold at *every*
configuration. The single failure is D1 at top-k=16: the oracle is worse than uniform but still
better than random under both fixed and adaptive thresholds. This failure is a genuine property
of the sparse oracle instrument, not a threshold-calibration artifact.

**Why it happens.** top-k changes the homophily distribution the oracle reads. At k=16 the
distribution tightens around h ≈ 0.5 (mean 0.52, only 26% of nodes below 0.4) so the three
homophily buckets no longer separate sentences; at k=32/64 it spans both fixed thresholds and
separates cleanly. Switching to tercile-based adaptive thresholds keeps all buckets populated
([3875, 3764, 3881]) but the oracle still cannot beat uniform.

## 3. Dilution ladder (k=32, seed 0)

| dilution | oracle_gain | label_free_gain |
|---|---|---|
| 0.00 | 0.0071 | 0.0135 |
| 0.25 | 0.0191 | 0.0524 |
| 0.50 | 0.0108 | 0.0578 |
| 0.75 | 0.0169 | 0.0481 |
| 1.00 | 0.0000 | 0.0852 |

**Anomaly resolved.** The label-free gain that *rises* toward dilution→1 looks like a violation of
the synthetic pattern (0.231 → 0.008). Diagnostics show it is an artifact of the ladder itself:
the dilution rewrite forces same-class edges (label homophily → 1.0), which plants degree/spectral
structure the structural router reads mechanically — independent of real attention. Concretely the
router's `feature_homophily` field stays ≈ 0.996 flat across all dilutions (L2 drift grows, but the
first feature column does not move), and at dilution=1 the label-free assignment tracks real
sentence classes (NMI ≈ 0.28) while oracle/random collapse to NMI ≈ 0. So the surviving gain is a
property of the planted re-wiring, not of real routing signal. The synthetic dilution decay is only
reproduced when class sizes are balanced; the real graph's long-tail sentence sizes expose this.

**Honest statement:** the dilution ladder is a valid instrument on balanced planted graphs but is
not a valid *erasure* test for real, imbalanced attention graphs. Robustness conclusions should
carry this caveat rather than the raw dilution curve.

## 4. Engineering notes

- Crash fix: TensorFlow 2.21 (pulled by `transformers` image transforms) clashed with torch on
  Windows (0xC0000005); removed tensorflow / tf_keras / keras.
- Sparse-graph fix: attention graphs need per-row top-k (default 32); without it the graph is
  ~99.8% dense and the oracle degenerates to one bucket.
- Sentence labels: real punctuation/newline boundaries from tokenizer offsets (regex split), not a
  character-gap heuristic.
- New oracle knob: `oracle_bucket_assignment(..., adaptive=True)` uses homophily terciles; exposed
  via `--adaptive-oracle`.
- New CLI conveniences: `--skip-dilution` (verdict-only, ~5 min vs ~49 min), `--no-progress`;
  per-cell/per-rung tqdm progress with ETA added to `run_protocol`.
- Runtime profile at k=32 full: model load 6 s, export 18 s, assignments 148 s (Laplacian eig +
  KMeans + RouterGNN), main protocol ~20-25 min (120 logistic-refit cells), dilution ladder ~35 min.
- Tests: 92 pass (`python -m pytest`).

## 5. Density-aware oracle sweep (step 1 follow-up; closes the k=16 question)

Goal: find an oracle calibration/extraction that beats uniform at top-k=16 and beat the fixed
thresholds at every k. All variants evaluated on the same k=16 graph with the 6×2 protocol
(primary sign/p/z held under the full 10×3 where noted).

**Result: every *honest* homophily-based oracle fails D1 at k=16.**

| Oracle variant (k=16) | usage hi/id/lo | D1 (o>u) | d_z | verdict |
|---|---|---|---|---|
| fixed 0.4/0.6 (repro) | [4697,3720,3103] | −0.0045 | −1.17 | ✗ |
| adaptive tercile (repro) | [3875,3764,3881] | −0.0029 | −0.94 | ✗ |
| Bayesian shrink, c=5 | [3927,3706,3887] | −0.0012 | −0.45 | ✗ |
| Bayesian shrink, c=20 | [3940,3675,3905] | −0.0009 | −0.27 | ✗ |
| confidence-gated (z=1) | [2062,7822,1636] | −0.0039 | −1.42 | ✗ |
| degree-winsorized (top 15% → identity) | [2350,8091,1079] | −0.0019 | −0.76 | ✗ |
| attention-weighted homophily, tercile | [3840,3840,3840] | −0.0038 | −0.98 | ✗ |
| attention-weighted homophily, shrink c=5 | [3840,3840,3840] | −0.0027 | −0.94 | ✗ |
| budget-normalized homophily, tercile | [4442,2749,4329] | −0.0081 | −2.84 | ✗ |
| budget-normalized homophily, 0.5/0.8 | [5328,3629,2563] | −0.0049 | −1.20 | ✗ |

Two apparent wins turned out to be artifacts, and their dissection is instructive:

- **top-1 split** (route on whether the *single strongest* attention edge is same-sentence) gave
  D1 = +0.0158 at both k=16 and k=32 (p=0.002, d_z≈18) — but its expert-0 "cluster" is exactly
  **one sentence** (376 nodes, single label), so the routing reduces to a trivial majority-label
  classifier scoring 100% on its own sentence while contributing nothing to structure. It would be
  dishonest to report this as signal. The k-invariant result also revealed that at both k the
  strongest edge is the *same* edge, so top-1 does not actually exercise density.

**The one legitimate win: budget × tercile.** A clean 2×2 at k=32 (full 10×3 protocol) isolates
which factor drives the oracle:

| k=32 oracle | fixed 0.4/0.6 | tercile |
|---|---|---|
| raw homophily | D1 +0.0063 · D3 +0.0208 | D1 +0.0062 · D3 +0.0215 |
| budget-normalized | D1 +0.0039 · D3 +0.0189 | **D1 +0.0100 · D3 +0.0244** |

Budget normalization alone *hurts* (it pushes most nodes into one bucket: usage [5915,3443,2162]
vs [2046,3840,5634]). The improvement only appears *combined with tercile thresholds*, which force
balanced usage — then budget-normalized homophily correctly rewards a short sentence that uses its
full in-sentence attention budget. **budget × tercile is the strongest oracle across all
configurations tested** (D1 d_z +2.96, D3 +0.0244). Exposed via `--budget-oracle`; at k≥32 this is
the recommended oracle setting.

**Conclusion.** The k=16 D1 failure is a property of the sparse oracle *instrument*, now verified
under ten distinct extractions/shiftings/gates. At k=16 the same-sentence signal is beneath the
neighbor-averaging noise floor (only ~1 in 16 edges is a reliable same-sentence edge), so no
calibration of a label-homophily oracle recovers it — while the *label-free* router (D2) stays
significant there (+0.0132..+0.0160). We keep the honest framing: **the router's signal survives
sparsity; the homophily oracle does not.** At k≥32, budget × tercile is the recommended oracle.

## 6. Bottom line

> The primary result is robust: real attention carries a label-free-detectable sentence-routing
> signal (D2, D3 pass at every density and seed; verdict Supported for top-k ≥ 32). The only
> caveat is D1 at top-k=16, which fails because the *oracle* instrument degenerates under sparse
> neighborhoods — the label-free router itself remains significant there.

## 7. Artifacts

| Run | JSON |
|---|---|
| k=32 baseline (full, with dilution) | `logs/real_attention/real_attention_HuggingFaceTB--SmolLM2-1.7B.json` (backup: `..._topk32.json`) |
| k=32 seed 7 | `logs/rk32_s7/real_attention/...` |
| k=64 | `logs/rk64/real_attention/...` |
| k=16 fixed thresholds | `logs/rk16/real_attention/...` |
| k=16 adaptive terciles | `logs/rk16_adapt/real_attention/...` |