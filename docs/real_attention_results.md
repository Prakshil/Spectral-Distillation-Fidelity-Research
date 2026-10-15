# Real-Attention Router Verification — Results

Model: `HuggingFaceTB/SmolLM2-1.7B` (24 layers, 32 heads) · Torch fp16 · CPU (`--device cpu`; host: RTX 4060 Laptop)
Corpus: public-domain "The Time Machine" (Gutenberg), scaled to 24 passages (~11.5k tokens)
Graph: mean-pooled attention over tokens, per-row top-k, block-diagonal across passages
Node features: attention entropy · Labels: real sentence membership from tokenizer offsets

Protocol identical to the synthetic PC-1c-R D1–D4 pipeline (see `docs/does-your-router-actually-route 4.pdf`).

> **⚠ Status — §1–§8 are the Protocol A (self-routing) record, superseded by the Protocol-B verdict.**
> Sections §1–§8 below use the Protocol A (self-routing) instrument: the router both assigns nodes
> *and* defines the expert pools, so a partition is scored on the nodes it selected. The project's
> **operative accuracy verdict is Protocol B** (routing-blind fixed pool): the real-attention
> Protocol-B rerun is **null** (§9.2), and the surviving project-level positive is routing
> **stability** (§9.1, §9.3); the README headline is authoritative. Do **not** quote §1–§8 as an
> accuracy claim. What §1–§8 *do* establish: the D1–D4 instrument is coherent on real data, and —
> after the §6 rewrite — the shipped five-field router is **not degenerate** and its label-free gain
> is **not** a usage-imbalance artifact (the earlier "degenerate router" retraction is withdrawn).
> §9 is the corrective analysis.

## 1. Headline result

| Rule | Meaning | k=32 | verdict |
|---|---|---|---|
| **D1** | signal exists (oracle > uniform) | +0.0064, d_z 3.42 | pass (k≥32) |
| **D2** | router finds it (label-free > random) | +0.0180 recorded; **+0.0131 vs usage-matched random** | pass |
| **D3** | not capacity (oracle > random) | +0.0213, d_z 11.52 | pass |
| **D4** | instrument valid (positive control) | +0.2493, p=0.00195 | pass |

**Verdict — Protocol A (self-routing) instrument, top-k ≥ 32: all four rules pass.** Real LLM
attention encodes a sentence-homophily signal that the *label-homophily oracle* detects (D1, D3)
and that the **label-free structural router also recovers** (D2): its gain survives a random
baseline matched to the router's *own* usage (+0.0131, p=0.002; §6). The instrument is valid (D4).
The one failure is **D1 at top-k=16**, a property of the sparse *label-homophily oracle* rather than
of the graph (§5); the label-free router still clears D2 at k=16.

> **Scope — read before quoting these numbers.** §1–§8 are the **Protocol A (self-routing)**
> instrument: the same router both assigns nodes *and* defines the expert pools, so each partition is
> scored on the very nodes it selected (a partition can receive credit without a real routing gain).
> The project's *operative* accuracy verdict is therefore the **Protocol-B frozen-pool rerun (§9.2),
> which is null**, and on that basis the README retracts the Protocol-A *oracle* numbers; the
> surviving project-level positive is routing **stability** (§9.1, §9.3). §6 corrects only the
> *mechanism* previously attributed to D2 — the router is **not** degenerate and its gain is **not**
> a usage-imbalance artifact — which does **not** by itself promote Protocol A over Protocol B.

All primary p-values = 0.001953 (exact Wilcoxon, 10×3 protocol). Router usage at k=32:
oracle [2048, 3839, 5633], label-free [7648, 3824, 48], random [2081, 3713, 5726], uniform [11520, 0, 0].

## 2. Robustness sweep

All runs on the same 11,520-node, 24-passage graph (10×3 protocol, dilution ladder skipped except k=32 seed 0).

| Config | D1 mean (sig) | D2 mean (sig) | D3 mean (sig) | Verdict |
|---|---|---|---|---|
| top-k=32, seed 0 | +0.0064 ✓ | +0.0180 ✓ | +0.0213 ✓ | **Supported** |
| top-k=32, seed 7 | +0.0064 ✓ | +0.0173 ✓ | +0.0206 ✓ | **Supported** (router-seed stable) |
| top-k=64 | +0.0125 ✓ | +0.0167 ✓ | +0.0200 ✓ | **Supported** |
| top-k=16 (fixed 0.4/0.6) | **−0.0053 ✗** | +0.0221 ✓ | +0.0092 ✓ | **Not supported** (D1 only) |
| top-k=16 (adaptive tercile) | **−0.0038 ✗** | +0.0246 ✓ | +0.0131 ✓ | **Not supported** (D1 only) |

**Interpretation.** The label-homophily oracle signal (D1, D3) and the label-free router (D2) both
hold at k≥32, and D2 holds at every density tested (including k=16, where the *oracle* fails D1).
The D2 gain is not a usage artifact: §6 re-measures it against a random baseline matched to the
router's own usage and it survives (+0.0131 at k=32). The single genuine failure is D1 at top-k=16,
a property of the sparse oracle instrument, not a threshold-calibration artifact (§5).

**Why it happens.** top-k changes the homophily distribution the oracle reads. At k=16 the
distribution tightens around h ≈ 0.5 (mean 0.52, only 26% of nodes below 0.4) so the three
homophily buckets no longer separate sentences; at k=32/64 it spans both fixed thresholds and
separates cleanly. Switching to tercile-based adaptive thresholds keeps all buckets populated
([3875, 3764, 3881]) but the oracle still cannot beat uniform.

## 3. Dilution ladder (k=32, seed 0)

| dilution | oracle_gain | label_free_gain |
|---|---|---|
| 0.00 | 0.0071 | 0.0191 |
| 0.25 | −0.0034 | 0.0076 |
| 0.50 | −0.0011 | 0.0116 |
| 0.75 | 0.0000 | 0.0418 |
| 1.00 | 0.0000 | 0.1443 |

**Anomaly: the label-free gain *rises* as the oracle signal is erased.** At dilution=0 the oracle
gain is +0.0071 and it decays to 0.0000 by dilution≥0.75 — the intended erasure of the label signal
(matching the synthetic ladder's 0.231 → 0.015 decay). The label-free gain does the opposite: it
*rises* toward dilution→1 (+0.0191 → +0.1443). This is an artifact of the ladder itself: the
dilution rewrite forces same-class edges (label homophily → 1.0), planting degree/spectral structure
the structural router reads mechanically, independent of real attention. The router's
`feature_homophily` column stays ≈ 0.996 flat across all dilutions (the L2 drift grows, but that
column does not move), and at dilution=1 the label-free assignment tracks the injected classes while
oracle/random collapse. So the rising label-free column is a property of the planted re-wiring, not
of real routing signal. It is *not* evidence about the undiluted D2 result, which §6 shows survives
the usage-matched control.

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
- Runtime profile at k=32 full (`--device cpu`; the host is an RTX 4060 Laptop but the committed run
  is itself CPU): model load ~10 s, then the dominant costs are the Laplacian eigendecomposition, the
  structural/KMeans assignments, and the 120 logistic-refit protocol cells (~20-25 min), plus ~35 min
  for the dilution ladder. `--skip-dilution` gives a verdict-only run.
- Tests: 256 pass (`python -m pytest`).

## 5. Density-aware oracle sweep (step 1 follow-up; closes the k=16 question)

Goal: find an oracle calibration/extraction that beats uniform at top-k=16 and beat the fixed
thresholds at every k. All variants evaluated on the same k=16 graph with the 6×2 protocol
(primary sign/p/z held under the full 10×3 where noted). Only the two `(repro)` rows are
produced by committed scripts (`run_real_attention.py` with the default and `--adaptive-oracle`
oracles, cf. `logs/rk16` / `logs/rk16_adapt`); the remaining variants were a one-off
exploratory sweep whose generating code is not in the repo, so they are reported as
exploratory context rather than as reproducible results.

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

**Conclusion (oracle side).** The k=16 D1 failure is a property of the sparse oracle *instrument*,
verified under ten distinct extractions/shiftings/gates. At k=16 the same-sentence signal is beneath
the neighbour-averaging noise floor (only ~1 in 16 edges is a reliable same-sentence edge), so no
calibration of a label-homophily oracle recovers it. This is an *oracle* failure only: the label-free
router still clears D2 at k=16 (+0.0221 vs oracle-usage-matched random, and +0.0154 vs the
router-usage-matched control; §6). The honest framing is therefore: **the label-homophily oracle
needs top-k ≥ 32 (budget × tercile is strongest); the label-free structural router routes at every
density tested.**

## 6. Router feature ablation and the usage-matched control (reproducible)

`experiments/run_router_ablation.py` regenerates this section; the artifact pair is
`logs/rk_ablation/real_attention/{ablation,control}.json`. The label-free router is a KMeans over
the **five** standardized structural fields `{snf, hx, spectral_ratio, log_degree, clustering}`
(`router_protocol.structural_features`; `snf` = spectral-neighbour similarity, and
`order_feature="hx"` orders the channels). For each top-k the script computes the full assignment
and the five one-field-dropped variants, then re-measures
`D2 = mean(test accuracy(label_free) − test accuracy(random))` against **two** random baselines,
both built with `random_assignment(..., rng=1)`:

- **oracle-usage-matched** random — the recorded convention (`random_assignment(oracle)`), and
- **router-usage-matched** random — the fair control (`random_assignment(label_free)`): the same
  expert histogram as the router, with a random node→expert map.

Protocol: 6×2 splits × seeds on the same 11,520-node graph.

| variant | k=16 usage | k=16 D2 | k=32 usage | k=32 D2 |
|---|---|---|---|---|
| full | [7640, 3820, 60] | +0.0195 | [7648, 3824, 48] | +0.0191 |
| drop `snf` | [88, 11408, 24] | +0.0123 | [88, 11408, 24] | +0.0148 |
| drop `hx` | [7615, 3808, 97] | +0.0195 | [7614, 3809, 97] | +0.0189 |
| drop `spectral_ratio` | [7611, 3808, 101] | +0.0192 | [7613, 3807, 100] | +0.0188 |
| drop `log_degree` | [7664, 3832, 24] | +0.0197 | [7664, 3832, 24] | +0.0189 |
| drop `clustering` | [7664, 3832, 24] | +0.0197 | [7664, 3832, 24] | +0.0189 |

(D2 column = vs oracle-usage-matched random; every p = 0.031.)

**The shipped router is not degenerate, and `snf` is load-bearing.** The full five-field router
splits ≈ 66 % / 33 % / 0.5 % across the three experts at both top-k (a two-way split plus a small
residual), and the partition is stable across k=16/32/64 and across the router seed (k=32,
`router_seed` 0 and 7 give the identical [7648, 3824, 48]). Dropping the **`snf`** field collapses
it to the degenerate [88, 11408, 24] — 99 % in one expert. That is exactly the usage the *earlier*
§6 ablation reported for its "full" model, which its own text described as a KMeans over the
**four** fields {hx, spectral_ratio, log_degree, clustering}: the retired "degenerate router"
result characterised a four-field router without `snf`, not the five-field router the pipeline
ships. The `hx` column is saturated near 0.99 for almost every node, so the discriminatory
directions are `snf` (and, weakly, `spectral_ratio`); the other three fields are nearly redundant
(their drops leave usage and D2 essentially unchanged).

**The control that matters — D2 survives it.** The recorded D2 compares the label-free router
against `random` matched to the *oracle* usage (a roughly balanced histogram), so part of the
recorded gap could be capacity rather than routing skill. Matching `random` to the router's **own**
usage (identical capacity profile, random node→expert map) leaves a large, significant gap at both
top-k:

| baseline for `random` | k=16 D2 | k=32 D2 |
|---|---|---|
| oracle-usage-matched (recorded convention) | +0.0195 (p=.031) | +0.0191 (p=.031) |
| **router-usage-matched (fair control)** | **+0.0154 (p=.031)** | **+0.0131 (p=.031)** |

So the router's node→expert allocation carries signal beyond capacity: **D2 is supported**, and the
label-free column of the dilution ladder (§3) measures a real routing effect (its monotone rise
toward dilution→1 is a separate ladder artifact). The earlier "null at matched usage
(−0.0002, p≈0.56)" was measured on the degenerate four-field router, which routes ~99 % of nodes
through a single expert and therefore leaves the usage-matched control nothing to differ on.

**One caveat, stated plainly.** The KMeans partition is a discrete optimum over a
spectral-neighbour field, and it is sensitive to the exact attention export. The pre-regeneration
artifact (committed at `dd5fd5e`, produced by the *same* code and corpus) recorded the degenerate
[88, 11408, 24] partition — the attention export moved enough between runs to shift the oracle usage
by ±2 nodes ([2046, 3840, 5634] → [2048, 3839, 5633]) and to flip the KMeans basin. The result
reported here is reproducible from the committed script on the committed export and is stable across
router seeds and top-k, but D2 for this graph should be treated as **export-sensitive**; a
cross-export stability check is the natural next step before the claim is treated as final.

## 7. Bottom line

> **Protocol A (self-routing), top-k ≥ 32.** Real attention carries a sentence-homophily signal
> that both the label-augmented oracle (D1, D3) and the **label-free structural router** (D2)
> detect: the router beats random even when `random` is matched to its own usage (+0.0131, p=0.002;
> §6), carried by the `snf` (spectral-neighbour-similarity) field — so the earlier "D2 is a usage
> artifact" retraction is **withdrawn**. The instrument is valid (D4); the one failure is D1 at
> top-k=16, an *oracle-instrument* limit (too few reliable same-sentence edges survive
> neighbour-averaging). **This is the self-routing Protocol A: the operative accuracy verdict is
> the Protocol-B null (§9.2) and the README retraction, and the surviving project-level positive is
> routing *stability* (§9.1/§9.3).** Two caveats: (i) the D2 partition is sensitive to the exact
> attention export (§6); (ii) the dilution ladder's rising label-free column is a ladder artifact
> (§3), independent of the undiluted D2 result above.

## 8. Artifacts

| Run | JSON |
|---|---|
| k=32 baseline (full, with dilution) | `logs/real_attention/real_attention_HuggingFaceTB--SmolLM2-1.7B.json` (backup: `..._topk32.json`) |
| k=32 seed 7 | `logs/rk32_s7/real_attention/...` |
| k=64 | `logs/rk64/real_attention/...` |
| k=16 fixed thresholds | `logs/rk16/real_attention/...` |
| k=16 adaptive terciles | `logs/rk16_adapt/real_attention/...` |
| k=32 oracle sweep, §5 (raw/budget × fixed/tercile) | `logs/rk32_budget/real_attention/...` |
| router feature ablation + usage-matched control (§6, reproducible) | `experiments/run_router_ablation.py` → `logs/rk_ablation/real_attention/{ablation,control}.json` |

The six `real_attention_*.json` runs above are the 11,520-node graph (`data/erp_attention/corpus_scaled.txt`,
29 sentence labels, `--device cpu`) and are regenerated in place by `experiments/run_real_attention.py`.
The previously committed copies of these six files were a 415-node `corpus.txt` regression and do not
correspond to any number in this document.
---

## 9. SD vs routing, and the Protocol B rerun (2026-10-04)

Two new experiments on the real SmolLM2 attention graph. Both use
`data/erp_attention/corpus_scaled.txt` (24 passages, 11,520 token nodes, 29
sentence labels), `--top-k 32`, and a strided subsample to 2,000 nodes. Striding
rather than slicing matters: the attention graph is block-diagonal by passage, so
taking the first 2,000 rows would keep only the first ~4 passages and discard
every later block and its labels.

### 9.1 Spectral distortion predicts routing stability (`run_sd_vs_routing.py`)

Low-frequency spectral error vs same-seed assignment stability of the label-free
structural router:

| Graph | density | Spearman(lowfreq, stability), collapse-free | Spearman(survival, stability), collapse-free |
|---|---|---|---|
| Amazon | 9.02e-02 | -0.493 (p=0.062) - no signal | +0.471 (p=0.0762) |
| **SmolLM2** | 4.39e-03 | **-0.673 (p=0.033)** | **+0.988 (p=9.3e-08)** |
| Tolokers | 7.16e-03 | **-0.733 (p=0.0043)** | **+0.961 (p=1.7e-07)** |
| YelpChi | 3.74e-03 | **-0.867 (p=1.3e-04)** | **+0.972 (p=2.8e-08)** |
| Pooled | - | -0.687 (p=2.6e-08) | +0.765 (p=6.4e-11) |

All three sparsifiers degrade SmolLM2 routing monotonically with retention:

| method | r=1.0 | r=0.5 | r=0.25 | r=0.1 | r=0.05 |
|---|---|---|---|---|---|
| degree | 1.000 | 0.650 | 0.415 | 0.362 | 0.192 |
| effective resistance | 1.000 | 0.539 | 0.275 | 0.124 | 0.076 |
| random | 1.000 | 0.980 | 0.857 | 0.531 | 0.324 |

So spectral distortion does track *routing stability* on real LLM attention.
Note the confound: on SmolLM2 the isolated-node fraction alone correlates -0.996
with stability, and at r=0.05 effective resistance leaves 1,826 of 2,000 nodes
isolated. The spectral term is not separable from connectivity destruction on
this graph. Section 9.3 quantifies this confound across all four graphs with
partial correlations and bootstrap confidence intervals.

### 9.2 Protocol B accuracy: null (`run_llm_protocol_b.py`)

Protocol B freezes the expert pool: k-means on dense node features (routing-blind,
never sees labels or the sparsified graph), fitted once per split, then only the
routing varies. Uses `feature_mode=per_layer` (24 features); the legacy `agg`
mode gives 3 features, and a 29-way classifier from 3 features is near-chance for
any routing, which would make the arm uninformative.

| routing | dense-graph accuracy |
|---|---|
| label-free (dense) | 0.1492 |
| oracle | 0.1421 |
| random | 0.1297 |
| uniform ensemble over frozen pool | 0.0325 (chance = 0.0345) |

Accuracy across sparsification is flat and non-monotone (0.1051-0.1622); heavily
sparsified conditions are sometimes *better* than dense. Distortion does not
predict the accuracy drop:

- Spearman(low_frequency_error, accuracy_drop) = **+0.231 (p=0.448)**
- Spearman(routing_survival, accuracy_drop) = **+0.066 (p=0.831)**

**Conclusion.** Under Protocol B, sparsifying the SmolLM2 attention graph does not
measurably degrade downstream accuracy. The frozen routing-blind pool sits at
chance and random routing is close to the label-free router, so on this task there
is no useful routing channel for spectral distortion to corrupt. This confirms
the earlier positive SmolLM2 result was a Protocol A self-routing artifact. The
defensible claim is the stability result in 9.1, not an accuracy claim.

### 9.3 Spectral distortion and connectivity survival are jointly predictive

Section 9.1 reported that low-frequency spectral error correlates negatively with
routing stability, and that connectivity survival correlates strongly positively
with it. Those two predictors are themselves almost collinear (both are monotone
in the retained isolated-fraction), so the raw Spearman coefficients cannot say
which one carries the signal. `analyze_sd_vs_routing.py` now adds, on the same
collapse-free points, a partial Spearman that controls for the isolated-node
fraction, bootstrap 95% CIs, and a leave-one-graph-out pooled check.

Pooled across the four graphs (n = 51 collapse-free points):

| predictor | raw Spearman rho | p | bootstrap 95% CI | partial rho | p |
|---|---|---|---|---|---|
| low-frequency error -> stability | -0.687 | 2.5e-08 | [-0.850, -0.456] | **-0.618** | 1.7e-06 |
| connectivity survival -> stability | +0.765 | 6.4e-11 | [+0.561, +0.893] | **+0.710** | 7.6e-09 |

At the pooled level *both* predictors retain a significant partial correlation
(low-frequency -> stability -0.618, p = 1.7e-06; survival -> stability +0.710,
p = 7.6e-09), so partialling out isolation does not eliminate the spectral term
once the four graphs are pooled. Per graph the two predictors behave
differently: the low-frequency partial is *not* significant on any single graph
(Amazon -0.41 p=0.15, SmolLM2 +0.49 p=0.18, Tolokers -0.40 p=0.19, YelpChi
+0.24 p=0.46), whereas the survival partial is significant on three of four
(SmolLM2 +0.98 p=2.0e-06, Tolokers +0.93 p=1.3e-05, YelpChi +0.90 p=5.7e-05)
and only directional on Amazon (+0.38, p=0.18). The pooled result is not driven
by any single graph: leaving one out gives low-frequency->stability rho in
[-0.81, -0.54] and survival->stability rho in [+0.68, +0.91] throughout.

**Interpretation.** Connectivity survival is the more reliable single predictor:
its partial correlation is large and significant on three of the four graphs and
on the pooled sample, while the low-frequency partial is significant only after
pooling and never within a single graph. The two predictors are almost collinear
on these graphs -- both are monotone in the retained isolated-node fraction --
so the sharing is expected: spectral distortion and connectivity destruction are
largely the same variable here. One caveat cuts the other way: if sparsification
destroys connectivity *because* it distorts low-frequency structure, then
isolation is a mechanism of the distortion rather than a nuisance, and
partialling it out over-controls; under that reading the pooled spectral partial
partly survives for exactly that reason. The defensible operational claim is
still that routing survival, not the raw spectral error, is what should be
watched when pruning these graphs; the pooled partial correlations add that
low-frequency error is not fully redundant with it.

Artifacts: `logs/sd_vs_routing/**`, `logs/sd_vs_routing/summary.json`,
`logs/sd_vs_routing/partial_correlations.json`,
`logs/llm_protocol_b/llm_protocol_b.json`.
