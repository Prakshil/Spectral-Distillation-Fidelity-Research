# Literature positioning

Where Spectral Distillation sits relative to the adjacent work, and what is
genuinely different. This is a positioning document, not a results document:
all quantitative claims it references are stated and sourced in
`docs/fixed_expert_protocol.md`, `docs/real_attention_results.md`, and
`docs/reproducibility.md`.

## 1. The problem this project addresses

Two literatures optimize different halves of the same pipeline and neither
measures the join:

1. **Graph sparsification** decides *which edges to keep* so that a graph is
   cheaper to run on, or so a GNN sees less noise. The guarantees are about the
   graph (spectral similarity, cut distortion, task accuracy), not about the
   router that will run on top of it.
2. **Graph Mixture-of-Experts (MoE) / adaptive propagation** decides *which
   expert handles which node*. The evaluations are end-task accuracy, and the
   assignment signal is usually learned or structural.

The gap is the interface: a per-node router consumes graph-derived features, so
*sparsification damages the router's input before the router ever runs*, and no
standard metric reports that damage in advance. This project defines a
**pre-training diagnostic for router fidelity** (spectral distortion, SD) and a
**router-evaluation protocol** that measures whether routing helps at all,
independent of the sparsifier.

## 2. Spectral sparsification theory (the foundation, not the contribution)

The bounds this project leans on are classical:

- Spielman & Srivastava (2011), *Graph sparsification by effective
  resistances* — sampling edges with probability `∝ W[i,j]·R_eff(i,j)` yields an
  ε-spectral sparsifier with `O(n log n / ε²)` edges.
- Spielman & Teng (2011), *Spectral sparsification of graphs*; Batson,
  Spielman & Srivastava (2012), *Twice-Ramanujan sparsifiers*.

Our use is standard: the effective-resistance sampling rule and the
`(1±ε)`-sandwich equivalence to eigenvalue preservation (`Part 5` of the
implementation guide). **We do not claim a new sparsifier.** The contribution is
the *metric and the protocol built on top of these bounds*, not the bounds
themselves.

## 3. GNN sparsification as speed / accuracy (adjacent, different objective)

- Ding et al. (2020), **FastGAT** — effective-resistance sparsification to speed
  up graph attention. Accelerator; no router-fidelity claim.
- Zhang et al., **MoG — Graph Sparsification via Mixture of Graphs**
  (ICLR 2025 Spotlight, arXiv:2405.14260) — multiple *sparsifier experts*, each a
  different pruning criterion (degree, Jaccard, **effective resistance**,
  gradient magnitude) and sparsity level, chosen per node by a router and mixed
  on the Grassmann manifold.

MoG is the closest sparsification neighbour: it also uses effective resistance
and MoE-style routing. The difference is direction of the claim. MoG routes over
*sparsifiers* to build a better graph and reports end-task accuracy; we treat the
sparse graph as given and ask *how much routing information it destroyed*, with a
metric measurable before training. MoG's router chooses among pruning policies;
our SD predicts whether any router can work on the resulting graph.

## 4. Spectrum-preserving sparsification / rewiring (adjacent, different target)

- Liang et al., **GOKU — Mitigating Over-Squashing in GNNs by
  Spectrum-Preserving Sparsification** (ICML 2025, PMLR v267; arXiv:2506.16110) —
  densify then re-sparsify (DSR), using an effective-resistance ISS instance with
  a feature-similarity prior `p_e ∝ (1+S_e)·R_e`, to raise the smallest
  eigenvalues while preserving the rest of the spectrum.

GOKU is the closest *spectral* neighbour: it explicitly preserves the Laplacian
spectrum. The distinction: GOKU preserves the spectrum as a *means* to better
message passing and reports accuracy/connectivity; we use spectral deviation as
a *diagnostic* and connect it to a router-information bound. GOKU asks "does the
rewired graph classify better"; we ask "does the spectral damage predict whether
the router's signal survived", which is a different measurable (and, crucially,
available pre-training).

## 5. Graph Mixture-of-Experts and spectral experts (adjacent, different objective)

- Luan et al. (2022), **ACM-GNN**; Hevapathige et al. (2026), **AD-GNN**;
  Zhang et al. (2025), **GraphRouter** — adaptive per-node propagation and
  routing.
- **Node-MoE** (arXiv:2406.03464) — gate features `[X, |AX−X|, |A²X−X|]` and
  differentiated ChebNetII filter initialisation. Our learned-router baseline
  reuses the *gate features* but not the expert construction (stated as a caveat
  in `docs/fixed_expert_protocol.md`).
- Wang et al., **GNNMoE — Mixture of Message-Passing Experts with Routing Entropy
  Regularization** (arXiv:2502.08083) — entropy-driven soft/top-k routing.
- **MORGAN — To Bridge Mixture of Experts and Spectral GNN** (AAAI 2026,
  doi:10.1609/aaai.v40i28.39553) — treats eigengraphs (outer products of
  Laplacian eigenvectors) as experts, partitions the spectrum into bands, one
  expert per band, with a learnable gate. **This is the single closest
  conceptual neighbour**: it makes the spectral-to-MoE mapping explicit.
- **SS-AdaMoE** (Entropy 28(3):355, 2026) — dual-domain (spatial + Bernstein
  spectral) experts with a global-prior gate.
- Yin et al., **Spectral Mixture-of-Experts for Continual Learning** (CVPR 2026) —
  disjoint spectral masks per expert.
- Kieu et al., **Eigenvectors of Experts are Training-free Non-collapsing Routers**
  (ICML 2026) — uses expert-weight eigenvectors as routers.

**MORGAN is the sharpest contrast.** MORGAN assigns experts *by* spectral band
and learns the gate; it assumes the spectrum is a good routing axis and never
measures whether the *sparsified/attention* graph preserves the spectrum a gate
needs. Our result is a cautionary complement: the spectrum is only a usable
routing axis to the extent SD is small (and, on the graphs tested, no
structural router beats a single global model). MORGAN's construction would be
affected by exactly the distortion our SD quantifies; the two are complementary,
not competing.

## 6. Router-evaluation methodology (our protocol's lineage)

The fixed-expert protocol is a correction to the standard self-routing
benchmark, in the lineage of:

- Roller et al. (2021), *Hash layers for large sparse models* — random routing
  as a floor.
- Dikkala et al. (2023) — do learnable routers recover planted structures.
- ERP•AI (2026), *Does your router actually route? A mechanism-validity
  protocol* — the D1–D4 structure and the planted positive control.
- *How much of the routing gap is real? Decomposing the router-to-oracle gap*
  (arXiv:2607.03436).

Our contribution here is specific and testable: the standard self-routing
protocol is **permutation-invariant** and grants **free credit to any
label-homogeneous partition**, so a router score and its negation score
identically. We show this (`docs/fixed_expert_protocol.md`, Result 1) and replace
it with a cross-fit protocol over a routing-blind frozen pool. The same protocol
is what produces the negative result; the methodological finding and the
empirical finding are the same artifact viewed twice.

## 7. Spectral / entropy signals inside LLMs (adjacent, different use)

- Thermodynamic signatures of reasoning (arXiv:2606.19404); LapEigvals (EMNLP
  2025); spectral guardrails (Noël 2026); CHIAR-Former (2026) — spectral or
  entropy features read *out of* a transformer for hallucination detection or
  entropy routing.
- FastGAT / attention-graph sparsification — attention graphs are already
  sparsified in practice.

Our twist: attention entropy is used as a **pre-construction** routing signal —
computed directly from the LLM, so graph distillation cannot corrupt it by
construction (`src/attention_entropy.py`). Prior work uses entropy *after* or
*inside* a model; we propose it as the signal that is provably outside the
distillation's blast radius.

## 8. Davis-Kahan and GNN stability (classical machinery, operational twist)

- Davis & Kahan (1970); Gama, Ribeiro & Bruna (2020, ICASSP); GCN stability
  analyses; **DISPEL-GNN** (Mathematics 14(4):602, 2026).

Davis-Kahan is classical and widely used for stability; our use is to turn the
ratio `SD / min-eigengap` (the DK numerator over the fragility denominator) into
an **operational, pre-training routing diagnostic**, plus a scoped Lean
formalization of the edge-removal corollary (Cauchy interlacing). The novelty is
the operationalization, not the inequality.

## 9. What is genuinely ours

1. **A pre-training router-fidelity metric.** SD (and its component-robust
   repairs `rank_matched_distortion`, `low_frequency_distortion`,
   `routing_survival`) is computed in milliseconds from the two Laplacians,
   before any GNN trains, and is tied to a routing-information bound
   (`docs/theorem_proofs.md`). MoG/GOKU optimize a graph; MORGAN/SS-AdaMoE learn a
   gate; none publishes a pre-training predictor of routing signal survival.
2. **Attention entropy as a pre-construction signal** that lives outside the
   distillation's reach by construction.
3. **A corrected router-evaluation protocol** that removes the permutation
   invariance and self-routing credit of the standard benchmark, with a positive
   control proving the harness can detect a win and both implicit-routing
   channels closed.
4. **A complete, honest empirical characterization**: on real fraud graphs and
   real LLM attention, structural / learned routers do **not** beat one global
   model, the oracle can be *negative*, and the one surviving positive is that
   SD/low-frequency distortion + connectivity survival **predict routing
   stability** (Spearman +0.78…+0.99; `docs/real_attention_results.md` §9.1).

## 10. Honest gaps versus the literature

- **Published routers are not yet implemented.** RouterGNN, Ada-Routing/AD-GNN,
  and Node-MoE's expert construction are not reproduced; the negative is against
  structural-score routers and a label-trained MLP gate, not against those
  methods (`docs/fixed_expert_protocol.md`, caveats). This is the largest open
  item.
- **The k=128 / YelpChi and k=32 real-attention rows are reduced-power** (fewer
  splits/seeds or node-capped) and are marked as such; the honest headline is
  full-power 0/26.
- **The accuracy claim under Protocol B is null**; only the stability claim
  (§9.1) is defensible, and it carries a connectivity confound on the SmolLM2
  graph (isolated-node fraction correlates −0.996 with stability).
- **The Lean formalization is a sketch**, not a compiled Mathlib contribution
  (Appendix B of the implementation guide).

## Reference key (venue/ID for the closest works)

| Work | Venue / ID | Relation |
|------|-----------|----------|
| Spielman & Srivastava | SIAM J. Comput. 2011 | ER sampling bound (used) |
| Ding et al., FastGAT | arXiv:2006.08796 | attention sparsification (used as baseline) |
| Zhang et al., MoG | ICLR 2025, arXiv:2405.14260 | sparsifier MoE (adjacent) |
| Liang et al., GOKU | ICML 2025, PMLR v267, arXiv:2506.16110 | spectrum-preserving rewiring (closest spectral) |
| MORGAN | AAAI 2026, doi:10.1609/aaai.v40i28.39553 | eigengraph-as-expert MoE (closest conceptual) |
| SS-AdaMoE | Entropy 28(3):355, 2026 | spatio-spectral MoE |
| Node-MoE | arXiv:2406.03464 | gate features reused |
| GNNMoE | arXiv:2502.08083 | entropy-routed MoE |
| Roller et al. | NeurIPS 2021 | random-routing floor |
| Dikkala et al. | 2023 | learned-router recovery |
| ERP•AI | Tech. Report 2026 | D1–D4 protocol |
| Davis & Kahan | SIAM J. Numer. Anal. 1970 | eigenvector rotation bound |
| DISPEL-GNN | Mathematics 14(4):602, 2026 | spectral stability GNN |
