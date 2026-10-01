# D2 router fix: label-free routing on real fraud graphs

## Summary

D2 asks whether a **label-free** router recovers routing gains comparable to the
label-using oracle. It failed on both real fraud graphs while synthetic and
planted-control experiments passed. The cause was a design flaw in the label-free
router, not a lack of signal.

The shipped router clustered nodes with a joint KMeans over five structural
features and then ordered the resulting clusters by their mean feature-space
homophily (`hx`). Two problems:

1. **Wrong proxy.** `hx` measures similarity in *feature* space, while routing
   utility is driven by similarity in *label* space. On Tolokers these are
   anti-correlated: Spearman `rho = -0.14` for `neigh_feat_sim`, and the legacy
   ordering key `hx` performs no better than a random router.
2. **Wrong partition.** Even with a correct ordering score, the joint KMeans
   partition is dominated by `log_degree` and `clustering`, so clusters do not
   align with the smoothness levels the routing channels need. Reordering alone
   gave `+0.0001` over random (`p = 0.63`) while directly bucketing the same
   score gave `+0.0050`.

The fix buckets the ordering score directly into adaptive quantiles, using the
same rule the oracle uses, and orders by a spectral neighborhood-smoothness
proxy instead of feature homophily.

## Diagnostic screen

Each candidate is a label-free node score; `rho` is its Spearman correlation
with the true label homophily (labels are used **only** for this diagnostic, never
by the router). Artifacts: `logs/{amazon,tolokers}_d2_candidates/candidates.json`.

| candidate | Tolokers rho | Amazon rho |
|---|---|---|
| `eig_nb_sim` | **+0.4184** | -0.0317 |
| `eig_centrality` | +0.3974 | **-0.5094** |
| `neigh_degree` | -0.3472 | +0.4492 |
| `neigh_feat_sim` | -0.1431 | +0.1910 |
| `local_clust` | +0.0667 | -0.3121 |
| `degree` | -0.0656 | +0.3124 |

Candidate scores are oriented by the sign of `rho` so that higher always means
more label-homophilic. Without this, a strongly anti-correlated candidate is fed
to the router reversed and can never work — this silently invalidated `neigh_degree`
on Amazon in an earlier run.

## Result

Full protocol, 10 splits x 3 seeds, frozen-permutation gate with 20 permutations.
`eig_nb_sim` with `proxy_quantile` bucketing.

| comparison | Tolokers | Amazon |
|---|---|---|
| D1 oracle vs uniform | +0.0365 (p 0.001953, dz 14.94) | +0.0028 (p 0.001953, dz 2.68) |
| **D2 label-free vs random** | **+0.0040 (p 0.001953, dz 4.24)** | +0.0004 (p 0.2324, dz 0.39) |
| D3 oracle vs random | +0.0368 (p 0.001953, dz 13.31) | +0.0038 (p 0.001953, dz 3.43) |
| verdict | **Supported** | **Not supported (D2)** |

Tolokers is the first real fraud graph to pass D1-D4 end to end. D2 recovers
about 11% of the oracle's gain, versus 0.3% under the legacy router.

**Amazon still fails D2**, and the diagnostic explains why: `eig_nb_sim` has no
correlation with label homophily there (`rho = -0.03`). The proxy that works on
one fraud graph does not transfer. The best Amazon candidate is
`eig_centrality` at `+0.00093` (`p = 0.0039`, Holm-corrected) against an oracle
ceiling of `+0.00396`, so even a proxy chosen per-graph recovers only ~23%.

The honest conclusion is that a single universal label-free proxy does not exist
across these graphs. The router now works where a good proxy exists, and the
failure mode is measurable in advance via the stage-1 screen.

**Third-graph update (YelpChi).** YelpChi passes D1–D4, but **under the legacy
router as well** (D2 +0.00575, CI [+0.00474, +0.00675], dz 4.33) — the fix gives
+0.00481 (CI [+0.00427, +0.00535], dz 6.73). The CIs overlap, the legacy mean is
higher, and under ER pruning the fix is non-significant at r=0.08 where legacy is
not. So the fix is a **Tolokers-only** improvement, not a two-of-three one. The
result that generalizes is weaker and more believable: *two of three real fraud
graphs have a recoverable label-free routing signal, and it does not matter much
which bucketing rule you use to find it.* See `docs/yelpchi_third_graph.md`.

## Reproduction

```bash
# full protocol with the fix
python spectral_distillation/experiments/run_real_fraud.py --dataset tolokers \
  --label-free-strategy proxy_quantile --order-feature eig_nb_sim \
  --splits 10 --seeds 3 --frozen-perms 20 --no-progress --out logs

# diagnostic screen (add --with-resistance for the exact dense pinvh candidate)
python spectral_distillation/experiments/run_d2_candidates.py --dataset tolokers \
  --splits 10 --seeds 3 --top-k-eval 4 --no-progress --out logs
```

Artifacts land in suffixed directories, `logs/{dataset}_fraud_protocol_eig_nb_sim_pq/`,
so the committed legacy-protocol results are never overwritten by a different
router. The runner defaults remain the legacy `kmeans` / `hx` configuration so
every previously committed artifact still reproduces exactly; the fix is opt-in.

## Caveats

- `sd_laplacian` is `0.0` in the ladder artifacts; that is a `--skip-sd` placeholder,
  not a measured distortion.
- The Amazon `eig_centrality` gain (`+0.00093`, dz 1.67) is Holm-significant but
  its `rho = -0.51` means the sign-flip orientation is doing real work. Treat it as
  suggestive rather than established.
- `eig_nb_sim` uses 8 nontrivial Laplacian modes (`--spectral-k`); the value is
  untuned and may be sensitive.
- Only two real fraud graphs have been tested. A third is required before claiming
  generality.
