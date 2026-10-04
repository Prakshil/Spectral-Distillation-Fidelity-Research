"""Real LLM attention export into protocol-ready graph artifacts.

Bridge between a real pretrained causal language model and the existing
D1-D4 router-protocol machinery (guide: "Attention Entropy as a
Pre-Construction Routing Signal"). ``export_attention`` runs the model on a
passage and records ``(L, n, n)`` attention probability matrices (heads
averaged); ``build_attention_graph`` reduces one or more passages to a
single weighted adjacency over tokens using the same aggregation operators
the synthetic pipeline relies on.

Two subtle design decisions shaped by the deconfounding invariants:

- Node features are the *attention-entropy* signal (mean / max / std across
  layers), i.e. a pre-construction signal that graph distillation cannot
  corrupt, mirroring the entropy features used by the synthetic control.
- Node labels are the token-to-sentence mapping. Sentence membership is a
  natural, real "regime": the oracle bucket assignment reads it via label
  homophily, and the label-free structural router must recover it from the
  graph alone, which keeps the oracle / label-free split honest.

Multi-passage graphs are merged as a block-diagonal adjacency, matching the
ERP-style premise that transactions interact only weakly across passages --
the same structure the synthetic attention generator mimics (Appendix C).

``transformers`` / ``torch`` are imported lazily so the rest of the package
still runs and its tests pass without the heavy dependency installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from spectral_distillation.src.attention_entropy import (
    aggregate_entropy_features,
    compute_attention_entropy,
)
from spectral_distillation.src.laplacian import build_adjacency

DEFAULT_CONTEXT = 512
PAD_PLACEHOLDER = "<_>"


@dataclass
class AttentionRecord:
    """Per-passage attention export: tokens, (L, n, n) layers, sentence labels."""

    tokens: list[str] = field(default_factory=list)
    attention: np.ndarray | None = None  # (L, n, n), rows normalized
    sentence_ids: np.ndarray | None = None  # (n,) int


def renormalize_attention(attention: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-normalize attention probabilities (rows sum to 1); guard zero rows."""
    attention = np.asarray(attention, dtype=float)
    row_sums = attention.sum(axis=-1, keepdims=True)
    row_sums = np.where(row_sums > eps, row_sums, 1.0)
    return attention / row_sums


def aggregate_attention(record: AttentionRecord, how: str = "mean") -> np.ndarray:
    """Reduce (L, n, n) attention to a single (n, n) weighted matrix."""
    A = renormalize_attention(record.attention)
    if how == "max":
        return np.max(A, axis=0)
    if how == "median":
        return np.median(A, axis=0)
    return np.mean(A, axis=0)


def _block_diag(blocks: list[np.ndarray]) -> np.ndarray:
    """Stack square blocks into a sparse-friendly block-diagonal matrix."""
    total = sum(b.shape[0] for b in blocks)
    out = np.zeros((total, total), dtype=float)
    offset = 0
    for b in blocks:
        n = b.shape[0]
        out[offset : offset + n, offset : offset + n] = b
        offset += n
    return out


def build_attention_graph(
    records: list[AttentionRecord] | AttentionRecord,
    how: str = "mean",
    max_edges_per_node: int | None = None,
    feature_mode: str = "agg",
) -> dict:
    """Build a protocol-ready graph dict from one or more exported passages.

    Returns the schema consumed by the D1-D4 pipeline: ``W`` (weighted
    adjacency), ``features`` (per-token attention-entropy features), ``y``
    (token -> sentence label ids), ``tokens``, and provenance metadata.
    Rows of every attention layer are used for entropy so the feature
    dimension is independent of passage count.

    ``feature_mode``:
      - ``"agg"`` (default, legacy): mean/max/std of the per-layer entropy, so
        ``d_features == 3`` regardless of model depth. This is what the
        committed artifacts used, and it keeps them reproducible.
      - ``"per_layer"``: one column per transformer layer (``d_features ==
        n_layers``). Needed for any downstream *accuracy* arm: with only 3
        features a classifier over 29 passage labels is near-chance no matter
        how good the routing is, which makes Protocol B uninformative.
    """
    if isinstance(records, AttentionRecord):
        records = [records]

    blocks, tokens, ys = [], [], []
    n_layers = records[0].attention.shape[0] if records[0].attention is not None else 1

    for rec in records:
        A = aggregate_attention(rec, how=how)
        if max_edges_per_node is not None and max_edges_per_node < A.shape[0]:
            cutoff = np.partition(A, -max_edges_per_node, axis=1)[:, -max_edges_per_node][:, None]
            A = np.where(A >= cutoff, A, 0.0)
        W = build_adjacency(A, symmetrize=True)
        blocks.append(W)
        tokens.extend(rec.tokens)
        ys.append(np.asarray(rec.sentence_ids, dtype=int))

    W = _block_diag(blocks)
    y = np.concatenate(ys) if len(ys) else np.zeros(W.shape[0], dtype=int)

    if feature_mode == "per_layer":
        entropies = [compute_attention_entropy(rec.attention).T for rec in records]
    elif feature_mode == "agg":
        entropies = [
            aggregate_entropy_features(compute_attention_entropy(rec.attention))
            for rec in records
        ]
    else:
        raise ValueError(
            f"feature_mode must be 'agg' or 'per_layer', got {feature_mode!r}"
        )
    features = np.concatenate(entropies, axis=0) if len(entropies) > 1 else entropies[0]

    return {
        "W": W,
        "features": features,
        "y": y,
        "tokens": list(tokens),
        "n_nodes": W.shape[0],
        "n_layers": n_layers,
        "d_features": features.shape[1],
        "feature_mode": feature_mode,
        "passages": len(records),
    }


def passage_to_sentence_ids(
    n_tokens: int,
    n_sentences: int | None = None,
    sentence_lengths: list[int] | None = None,
) -> np.ndarray:
    """Map tokens to sentence ids (int labels) from tokenizer offsets.

    If ``sentence_lengths`` (token counts per sentence) is given it is used
    directly; otherwise ``n_tokens`` are split into ``n_sentences`` roughly
    equal contiguous blocks. This mirrors how a real tokenizer would assign
    each token to the sentence that contains its character span.
    """
    if sentence_lengths is not None:
        ids = []
        for s, length in enumerate(sentence_lengths):
            ids.extend([s] * max(0, int(length)))
        ids = ids[:n_tokens]
        return np.asarray(ids, dtype=int)

    n_sentences = n_sentences or max(1, int(np.ceil(n_tokens / 48)))
    boundaries = np.linspace(0, n_tokens, n_sentences + 1, dtype=int)
    ids = np.zeros(n_tokens, dtype=int)
    for s in range(n_sentences):
        ids[boundaries[s] : boundaries[s + 1]] = s
    return ids


def split_dataset(text: str, max_tokens: int = 480) -> list[str]:
    """Split raw corpus text into a list of passages bounded by ``max_tokens``.

    Passages are provisional chunks; the tokenizer enforces the true budget.
    Non-empty, whitespace-squeezed lines are grouped so each passage is a
    contiguous slice of the source document(s).
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    passages: list[str] = []
    current: list[str] = []
    budget = 0
    for ln in lines:
        est = len(ln.split())
        if current and budget + est > max_tokens:
            passages.append(" ".join(current))
            current, budget = [], 0
        current.append(ln)
        budget += est
    if current:
        passages.append(" ".join(current))
    return passages


def export_attention(
    model,
    tokenizer,
    text: str | list[str],
    max_tokens: int = 480,
    device: str = "cuda",
) -> list[AttentionRecord]:
    """Run ``model`` on ``text`` and record per-layer attention per passage.

    ``text`` may be a single string (treated as one passage) or a list of
    passages. Each passage is tokenized and truncated to ``max_tokens``,
    forwarded once, and its ``(L, n, n)`` attention (heads averaged, rows
    normalized) plus token strings and sentence labels recorded. Sentences
    are inferred from the tokenizer's per-token offsets so labels come from
    the real text structure, not from a synthetic label generator.
    """
    import torch

    if isinstance(text, str):
        text = [text]

    records: list[AttentionRecord] = []
    for passage in text:
        encoding = tokenizer(
            passage,
            max_length=max_tokens,
            truncation=True,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        input_ids = encoding["input_ids"]
        offsets = encoding.get("offset_mapping")
        n_tokens = len(input_ids)

        if n_tokens == 0:
            continue

        # Sentence boundaries from real text structure (punctuation/newlines).
        sentence_ids = (
            _sentence_ids_from_offsets(passage, offsets, n_tokens) if offsets else None
        )
        if sentence_ids is None:
            sentence_ids = passage_to_sentence_ids(n_tokens, n_sentences=min(6, max(1, n_tokens // 48)))

        input_tensor = torch.as_tensor([input_ids], dtype=torch.long, device=device)
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_tensor, output_attentions=True, use_cache=False)

        attns = outputs.attentions
        if attns is None:
            raise RuntimeError("model returned no attention; enable output_attentions")

        layers = []
        for layer_attn in attns:
            A = layer_attn.detach().float().cpu().numpy()[0]  # (H, n, n)
            A = A.mean(axis=0)
            layers.append(renormalize_attention(A))
        stacked = np.stack(layers, axis=0)

        tokens = [
            (tokenizer.decode([tid]).strip() or PAD_PLACEHOLDER)
            for tid in input_ids
        ]

        records.append(
            AttentionRecord(
                tokens=tokens,
                attention=stacked,
                sentence_ids=np.asarray(sentence_ids[:n_tokens], dtype=int),
            )
        )
    return records


def _sentence_ids_from_offsets(text: str, offsets, n_tokens: int) -> np.ndarray:
    """Map tokens to sentence ids using real punctuation/newline boundaries.

    A sentence break is recorded at any ``.``, ``!`` or ``?`` (or blank line)
    in ``text``; each token is assigned the id of the sentence whose character
    span contains the token's start offset. This uses the real text structure
    instead of synthetic chunking.
    """
    breaks = set()
    for m in re.finditer(r"[.!?]+\s+|[.!?]+$", text):
        breaks.add(m.end() - 1)
    breaks.update(m.start() for m in re.finditer(r"\n\s*\n", text))
    breaks.add(len(text))
    ordered = sorted(b for b in breaks if b <= len(text))

    import bisect

    ids = np.zeros(n_tokens, dtype=int)
    for i in range(n_tokens):
        start = offsets[i][0] if offsets[i] else 0
        ids[i] = bisect.bisect_left(ordered, start)
    return ids