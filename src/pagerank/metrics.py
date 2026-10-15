"""Ranking helpers for score vectors.

Notebook 02 compares the PageRank ordering against the in-degree ordering, so
the tie-breaking rule has to be deterministic: an ordering that changed between
runs would make the comparison table unreproducible. Ties are therefore broken
by ascending node index, which is stable and independent of the sort backend.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def ranking(scores: np.ndarray) -> np.ndarray:
    """Return node indices ordered by score, highest first.

    Ties are broken by ascending index, deterministically.

    Args:
        scores: Score per node, shape ``(n,)``.

    Returns:
        An int64 array of node indices, length ``n``, such that
        ``scores[result]`` is non-increasing and equal scores appear in
        ascending index order.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    # lexsort sorts by the last key first: primary key -scores (descending
    # score), secondary key the index (ascending) to break ties.
    return np.lexsort((np.arange(n), -scores))


def top_k(
    scores: np.ndarray,
    k: int,
    labels: Sequence[Any] | None = None,
) -> list[tuple[Any, float]]:
    """Return the ``k`` highest-scoring nodes as ``(label, score)`` pairs.

    Args:
        scores: Score per node, shape ``(n,)``.
        k: Number of entries to return; the result has length ``min(k, n)``.
        labels: Optional labels indexed by node; when ``None`` the node index
            is used as the label.

    Returns:
        A list of ``(label_or_index, score)`` pairs, highest score first, with
        ties broken by ascending index.

    Raises:
        ValueError: If ``k`` is negative or ``labels`` has the wrong length.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")

    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    if labels is not None and len(labels) != n:
        raise ValueError(
            f"labels has length {len(labels)}, expected {n}"
        )

    order = ranking(scores)[: min(k, n)]
    if labels is None:
        return [(int(i), float(scores[i])) for i in order]
    return [(labels[i], float(scores[i])) for i in order]
