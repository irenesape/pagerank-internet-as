"""Ranking helpers for score vectors.

Notebook 02 compares the PageRank ordering against the in-degree ordering, so
the tie-breaking rule has to be deterministic: an ordering that changed between
runs would make the comparison table unreproducible. Ties are therefore broken
by ascending node index, which is stable and independent of the sort backend.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import scipy.stats


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


# --------------------------------------------------------------------------
# Comparing two rankings.
#
# Notebook 02 sets the PageRank ordering against the in-degree ordering, which
# needs more than a top-k list: the rank every node holds under each measure,
# how much the two top-k sets overlap, a single rank-correlation number, and
# the nodes that move the most between the two. These four helpers provide
# exactly that, all resting on the deterministic `ranking` above so their
# output is reproducible.
# --------------------------------------------------------------------------


def rank_positions(scores: np.ndarray) -> np.ndarray:
    """Return the 1-based rank of every node under :func:`ranking`.

    The inverse of the permutation ``ranking`` returns: where ``ranking`` lists
    node indices best-first, this lists, for each node index, the position it
    holds (1 = highest score). The two satisfy
    ``rank_positions(s)[ranking(s)[i]] == i + 1``.

    Args:
        scores: Score per node, shape ``(n,)``.

    Returns:
        An int64 array of shape ``(n,)``; entry ``i`` is node ``i``'s 1-based
        rank, ties broken by ascending index (the same rule as ``ranking``).
    """
    order = ranking(scores)
    n = order.shape[0]
    positions = np.empty(n, dtype=np.int64)
    positions[order] = np.arange(1, n + 1, dtype=np.int64)
    return positions


def rank_positions_ties(scores: np.ndarray) -> np.ndarray:
    """Return 1-based ranks where tied scores share the minimum rank.

    Unlike :func:`rank_positions`, which breaks ties by index and so assigns a
    distinct position to every node, this gives every node with the same score
    the same rank: the best position that group occupies. It uses
    ``scipy.stats.rankdata`` with ``method="min"`` on the negated scores (so the
    highest score ranks 1).

    This is the right rank when a large block of nodes genuinely ties. The
    customer-cone comparison needs it because most ASes have cone size 1: an
    index-tie-broken rank would spread those identical cones across tens of
    thousands of arbitrary positions and manufacture rank "shifts" that are
    pure index artefacts.

    Args:
        scores: Score per node, shape ``(n,)``. Higher is better.

    Returns:
        An int64 array of shape ``(n,)``; entry ``i`` is node ``i``'s 1-based
        rank with ties sharing the smallest rank in their group.
    """
    scores = np.asarray(scores, dtype=np.float64)
    ranks = scipy.stats.rankdata(-scores, method="min")
    return ranks.astype(np.int64)


def top_k_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Fraction of the top ``k`` shared by two score vectors.

    ``|top_k(a) ∩ top_k(b)| / k``, with each top-k set taken from
    :func:`ranking` (so ties break by ascending index). A value of 1.0 means
    the two measures pick the same set of ``k`` nodes, regardless of their
    order within it.

    Args:
        a: First score vector, shape ``(n,)``.
        b: Second score vector, same shape as ``a``.
        k: Size of the top set to compare; must be in ``1 .. n``.

    Returns:
        The overlap as a float in ``[0, 1]``.

    Raises:
        ValueError: If the vectors differ in length, or ``k`` is not in
            ``1 .. n``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got {a.shape} and {b.shape}")
    n = a.shape[0]
    if not (1 <= k <= n):
        raise ValueError(f"k must be in 1..{n}, got {k}")

    top_a = set(int(i) for i in ranking(a)[:k])
    top_b = set(int(i) for i in ranking(b)[:k])
    return len(top_a & top_b) / k


def kendall_tau(
    a: np.ndarray,
    b: np.ndarray,
    subset: np.ndarray | None = None,
) -> float:
    """Kendall's tau-b between two score vectors, optionally over a subset.

    A rank-correlation in ``[-1, 1]``: 1 for identical orderings, -1 for
    reversed. The tau-b variant corrects for ties, which matter here because
    in-degree is a small integer with many ties. This is the one place the
    project leans on :mod:`scipy.stats` rather than its own code: the
    "from scratch" claim in the README is about the PageRank solver, not about
    a standard statistic, so using ``scipy.stats.kendalltau`` is in scope.

    Args:
        a: First score vector, shape ``(n,)``.
        b: Second score vector, same shape as ``a``.
        subset: Optional index or boolean array selecting the nodes to compare;
            ``None`` compares all nodes.

    Returns:
        The tau-b statistic as a float.

    Raises:
        ValueError: If the vectors differ in length.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got {a.shape} and {b.shape}")
    if subset is not None:
        a = a[subset]
        b = b[subset]
    return float(scipy.stats.kendalltau(a, b).statistic)


def largest_rank_shifts(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """Return the ``k`` node indices whose rank moves most between two vectors.

    The shift of a node is ``|rank_positions(a) - rank_positions(b)|``; this
    returns the ``k`` nodes with the largest shift, most displaced first, ties
    broken by ascending index. These are the ASes the two measures disagree
    about the most, which notebook 02 inspects for the structural reason.

    Args:
        a: First score vector, shape ``(n,)``.
        b: Second score vector, same shape as ``a``.
        k: Number of nodes to return; clipped to ``n`` if larger.

    Returns:
        An int64 array of node indices, length ``min(k, n)``, ordered by
        descending shift then ascending index.

    Raises:
        ValueError: If the vectors differ in length, or ``k`` is negative.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got {a.shape} and {b.shape}")
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")

    shift = np.abs(rank_positions(a) - rank_positions(b))
    n = shift.shape[0]
    # lexsort: last key is primary. Sort by -shift (descending), ties by index
    # (ascending via the arange secondary key).
    order = np.lexsort((np.arange(n), -shift))
    return order[: min(k, n)].astype(np.int64)
