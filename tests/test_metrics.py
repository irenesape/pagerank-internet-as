"""Tests for :mod:`pagerank.metrics`.

The ordering and tie-breaking are the load-bearing properties: the ranking
comparison in notebook 02 depends on them being deterministic.
"""

import numpy as np
import pytest

from pagerank.metrics import ranking, top_k


def test_ranking_orders_by_descending_score() -> None:
    """Highest score first."""
    scores = np.array([0.1, 0.5, 0.3, 0.05])
    assert ranking(scores).tolist() == [1, 2, 0, 3]


def test_ranking_breaks_ties_by_ascending_index() -> None:
    """Equal scores keep ascending index order, deterministically."""
    scores = np.array([0.3, 0.3, 0.3, 0.1])
    assert ranking(scores).tolist() == [0, 1, 2, 3]

    scores = np.array([0.1, 0.4, 0.4, 0.4])
    assert ranking(scores).tolist() == [1, 2, 3, 0]


def test_top_k_without_labels_uses_indices() -> None:
    """Default labels are the node indices, as (index, score) pairs."""
    scores = np.array([0.1, 0.5, 0.3, 0.05])
    assert top_k(scores, 2) == [(1, 0.5), (2, 0.3)]


def test_top_k_with_labels() -> None:
    """Labels are carried through in the result."""
    scores = np.array([0.1, 0.5, 0.3, 0.05])
    labels = ["a", "b", "c", "d"]
    assert top_k(scores, 2, labels=labels) == [("b", 0.5), ("c", 0.3)]


def test_top_k_larger_than_n_returns_all() -> None:
    """k > n yields every node, not an error."""
    scores = np.array([0.6, 0.4])
    result = top_k(scores, 10)
    assert result == [(0, 0.6), (1, 0.4)]
    assert len(result) == 2


def test_top_k_breaks_ties_by_index() -> None:
    """Tie-breaking carries into the top-k selection."""
    scores = np.array([0.5, 0.5, 0.5])
    assert top_k(scores, 2) == [(0, 0.5), (1, 0.5)]


def test_top_k_zero_is_empty() -> None:
    """k == 0 returns an empty list."""
    assert top_k(np.array([0.5, 0.5]), 0) == []


def test_top_k_rejects_bad_arguments() -> None:
    """Negative k and mismatched labels are errors."""
    scores = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="k must be non-negative"):
        top_k(scores, -1)
    with pytest.raises(ValueError, match="labels has length"):
        top_k(scores, 1, labels=["only-one"])
