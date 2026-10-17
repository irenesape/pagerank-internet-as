"""Tests for :mod:`pagerank.metrics`.

The ordering and tie-breaking are the load-bearing properties: the ranking
comparison in notebook 02 depends on them being deterministic.
"""

import numpy as np
import pytest

from pagerank.metrics import (
    kendall_tau,
    largest_rank_shifts,
    rank_positions,
    rank_positions_ties,
    ranking,
    top_k,
    top_k_overlap,
)


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


# --------------------------------------------------------------------------
# Comparison metrics
# --------------------------------------------------------------------------


def test_rank_positions_inverts_the_ranking() -> None:
    """Each node's 1-based rank, with a tie broken by ascending index.

    scores = [0.1, 0.5, 0.5, 0.2, 0.05]. ranking is [1, 2, 3, 0, 4] (the two
    0.5s keep index order), so the positions are:
      node 0 -> 4th, node 1 -> 1st, node 2 -> 2nd, node 3 -> 3rd, node 4 -> 5th.
    """
    scores = np.array([0.1, 0.5, 0.5, 0.2, 0.05])
    positions = rank_positions(scores)
    assert positions.tolist() == [4, 1, 2, 3, 5]

    # The defining relation against ranking.
    order = ranking(scores)
    for i, node in enumerate(order):
        assert positions[node] == i + 1


def test_rank_positions_ties_share_the_minimum_rank() -> None:
    """A three-way tie shares the best position the group occupies.

    scores = [0.9, 0.5, 0.5, 0.5, 0.2]. The top node ranks 1; the three 0.5s
    all tie for the next positions (2, 3, 4) and so all take the minimum, 2;
    the 0.2 node follows at 5. Contrast with rank_positions, which would spread
    the tie over 2, 3, 4 by index.
    """
    scores = np.array([0.9, 0.5, 0.5, 0.5, 0.2])
    assert rank_positions_ties(scores).tolist() == [1, 2, 2, 2, 5]
    # The index-tie-broken version does spread them, for contrast.
    assert rank_positions(scores).tolist() == [1, 2, 3, 4, 5]


def test_top_k_overlap_counts_the_shared_members() -> None:
    """Two vectors sharing exactly 2 of their top 3 give 2/3.

    a top 3 (indices): {1, 2, 0}. b top 3: {1, 2, 4}. Shared: {1, 2}, so 2/3.
    """
    a = np.array([0.30, 0.90, 0.80, 0.10, 0.05])
    b = np.array([0.05, 0.90, 0.80, 0.10, 0.30])
    assert top_k_overlap(a, b, 3) == pytest.approx(2 / 3)


def test_top_k_overlap_is_one_for_the_same_top_set() -> None:
    """Identical orderings overlap completely."""
    a = np.array([0.4, 0.3, 0.2, 0.1])
    assert top_k_overlap(a, a, 2) == 1.0


def test_top_k_overlap_rejects_bad_k() -> None:
    """k must lie in 1..n and the vectors must match in length."""
    a = np.array([0.4, 0.3, 0.2])
    with pytest.raises(ValueError, match="k must be in"):
        top_k_overlap(a, a, 0)
    with pytest.raises(ValueError, match="k must be in"):
        top_k_overlap(a, a, 4)
    with pytest.raises(ValueError, match="same shape"):
        top_k_overlap(a, np.array([0.1, 0.2]), 1)


def test_kendall_tau_is_one_for_identical_orderings() -> None:
    """A vector against itself correlates perfectly."""
    a = np.array([0.1, 0.4, 0.2, 0.9, 0.3])
    assert kendall_tau(a, a) == pytest.approx(1.0)


def test_kendall_tau_is_minus_one_for_a_reversed_ordering() -> None:
    """A strictly reversed ordering anti-correlates perfectly."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([4.0, 3.0, 2.0, 1.0])
    assert kendall_tau(a, b) == pytest.approx(-1.0)


def test_kendall_tau_hand_checked_permutation() -> None:
    """A 3-element case worked out by hand.

    a = [3, 2, 1] ranks the nodes 0 > 1 > 2. b = [3, 1, 2] ranks 0 > 2 > 1.
    Of the three pairs, (0,1) and (0,2) are concordant and (1,2) is discordant,
    with no ties, so tau-b = (2 - 1) / 3 = 1/3.
    """
    a = np.array([3.0, 2.0, 1.0])
    b = np.array([3.0, 1.0, 2.0])
    assert kendall_tau(a, b) == pytest.approx(1 / 3)


def test_kendall_tau_over_a_subset() -> None:
    """The subset selects which nodes enter the correlation.

    Over all four nodes a and b disagree; restricted to the first three, where
    both order 0 > 1 > 2, they agree perfectly.
    """
    a = np.array([3.0, 2.0, 1.0, 0.0])
    b = np.array([3.0, 2.0, 1.0, 9.0])
    subset = np.array([0, 1, 2])
    assert kendall_tau(a, b, subset=subset) == pytest.approx(1.0)


def test_largest_rank_shifts_finds_the_big_mover() -> None:
    """A node that jumps from last to first is the most displaced.

    a = [6,5,4,3,2,1] ranks node i at position i+1. b = [5,4,3,2,1,6] moves
    node 5 from rank 6 to rank 1 (shift 5); every other node shifts by 1.
    So node 5 comes first; the remaining order is by ascending index, since
    they all tie at shift 1.
    """
    a = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    b = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 6.0])
    shifts = largest_rank_shifts(a, b, 3)
    assert shifts.tolist() == [5, 0, 1]


def test_largest_rank_shifts_clips_k_and_validates() -> None:
    """k larger than n returns all nodes; a negative k and a shape mismatch
    are errors."""
    a = np.array([3.0, 2.0, 1.0])
    b = np.array([1.0, 2.0, 3.0])
    assert largest_rank_shifts(a, b, 10).shape[0] == 3
    with pytest.raises(ValueError, match="k must be non-negative"):
        largest_rank_shifts(a, b, -1)
    with pytest.raises(ValueError, match="same shape"):
        largest_rank_shifts(a, np.array([1.0, 2.0]), 1)
