"""Tests for :mod:`pagerank.graph`.

These check the sparse structure itself -- shapes, dtypes, column sums of the
stored transpose, and the counting of removed self-loops and duplicates --
independently of any PageRank computation. The two hand-checked fixtures from
``tests/test_toy_graph.py`` are reused so the structure is validated on the
same graphs the solver is validated on.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from pagerank.graph import DirectedGraph, from_edge_list

from test_toy_graph import (
    DANGLING_EDGES,
    DANGLING_N,
    TOY_EDGES,
    TOY_N,
)


def test_field_shapes_and_dtypes() -> None:
    """Every field has the shape and dtype the solver relies on."""
    graph = from_edge_list(TOY_EDGES, TOY_N)

    assert isinstance(graph, DirectedGraph)
    assert graph.n == TOY_N
    assert isinstance(graph.transpose, sp.csr_matrix)
    assert graph.transpose.shape == (TOY_N, TOY_N)
    assert graph.transpose.dtype == np.float64

    assert graph.out_degree.shape == (TOY_N,)
    assert graph.in_degree.shape == (TOY_N,)
    assert graph.out_degree.dtype == np.int64
    assert graph.in_degree.dtype == np.int64


def test_degrees_match_the_toy_fixture() -> None:
    """Out- and in-degrees are counted correctly."""
    graph = from_edge_list(TOY_EDGES, TOY_N)

    # TOY_EDGES: 0->1, 0->2, 1->2, 2->0, 3->2
    assert graph.out_degree.tolist() == [2, 1, 1, 1]
    assert graph.in_degree.tolist() == [1, 1, 3, 0]
    assert graph.n_edges == len(TOY_EDGES)


def test_transpose_columns_sum_to_one_for_non_dangling_nodes() -> None:
    """Column i of P^T holds node i's out-distribution, so it sums to 1.

    The stored matrix is P^T, so a non-dangling node's out-links live in its
    *column*; those entries are the 1/out-degree weights and must sum to 1.
    """
    graph = from_edge_list(DANGLING_EDGES, DANGLING_N)
    column_sums = np.asarray(graph.transpose.sum(axis=0)).ravel()

    for node in range(DANGLING_N):
        if graph.dangling[node]:
            assert column_sums[node] == 0.0
        else:
            assert column_sums[node] == pytest.approx(1.0, abs=1e-12)


def test_dangling_mask_matches_each_fixture() -> None:
    """The dangling mask is exactly out_degree == 0 for both fixtures."""
    toy = from_edge_list(TOY_EDGES, TOY_N)
    assert toy.dangling.tolist() == [False, False, False, False]

    dang = from_edge_list(DANGLING_EDGES, DANGLING_N)
    # DANGLING_EDGES: node 3 is the only sink.
    assert dang.dangling.tolist() == [False, False, False, True]


def test_specific_transpose_entries() -> None:
    """Spot-check the 1/out-degree weights on the toy graph.

    Node 0 has out-degree 2 (0->1, 0->2), so transpose[1, 0] == transpose[2, 0]
    == 1/2. Node 2 has out-degree 1 (2->0), so transpose[0, 2] == 1.
    """
    graph = from_edge_list(TOY_EDGES, TOY_N)
    dense = graph.transpose.toarray()

    assert dense[1, 0] == pytest.approx(0.5)
    assert dense[2, 0] == pytest.approx(0.5)
    assert dense[0, 2] == pytest.approx(1.0)
    # Dangling column of DANGLING_EDGES is all zero -- checked above; here the
    # toy graph has none, so no all-zero column.


def test_self_loops_are_dropped_and_counted() -> None:
    """Self-loops are removed by default and reported, never kept."""
    edges = [(0, 1), (1, 1), (2, 2), (1, 0)]
    graph = from_edge_list(edges, 3)

    assert graph.n_self_loops_dropped == 2
    assert graph.n_edges == 2
    assert graph.out_degree.tolist() == [1, 1, 0]
    assert graph.dangling.tolist() == [False, False, True]


def test_self_loops_are_kept_when_requested() -> None:
    """With drop_self_loops=False the loop stays and is not counted dropped."""
    edges = [(0, 1), (1, 1)]
    graph = from_edge_list(edges, 2, drop_self_loops=False)

    assert graph.n_self_loops_dropped == 0
    assert graph.out_degree.tolist() == [1, 1]
    # transpose[1, 1] = 1 / out_degree[1] = 1.
    assert graph.transpose.toarray()[1, 1] == pytest.approx(1.0)


def test_duplicate_edges_are_dropped_and_counted() -> None:
    """Exact duplicate pairs are removed, not summed into a larger weight."""
    edges = [(0, 1), (0, 1), (0, 2), (0, 2), (0, 2)]
    graph = from_edge_list(edges, 3)

    # Three duplicates removed: one extra 0->1, two extra 0->2.
    assert graph.n_duplicates_dropped == 3
    assert graph.out_degree.tolist() == [2, 0, 0]
    dense = graph.transpose.toarray()
    # Weight is 1/2 each, not 2/3 or 1/5 -- duplicates were dropped, not summed.
    assert dense[1, 0] == pytest.approx(0.5)
    assert dense[2, 0] == pytest.approx(0.5)


def test_self_loops_counted_before_duplicates() -> None:
    """A duplicated self-loop counts once as a loop, not as a duplicate."""
    edges = [(0, 0), (0, 0), (0, 1)]
    graph = from_edge_list(edges, 2)

    assert graph.n_self_loops_dropped == 2
    assert graph.n_duplicates_dropped == 0
    assert graph.out_degree.tolist() == [1, 0]


def test_out_of_range_edge_raises_with_the_offending_pair() -> None:
    """An endpoint outside [0, n) is a ValueError naming the pair."""
    with pytest.raises(ValueError, match=r"\(1, 5\)"):
        from_edge_list([(0, 1), (1, 5)], 3)

    with pytest.raises(ValueError, match=r"\(-1, 0\)"):
        from_edge_list([(-1, 0)], 3)


def test_empty_graph_is_all_dangling() -> None:
    """No edges: every node is dangling, transpose is all zero."""
    graph = from_edge_list([], 3)

    assert graph.n_edges == 0
    assert graph.out_degree.tolist() == [0, 0, 0]
    assert graph.in_degree.tolist() == [0, 0, 0]
    assert graph.dangling.tolist() == [True, True, True]
    assert graph.transpose.nnz == 0


def test_array_input_matches_the_list_of_tuples() -> None:
    """An (m, 2) array builds the same graph as the equivalent list of pairs.

    The array form is the fast path used for the real 884k-edge graph; it must
    produce a DirectedGraph identical in every field to the Python-level
    intake, so switching to it changes nothing about the result.
    """
    from_list = from_edge_list(TOY_EDGES, TOY_N)
    from_array = from_edge_list(np.array(TOY_EDGES, dtype=np.int64), TOY_N)

    assert from_array.n == from_list.n
    assert from_array.out_degree.tolist() == from_list.out_degree.tolist()
    assert from_array.in_degree.tolist() == from_list.in_degree.tolist()
    assert from_array.n_self_loops_dropped == from_list.n_self_loops_dropped
    assert from_array.n_duplicates_dropped == from_list.n_duplicates_dropped
    # Same sparse transpose, entry for entry.
    assert np.array_equal(
        from_array.transpose.toarray(), from_list.transpose.toarray()
    )


def test_array_input_with_the_wrong_width_raises() -> None:
    """An array that is not (m, 2) -- e.g. (m, 3) -- is a ValueError."""
    bad = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    with pytest.raises(ValueError, match=r"shape \(m, 2\)"):
        from_edge_list(bad, 4)
