"""Tests for the dense, course-notation PageRank construction.

These check the structural properties the lecture's chain ``A -> H -> S -> G``
must have (shapes, the stochastic row sums, strict positivity of ``G``,
eigenvalue 1) and that the dense eigenvector agrees with the project's sparse
solver on a small graph. The lecture's specific worked examples and their
printed values are checked separately in :mod:`tests.test_course_examples`.
"""

from __future__ import annotations

import numpy as np
import pytest

from pagerank.dense import (
    adjacency_matrix,
    google_matrix,
    hyperlink_matrix,
    pagerank_eigenvector,
    stochastic_matrix,
)
from pagerank.graph import from_edge_list
from pagerank.power import pagerank_power
from tests.test_toy_graph import (
    ALPHA,
    DANGLING_EDGES,
    DANGLING_N,
    TOY_EDGES,
    TOY_N,
    TOY_PAGERANK,
)


def test_adjacency_matrix_shape_and_entries() -> None:
    """A[i, j] is 1 exactly on the edges i -> j, in row convention."""
    a = adjacency_matrix(TOY_EDGES, TOY_N)
    assert a.shape == (TOY_N, TOY_N)
    for tail, head in TOY_EDGES:
        assert a[tail, head] == 1.0
    assert a.sum() == len(TOY_EDGES)  # no duplicates in the fixture


def test_adjacency_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        adjacency_matrix([(0, 5)], 4)


def test_hyperlink_rows_are_stochastic_except_dangling() -> None:
    """Non-dangling rows of H sum to 1; a dangling row is all zero."""
    a = adjacency_matrix(DANGLING_EDGES, DANGLING_N)
    h = hyperlink_matrix(a)
    out_degree = a.sum(axis=1)
    for i in range(DANGLING_N):
        if out_degree[i] > 0:
            assert h[i].sum() == pytest.approx(1.0)
        else:
            assert np.all(h[i] == 0.0)  # node 3 is dangling


def test_stochastic_matrix_rows_sum_to_one() -> None:
    """S repairs the dangling row so every row sums to 1."""
    a = adjacency_matrix(DANGLING_EDGES, DANGLING_N)
    s = stochastic_matrix(hyperlink_matrix(a))
    assert np.allclose(s.sum(axis=1), 1.0)
    # The repaired (dangling) row is uniform 1/N.
    assert np.allclose(s[3], 1.0 / DANGLING_N)


def test_google_matrix_rows_sum_to_one_and_strictly_positive() -> None:
    """G is row-stochastic and, for 0 < alpha < 1, strictly positive."""
    a = adjacency_matrix(DANGLING_EDGES, DANGLING_N)
    g = google_matrix(stochastic_matrix(hyperlink_matrix(a)), alpha=0.85)
    assert np.allclose(g.sum(axis=1), 1.0)
    assert np.all(g > 0.0)


def test_pagerank_eigenvector_is_a_distribution() -> None:
    """The dominant left eigenvector is non-negative and sums to 1."""
    a = adjacency_matrix(TOY_EDGES, TOY_N)
    g = google_matrix(stochastic_matrix(hyperlink_matrix(a)), alpha=float(ALPHA))
    pi = pagerank_eigenvector(g)
    assert pi.sum() == pytest.approx(1.0)
    assert np.all(pi >= -1e-15)


def test_pagerank_eigenvector_rejects_non_stochastic() -> None:
    """A matrix whose dominant eigenvalue is not 1 is rejected."""
    with pytest.raises(ValueError):
        pagerank_eigenvector(np.array([[2.0, 0.0], [0.0, 0.5]]))


def test_dense_eigenvector_matches_sparse_solver_on_toy_graph() -> None:
    """The dense eigenvector equals the sparse power-method vector to 1e-10."""
    a = adjacency_matrix(TOY_EDGES, TOY_N)
    g = google_matrix(stochastic_matrix(hyperlink_matrix(a)), alpha=float(ALPHA))
    pi = pagerank_eigenvector(g)
    sparse = pagerank_power(
        from_edge_list(TOY_EDGES, TOY_N), alpha=float(ALPHA), tol=1e-13
    ).vector
    assert np.allclose(pi, sparse, atol=1e-10)
    # And both match the hand-derived fixture.
    assert np.allclose(pi, TOY_PAGERANK, atol=1e-10)


def test_dense_matches_sparse_on_the_dangling_fixture() -> None:
    """Dangling repair in the dense chain agrees with the sparse solver."""
    a = adjacency_matrix(DANGLING_EDGES, DANGLING_N)
    g = google_matrix(stochastic_matrix(hyperlink_matrix(a)), alpha=float(ALPHA))
    pi = pagerank_eigenvector(g)
    sparse = pagerank_power(
        from_edge_list(DANGLING_EDGES, DANGLING_N), alpha=float(ALPHA), tol=1e-13
    ).vector
    assert np.allclose(pi, sparse, atol=1e-10)
