"""The lecture's worked PageRank examples, reproduced by the project's code.

Lesson 4 solves four PageRank examples on the slides. This module encodes each
one -- its node order, its edges as ``tail -> head``, its damping factor and the
values printed on the slide -- and checks three things against them:

* the dense construction in the course's notation
  (:func:`pagerank.pagerank_eigenvector` on ``G``) reproduces the printed
  vector to the precision the slide prints it;
* the project's sparse, matrix-free solver (:func:`pagerank.pagerank_power`)
  agrees with the dense eigenvector to ``1e-10``, which is the real point --
  the number the notebooks report comes from the sparse solver, and this is
  the evidence it computes the same thing the lecture does;
* for the first example, the lecture's own observation that ranking by
  PageRank and ranking by in-degree disagree.

The printed slide values are rounded; where the slide's last printed digit
differs from the exact vector by one unit (the 6-node ``D`` prints ``0.383``
while the exact value is ``0.38367``), the comparison against the *printed*
value carries a tolerance of one unit in the last printed place, while the
dense-versus-sparse comparison stays tight. No claim is made that a printed
digit is exact.

The exercise sheet is deliberately not touched: only the examples solved on
the lecture slides appear here.
"""

from fractions import Fraction

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


def _dense_pagerank(
    edges: list[tuple[int, int]], n: int, alpha: float
) -> np.ndarray:
    """Run the full course construction ``A -> H -> S -> G -> pi`` on an example."""
    a = adjacency_matrix(edges, n)
    h = hyperlink_matrix(a)
    s = stochastic_matrix(h)
    g = google_matrix(s, alpha)
    return pagerank_eigenvector(g)


def _sparse_pagerank(
    edges: list[tuple[int, int]], n: int, alpha: float
) -> np.ndarray:
    """Run the project's power-iteration solver on an example, tightly converged."""
    return pagerank_power(
        from_edge_list(edges, n), alpha=alpha, tol=1e-14, max_iter=5000
    ).vector


def _in_degrees(edges: list[tuple[int, int]], n: int) -> np.ndarray:
    into = np.zeros(n, dtype=np.int64)
    for _, head in edges:
        into[head] += 1
    return into


# ==========================================================================
# Example 1 -- the 4-node "random surfer, simple case", no teleportation.
#
# Nodes W X Y Z (indices 0 1 2 3). With no teleport the Google matrix is the
# stochastic matrix S itself (alpha = 1); the graph is strongly connected, so
# S is irreducible and has the exact stationary vector (4, 6, 9, 12) / 31.
# The slide also contrasts the PageRank order with the in-degree order: Y has
# in-degree 3 and leads by in-degree, while Z leads by PageRank.
# ==========================================================================

FOUR_EDGES: list[tuple[int, int]] = [
    (0, 1),  # W -> X
    (0, 2),  # W -> Y
    (1, 2),  # X -> Y
    (1, 3),  # X -> Z
    (2, 3),  # Y -> Z
    (3, 0),  # Z -> W
    (3, 1),  # Z -> X
    (3, 2),  # Z -> Y
]
FOUR_N = 4
FOUR_EXACT: tuple[Fraction, ...] = (
    Fraction(4, 31),
    Fraction(6, 31),
    Fraction(9, 31),
    Fraction(12, 31),
)


def test_four_node_exact_vector_is_a_distribution() -> None:
    """The slide's exact answer (4, 6, 9, 12)/31 sums to 1 with no slack."""
    assert sum(FOUR_EXACT) == 1


def test_four_node_dense_matches_the_exact_fractions() -> None:
    """With no teleport (alpha = 1) the dense eigenvector is (4, 6, 9, 12)/31.

    The graph is strongly connected, so S is irreducible and primitive and the
    stationary vector is exact and unique even without the teleport term.
    """
    vec = _dense_pagerank(FOUR_EDGES, FOUR_N, alpha=1.0)
    expected = np.array([float(f) for f in FOUR_EXACT])
    np.testing.assert_allclose(vec, expected, atol=1e-12)


def test_four_node_pagerank_and_indegree_rank_differently() -> None:
    """The lecture's point: in-degree and PageRank disagree on the 4-node graph.

    By in-degree Y (index 2, in-degree 3) leads; by PageRank Z (index 3) leads.
    The two full orderings are not the same.
    """
    indeg = _in_degrees(FOUR_EDGES, FOUR_N)
    pr = np.array([float(f) for f in FOUR_EXACT])

    assert int(indeg.argmax()) == 2  # Y leads by in-degree
    assert int(pr.argmax()) == 3  # Z leads by PageRank

    indeg_order = sorted(range(FOUR_N), key=lambda i: indeg[i], reverse=True)
    pr_order = sorted(range(FOUR_N), key=lambda i: pr[i], reverse=True)
    assert indeg_order != pr_order


# ==========================================================================
# Example 2 -- the 5-node "general case", alpha = 0.85.
#
# Nodes V W X Y Z (indices 0 1 2 3 4). V is dangling (no out-edge), so this
# example exercises the dangling repair. Printed PageRank, in node order:
# 0.1219 0.1219 0.1737 0.2475 0.3350.
# ==========================================================================

FIVE_EDGES: list[tuple[int, int]] = [
    (1, 2),  # W -> X
    (1, 3),  # W -> Y
    (2, 3),  # X -> Y
    (2, 4),  # X -> Z
    (3, 4),  # Y -> Z
    (4, 0),  # Z -> V
    (4, 1),  # Z -> W
    (4, 2),  # Z -> X
    (4, 3),  # Z -> Y
]
FIVE_N = 5
FIVE_ALPHA = 0.85
FIVE_PRINTED = np.array([0.1219, 0.1219, 0.1737, 0.2475, 0.3350])


# ==========================================================================
# Example 3 -- the 6-node exercise, alpha = 0.9.
#
# Nodes A..F (indices 0..5). C is dangling. Printed PageRank, in node order:
# 0.037 0.042 0.054 0.383 0.197 0.286. The exact D value is 0.38367, which
# rounds to 0.384; the slide prints 0.383, so the comparison against the
# printed value carries a one-unit tolerance in the last printed place.
# ==========================================================================

SIX_EDGES: list[tuple[int, int]] = [
    (0, 1),  # A -> B
    (0, 2),  # A -> C
    (1, 0),  # B -> A
    (1, 2),  # B -> C
    (1, 3),  # B -> D
    (3, 4),  # D -> E
    (3, 5),  # D -> F
    (4, 3),  # E -> D
    (4, 5),  # E -> F
    (5, 3),  # F -> D
]
SIX_N = 6
SIX_ALPHA = 0.9
SIX_PRINTED = np.array([0.037, 0.042, 0.054, 0.383, 0.197, 0.286])


# ==========================================================================
# Example 4 -- the 8-node exercise, alpha = 0.9.
#
# Nodes A..H (indices 0..7). Printed PageRank, in node order:
# 0.023 0.023 0.212 0.108 0.157 0.212 0.108 0.157.
# ==========================================================================

EIGHT_EDGES: list[tuple[int, int]] = [
    (0, 1),  # A -> B
    (0, 2),  # A -> C
    (1, 0),  # B -> A
    (1, 5),  # B -> F
    (2, 3),  # C -> D
    (2, 4),  # C -> E
    (3, 2),  # D -> C
    (3, 4),  # D -> E
    (4, 2),  # E -> C
    (5, 6),  # F -> G
    (5, 7),  # F -> H
    (6, 5),  # G -> F
    (6, 7),  # G -> H
    (7, 5),  # H -> F
]
EIGHT_N = 8
EIGHT_ALPHA = 0.9
EIGHT_PRINTED = np.array(
    [0.023, 0.023, 0.212, 0.108, 0.157, 0.212, 0.108, 0.157]
)


# --------------------------------------------------------------------------
# The three-decimal and four-decimal examples, parametrised.
#
# `printed_atol` is one unit in the last printed place plus a hair, so a slide
# that rounded its last digit the other way (the 6-node D) still passes against
# the *printed* value. The dense-versus-sparse check is independent and tight.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "edges, n, alpha, printed, printed_atol",
    [
        pytest.param(FIVE_EDGES, FIVE_N, FIVE_ALPHA, FIVE_PRINTED, 5e-5,
                     id="5-node-alpha0.85"),
        pytest.param(SIX_EDGES, SIX_N, SIX_ALPHA, SIX_PRINTED, 1.5e-3,
                     id="6-node-alpha0.9"),
        pytest.param(EIGHT_EDGES, EIGHT_N, EIGHT_ALPHA, EIGHT_PRINTED, 5e-4,
                     id="8-node-alpha0.9"),
    ],
)
def test_dense_matches_the_printed_slide_values(
    edges: list[tuple[int, int]],
    n: int,
    alpha: float,
    printed: np.ndarray,
    printed_atol: float,
) -> None:
    """The dense construction reproduces each slide's printed PageRank vector."""
    vec = _dense_pagerank(edges, n, alpha)
    np.testing.assert_allclose(vec, printed, atol=printed_atol)


@pytest.mark.parametrize(
    "edges, n, alpha, printed, printed_atol",
    [
        pytest.param(FIVE_EDGES, FIVE_N, FIVE_ALPHA, FIVE_PRINTED, 5e-5,
                     id="5-node-alpha0.85"),
        pytest.param(SIX_EDGES, SIX_N, SIX_ALPHA, SIX_PRINTED, 1.5e-3,
                     id="6-node-alpha0.9"),
        pytest.param(EIGHT_EDGES, EIGHT_N, EIGHT_ALPHA, EIGHT_PRINTED, 5e-4,
                     id="8-node-alpha0.9"),
    ],
)
def test_sparse_solver_matches_the_printed_slide_values(
    edges: list[tuple[int, int]],
    n: int,
    alpha: float,
    printed: np.ndarray,
    printed_atol: float,
) -> None:
    """The project's sparse solver reproduces each slide's printed vector."""
    vec = _sparse_pagerank(edges, n, alpha)
    np.testing.assert_allclose(vec, printed, atol=printed_atol)


@pytest.mark.parametrize(
    "edges, n, alpha",
    [
        pytest.param(FIVE_EDGES, FIVE_N, FIVE_ALPHA, id="5-node-alpha0.85"),
        pytest.param(SIX_EDGES, SIX_N, SIX_ALPHA, id="6-node-alpha0.9"),
        pytest.param(EIGHT_EDGES, EIGHT_N, EIGHT_ALPHA, id="8-node-alpha0.9"),
    ],
)
def test_sparse_solver_agrees_with_the_dense_eigenvector(
    edges: list[tuple[int, int]], n: int, alpha: float
) -> None:
    """Sparse solver and dense eigenvector agree to 1e-10 on every example.

    This is the load-bearing check: the dense construction is the course's
    definition written out, and the sparse, matrix-free solver is what every
    reported number actually comes from. Their agreement to 1e-10 is the
    evidence the two compute the same thing.
    """
    dense = _dense_pagerank(edges, n, alpha)
    sparse = _sparse_pagerank(edges, n, alpha)
    np.testing.assert_allclose(sparse, dense, atol=1e-10)
