"""Reference toy graph for validating the PageRank implementation.

This module is the numerical ground truth of the project. It contains a graph
small enough to solve by hand, its exact PageRank vector as rationals, and
tests that the ground truth itself is correct -- independently of the code in
``src/pagerank``.

Why bother proving the fixture? Because a test is only as trustworthy as its
expected values. If the constants below were wrong, every later test would
happily confirm a wrong implementation.

Toy graph (directed, 4 nodes, no dangling nodes)::

        0 ---> 1
        |      |
        v      v
        2 <----+   <--- 3
        |
        +----> 0        (2 -> 0 closes the cycle)

    edges: 0->1, 0->2, 1->2, 2->0, 3->2
    out-degrees: 0:2, 1:1, 2:1, 3:1

Node 3 has no incoming edge, so it can only ever hold the teleport mass
(1-alpha)/n. That makes it an easy sanity check by eye. Node 2 has three
incoming edges and should rank first.

Hand derivation (alpha = 17/20, n = 4, c = (1-alpha)/n = 3/80):

    p0 = c + alpha * p2
    p1 = c + alpha * p0/2
    p2 = c + alpha * (p0/2 + p1 + p3)
    p3 = c

Substituting p2 = (p0 - c)/alpha into the third equation gives

    p0 * [1 - alpha^2 (1 + alpha)/2] = c * (1 + alpha + 2 alpha^2)
    p0 = 1977/5307 = 659/1769

and back-substitution yields the remaining three components.
"""

from fractions import Fraction

import pytest

# --------------------------------------------------------------------------
# Fixture: the toy graph and its exact PageRank vector
# --------------------------------------------------------------------------

TOY_EDGES: list[tuple[int, int]] = [(0, 1), (0, 2), (1, 2), (2, 0), (3, 2)]
TOY_N = 4
ALPHA = Fraction(85, 100)

#: Exact PageRank of TOY_EDGES at ALPHA with uniform teleportation.
TOY_PAGERANK_EXACT: tuple[Fraction, ...] = (
    Fraction(659, 1769),      # 0.372526851...
    Fraction(27713, 141520),  # 0.195823911...
    Fraction(2789, 7076),     # 0.394149236...
    Fraction(3, 80),          # 0.0375 exactly = (1-alpha)/n
)

#: Same vector as floats, for comparing against numerical implementations.
TOY_PAGERANK = tuple(float(p) for p in TOY_PAGERANK_EXACT)

#: A directed 3-cycle is vertex-transitive, so its PageRank is uniform for
#: *every* value of alpha. Useful as a first smoke test.
CYCLE_EDGES: list[tuple[int, int]] = [(0, 1), (1, 2), (2, 0)]
CYCLE_N = 3
CYCLE_PAGERANK = (1 / 3, 1 / 3, 1 / 3)


def _out_degrees(edges: list[tuple[int, int]], n: int) -> list[int]:
    out = [0] * n
    for tail, _ in edges:
        out[tail] += 1
    return out


# --------------------------------------------------------------------------
# Tests that validate the ground truth itself (no project code involved)
# --------------------------------------------------------------------------


def test_expected_vector_is_a_probability_distribution() -> None:
    """The exact solution must sum to 1 with no rounding slack at all."""
    assert sum(TOY_PAGERANK_EXACT) == 1


def test_expected_vector_satisfies_the_pagerank_fixed_point() -> None:
    """Check p = (1-a)/n + a * P^T p exactly, component by component.

    Done in rational arithmetic, so this passes with zero tolerance or not
    at all. This is what makes TOY_PAGERANK_EXACT trustworthy.
    """
    out_deg = _out_degrees(TOY_EDGES, TOY_N)
    teleport = (1 - ALPHA) / TOY_N
    p = TOY_PAGERANK_EXACT

    for node in range(TOY_N):
        inflow = sum(
            p[tail] / out_deg[tail] for tail, head in TOY_EDGES if head == node
        )
        assert p[node] == teleport + ALPHA * inflow, f"node {node} is inconsistent"


def test_node_without_incoming_edges_holds_only_teleport_mass() -> None:
    """Node 3 is unreachable, so its rank is exactly (1-alpha)/n."""
    assert TOY_PAGERANK_EXACT[3] == (1 - ALPHA) / TOY_N


def test_ranking_order_is_the_expected_one() -> None:
    """Node 2 (three in-links) first, node 3 (none) last."""
    order = sorted(range(TOY_N), key=lambda i: TOY_PAGERANK_EXACT[i], reverse=True)
    assert order == [2, 0, 1, 3]


def test_networkx_agrees_with_the_hand_computed_values() -> None:
    """Independent cross-check with a third-party implementation.

    This is the only role networkx plays in this project: validating small
    graphs. It never computes any result reported in the notebooks.
    """
    nx = pytest.importorskip("networkx")

    graph = nx.DiGraph()
    graph.add_nodes_from(range(TOY_N))
    graph.add_edges_from(TOY_EDGES)

    reference = nx.pagerank(graph, alpha=float(ALPHA), tol=1e-12)

    for node in range(TOY_N):
        assert reference[node] == pytest.approx(TOY_PAGERANK[node], abs=1e-9)


# --------------------------------------------------------------------------
# Tests for the implementation in src/pagerank -- skipped until it exists
# --------------------------------------------------------------------------


def test_power_iteration_matches_the_toy_graph() -> None:
    """Contract this test will enforce once the solver is written:

    * accepts the graph in whatever representation you settle on
    * returns a vector summing to 1
    * agrees with TOY_PAGERANK to ~1e-10 when the tolerance is tight
    * reproduces CYCLE_PAGERANK for the 3-cycle at any alpha

    Wire the call below to your final signature and delete the skip.
    """
    pytest.importorskip("pagerank")
    pytest.skip("power iteration not implemented yet")
