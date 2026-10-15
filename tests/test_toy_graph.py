"""Reference toy graphs for validating the PageRank implementation.

This module is the numerical ground truth of the project. It contains graphs
small enough to solve by hand, their exact PageRank vectors as rationals, and
tests that the ground truth itself is correct -- independently of the code in
``src/pagerank``.

Why bother proving the fixture? Because a test is only as trustworthy as its
expected values. If the constants below were wrong, every later test would
happily confirm a wrong implementation.

Two fixtures live here, for two different jobs:

``TOY_EDGES``
    Every node has at least one out-link, so the transition matrix is
    stochastic and the standard fixed point applies with no special cases.
    Fully derived and cross-checked.

``DANGLING_EDGES``
    Contains a node with no out-links. Project 3.4 requires dangling-node
    treatment, so the suite has to exercise it; the first fixture deliberately
    cannot. Its expected vectors are still ``TODO`` -- see below.

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


def _in_degrees(edges: list[tuple[int, int]], n: int) -> list[int]:
    into = [0] * n
    for _, head in edges:
        into[head] += 1
    return into


def _dangling_nodes(edges: list[tuple[int, int]], n: int) -> list[int]:
    """Nodes with no outgoing edge, in ascending order."""
    out_deg = _out_degrees(edges, n)
    return [node for node in range(n) if out_deg[node] == 0]


# --------------------------------------------------------------------------
# Second fixture: a graph that does have a dangling node
# --------------------------------------------------------------------------
#
# Dangling graph (directed, 4 nodes, node 3 dangling)::
#
#         0 ---> 1 ---> 3        node 3 has no outgoing edge
#         |      |      ^
#         |      v      |
#         +----> 2 -----+
#
#     edges: 0->1, 0->2, 1->2, 1->3, 2->3
#     out-degrees: 0:2, 1:2, 2:1, 3:0
#     in-degrees:  0:0, 1:1, 2:2, 3:2
#
# Two properties are load-bearing and neither is accidental:
#
# * **Node 3 is a genuine sink, not an isolated node.** It has no out-links but
#   two in-links, so probability mass flows into it and then has nowhere to go.
#   An isolated node would also be dangling, but it would never accumulate the
#   mass that makes the leak observable, so it would not exercise the
#   treatment. The first fixture already has the isolated-ish case covered from
#   the other side (its node 3 has no *in*-links).
# * **Node 0 is unreachable.** It receives nothing through any edge, so
#   whatever rank it ends up with is precisely the floor the convention
#   imposes: the teleport mass plus whatever share of the leaked mass the
#   convention hands it. That makes node 0 a direct read-out of which
#   convention is in force, and it should be last in the ranking under all of
#   them.
#
# Without a dangling-node convention the row-normalised transition matrix is
# only *substochastic*: the row for node 3 is all zeros, mass leaks out, and
# the iteration converges to something that does not sum to 1. Each convention
# repairs that differently, so there is no convention-independent ground truth
# here. Any derived vector must name its convention, and the obvious trap is
# deriving under one and testing an implementation that uses another.
#
# Note, for the uniform teleport vector used throughout this project, that
# redistributing the leaked mass *uniformly* and redistributing it *along the
# teleport vector* are the same operation. They only diverge under personalised
# PageRank, which belongs to extension 7. So this fixture distinguishes two
# distinct answers, not three.

DANGLING_EDGES: list[tuple[int, int]] = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
DANGLING_N = 4

# ==========================================================================
# TODO(derivation): the two vectors below are NOT filled in yet, on purpose.
#
# Deriving them by hand at ALPHA = 17/20 is part of the assessed work, and the
# entire value of TOY_PAGERANK_EXACT is that it was proved before any solver
# existed. A constant produced by anything other than that derivation would
# look identical and be worth nothing, so these stay empty until the hand
# derivation is done and independently checked.
#
# What to fill in, in exact Fraction arithmetic, for each convention:
#
#   REDISTRIBUTE   repair the matrix first: every dangling row is replaced by
#                  the teleport distribution, making it stochastic, and then
#                  the ordinary fixed point p = (1-a)/n + a P^T p applies.
#                  Result sums to exactly 1.
#
#   RENORMALISE    leave the matrix substochastic, iterate, and rescale to
#                  unit L1 norm. The fixed point is the normalised dominant
#                  eigenvector of the damped substochastic operator, which is
#                  NOT the same vector as above.
#
# Then delete the skips on the tests at the bottom of this module.
# ==========================================================================

#: Exact PageRank of DANGLING_EDGES at ALPHA, dangling mass redistributed.
DANGLING_PAGERANK_EXACT_REDISTRIBUTE: tuple[Fraction, ...] | None = None  # TODO

#: Exact PageRank of DANGLING_EDGES at ALPHA, substochastic then renormalised.
DANGLING_PAGERANK_EXACT_RENORMALISE: tuple[Fraction, ...] | None = None  # TODO


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
# Structural tests for the dangling fixture
#
# These assert the shape of the graph, not its PageRank, so they run now and
# are independent of the derivation still outstanding above. Their job is to
# stop the fixture from quietly losing the property it exists for.
# --------------------------------------------------------------------------


def test_first_fixture_has_no_dangling_nodes() -> None:
    """TOY_EDGES must stay dangling-free -- it is the clean baseline.

    If a future edit introduced a dangling node here, the hand-derived
    TOY_PAGERANK_EXACT would silently stop being the right answer, since it
    was derived from the stochastic fixed point with no repair term.
    """
    assert _dangling_nodes(TOY_EDGES, TOY_N) == []


def test_dangling_fixture_has_exactly_one_dangling_node() -> None:
    """Node 3, and only node 3, has no outgoing edge."""
    assert _dangling_nodes(DANGLING_EDGES, DANGLING_N) == [3]


def test_dangling_node_is_a_sink_and_not_isolated() -> None:
    """The dangling node must receive mass, or it tests nothing.

    out-degree 0 makes it dangling; in-degree > 0 is what makes the leak
    observable and therefore makes the convention matter.
    """
    out_deg = _out_degrees(DANGLING_EDGES, DANGLING_N)
    in_deg = _in_degrees(DANGLING_EDGES, DANGLING_N)

    assert out_deg[3] == 0
    assert in_deg[3] > 0


def test_dangling_fixture_has_an_unreachable_node() -> None:
    """Node 0 receives nothing, so its rank reads out the convention's floor."""
    assert _in_degrees(DANGLING_EDGES, DANGLING_N)[0] == 0


def test_dangling_fixture_is_a_well_formed_simple_digraph() -> None:
    """No self-loops, no duplicate edges, all endpoints in range."""
    assert len(set(DANGLING_EDGES)) == len(DANGLING_EDGES)
    for tail, head in DANGLING_EDGES:
        assert tail != head
        assert 0 <= tail < DANGLING_N
        assert 0 <= head < DANGLING_N


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


def test_dangling_ground_truth_is_internally_consistent() -> None:
    """Proof obligation for the dangling fixture, mirroring the one above.

    Once DANGLING_PAGERANK_EXACT_REDISTRIBUTE is derived, this is where it gets
    proved rather than trusted, in Fraction arithmetic and with zero tolerance:

    * sums to exactly 1
    * satisfies the fixed point of the *repaired* matrix, where node 3's row is
      the teleport distribution
    * puts node 0 last, at exactly the floor value the convention implies

    Same obligation for DANGLING_PAGERANK_EXACT_RENORMALISE under its own
    operator, and the two vectors must come out different -- if they agree,
    one of the two derivations has used the wrong convention.
    """
    if DANGLING_PAGERANK_EXACT_REDISTRIBUTE is None:
        pytest.skip("dangling ground truth not derived yet (see TODO above)")


def test_power_iteration_handles_the_dangling_node() -> None:
    """Contract this test will enforce once the solver is written:

    * agrees with the derived vector for the convention it implements, to
      ~1e-10 at a tight tolerance
    * returns a vector summing to 1 even though the raw matrix is
      substochastic -- the leak must be repaired, not ignored
    * exposes the convention as an explicit argument, so both vectors above
      can be reproduced from the same code path
    * ranks node 0 last

    Wire the call below to the final signature and delete the skip.
    """
    pytest.importorskip("pagerank")
    pytest.skip("dangling-node treatment not implemented yet")
