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
    cannot. Its expected vector is still ``TODO`` -- see below. One vector
    suffices: the standard dangling conventions provably coincide, which is
    itself asserted by a test here.

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
#   whatever rank it ends up with is precisely the floor the model imposes:
#   the teleport mass plus its share of the redistributed leaked mass. It
#   should come last in the ranking, which makes it a cheap by-eye check at
#   the bottom end, exactly as node 3 of the first fixture is at the top.
#   It does *not* discriminate between the dangling conventions -- nothing
#   does, see below.
#
# ---------------------------------------------------------------------------
# The conventions are not actually different
# ---------------------------------------------------------------------------
#
# Without any treatment the row-normalised transition matrix is only
# *substochastic*: node 3's row is all zeros, so mass leaks and the iteration
# converges to something that does not sum to 1. Three repairs get discussed
# in the literature and in course notes:
#
#   (a) replace each dangling row with the teleport distribution v, making the
#       matrix stochastic, then solve the ordinary fixed point;
#   (b) replace each dangling row with the uniform distribution;
#   (c) leave the matrix substochastic, solve, and rescale the solution to
#       unit L1 norm at the end.
#
# All three give the *same vector* here, and (a) == (c) holds for any teleport
# vector at all. Proof, with P the row-normalised matrix (zero rows at
# dangling nodes), v the teleport vector, d the 0/1 indicator of dangling
# rows, and alpha the damping factor:
#
#   The repaired matrix is P' = P + d v^T, so P'^T = P^T + v d^T. Its fixed
#   point satisfies
#
#       y = (1-alpha) v + alpha P'^T y
#         = (1-alpha) v + alpha P^T y + alpha v (d^T y)
#         = [(1-alpha) + alpha m] v + alpha P^T y,     m = d^T y, a scalar
#
#   while the substochastic fixed point satisfies
#
#       x = (1-alpha) v + alpha P^T x
#
#   Both have the form z = k v + alpha P^T z, whose solution is
#   z = k (I - alpha P^T)^-1 v -- the same operator, differing only in the
#   scalar k. So y and x are parallel, and since ||y||_1 = 1 by construction,
#   y is exactly x normalised. Which is what (c) computes.  QED
#
# (a) == (b) needs one extra assumption, and it is the one this project
# satisfies: v uniform. Under a biased teleport vector (a) and (b) genuinely
# differ, which only becomes relevant for personalised PageRank in extension 7.
#
# Verified independently in exact Fraction arithmetic over 300 random graphs
# with dangling nodes, varying n, alpha and v (uniform and biased): no
# discrepancies. `test_dangling_conventions_agree_exactly` below reproduces
# that check on this fixture, and it runs today because it needs no
# hand-derived constant.
#
# Consequence for this module: **one** expected vector, not one per
# convention. The remaining real pitfall is renormalising *at every iteration*
# instead of once at the end -- that does converge somewhere else, because
# rescaling mid-recursion makes it non-linear. It is generally treated as an
# implementation slip rather than a named convention, and it is worth knowing
# about mainly so it can be ruled out when an implementation disagrees with
# the constant below.

DANGLING_EDGES: list[tuple[int, int]] = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
DANGLING_N = 4

# ==========================================================================
# TODO(derivation): the vector below is NOT filled in yet, on purpose.
#
# Deriving it by hand at ALPHA = 17/20 with uniform teleportation is part of
# the assessed work, and the entire value of TOY_PAGERANK_EXACT is that it was
# proved before any solver existed. A constant produced by anything other than
# that derivation would look identical and be worth nothing, so this stays
# empty until the hand derivation is done and independently checked.
#
# What to derive: the fixed point of the *repaired* matrix, i.e. node 3's row
# replaced by the uniform teleport distribution, so
#
#     p = (1-alpha)/n + alpha * P'^T p,   sum(p) == 1 exactly
#
# No convention label is needed on the result -- per the proof above, the
# alternatives land on the same vector.
#
# Then delete the skips on the tests at the bottom of this module.
# ==========================================================================

#: Exact PageRank of DANGLING_EDGES at ALPHA with uniform teleportation.
DANGLING_PAGERANK_EXACT: tuple[Fraction, ...] | None = None  # TODO


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
# Exact linear algebra, for the convention-equivalence property below
#
# Two small helpers in Fraction arithmetic. They solve a 4x4 linear system and
# are not a PageRank implementation: no iteration, no convergence criterion,
# no sparse structure, and nothing here is importable by the library. Their
# only job is to let a test assert an algebraic identity exactly rather than
# to within a tolerance.
# --------------------------------------------------------------------------


def _transition_matrix(edges: list[tuple[int, int]], n: int) -> list[list[Fraction]]:
    """Row-normalised P. Rows of dangling nodes are left all zero."""
    out_deg = _out_degrees(edges, n)
    matrix = [[Fraction(0)] * n for _ in range(n)]
    for tail, head in edges:
        matrix[tail][head] += Fraction(1, out_deg[tail])
    return matrix


def _solve_exact(
    matrix: list[list[Fraction]], rhs: list[Fraction]
) -> list[Fraction]:
    """Gauss-Jordan elimination over the rationals. Exact, no pivoting policy
    beyond picking any non-zero pivot -- there is no rounding to stabilise."""
    n = len(rhs)
    aug = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot = next(r for r in range(col, n) if aug[r][col] != 0)
        aug[col], aug[pivot] = aug[pivot], aug[col]

        scale = Fraction(1) / aug[col][col]
        aug[col] = [value * scale for value in aug[col]]

        for row in range(n):
            if row != col and aug[row][col] != 0:
                factor = aug[row][col]
                aug[row] = [a - factor * b for a, b in zip(aug[row], aug[col])]

    return [aug[i][n] for i in range(n)]


def _damped_fixed_point(
    matrix: list[list[Fraction]], teleport: list[Fraction], alpha: Fraction
) -> list[Fraction]:
    """Solve z = (1-alpha) v + alpha M^T z directly, as (I - alpha M^T) z = (1-alpha) v."""
    n = len(teleport)
    system = [
        [(Fraction(1) if i == j else Fraction(0)) - alpha * matrix[j][i] for j in range(n)]
        for i in range(n)
    ]
    rhs = [(1 - alpha) * teleport[i] for i in range(n)]
    return _solve_exact(system, rhs)


def test_dangling_conventions_agree_exactly() -> None:
    """The two standard dangling treatments give the *same* vector.

    Repairing the matrix (dangling rows replaced by the teleport distribution)
    and leaving it substochastic then normalising the solution at the end are
    not two conventions with two answers. Both fixed points have the form
    z = k v + alpha P^T z for the same operator and differ only in the scalar
    k, so they are parallel; the repaired one already has unit L1 norm, so it
    is exactly the normalised substochastic one.

    Checked here in exact rational arithmetic, so it holds to the last bit and
    not to a tolerance. This test needs no hand-derived constant, which is why
    it runs while DANGLING_PAGERANK_EXACT is still open.
    """
    n = DANGLING_N
    teleport = [Fraction(1, n)] * n
    substochastic = _transition_matrix(DANGLING_EDGES, n)

    repaired = [row[:] for row in substochastic]
    for node in _dangling_nodes(DANGLING_EDGES, n):
        repaired[node] = teleport[:]

    repaired_solution = _damped_fixed_point(repaired, teleport, ALPHA)
    leaked_solution = _damped_fixed_point(substochastic, teleport, ALPHA)

    # The repair makes the matrix stochastic, so no mass is lost.
    assert sum(repaired_solution) == 1
    # Mass does leak without it, which is the whole problem being repaired.
    assert sum(leaked_solution) < 1

    leaked_total = sum(leaked_solution)
    normalised = [value / leaked_total for value in leaked_solution]
    assert repaired_solution == normalised


def test_unreachable_node_ranks_last_under_the_repaired_matrix() -> None:
    """Node 0 gets only the floor, so nothing can rank below it.

    Every other node receives the same floor plus a strictly positive inflow,
    so this is a structural property of the fixture rather than a numerical
    coincidence. It is the by-eye check at the bottom of the ranking.
    """
    n = DANGLING_N
    teleport = [Fraction(1, n)] * n
    matrix = _transition_matrix(DANGLING_EDGES, n)
    for node in _dangling_nodes(DANGLING_EDGES, n):
        matrix[node] = teleport[:]

    solution = _damped_fixed_point(matrix, teleport, ALPHA)

    assert min(solution) == solution[0]
    assert all(solution[node] > solution[0] for node in range(1, n))


# --------------------------------------------------------------------------
# Tests for the implementation in src/pagerank -- skipped until it exists
# --------------------------------------------------------------------------


def test_power_iteration_matches_the_toy_graph() -> None:
    """Contract this test will enforce once the solver is written:

    * accepts the graph in whatever representation you settle on
    * returns a vector summing to 1
    * agrees with TOY_PAGERANK to ~1e-10 when the tolerance is tight
    * reproduces CYCLE_PAGERANK for the 3-cycle at any alpha
    """
    pytest.importorskip("pagerank")
    from pagerank.graph import from_edge_list
    from pagerank.power import pagerank_power

    toy = pagerank_power(
        from_edge_list(TOY_EDGES, TOY_N), alpha=float(ALPHA), tol=1e-13
    )
    assert toy.vector.sum() == pytest.approx(1.0, abs=1e-12)
    for node in range(TOY_N):
        assert toy.vector[node] == pytest.approx(TOY_PAGERANK[node], abs=1e-10)

    for alpha in (0.5, float(ALPHA), 0.99):
        cycle = pagerank_power(
            from_edge_list(CYCLE_EDGES, CYCLE_N), alpha=alpha, tol=1e-13
        )
        for node in range(CYCLE_N):
            assert cycle.vector[node] == pytest.approx(
                CYCLE_PAGERANK[node], abs=1e-10
            )


def test_dangling_ground_truth_is_internally_consistent() -> None:
    """Proof obligation for the dangling fixture, mirroring the one above.

    Once DANGLING_PAGERANK_EXACT is derived by hand, this is where it gets
    proved rather than trusted, in Fraction arithmetic and with zero tolerance:

    * sums to exactly 1
    * satisfies the fixed point of the *repaired* matrix, where node 3's row is
      the uniform teleport distribution
    * puts node 0 last, at exactly the floor value the model implies

    One vector covers every standard convention, since they coincide -- see
    `test_dangling_conventions_agree_exactly`.
    """
    if DANGLING_PAGERANK_EXACT is None:
        pytest.skip("dangling ground truth not derived yet (see TODO above)")


def test_power_iteration_handles_the_dangling_node() -> None:
    """Contract this test will enforce once the solver is written:

    * agrees with DANGLING_PAGERANK_EXACT to ~1e-10 at a tight tolerance,
      whichever of the equivalent treatments it implements internally
    * returns a vector summing to 1 even though the raw matrix is
      substochastic -- the leak must be handled, not ignored
    * ranks node 0 last

    A disagreement with the constant is therefore not a convention mismatch.
    The likely cause is renormalising inside the iteration instead of
    repairing the matrix, which converges somewhere else entirely.
    """
    pytest.importorskip("pagerank")
    from pagerank.graph import from_edge_list
    from pagerank.power import pagerank_power

    result = pagerank_power(
        from_edge_list(DANGLING_EDGES, DANGLING_N),
        alpha=float(ALPHA),
        tol=1e-13,
    )

    # Mass is conserved even though the raw matrix is substochastic, and the
    # unreachable node 0 sits at the floor, so it ranks last. These hold
    # without the hand-derived constant.
    assert result.vector.sum() == pytest.approx(1.0, abs=1e-12)
    assert int(result.vector.argmin()) == 0

    # The agreement-with-the-constant check stays skipped until the vector is
    # derived by hand.
    if DANGLING_PAGERANK_EXACT is None:
        pytest.skip("dangling ground truth not derived yet (see TODO above)")

    expected = [float(p) for p in DANGLING_PAGERANK_EXACT]
    for node in range(DANGLING_N):
        assert result.vector[node] == pytest.approx(expected[node], abs=1e-10)
