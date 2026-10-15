"""Tests for :mod:`pagerank.power`.

The power solver is validated against the hand-derived toy vector, against
networkx on the dangling fixture, and against two structural facts: a directed
cycle is uniform at every alpha, and the residual contracts at a rate bounded
by alpha. The dangling fixture also checks that mass is conserved despite the
raw matrix being substochastic.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from pagerank.graph import DirectedGraph, from_edge_list
from pagerank.power import pagerank_power

from test_toy_graph import (
    ALPHA,
    CYCLE_EDGES,
    CYCLE_N,
    DANGLING_EDGES,
    DANGLING_N,
    TOY_EDGES,
    TOY_N,
    TOY_PAGERANK,
)


@pytest.mark.parametrize("alpha", [0.5, 0.85, 0.99])
def test_directed_cycle_is_uniform_for_every_alpha(alpha: float) -> None:
    """A 3-cycle is vertex-transitive, so PageRank is uniform at any alpha."""
    graph = from_edge_list(CYCLE_EDGES, CYCLE_N)
    result = pagerank_power(graph, alpha=alpha, tol=1e-13)

    assert result.vector == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-10)
    assert result.vector.sum() == pytest.approx(1.0, abs=1e-12)


def test_toy_graph_matches_the_hand_derived_vector() -> None:
    """TOY_EDGES agrees with the exact fixture to 1e-10 at a tight tolerance."""
    graph = from_edge_list(TOY_EDGES, TOY_N)
    result = pagerank_power(graph, alpha=float(ALPHA), tol=1e-13)

    assert result.vector == pytest.approx(TOY_PAGERANK, abs=1e-10)
    assert result.converged


def test_dangling_graph_agrees_with_networkx() -> None:
    """DANGLING_EDGES agrees with networkx, sums to 1, ranks node 0 last.

    networkx handles dangling nodes by the same repair this solver uses, so
    agreement to 1e-9 confirms the dangling term is correct.
    """
    nx = pytest.importorskip("networkx")

    graph = from_edge_list(DANGLING_EDGES, DANGLING_N)
    result = pagerank_power(graph, alpha=float(ALPHA), tol=1e-13)

    reference = nx.DiGraph()
    reference.add_nodes_from(range(DANGLING_N))
    reference.add_edges_from(DANGLING_EDGES)
    nx_pr = nx.pagerank(reference, alpha=float(ALPHA), tol=1e-12)
    expected = [nx_pr[node] for node in range(DANGLING_N)]

    assert result.vector == pytest.approx(expected, abs=1e-9)
    assert result.vector.sum() == pytest.approx(1.0, abs=1e-12)
    # Node 0 is unreachable, so it sits at the model's floor and ranks last.
    assert int(np.argmin(result.vector)) == 0


def test_residuals_are_positive_and_last_is_below_tol_when_converged() -> None:
    """Each residual is positive; the final one is below tol on convergence."""
    graph = from_edge_list(TOY_EDGES, TOY_N)
    result = pagerank_power(graph, alpha=float(ALPHA), tol=1e-12)

    assert result.converged
    assert all(r > 0 for r in result.residuals)
    assert result.residuals[-1] < 1e-12
    assert len(result.residuals) == result.iterations


def test_max_iter_cap_returns_not_converged() -> None:
    """Hitting max_iter yields converged=False with that many residuals."""
    graph = from_edge_list(TOY_EDGES, TOY_N)
    result = pagerank_power(graph, alpha=float(ALPHA), tol=1e-13, max_iter=3)

    assert not result.converged
    assert result.iterations == 3
    assert len(result.residuals) == 3


def test_residual_contracts_at_a_rate_bounded_by_alpha() -> None:
    """The L1 residual shrinks by at most about alpha per step.

    The damped operator is a contraction with factor alpha, so late residual
    ratios approach alpha from below. Allow a small slack for the transient.
    """
    alpha = float(ALPHA)
    graph = from_edge_list(TOY_EDGES, TOY_N)
    result = pagerank_power(graph, alpha=alpha, tol=1e-14, max_iter=200)

    residuals = result.residuals
    for k in range(2, len(residuals) - 1):
        if residuals[k] == 0:
            break
        ratio = residuals[k + 1] / residuals[k]
        assert ratio <= alpha + 0.05, f"ratio {ratio} at step {k}"


def test_explicit_uniform_teleport_reproduces_the_default() -> None:
    """Passing the uniform vector as teleport equals the None default."""
    graph = from_edge_list(TOY_EDGES, TOY_N)
    uniform = np.full(TOY_N, 1.0 / TOY_N)

    default = pagerank_power(graph, alpha=float(ALPHA), tol=1e-13)
    explicit = pagerank_power(
        graph, alpha=float(ALPHA), tol=1e-13, teleport=uniform
    )

    assert explicit.vector == pytest.approx(default.vector, abs=1e-14)


def test_bad_teleport_raises() -> None:
    """A teleport vector that is not a distribution is rejected."""
    graph = from_edge_list(TOY_EDGES, TOY_N)

    with pytest.raises(ValueError, match="sum to 1"):
        pagerank_power(graph, teleport=np.array([0.5, 0.1, 0.1, 0.1]))

    with pytest.raises(ValueError, match="non-negative"):
        pagerank_power(graph, teleport=np.array([1.2, -0.2, 0.0, 0.0]))

    with pytest.raises(ValueError, match="shape"):
        pagerank_power(graph, teleport=np.array([0.5, 0.5]))


def test_alpha_out_of_range_raises() -> None:
    """alpha must lie in [0, 1)."""
    graph = from_edge_list(TOY_EDGES, TOY_N)

    with pytest.raises(ValueError, match="alpha"):
        pagerank_power(graph, alpha=1.0)
    with pytest.raises(ValueError, match="alpha"):
        pagerank_power(graph, alpha=-0.1)


def test_empty_graph_raises() -> None:
    """n == 0 has no distribution to return."""
    graph = from_edge_list([], 0)
    with pytest.raises(ValueError, match="empty graph"):
        pagerank_power(graph)


def test_mass_drift_raises_instead_of_being_normalised_away() -> None:
    """An inconsistent graph must fail loudly, not return a plausible vector.

    Built by hand so that the degree arrays contradict the matrix: edge 0->1
    is in ``transpose``, but ``out_degree`` claims node 0 has no out-links, so
    the solver also counts node 0's mass as dangling and redistributes it a
    second time. One step from the uniform start then sums to
    ``1 + alpha * 0.5 = 1.425``. Normalising that away at the end would hide
    the bug, which is why the check is an explicit raise and not an assert
    that ``python -O`` would remove.
    """
    transpose = sp.csr_matrix(np.array([[0.0, 0.0], [1.0, 0.0]]))
    inconsistent = DirectedGraph(
        n=2,
        transpose=transpose,
        out_degree=np.array([0, 0], dtype=np.int64),
        in_degree=np.array([0, 1], dtype=np.int64),
        n_duplicates_dropped=0,
        n_self_loops_dropped=0,
    )

    with pytest.raises(RuntimeError, match="mass drifted to 1.425"):
        pagerank_power(inconsistent, alpha=0.85, max_iter=1)
