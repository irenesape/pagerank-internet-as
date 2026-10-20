"""Tests for the Gauss-Seidel PageRank solver.

Gauss-Seidel solves the linear form ``(I - alpha P^T) x = (1 - alpha) v`` of
PageRank, where the power method iterates the Markov chain. The two must agree
on the vector they return (they solve the same system); they need not take the
same number of iterations, which is itself one of the things the convergence
notebook measures.

The toy and dangling fixtures, their exact vectors and the convention proof
live in :mod:`tests.test_toy_graph`; this module imports the fixtures rather
than restating them.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve_triangular

from pagerank.gauss_seidel import pagerank_gauss_seidel
from pagerank.graph import from_edge_list
from pagerank.power import pagerank_power
from tests.test_toy_graph import (
    ALPHA,
    CYCLE_EDGES,
    CYCLE_N,
    CYCLE_PAGERANK,
    DANGLING_EDGES,
    DANGLING_N,
    TOY_EDGES,
    TOY_N,
    TOY_PAGERANK,
)


# --------------------------------------------------------------------------
# Correctness against the known vectors
# --------------------------------------------------------------------------


def test_matches_the_toy_graph_exact_vector() -> None:
    """Agrees with the hand-derived toy vector to 1e-10 at a tight tol."""
    result = pagerank_gauss_seidel(
        from_edge_list(TOY_EDGES, TOY_N), alpha=float(ALPHA), tol=1e-13
    )
    assert result.vector.sum() == pytest.approx(1.0, abs=1e-12)
    for node in range(TOY_N):
        assert result.vector[node] == pytest.approx(TOY_PAGERANK[node], abs=1e-10)


def test_three_cycle_is_uniform_for_several_alpha() -> None:
    """A directed 3-cycle is vertex-transitive: uniform rank for every alpha."""
    graph = from_edge_list(CYCLE_EDGES, CYCLE_N)
    for alpha in (0.5, float(ALPHA), 0.99):
        result = pagerank_gauss_seidel(graph, alpha=alpha, tol=1e-13)
        for node in range(CYCLE_N):
            assert result.vector[node] == pytest.approx(
                CYCLE_PAGERANK[node], abs=1e-10
            )


def test_agrees_with_power_on_the_dangling_fixture() -> None:
    """Same vector as the power method on the dangling graph, summing to 1.

    This is the dangling treatment under test: the substochastic solve
    normalised once at the end must land on the power method's repaired-matrix
    vector (``test_dangling_conventions_agree_exactly`` proves they coincide),
    and the unreachable node 0 must rank last.
    """
    graph = from_edge_list(DANGLING_EDGES, DANGLING_N)
    gs = pagerank_gauss_seidel(graph, alpha=float(ALPHA), tol=1e-13)
    power = pagerank_power(graph, alpha=float(ALPHA), tol=1e-13)

    assert gs.vector.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.abs(gs.vector - power.vector).max() < 1e-10
    assert int(gs.vector.argmin()) == 0


# --------------------------------------------------------------------------
# The sweep is genuine Gauss-Seidel, not a black-box solve
# --------------------------------------------------------------------------


def test_first_three_sweeps_equal_a_scalar_reference_loop() -> None:
    """The triangular-solve sweep reproduces the scalar Gauss-Seidel iterates.

    A ten-line scalar reference loop on the toy graph, element for element, for
    the first three sweeps. This is what makes the ``(D - L)^{-1}`` formulation
    trustworthy as *the* Gauss-Seidel method and not merely a linear solve that
    happens to reach the same fixed point.
    """
    alpha = float(ALPHA)
    graph = from_edge_list(TOY_EDGES, TOY_N)
    n = graph.n
    pt = graph.transpose.toarray()
    v = np.full(n, 1.0 / n)

    # Scalar Gauss-Seidel: sweep i in index order, use new x_j for j < i and
    # old x_j for j > i, exactly the update in the module docstring.
    x_ref = v.copy()
    sweeps_ref = []
    for _ in range(3):
        x = x_ref.copy()
        for i in range(n):
            off = sum(pt[i, j] * x[j] for j in range(n) if j != i)
            x[i] = ((1 - alpha) * v[i] + alpha * off) / (1 - alpha * pt[i, i])
        x_ref = x
        sweeps_ref.append(x_ref.copy())

    # The implementation, stopped after each of the first three sweeps.
    for k in range(1, 4):
        got = pagerank_gauss_seidel(graph, alpha=alpha, max_iter=k, tol=0.0)
        # tol=0.0 never triggers, so it runs exactly k sweeps; compare the
        # un-normalised iterate by rescaling the returned (normalised) one is
        # not possible, so instead compare against the normalised reference.
        ref = sweeps_ref[k - 1]
        np.testing.assert_allclose(got.vector, ref / ref.sum(), atol=1e-15)


# --------------------------------------------------------------------------
# Stopping behaviour and validation, mirroring the power method
# --------------------------------------------------------------------------


def test_hitting_max_iter_reports_not_converged() -> None:
    """``max_iter=2`` returns ``converged=False`` with exactly two residuals."""
    result = pagerank_gauss_seidel(
        from_edge_list(TOY_EDGES, TOY_N), alpha=float(ALPHA), max_iter=2, tol=1e-13
    )
    assert result.converged is False
    assert result.iterations == 2
    assert len(result.residuals) == 2


def test_rejects_alpha_out_of_range() -> None:
    graph = from_edge_list(TOY_EDGES, TOY_N)
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            pagerank_gauss_seidel(graph, alpha=bad)


def test_rejects_empty_graph() -> None:
    with pytest.raises(ValueError, match="empty graph"):
        pagerank_gauss_seidel(from_edge_list([], 0))


def test_rejects_a_teleport_that_is_not_a_distribution() -> None:
    graph = from_edge_list(TOY_EDGES, TOY_N)
    with pytest.raises(ValueError, match="teleport"):
        pagerank_gauss_seidel(graph, teleport=np.array([0.5, 0.2, 0.2, 0.2]))
    with pytest.raises(ValueError, match="teleport"):
        pagerank_gauss_seidel(graph, teleport=np.array([-1.0, 1.0, 0.5, 0.5]))


# --------------------------------------------------------------------------
# Convergence rate: each method contracts at the spectral radius of its own
# iteration matrix, and which one is smaller is a property of the graph and
# the node ordering, not of the method. These two tests pin that down on the
# two fixtures -- the unfavourable case (toy) and the favourable one (acyclic
# fixture in topological order) -- so the notebook's framing is backed by the
# suite rather than asserted.
# --------------------------------------------------------------------------


def test_gauss_seidel_rate_matches_its_spectral_radius_on_the_toy_graph() -> None:
    """Gauss-Seidel is slower than the power method on the toy graph, by theory.

    The power method's asymptotic rate is ``abs(lambda_2)`` of the repaired
    Google matrix; Gauss-Seidel's is ``rho(M^-1 U)`` where ``A = I - alpha P^T``
    splits as ``M = tril(A)``, ``U = -triu(A, 1)``. On the toy graph in ASN
    (index) order the Gauss-Seidel iteration matrix contracts *less* per step
    (0.668 against 0.601), so Gauss-Seidel takes more iterations, not fewer.
    This is the measured reversal of the "typically fewer sweeps" intuition:
    the advantage is a property of the ordering relative to the link structure,
    which the toy graph does not have.
    """
    alpha = float(ALPHA)
    graph = from_edge_list(TOY_EDGES, TOY_N)
    n = graph.n
    pt = graph.transpose.toarray()  # P^T, dense 4x4
    p = pt.T  # row-normalised P; the toy graph has no dangling row
    v = np.full(n, 1.0 / n)

    # Power method rate: subdominant eigenvalue of the Google matrix
    # G = alpha P + (1 - alpha) 1 v^T (no dangling repair needed here).
    google = alpha * p + (1.0 - alpha) * np.outer(np.ones(n), v)
    google_eigs = np.sort(np.abs(np.linalg.eigvals(google)))[::-1]
    lambda_2 = google_eigs[1]

    # Gauss-Seidel rate: spectral radius of M^-1 U.
    a = np.eye(n) - alpha * pt
    m = np.tril(a)
    u = -np.triu(a, 1)
    rho_gs = np.max(np.abs(np.linalg.eigvals(np.linalg.solve(m, u))))

    gs_iters = pagerank_gauss_seidel(graph, alpha=alpha, tol=1e-10).iterations
    power_iters = pagerank_power(graph, alpha=alpha, tol=1e-10).iterations

    assert rho_gs > lambda_2, (
        f"expected Gauss-Seidel rate {rho_gs:.4f} to exceed power-method rate "
        f"{lambda_2:.4f} on the toy graph"
    )
    assert gs_iters > power_iters, (
        f"Gauss-Seidel took {gs_iters} iterations, power method {power_iters}; "
        "Gauss-Seidel should be slower on this graph and ordering"
    )


def test_gauss_seidel_is_exact_after_one_sweep_on_an_acyclic_graph_in_topological_order() -> None:
    """One sweep solves an acyclic graph whose indices are a topological order.

    ``DANGLING_EDGES`` is acyclic and every edge runs tail < head, so the node
    indices already are a topological order. The Gauss-Seidel sweep is a
    lower-triangular solve, which for that ordering resolves every unknown in a
    single forward pass: the first sweep lands on the exact solution of
    ``(I - alpha P^T) x = (1 - alpha) v`` before any normalisation. The run then
    needs one more sweep to see a (zero) residual below ``tol`` and stop, so it
    converges in at most two sweeps -- the favourable case the toy graph lacks.
    """
    alpha = float(ALPHA)
    graph = from_edge_list(DANGLING_EDGES, DANGLING_N)
    n = graph.n
    pt = graph.transpose
    v = np.full(n, 1.0 / n)

    # One sweep from x0 = v, un-normalised, with the implementation's own split.
    a = (sp.identity(n, dtype=np.float64, format="csr") - alpha * pt).tocsr()
    m = sp.tril(a, k=0, format="csr")
    upper = (alpha * sp.triu(pt, k=1)).tocsr()
    x_one_sweep = spsolve_triangular(m, (1.0 - alpha) * v + upper @ v, lower=True)

    # Exact solution of the same substochastic system, un-normalised.
    x_exact = np.linalg.solve(a.toarray(), (1.0 - alpha) * v)

    np.testing.assert_allclose(x_one_sweep, x_exact, atol=1e-14)

    result = pagerank_gauss_seidel(graph, alpha=alpha, tol=1e-10)
    assert result.converged is True
    assert result.iterations <= 2, (
        f"expected convergence in at most two sweeps, took {result.iterations}"
    )
