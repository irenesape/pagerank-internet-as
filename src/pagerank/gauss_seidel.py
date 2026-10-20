"""PageRank by Gauss-Seidel on the linear system, with dangling treatment.

The power method iterates the Markov chain. Gauss-Seidel instead solves the
*linear* form of PageRank directly. With uniform teleport ``v``, damping
``alpha`` and the substochastic transpose ``P^T`` (dangling columns zero,
stored in :attr:`DirectedGraph.transpose`), the PageRank vector is the fixed
point of

    x = (1 - alpha) v + alpha P^T x        i.e.   (I - alpha P^T) x = (1 - alpha) v

Write ``A = I - alpha P^T``. Gauss-Seidel sweeps the entries in index order and
uses each freshly computed value as soon as it is available:

    x_i  <-  ( (1 - alpha) v_i + alpha * sum_{j != i} PT[i, j] x_j )
             / (1 - alpha * PT[i, i])

``PT[i, i]`` is zero here because :func:`from_edge_list` drops self-loops, so
the denominator is 1; the code keeps it so the update stays correct if
self-loops are ever retained.

**How the sweep is evaluated.** Split ``A = (D - L) - U`` with ``D - L`` its
lower triangle including the diagonal and ``U`` the strictly upper part (sign
flipped). One Gauss-Seidel sweep is then the single lower-triangular solve

    x_new = (D - L)^{-1} ( (1 - alpha) v + U x_old )

which produces exactly the iterates of the scalar loop above -- the lower
entries are resolved inside the forward substitution, the upper entries come
from ``x_old``. Here ``M = D - L = tril(A)`` and ``U = alpha * triu(P^T, 1)``.
``M`` and ``U`` are built once; each sweep is one
:func:`scipy.sparse.linalg.spsolve_triangular`.

**Dangling nodes.** ``P^T`` is substochastic (a dangling node contributes a
zero column), so this solve leaks mass by design and its solution does *not*
sum to 1. Normalising it once at the end is exactly the dangling treatment:
``tests/test_toy_graph.py`` (``test_dangling_conventions_agree_exactly``)
proves that the substochastic solution normalised to unit L1 norm equals the
PageRank vector with dangling mass redistributed along ``v``. That is why the
stopping rule runs on the un-normalised sweeps and the single division happens
after the loop, never inside it.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve_triangular

from pagerank.graph import DirectedGraph
from pagerank.power import PageRankResult, _as_distribution


def pagerank_gauss_seidel(
    graph: DirectedGraph,
    alpha: float = 0.85,
    *,
    tol: float = 1e-10,
    max_iter: int = 1000,
    teleport: np.ndarray | None = None,
    x0: np.ndarray | None = None,
) -> PageRankResult:
    """Compute PageRank of ``graph`` by Gauss-Seidel on the linear system.

    Solves ``(I - alpha P^T) x = (1 - alpha) v`` by Gauss-Seidel sweeps, each a
    lower-triangular solve (see the module docstring for the derivation and for
    why the dangling treatment is a single final normalisation). The signature,
    validation and return type match :func:`pagerank.power.pagerank_power`, so
    the two solvers are interchangeable.

    Args:
        graph: The directed graph; its ``transpose`` is ``P^T``.
        alpha: Damping factor, ``0 <= alpha < 1``.
        tol: Convergence threshold on the L1 residual between successive sweeps.
        max_iter: Maximum number of sweeps before giving up (returns
            ``converged=False``; does not raise).
        teleport: Teleport distribution ``v``; uniform ``1/n`` when ``None``.
        x0: Initial iterate; the teleport distribution when ``None``.

    Returns:
        A :class:`PageRankResult`. ``vector`` sums to 1. ``residuals`` holds the
        L1 distance between successive (un-normalised) sweeps, one per sweep.

    Raises:
        ValueError: If ``graph.n == 0``, ``alpha`` is outside ``[0, 1)``, or
            ``teleport``/``x0`` is not a valid probability vector.
    """
    n = graph.n
    if n == 0:
        raise ValueError("cannot run PageRank on an empty graph (n == 0)")
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f"alpha must satisfy 0 <= alpha < 1, got {alpha!r}")

    v = _as_distribution(teleport, n, "teleport")
    x = _as_distribution(x0, n, "x0").copy()

    pt = graph.transpose

    # A = I - alpha P^T. Split A = (D - L) - U:
    #   M = D - L = tril(A)            (lower triangle including the diagonal)
    #   U = alpha * triu(P^T, 1)       (the strictly upper part of A, negated)
    # so one sweep is the lower-triangular solve
    #   M x_new = (1 - alpha) v + U x_old.
    identity = sp.identity(n, dtype=np.float64, format="csr")
    a = (identity - alpha * pt).tocsr()
    m = sp.tril(a, k=0, format="csr")
    upper = (alpha * sp.triu(pt, k=1)).tocsr()

    rhs_const = (1.0 - alpha) * v

    residuals: list[float] = []
    converged = False
    for _ in range(max_iter):
        rhs = rhs_const + upper @ x
        x_new = spsolve_triangular(m, rhs, lower=True)
        r = float(np.abs(x_new - x).sum())
        residuals.append(r)
        x = x_new
        if r < tol:
            converged = True
            break

    # The substochastic solve leaks mass (P^T has zero columns at dangling
    # nodes), so x does not sum to 1 here -- unlike the power method, where it
    # does at every step. Normalising once is the dangling treatment itself:
    # tests/test_toy_graph.py::test_dangling_conventions_agree_exactly proves
    # the normalised substochastic solution equals the repaired-matrix vector.
    total = float(x.sum())
    if total <= 0.0:
        raise RuntimeError(
            f"Gauss-Seidel solution has non-positive mass {total!r}"
        )
    x = x / total

    return PageRankResult(
        vector=x,
        iterations=len(residuals),
        residuals=tuple(residuals),
        converged=converged,
        alpha=alpha,
    )
