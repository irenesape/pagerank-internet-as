"""PageRank by power iteration, with dangling-node redistribution.

The iteration is the damped update

    x_new = alpha * P^T x + (alpha * m + (1 - alpha)) * v

where ``v`` is the teleport distribution, ``m`` is the probability mass sitting
on dangling nodes at this step, and ``P^T`` is the stored transpose of the
row-normalised transition matrix. The ``alpha * m`` term is the dangling-node
treatment: a dangling node's row of ``P`` is zero, so without this term its
mass would simply vanish and the iterate would stop summing to 1. Adding
``alpha * m`` back along ``v`` is algebraically identical to replacing each
dangling row of ``P`` with ``v`` (the "repair the matrix" construction), which
keeps the matrix stochastic.

The mass is redistributed *inside* the update and the vector is normalised only
once, at the end, to clear floating-point drift. It is never renormalised
inside the loop: rescaling mid-recursion makes the map non-linear and converges
to a different vector. See ``tests/test_toy_graph.py`` --
``test_dangling_conventions_agree_exactly`` proves repairing-then-solving and
solving-then-normalising coincide, while the module docstring there records
that per-iteration rescaling does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pagerank.graph import DirectedGraph


@dataclass(frozen=True)
class PageRankResult:
    """Outcome of a power-iteration run.

    Attributes:
        vector: PageRank vector, float64, shape ``(n,)``, sums to 1.
        iterations: Number of update steps performed.
        residuals: L1 norm ``||x_{k+1} - x_k||_1`` for each step performed, in
            order. One entry per iteration.
        converged: True if the run stopped because the residual fell below
            ``tol``; False if it hit ``max_iter`` first.
        alpha: The damping factor used.
    """

    vector: np.ndarray
    iterations: int
    residuals: tuple[float, ...]
    converged: bool
    alpha: float


def _as_distribution(
    candidate: np.ndarray | None,
    n: int,
    name: str,
) -> np.ndarray:
    """Return a validated probability vector of length ``n``.

    ``None`` yields the uniform distribution. Otherwise the vector must be
    non-negative and sum to 1 within 1e-12; it is normalised if the sum is
    within tolerance of 1, and rejected otherwise.
    """
    if candidate is None:
        return np.full(n, 1.0 / n, dtype=np.float64)

    vec = np.asarray(candidate, dtype=np.float64)
    if vec.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {vec.shape}")
    if np.any(vec < 0):
        raise ValueError(f"{name} must be non-negative")

    total = float(vec.sum())
    if abs(total - 1.0) > 1e-12:
        raise ValueError(f"{name} must sum to 1 (sums to {total!r})")
    return vec / total


def pagerank_power(
    graph: DirectedGraph,
    alpha: float = 0.85,
    *,
    tol: float = 1e-10,
    max_iter: int = 1000,
    teleport: np.ndarray | None = None,
    x0: np.ndarray | None = None,
) -> PageRankResult:
    """Compute PageRank of ``graph`` by the damped power method.

    Args:
        graph: The directed graph; its ``transpose`` is ``P^T``.
        alpha: Damping factor, ``0 <= alpha < 1``.
        tol: Convergence threshold on the L1 residual between iterates.
        max_iter: Maximum number of iterations before giving up (returns
            ``converged=False``; does not raise).
        teleport: Teleport distribution ``v``; uniform ``1/n`` when ``None``.
        x0: Initial iterate; the teleport distribution when ``None``.

    Returns:
        A :class:`PageRankResult`. ``vector`` sums to 1.

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

    dangling = graph.dangling
    pt = graph.transpose

    residuals: list[float] = []
    converged = False
    for _ in range(max_iter):
        m = float(x[dangling].sum())
        x_new = alpha * (pt @ x) + (alpha * m + (1.0 - alpha)) * v
        r = float(np.abs(x_new - x).sum())
        residuals.append(r)
        x = x_new
        if r < tol:
            converged = True
            break

    # Analytically the iterate sums to 1 at every step (that is the point of
    # the dangling repair), so this only removes floating-point drift. If the
    # sum has drifted meaningfully the update is wrong, so check first. An
    # explicit raise rather than an assert: asserts are stripped under -O,
    # and this is an invariant of the library, not a debugging aid.
    total = float(x.sum())
    if abs(total - 1.0) >= 1e-9:
        raise RuntimeError(
            f"iterate mass drifted to {total!r}; the dangling update is wrong"
        )
    x = x / total

    return PageRankResult(
        vector=x,
        iterations=len(residuals),
        residuals=tuple(residuals),
        converged=converged,
        alpha=alpha,
    )
