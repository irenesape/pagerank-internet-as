"""Dense PageRank construction in the course's matrix notation.

Lesson 4 builds PageRank as a chain of dense matrices on a graph of a handful
of nodes: the adjacency matrix ``A``, the row-stochastic hyperlink matrix
``H = diag(D)^-1 A``, the stochastic matrix ``S`` that repairs dangling rows,
the Google matrix ``G = alpha S + (1 - alpha)/N 1 1^T``, and PageRank as the
dominant left eigenvector of ``G``. This module implements exactly that chain,
in the slides' *row* convention (``A[i, j] = 1`` for an edge ``i -> j``, the
vector acting on the left), so the lecture's worked examples can be reproduced
step by step and checked against the project's sparse solver.

It is for small graphs only. Every function returns a dense ``n x n`` array, so
nothing here is ever applied to the real AS graph: ``G`` alone would be
``80_510**2`` entries. The sparse, matrix-free :func:`pagerank.pagerank_power`
is what computes every reported number; this module exists to show the
construction and to validate that solver on graphs small enough to write out.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def adjacency_matrix(
    edges: Iterable[tuple[int, int]] | np.ndarray, n: int
) -> np.ndarray:
    """Dense adjacency matrix ``A`` in the course's row convention.

    ``A[i, j] = 1`` when there is an edge ``i -> j`` and 0 otherwise, so the
    out-links of node ``i`` are the non-zeros of row ``i``. This is the matrix
    the lecture calls ``A``; duplicate edges collapse to a single 1 and
    self-loops are kept as written (the lecture examples contain neither).

    Args:
        edges: Iterable of ``(tail, head)`` pairs or an ``(m, 2)`` int array,
            each read as an edge ``tail -> head``.
        n: Number of nodes; indices must lie in ``[0, n)``.

    Returns:
        A float64 array of shape ``(n, n)`` with entries 0 or 1.

    Raises:
        ValueError: If ``n < 0`` or an endpoint is outside ``[0, n)``.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    arr = np.asarray(list(edges), dtype=np.int64).reshape(-1, 2) if not isinstance(
        edges, np.ndarray
    ) else np.asarray(edges, dtype=np.int64).reshape(-1, 2)

    a = np.zeros((n, n), dtype=np.float64)
    if arr.size:
        if (arr < 0).any() or (arr >= n).any():
            raise ValueError(f"an edge endpoint is outside [0, {n})")
        a[arr[:, 0], arr[:, 1]] = 1.0
    return a


def hyperlink_matrix(a: np.ndarray) -> np.ndarray:
    """Row-stochastic hyperlink matrix ``H = diag(D)^-1 A``.

    Each row of ``A`` is divided by that node's out-degree, so a non-dangling
    row of ``H`` sums to 1 and distributes a surfer's next step uniformly over
    the out-links. A dangling node (an all-zero row of ``A``) has out-degree 0;
    dividing by ``max(out_degree, 1)`` leaves its row all zero, which is why
    ``H`` is only *sub*-stochastic in general and needs the repair in
    :func:`stochastic_matrix`. This is the lecture's ``H``.

    Args:
        a: An ``(n, n)`` adjacency matrix, as from :func:`adjacency_matrix`.

    Returns:
        A float64 ``(n, n)`` array; row ``i`` sums to 1 unless node ``i`` is
        dangling, in which case it is all zero.
    """
    out_degree = a.sum(axis=1)
    # Divide by max(out_degree, 1): dangling rows (out_degree 0) stay zero
    # rather than becoming NaN, exactly as the sub-stochastic H requires.
    denom = np.where(out_degree > 0, out_degree, 1.0)
    return a / denom[:, None]


def stochastic_matrix(h: np.ndarray) -> np.ndarray:
    """Stochastic matrix ``S = H + (1/N) w 1^T`` repairing dangling rows.

    The lecture makes ``H`` stochastic by replacing each dangling (all-zero)
    row with the uniform distribution ``1/N``: ``w`` is the 0/1 indicator of
    dangling rows and ``w 1^T`` the outer product that writes ``1/N`` into each
    such row. Every row of ``S`` then sums to 1. This is the lecture's ``S``.

    Args:
        h: An ``(n, n)`` hyperlink matrix from :func:`hyperlink_matrix`.

    Returns:
        A row-stochastic float64 ``(n, n)`` array.
    """
    n = h.shape[0]
    row_sums = h.sum(axis=1)
    w = (row_sums == 0).astype(np.float64)  # 1 for dangling rows
    return h + np.outer(w, np.full(n, 1.0 / n))


def google_matrix(s: np.ndarray, alpha: float) -> np.ndarray:
    """Google matrix ``G = alpha S + (1 - alpha)/N 1 1^T``.

    The teleportation step mixes the stochastic matrix ``S`` with the uniform
    jump ``(1/N) 1 1^T`` in proportion ``alpha`` to ``1 - alpha``. With
    ``0 < alpha < 1`` every entry of ``G`` is strictly positive, so ``G`` is
    primitive (irreducible and aperiodic); by Perron-Frobenius it then has a
    unique positive dominant left eigenvector with eigenvalue 1, which is
    PageRank. This is the lecture's ``G``.

    Args:
        s: An ``(n, n)`` stochastic matrix from :func:`stochastic_matrix`.
        alpha: Damping factor, ``0 <= alpha <= 1``.

    Returns:
        A row-stochastic float64 ``(n, n)`` array; strictly positive when
        ``0 < alpha < 1``.
    """
    n = s.shape[0]
    return alpha * s + (1.0 - alpha) / n * np.ones((n, n), dtype=np.float64)


def pagerank_eigenvector(g: np.ndarray) -> np.ndarray:
    """PageRank as the dominant left eigenvector of ``G``, normalised.

    The lecture defines PageRank as the row vector ``pi`` with ``pi G = pi``
    and ``pi 1 = 1``: the left eigenvector of ``G`` for eigenvalue 1. A left
    eigenvector of ``G`` is a right eigenvector of ``G^T``, so this takes the
    eigendecomposition of ``G^T``, selects the eigenvalue closest to 1, keeps
    the real part, fixes the sign to make the vector non-negative, and
    normalises it to sum 1. For a primitive ``G`` that eigenvalue is exactly 1
    and the vector is strictly positive (Perron-Frobenius).

    Args:
        g: An ``(n, n)`` Google matrix from :func:`google_matrix`.

    Returns:
        A float64 ``(n,)`` probability vector summing to 1.

    Raises:
        ValueError: If the dominant eigenvalue is not 1 within 1e-12, which
            means ``g`` is not a valid (primitive stochastic) Google matrix.
    """
    eigenvalues, eigenvectors = np.linalg.eig(g.T)
    # The dominant eigenvalue of a stochastic matrix is 1; pick the one nearest.
    dominant = int(np.argmin(np.abs(eigenvalues - 1.0)))
    lead = eigenvalues[dominant]
    if abs(lead.real - 1.0) > 1e-12 or abs(lead.imag) > 1e-12:
        raise ValueError(
            f"dominant eigenvalue is {lead!r}, not 1; G is not a valid "
            "primitive stochastic matrix"
        )
    vec = eigenvectors[:, dominant].real
    # Eigenvectors come back with arbitrary sign/scale; make it a distribution.
    total = vec.sum()
    if total == 0.0:
        raise ValueError("dominant eigenvector sums to zero; cannot normalise")
    return vec / total
