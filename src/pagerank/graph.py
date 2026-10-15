"""Sparse directed-graph structure for PageRank.

PageRank's power step is ``x <- alpha * P^T x + ...`` where ``P`` is the
row-normalised transition matrix: ``P[i, j] = 1 / out_degree(i)`` for every
edge ``i -> j``. The iteration never needs ``P`` itself, only ``P^T x``, so
this module stores the *transpose* ``P^T`` in CSR form. A CSR matrix-vector
product is the fast path, and storing the matrix the solver actually
multiplies avoids a transpose on every call.

The graph object is a faithful description of the data: dangling nodes (no
out-links) leave a zero column in ``P^T`` and nothing is written into them
here. Redistributing their mass is the solver's responsibility, not the
graph's, so that the structure can be reused unchanged for in-degree
comparisons and for any teleport convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class DirectedGraph:
    """A directed graph as the sparse structure PageRank needs.

    Attributes:
        n: Number of nodes. Node ids are ``0 .. n-1``.
        transpose: ``P^T`` in CSR form, shape ``(n, n)``. The entry
            ``transpose[j, i]`` equals ``1 / out_degree[i]`` for every edge
            ``i -> j``. Columns of dangling nodes are entirely zero. Stored
            transposed on purpose: the power step multiplies ``P^T x`` and CSR
            mat-vec on the stored matrix is the fast path.
        out_degree: Out-degree of each node after cleaning, int64, shape ``(n,)``.
        in_degree: In-degree of each node after cleaning, int64, shape ``(n,)``.
        n_duplicates_dropped: Number of exact duplicate ``(tail, head)`` pairs
            removed. Duplicates are dropped, never summed into a weight.
        n_self_loops_dropped: Number of self-loops removed.
    """

    n: int
    transpose: sp.csr_matrix
    out_degree: np.ndarray
    in_degree: np.ndarray
    n_duplicates_dropped: int
    n_self_loops_dropped: int

    @property
    def dangling(self) -> np.ndarray:
        """Boolean mask of nodes with no outgoing edge, shape ``(n,)``."""
        return self.out_degree == 0

    @property
    def n_edges(self) -> int:
        """Number of edges after removing self-loops and duplicates."""
        return int(self.out_degree.sum())


def from_edge_list(
    edges: Iterable[tuple[int, int]],
    n: int,
    *,
    drop_self_loops: bool = True,
) -> DirectedGraph:
    """Build a :class:`DirectedGraph` from an edge list and a node count.

    Args:
        edges: Any iterable of ``(tail, head)`` integer pairs, ``tail -> head``.
        n: Number of nodes. Every endpoint must satisfy ``0 <= id < n``.
        drop_self_loops: If true (default), remove edges with ``tail == head``
            and count them in ``n_self_loops_dropped``.

    Returns:
        A :class:`DirectedGraph` whose ``transpose`` is the row-normalised
        transition matrix transposed. Self-loops (optionally) and exact
        duplicate pairs are removed and counted; duplicate edges are never
        silently summed into a larger weight.

    Raises:
        ValueError: If ``n < 0``, or if any endpoint is out of range (the
            offending pair is named in the message).
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    arr = np.fromiter(
        (coord for pair in edges for coord in pair),
        dtype=np.int64,
    )
    if arr.size % 2 != 0:
        raise ValueError("edges must be an iterable of (tail, head) pairs")
    arr = arr.reshape(-1, 2)

    # Validate ranges before doing any arithmetic, naming the offending pair.
    if arr.size:
        bad = (arr < 0) | (arr >= n)
        if bad.any():
            row = int(np.argmax(bad.any(axis=1)))
            tail, head = int(arr[row, 0]), int(arr[row, 1])
            raise ValueError(
                f"edge ({tail}, {head}) has an endpoint outside [0, {n})"
            )

    # Remove self-loops first, then exact duplicate pairs; count both.
    n_self_loops_dropped = 0
    if drop_self_loops and arr.size:
        keep = arr[:, 0] != arr[:, 1]
        n_self_loops_dropped = int((~keep).sum())
        arr = arr[keep]

    n_duplicates_dropped = 0
    if arr.size:
        unique = np.unique(arr, axis=0)
        n_duplicates_dropped = int(arr.shape[0] - unique.shape[0])
        arr = unique

    tails = arr[:, 0]
    heads = arr[:, 1]

    out_degree = np.bincount(tails, minlength=n).astype(np.int64)
    in_degree = np.bincount(heads, minlength=n).astype(np.int64)

    # transpose[head, tail] = 1 / out_degree[tail]. Every surviving edge has a
    # non-dangling tail by construction, so out_degree[tails] is never zero.
    data = 1.0 / out_degree[tails].astype(np.float64)
    transpose = sp.coo_matrix(
        (data, (heads, tails)),
        shape=(n, n),
        dtype=np.float64,
    ).tocsr()

    return DirectedGraph(
        n=n,
        transpose=transpose,
        out_degree=out_degree,
        in_degree=in_degree,
        n_duplicates_dropped=n_duplicates_dropped,
        n_self_loops_dropped=n_self_loops_dropped,
    )
