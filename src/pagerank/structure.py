"""Reducibility structure of a directed graph: SCCs and closed classes.

Lesson 4 explains why the Google matrix must be *primitive* for PageRank to be
unique, and names the two failure modes of the underlying chain: dangling
nodes and disconnected (reducible) structure. This module measures the second
one directly. It decomposes a :class:`~pagerank.graph.DirectedGraph` into its
strongly connected components and marks each one *closed* (a recurrent class:
no edge leaves it) or not.

In Markov-chain terms a closed SCC is a recurrent class and every other SCC is
transient: a random surfer who leaves a transient node can never return to it.
On the real AS graph this is the honest picture behind teleportation -- without
the teleport term the stationary mass concentrates on the closed class and
every transient AS gets zero; the teleport term is what makes the chain
irreducible and aperiodic, hence the answer positive and unique.

Nothing here computes PageRank; it only reports structure, using
``scipy.sparse.csgraph``. It is cheap enough to run on the full graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import connected_components

from pagerank.graph import DirectedGraph


@dataclass(frozen=True)
class GraphStructure:
    """The strongly/weakly connected decomposition of a directed graph.

    Attributes:
        n_weak_components: Number of weakly connected components (ignoring edge
            direction).
        n_strong_components: Number of strongly connected components (SCCs).
        labels: SCC label per node, int, shape ``(n,)``; nodes with the same
            label are in the same SCC.
        closed: Boolean per SCC, shape ``(n_strong_components,)``. ``True`` when
            no edge leaves that SCC (a closed / recurrent class).
    """

    n_weak_components: int
    n_strong_components: int
    labels: np.ndarray
    closed: np.ndarray

    @property
    def n_closed_classes(self) -> int:
        """How many SCCs are closed (recurrent classes)."""
        return int(self.closed.sum())

    @property
    def transient_mask(self) -> np.ndarray:
        """Per-node boolean: ``True`` when the node's SCC is not closed.

        A transient node is one a random surfer can leave and never return to.
        """
        return ~self.closed[self.labels]

    @property
    def largest_closed_class_size(self) -> int:
        """Number of nodes in the largest closed SCC, 0 if there is none."""
        if self.n_closed_classes == 0:
            return 0
        counts = np.bincount(self.labels, minlength=self.n_strong_components)
        return int(counts[self.closed].max())


def graph_structure(graph: DirectedGraph) -> GraphStructure:
    """Compute the SCC decomposition and closed classes of ``graph``.

    The graph stores ``P^T`` (entry ``[head, tail]`` for an edge
    ``tail -> head``); its transpose recovers the adjacency pattern
    ``[tail, head]``, which is what ``connected_components`` needs. An SCC is
    *closed* when no edge crosses from it to a node in a different SCC: taking
    the edge endpoints ``(tail, head)``, an SCC is open as soon as one of its
    edges lands in another SCC.

    Args:
        graph: The directed graph to analyse.

    Returns:
        A :class:`GraphStructure` with weak- and strong-component counts, the
        per-node SCC labels, and the closed/open flag per SCC.

    Raises:
        ValueError: If ``graph.n == 0``.
    """
    n = graph.n
    if n == 0:
        raise ValueError("cannot analyse the structure of an empty graph (n == 0)")

    # transpose is P^T with a non-zero at [head, tail]; transposing it back
    # gives the adjacency pattern (non-zero at [tail, head]).
    adjacency = graph.transpose.T.tocoo()

    n_strong, labels = connected_components(
        adjacency, directed=True, connection="strong"
    )
    n_weak, _ = connected_components(
        adjacency, directed=True, connection="weak"
    )

    # An SCC is closed unless one of its edges crosses into another SCC.
    closed = np.ones(n_strong, dtype=bool)
    tail_label = labels[adjacency.row]
    head_label = labels[adjacency.col]
    crossing = tail_label != head_label
    closed[tail_label[crossing]] = False

    return GraphStructure(
        n_weak_components=int(n_weak),
        n_strong_components=int(n_strong),
        labels=labels,
        closed=closed,
    )
