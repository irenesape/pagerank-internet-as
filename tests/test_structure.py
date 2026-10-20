"""Tests for the SCC / closed-class structure helper.

These check :func:`pagerank.graph_structure` on graphs small enough to reason
about by eye: the two project fixtures, the lecture's 8-node example (which has
two closed classes), and a directed 3-cycle (strongly connected, nothing
transient). The expected decompositions are properties of the graphs, not of
the implementation.
"""

import numpy as np

from pagerank.graph import from_edge_list
from pagerank.structure import GraphStructure, graph_structure

# Fixtures reused from the toy-graph module, restated here so this test reads
# on its own.
TOY_EDGES = [(0, 1), (0, 2), (1, 2), (2, 0), (3, 2)]
TOY_N = 4

DANGLING_EDGES = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
DANGLING_N = 4

CYCLE_EDGES = [(0, 1), (1, 2), (2, 0)]
CYCLE_N = 3

# The lecture's 8-node example (A..H), two closed classes {C,D,E} and {F,G,H}.
EIGHT_EDGES = [
    (0, 1), (0, 2),            # A -> B, A -> C
    (1, 0), (1, 5),            # B -> A, B -> F
    (2, 3), (2, 4),            # C -> D, C -> E
    (3, 2), (3, 4),            # D -> C, D -> E
    (4, 2),                    # E -> C
    (5, 6), (5, 7),            # F -> G, F -> H
    (6, 5), (6, 7),            # G -> F, G -> H
    (7, 5),                    # H -> F
]
EIGHT_N = 8


def _closed_class_members(s: GraphStructure) -> list[set[int]]:
    """The node sets of the closed SCCs, each as a set, order-independent."""
    members: list[set[int]] = []
    for scc in range(s.n_strong_components):
        if s.closed[scc]:
            members.append({i for i in range(len(s.labels)) if s.labels[i] == scc})
    return members


def test_toy_graph_has_one_closed_class_and_one_transient_node() -> None:
    """Toy graph: {0,1,2} is the single closed class; node 3 is transient."""
    s = graph_structure(from_edge_list(TOY_EDGES, TOY_N))

    assert s.n_weak_components == 1
    assert s.n_closed_classes == 1
    assert _closed_class_members(s) == [{0, 1, 2}]
    # Node 3 reaches the core but nothing returns to it.
    np.testing.assert_array_equal(
        s.transient_mask, np.array([False, False, False, True])
    )
    assert s.largest_closed_class_size == 3


def test_dangling_fixture_makes_the_sink_its_own_closed_class() -> None:
    """Dangling graph: node 3 (the sink) is a closed class; the rest transient.

    A dangling node has no out-edge, so no edge leaves the singleton SCC {3};
    it is therefore closed on its own, and every other node leads toward it.
    """
    s = graph_structure(from_edge_list(DANGLING_EDGES, DANGLING_N))

    assert s.n_closed_classes == 1
    assert _closed_class_members(s) == [{3}]
    np.testing.assert_array_equal(
        s.transient_mask, np.array([True, True, True, False])
    )


def test_eight_node_lecture_example_has_two_closed_classes() -> None:
    """Lecture 8-node example: closed classes {C,D,E} and {F,G,H}; A,B transient."""
    s = graph_structure(from_edge_list(EIGHT_EDGES, EIGHT_N))

    assert s.n_closed_classes == 2
    members = _closed_class_members(s)
    assert {frozenset(m) for m in members} == {
        frozenset({2, 3, 4}),  # C, D, E
        frozenset({5, 6, 7}),  # F, G, H
    }
    # A and B lead into the two cores but neither core returns to them.
    transient_nodes = {i for i in range(EIGHT_N) if s.transient_mask[i]}
    assert transient_nodes == {0, 1}


def test_directed_cycle_is_one_closed_class_with_nothing_transient() -> None:
    """A 3-cycle is strongly connected: one closed class, no transient node."""
    s = graph_structure(from_edge_list(CYCLE_EDGES, CYCLE_N))

    assert s.n_strong_components == 1
    assert s.n_weak_components == 1
    assert s.n_closed_classes == 1
    assert not s.transient_mask.any()
    assert s.largest_closed_class_size == CYCLE_N


def test_labels_and_transient_mask_have_node_length() -> None:
    """Shape contract: one label and one transient flag per node."""
    s = graph_structure(from_edge_list(EIGHT_EDGES, EIGHT_N))
    assert s.labels.shape == (EIGHT_N,)
    assert s.transient_mask.shape == (EIGHT_N,)
    assert s.closed.shape == (s.n_strong_components,)
