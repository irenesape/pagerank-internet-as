"""PageRank on the AS-level Internet topology.

Reusable, notebook-independent code for the INE course project. Everything
numerically relevant is implemented from scratch on top of NumPy/SciPy sparse
matrices; networkx is used only to cross-check small graphs in the test suite.

Modules are added here as they are written (graph construction, power
iteration, Gauss-Seidel, local push, metrics).

So far this provides graph construction (:class:`DirectedGraph`,
:func:`from_edge_list`), the power-iteration solver with dangling-node
treatment (:class:`PageRankResult`, :func:`pagerank_power`), deterministic
ranking helpers (:func:`ranking`, :func:`top_k`, :func:`rank_positions`,
:func:`top_k_overlap`, :func:`kendall_tau`, :func:`largest_rank_shifts`), and
reading and modelling
the CAIDA AS-relationship snapshots (:func:`read_as_rel`,
:func:`build_edge_list`, :func:`save_build`, :func:`load_build`) and the CAIDA
customer-cone sizes, both as shipped (:func:`read_ppdc_ases`) and recomputed
from the relationships (:func:`customer_cone_sizes`).
"""

from pagerank.caida import (
    C2P,
    P2P,
    AsRelSnapshot,
    CustomerCones,
    EdgeListBuild,
    build_edge_list,
    customer_cone_sizes,
    index_of,
    load_build,
    read_as_rel,
    read_ppdc_ases,
    save_build,
)
from pagerank.graph import DirectedGraph, from_edge_list
from pagerank.metrics import (
    kendall_tau,
    largest_rank_shifts,
    rank_positions,
    ranking,
    top_k,
    top_k_overlap,
)
from pagerank.power import PageRankResult, pagerank_power

__version__ = "0.1.0"

__all__: list[str] = [
    "DirectedGraph",
    "from_edge_list",
    "PageRankResult",
    "pagerank_power",
    "ranking",
    "top_k",
    "rank_positions",
    "top_k_overlap",
    "kendall_tau",
    "largest_rank_shifts",
    "C2P",
    "P2P",
    "AsRelSnapshot",
    "read_as_rel",
    "EdgeListBuild",
    "build_edge_list",
    "index_of",
    "save_build",
    "load_build",
    "CustomerCones",
    "read_ppdc_ases",
    "customer_cone_sizes",
]
