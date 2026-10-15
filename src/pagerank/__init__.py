"""PageRank on the AS-level Internet topology.

Reusable, notebook-independent code for the INE course project. Everything
numerically relevant is implemented from scratch on top of NumPy/SciPy sparse
matrices; networkx is used only to cross-check small graphs in the test suite.

Modules are added here as they are written (graph construction, power
iteration, Gauss-Seidel, local push, metrics).

So far this provides graph construction (:class:`DirectedGraph`,
:func:`from_edge_list`), the power-iteration solver with dangling-node
treatment (:class:`PageRankResult`, :func:`pagerank_power`), and deterministic
ranking helpers (:func:`ranking`, :func:`top_k`).
"""

from pagerank.graph import DirectedGraph, from_edge_list
from pagerank.metrics import ranking, top_k
from pagerank.power import PageRankResult, pagerank_power

__version__ = "0.1.0"

__all__: list[str] = [
    "DirectedGraph",
    "from_edge_list",
    "PageRankResult",
    "pagerank_power",
    "ranking",
    "top_k",
]
