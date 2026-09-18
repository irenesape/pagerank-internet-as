"""PageRank on the AS-level Internet topology.

Reusable, notebook-independent code for the INE course project. Everything
numerically relevant is implemented from scratch on top of NumPy/SciPy sparse
matrices; networkx is used only to cross-check small graphs in the test suite.

Modules are added here as they are written (graph construction, power
iteration, Gauss-Seidel, local push, metrics).
"""

__version__ = "0.1.0"

__all__: list[str] = []
