# Who Runs the Internet? PageRank on the AS-Level Internet Topology

Applying PageRank, implemented from scratch with sparse linear algebra, to the
real graph of ~75,000 autonomous systems that make up the Internet, in order to
ask which networks are structurally central and how fast the underlying
numerical methods actually converge.

## Why this graph

The Internet is not one network. It is tens of thousands of independently
operated networks (autonomous systems, ASes) glued together by commercial
agreements: an AS either pays another for transit (customer-to-provider) or
exchanges traffic with it as an equal (peer-to-peer). That web of business
relationships is a directed graph, and its structure decides who is central to
global reachability. PageRank is a natural lens on it, and unlike the web graph
it is small enough to study exactly while still being large enough that
algorithmic choices matter.

## Research questions

1. **Ranking.** Which ASes does PageRank identify as structurally central, and
   how does that ranking compare with [CAIDA AS Rank](https://asrank.caida.org/),
   which orders ASes by customer cone size? Where the two disagree, why?
2. **Convergence.** How many iterations do the power method and Gauss-Seidel
   need as a function of the damping factor `alpha`, and does the observed rate
   match the theoretical bound of `alpha` per iteration?
3. **Local computation.** How much accuracy does the local "push" algorithm for
   personalised PageRank give up in exchange for touching only a fraction of
   the graph?

## Scope and constraints

Everything numerical is implemented from scratch in Python on top of NumPy and
SciPy sparse matrices: graph construction, the power method, Gauss-Seidel, and
the local push algorithm. `networkx` appears only in the test suite, to
cross-check results on graphs small enough to verify by hand. It never produces
any number reported in the notebooks.

The emphasis is on data structures and algorithms, numerical methods, and
complexity analysis rather than on tooling.

## Academic context

Programming project for **Internet and Networked Economy (INE)**, Master's
degree in Advanced Telecommunication Technologies (MATT), taught at the ETSETB,
Universitat Politecnica de Catalunya. Worth 30% of the course grade. Delivered
as a narrative Jupyter notebook and presented on 10 December 2026.

Author: Irene Sarrias Perez.

## Data

The analysis uses the public **CAIDA AS Relationships** dataset (serial-1),
which provides monthly snapshots of inferred customer-provider and
peer-to-peer links between ASes.

The raw files are **not** redistributed in this repository: they are governed by
CAIDA's acceptable use agreement, and they are large. See
[`data/README.md`](data/README.md) for how to obtain them, the terms that apply,
and the required citation.

## Repository layout

```
├── data/
│   ├── raw/            CAIDA snapshots as downloaded (git-ignored)
│   ├── processed/      derived edge lists and index maps (git-ignored)
│   └── README.md       provenance, terms of use, citation
├── notebooks/          narrative analysis, numbered in reading order
│   ├── 00_toy_graph.ipynb           hand-checkable graph, validates the code
│   ├── 01_data_preparation.ipynb    parsing, cleaning, sparse construction
│   ├── 02_ranking_vs_asrank.ipynb   research question 1
│   ├── 03_convergence.ipynb         research question 2
│   └── 04_local_push.ipynb          research question 3
├── src/pagerank/       reusable, importable implementation
├── tests/              pytest suite, ground truth computed by hand
├── results/
│   ├── figures/        plots referenced by the notebooks and this README
│   └── tables/         small result tables (CSV)
├── LAB_NOTEBOOK.md     dated log of decisions, failures and fixes
├── pyproject.toml      package metadata, dependency ranges, pytest config
├── requirements.txt    environment definition (ranges, for installing)
└── requirements-lock.txt   exact pins (for reproducing the results)
```

The split matters: code that lives in `src/` is testable and reusable across
notebooks, whereas code written inside a notebook cell is neither. Notebooks
narrate and plot; they delegate computation.

## Reproducing

Requires **Python 3.11**, the interpreter these results were produced with.

```bash
git clone https://github.com/irenesape/pagerank-internet-as.git
cd pagerank-internet-as

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

That installs the project in editable mode, so notebooks can `from pagerank
import ...` without any path manipulation.

Verify the environment:

```bash
pytest
```

One test is reported as skipped until the solver it targets is written; the
remaining tests validate the ground truth the solver will be checked against.

### Which requirements file to use

| File | What it pins | Use it to |
| --- | --- | --- |
| `requirements.txt` | Nothing directly: it installs `-e .[dev]` and defers to the version *ranges* in `pyproject.toml` | **Install and work on the project.** pip resolves the newest compatible release of each dependency. |
| `requirements-lock.txt` | Every package, transitive dependencies included, at one exact version, with the interpreter and platform recorded in its header | **Reproduce the published results**, or bisect a discrepancy between your run and the reported numbers. |

Ranges express what the code is *compatible with*; the lock file records what
was *actually installed*. Only the second reproduces a result, which is why
both are tracked.

Then obtain the dataset as described in [`data/README.md`](data/README.md),
place it under `data/raw/`, and run the notebooks in numerical order.

## Results

_To be completed. This section will summarise the answers to the three research
questions, with figures from `results/figures/` and the ranking comparison
table._

## References

- L. Page, S. Brin, R. Motwani, T. Winograd. *The PageRank Citation Ranking:
  Bringing Order to the Web.* Stanford InfoLab, 1999.
- A. N. Langville, C. D. Meyer. *Google's PageRank and Beyond: The Science of
  Search Engine Rankings.* Princeton University Press, 2006.
- R. Andersen, F. Chung, K. Lang. *Local Graph Partitioning using PageRank
  Vectors.* FOCS, 2006.
- M. Luckie, B. Huffaker, k. claffy, A. Dhamdhere, V. Giotsas. *AS
  Relationships, Customer Cones, and Validation.* IMC, 2013.

## License

Code in this repository is released under the [MIT License](LICENSE). The CAIDA
dataset is not covered by that licence and remains subject to its own terms.
