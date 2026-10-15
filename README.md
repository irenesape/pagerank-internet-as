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

**This is not the web graph.** Its nodes are network operators, not pages, and
its edges are commercial agreements, not hyperlinks; the web PageRank was
invented for is content carried *over* this topology, one layer up. The point
is worth stating because "the Internet" and "the web" are used
interchangeably in ordinary speech, and the brief asks for a non-web graph. By
that criterion the AS topology is further from the web graph than the social
networks the brief offers as examples: a social graph is still an application
populated by users, whereas this is the routing and commercial substrate
underneath.

### Why CAIDA and not a SNAP dataset

The project brief suggests [SNAP](https://snap.stanford.edu/data/) as the source
for a non-web graph. The AS topology was chosen instead, on four grounds:

- **Size.** ~75,000 nodes and ~526,000 relationships: large enough that sparse
  storage and iteration counts genuinely matter, small enough to solve to high
  precision on a laptop and to inspect node by node.
- **Natural direction.** The graph is directed by its economics, not by a
  convention imposed after the fact. A customer-provider link is asymmetric
  because money and dependency flow one way, which is the property PageRank
  needs.
- **Real dangling nodes.** ASes with no out-links occur in the data rather than
  having to be constructed, so the dangling-node treatment is exercised by the
  dataset itself instead of by a synthetic corner case.
- **An independent expert ranking exists.** CAIDA publishes customer cone sizes
  for the same snapshot, which gives the interpretation of the top-ranked nodes
  something external to be checked against. Most SNAP graphs ship no reference
  ordering at all.

None of this is a dependency of the code. Everything in `src/pagerank/`
operates on an edge list plus a node-index map, so a SNAP graph can be fed to
the same solvers by swapping the loader. CAIDA supplies the data; it does not
shape the implementation.

## Research questions

The first block is the set of requirements the project brief imposes. The second
is extra work built on top of them.

### Core

1. **Modelling.** How is a dataset of a non-web network — the commercial
   relationships between autonomous systems — turned into a directed graph
   suitable for PageRank? Which direction does a link's vote run in, and which
   of those choices are facts from the data rather than modelling decisions?
2. **Dangling nodes.** Which ASes end up with no out-links under the chosen
   edge direction, how much probability mass do they hold, and what has to be
   done about the fact that the transition matrix is substochastic without
   them? Several treatments are in circulation — replace each dangling row
   with the teleport distribution, replace it with the uniform distribution,
   or leave the matrix as it is and renormalise the solution. Do they actually
   disagree? Showing *whether* the choice matters, and why, is part of the
   answer rather than a preliminary to it.
3. **Interpretation.** What does a high PageRank *mean* for an AS? Which
   networks does it put on top, and is that consistent with what the topology
   is known to look like — in particular with the 18-ASN transit-free clique
   that CAIDA's own inference forces?
4. **PageRank versus in-degree.** How does the PageRank ordering differ from
   simply ranking ASes by in-degree, and what explains the disagreements? This
   is the central question of the project: in-degree is the baseline PageRank
   has to justify itself against, since any node-importance claim that
   in-degree already captures is not worth an eigenvector computation.

### Extensions

Above the brief, and explicitly cuttable. The rubric notes that complexity is
not rewarded for its own sake — a simple, correct, well-evaluated
implementation scores better than an elaborate one that is poorly explained —
so these are only worth doing once the core is finished and written up.

5. **Customer cone comparison.** How does the PageRank ordering compare with
   [CAIDA AS Rank](https://asrank.caida.org/), which orders ASes by customer
   cone size? Where the two disagree, why?
6. **Convergence.** How many iterations do the power method and Gauss-Seidel
   need as a function of the damping factor `alpha`, and does the observed rate
   match the theoretical bound of `alpha` per iteration?
7. **Local computation.** How much accuracy does the local "push" algorithm for
   personalised PageRank give up in exchange for touching only a fraction of
   the graph?

**Order of removal if the schedule tightens: 7, then 6, then 5.** Local push
goes first — it is the largest piece of implementation and the one least
connected to anything the brief asks for. Convergence goes second; it is
cheaper, since the iteration counts fall out of the solver the core already
needs, but it answers a question about the method rather than about the graph.
The customer cone comparison is the last to go, because it is the only extension
that feeds back into a core requirement: an independent expert ranking is
evidence for or against the interpretation in question 3.

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
Polytechnic University of Catalonia (UPC). Worth 30% of the course grade. Delivered
as a narrative Jupyter notebook and presented on 10 December 2026.

The work is **Project 3.4: PageRank on a Non-Web Graph** from the closed list of
fifteen projects published with the course guidelines. The brief sets four
mandatory requirements:

1. Obtain a dataset of a network that is **not** the web and model it as a
   directed graph.
2. Apply an **own implementation** of PageRank, **including the treatment of
   dangling nodes** — Project 3.2 (*Handling Dangling Nodes*) is folded into
   3.4 as standard, so this is a requirement and not an optional refinement.
3. Interpret the top-ranked nodes: explain what a high PageRank means in this
   particular context.
4. Compare the resulting ranking against **in-degree** and discuss the
   differences.

The rubric states that complexity is not rewarded in itself: a simple, correct
and well-evaluated implementation scores higher than a complex one that is
badly explained. The research questions above are organised accordingly — the
four requirements as Core, everything else marked as an Extension.

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
│   ├── 00_toy_graph.ipynb                 hand-checkable graph, validates the code
│   ├── 01_data_preparation.ipynb          parsing, cleaning, sparse construction
│   ├── 02_ranking_and_indegree.ipynb      core: questions 1-4
│   ├── 03_convergence.ipynb               extension: question 6
│   ├── 04_local_push.ipynb                extension: question 7
│   └── 05_customer_cone_comparison.ipynb  extension: question 5
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

Three tests report as skipped: two target the solver, which is not written yet,
and one targets the dangling-node ground truth, which is still being derived by
hand. The rest validate the fixtures and the exact values the solver will be
checked against.

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

_To be completed. This section will summarise the answers to the four core
research questions first — the top-ranked ASes and their interpretation, the
dangling-node treatment, and the PageRank-versus-in-degree comparison — with
figures from `results/figures/` and the ranking tables from `results/tables/`.
Any extension that survives the schedule is reported after them, clearly
separated from the required work._

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
