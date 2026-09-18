# Lab notebook

Dated log of decisions, dead ends and fixes. The course presentation asks for a
retrospective, and retrospectives written from memory in December are fiction.
This file is written as things happen.

**How to use it:** one entry per working session, newest at the bottom, dated
`YYYY-MM-DD`. Record *why* a decision was taken, not just what was decided —
the reasoning is what you will need in the presentation. When something breaks,
write down the symptom, the cause and the fix, even if the fix was trivial.
Especially when it was embarrassing.

---

## 2026-09-18 — Repository skeleton

Set up the project scaffolding. No algorithm code yet, by design: the
implementation is the graded work.

### Decisions

- **Layout: `src/` package + numbered notebooks + separate `data/` and
  `results/`.** Computation lives in `src/pagerank/` so it can be imported and
  tested; notebooks narrate and plot. Code written inside notebook cells cannot
  be unit tested or reused across notebooks, which is exactly what a project
  with three research questions needs.
- **`pyproject.toml` with editable install.** `pip install -e .` makes
  `from pagerank import ...` work from any notebook with no
  `sys.path.append`. Path hacks break the moment someone clones the repo into a
  different directory, which matters because this repo is meant to be read by
  outsiders.
- **Environment: `venv` + `requirements.txt`, not conda.** Three options were on
  the table. A *virtual environment* is an isolated interpreter plus its
  packages, so this project's numpy cannot collide with another project's.
  *`requirements.txt`* is just a text list of what to install into it — it is
  not isolation, it is the recipe. *conda* is a separate package manager that
  also installs non-Python binaries and manages the interpreter itself. conda
  earns its complexity when you need compiled dependencies outside PyPI (CUDA,
  GDAL, R). Here the whole stack ships as PyPI wheels, so `venv` is the
  standard, lighter, more portable choice, and `python -m venv` needs nothing
  installed beyond Python itself.
- **Version ranges live only in `pyproject.toml`.** `requirements.txt` reduces
  to `-e .[dev]`. Declaring the same ranges in two files guarantees they drift
  apart. Exact reproducibility comes from `requirements-lock.txt`
  (`pip freeze`), not from the ranges.
- **Licence: MIT.** Permissive, short, well understood, and it lets a reader
  reuse the code with attribution. Apache-2.0 adds a patent grant and
  change-documentation duties this project does not need; GPL-3.0 would force
  derivative works to stay GPL; CC-BY is meant for creative works and data, not
  software; no licence at all would mean nobody may legally reuse it. The MIT
  grant covers **this code only** — the CAIDA data has its own terms.
- **`data/` is git-ignored, `results/` is not.** CAIDA's acceptable use
  agreement does not grant redistribution rights, and the snapshots are large;
  git keeps every version of every file forever, so a committed dataset is
  effectively permanent. Figures and small CSV tables *are* tracked, because the
  README links them and a reader browsing on GitHub will not run anything.
- **Dataset variant: CAIDA `serial-1`.** It is the variant AS Rank is built on,
  which keeps research question 1 an apples-to-apples comparison. `serial-2`
  mixes in peering links from a different measurement pipeline.
- **Notebook `00_toy_graph.ipynb` before touching real data.** Validate on a
  4-node graph solvable by hand, then scale to 75k nodes. Debugging a wrong
  ranking on the real graph without a known-good baseline is guesswork.
- **Tests from day one, not later.** `tests/test_toy_graph.py` holds the exact
  PageRank of the toy graph as rationals: `p = (659/1769, 27713/141520,
  2789/7076, 3/80)` for `alpha = 0.85`. Verified two ways — exact rational
  arithmetic against the fixed-point equations, and a numerical cross-check
  with `networkx`. Node 3 has no in-links, so its rank must be exactly
  `(1-alpha)/n = 3/80`, which is a good by-eye sanity check. The suite is green
  now; the test that will exercise the real solver is skipped until the solver
  exists.

### Notebook output policy

Notebooks are JSON files that store their outputs inline, including plots as
base64 blobs. Every re-run rewrites `execution_count` fields and image payloads
even when nothing meaningful changed, so committed outputs make diffs
unreadable, cause merge conflicts that cannot be resolved by hand, and grow the
repository permanently.

So, two modes:

- **While developing:** clear all outputs before committing (Kernel → Restart
  Kernel and Clear All Outputs, or `nbstripout` as a pre-commit hook). Diffs
  then show only the code and prose that actually changed.
- **For the final deliverable:** commit the notebooks **fully executed**, top to
  bottom, from a single clean run (Restart Kernel and Run All). The submission
  is a narrative notebook, and most readers will open it on GitHub without
  running anything. An unexecuted notebook looks like an empty
  file to them.

Procedure for the final version: freeze the code, restart and run all, check
every figure rendered, commit as a single dedicated commit, and tag it. Do not
mix that commit with code changes.

### Open items

- **Blocked: Python 3.12 is not installed on this machine.** `python3.12` is not
  on `PATH`; Homebrew has 3.11 and 3.14, and the system ships 3.9. Cannot use
  3.14: the pinned `numpy>=1.26,<2.1` has no wheels for it and would try to
  compile from source. Options are `brew install python@3.12` (matches the
  stated target) or falling back to 3.11, which works with the current ranges.
  Environment creation is on hold until this is settled.
- Generate `requirements-lock.txt` with `pip freeze` right after the first
  successful install.
- Decide and document the dangling-node convention (ASes with no out-links).
  Deliberately left out of the toy graph so the choice is explicit and tested
  separately rather than absorbed silently into the solver.
- Fix the repository URL in `pyproject.toml` and `README.md` once the GitHub
  remote exists — currently a placeholder.
- Deadline reminder: the repository must be presentable by **30 November 2026**,
  ten days before the course presentation on 10 December. The README and
  results sections need to be finished against the November date, not the
  December one.
