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
- **Dataset variant: CAIDA `serial-1`.** Longer series (1998 onwards) and a
  single measurement pipeline, BGP table snapshots, whereas `serial-2` merges in
  peering links inferred from BGP communities and traceroute. One pipeline means
  a surprising result can be attributed to the graph rather than to the mixture
  of methods behind it. Note that the first draft justified this by claiming AS
  Rank is built on `serial-1`; the dataset page does not say that, only that the
  `serial-1` inference method resembles the one in the AS Rank paper. Claim
  withdrawn and flagged for verification in `data/README.md`, because research
  question 1 rests on it.
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

### Interpreter: Python 3.11, not 3.12 (resolved)

The plan said 3.12. The machine does not have it: `python3.12` is absent from
`PATH`, Homebrew provides 3.11 and 3.14, and the system interpreter is 3.9.
Settled on **CPython 3.11.2** (Homebrew), with the virtual environment in
`.venv/`.

- **Why not 3.12.** It would have to be installed, and it buys nothing here.
  Every dependency in this stack — numpy, scipy, pandas, matplotlib, jupyter,
  networkx, pytest — publishes arm64 wheels for 3.11 just as it does for 3.12,
  and the project uses no 3.12-only language feature. The governing principle
  is that the documented version must be the version actually used; documenting
  3.12 while running something else is exactly the kind of quiet mismatch that
  makes a repository unreproducible.
- **Why not 3.14.** numpy in the pinned range publishes no wheels for 3.14, so
  pip would fall back to building numpy from source. That needs a C and Fortran
  toolchain, takes a long time, and fails in ways unrelated to this project.
- Updated `README.md`, `requirements.txt` and `pyproject.toml` to say 3.11.

Installed and verified: numpy 2.0.2, scipy 1.17.1, pandas 2.3.3,
matplotlib 3.11.2, networkx 3.6.1, pytest 8.4.2. `import pagerank` resolves to
`src/pagerank/__init__.py` from inside `notebooks/`, confirming the editable
install works without path hacks.

First test run: **5 passed, 1 skipped**. The networkx cross-check agreed with
the hand-derived rationals to 1e-9, so the ground truth is now confirmed by two
independent routes.

### Gotcha: pip freeze and editable installs

`pip freeze` writes the editable install as an absolute path
(`-e /Users/<user>/Development/pagerank-internet-as`), which resolves
only on this laptop. A lock file containing that line is worthless to anyone
else. Fixed by generating the lock with `pip freeze --exclude-editable` and
adding a relative `-e .` by hand. Recorded in the header of
`requirements-lock.txt` so the next regeneration does not reintroduce the bug.

Division of labour between the two files, now documented in the README:
`requirements.txt` installs (ranges, newest compatible release),
`requirements-lock.txt` reproduces (exact pins plus interpreter and platform).

### Review pass: dependency bounds

The `numpy>=1.26,<2.1` range was a leftover from the interpreter discussion, not
a real constraint. Nothing in the stack asks for it: scipy 1.17.1 declares
`numpy>=1.26.4,<2.7`, and pandas, matplotlib and networkx declare floors only.
Raised to `numpy>=2.0,<3.0`.

The floor moved up as well as the cap. The old range straddled the numpy 1.x/2.x
boundary, which was a genuine break — removed scalar aliases such as
`np.float_`, changed `copy=False` semantics in `np.array`, new scalar promotion
rules under NEP 50. Claiming compatibility with both majors means testing
against both, which buys nothing for a single-author project. Bounds now sit at
major-version boundaries, which is the only place these projects allow
themselves to break API. The tight upper bound arrives transitively from scipy,
so repeating it here would only go stale.

numpy upgraded 2.0.2 → 2.4.6. Checked for ABI breakage, since scipy ships
extension modules compiled against numpy: `scipy.sparse` matrix-vector products
and `scipy.sparse.linalg.norm` behave, tests still report 5 passed and 1
skipped, and `requirements-lock.txt` was regenerated and its resolution
re-validated with `pip install --dry-run`.

Done now rather than in November on purpose: there are no published results to
invalidate yet, so this is the cheapest moment for a dependency bump.

**Gotcha.** `pip install --upgrade -r requirements.txt` did not move numpy at
all. pip's default upgrade strategy is `only-if-needed`, and numpy 2.0.2 already
satisfied the widened range, so nothing looked out of date. Upgrading a
transitive dependency requires naming it (`pip install --upgrade numpy`) or
passing `--upgrade-strategy eager`, which churns the whole tree. Worth
remembering before concluding that a widened bound had no effect.

**Verified.** `python -V` inside `.venv` reports exactly `Python 3.11.2`, which
matches the version recorded above and in the lock file header.

### Branching

`main` plus short-lived, single-purpose branches (`feat/power-iteration`,
`feat/local-push`, `exp/gauss-seidel-ordering`), merged once `pytest` is green,
with annotated tags at milestones. Self-directed pull requests only for the two
large branches, where a written summary is worth the effort; small changes go
straight to a branch and merge.

The reason is not conflict avoidance — there is only one author — but that
`main` must stay presentable at all times, since the repository is public and
`main` is what a reader sees first. gitflow with `develop` and
`release/*` is rejected: it exists to coordinate teams and versioned releases.

Note that notebooks are the worst case for branching: two branches touching the
same `.ipynb` conflict inside JSON with base64 payloads and cannot be merged by
hand. The cleared-output policy above is what keeps branches usable.

### Open items

- Decide and document the dangling-node convention (ASes with no out-links).
  Deliberately left out of the toy graph so the choice is explicit and tested
  separately rather than absorbed silently into the solver.
- Verify which AS Relationships variant the AS Rank web application is computed
  from, and which of the two citation wordings applies. Both are flagged in
  `data/README.md`; record the answers here once confirmed.
- **Possible future improvement: `pyenv`.** Deferred deliberately. It manages
  several Python versions side by side and pins one per project via a
  `.python-version` file, which is the right tool once there are projects
  demanding different interpreters. Today there is one project and one
  interpreter, so it would be a dependency to learn with no problem to solve.
  Revisit when a second project needs a different version — at that point the
  fix is `pyenv install` plus a `.python-version` file, and nothing in this
  repository has to change.
- Regenerate `requirements-lock.txt` (with `--exclude-editable`) after any
  intentional dependency upgrade, and before the final submission.
- Deadline reminder: the repository must be presentable by **25 November 2026**.
  That is the working target, deliberately well ahead of the course
  presentation on **10 December**: a repository that is only ready on the day
  leaves no room for a broken figure, a last-minute rerun, or the write-up
  reading worse than expected. The README and results sections are finished
  against 25 November.
