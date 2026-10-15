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
---

## 2026-09-18 — First inspection of the CAIDA dataset

Downloaded the snapshot and read the format documentation before writing any
parser. The point of this pass was to find out what the file actually contains,
because most of the design decisions for the graph construction turn out to be
forced by the data rather than chosen freely.

### Data downloaded

Snapshot **20260901**, `serial-1`, the most recent available. Files now in
`data/raw/` (git-ignored, as always):

| File | Size | What it is |
| --- | --- | --- |
| `20260901.as-rel.bz2` | 1,705,904 bytes | the relationship graph |
| `20260901.ppdc-ases.bz2` | 1,709,605 bytes | customer cones per AS |
| `caida_serial-1_README.txt` | — | the README from the `serial-1` directory |

The customer cone file changes the plan for research question 1: it is the
reference ranking, used instead of the AS Rank API. That removes a network
dependency and a moving target from the comparison, since the cones ship
alongside the exact snapshot the PageRank vector is computed from rather than
being fetched from a service that updates on its own schedule.

The `serial-1` README is stored next to the data on purpose. It documents the
file format, and a format description that lives only in a browser tab is a
format description that is gone by November.

**Filenames are not what the documentation says.** the downloaded files lack the .txt
component while downloading, so the files are `20260901.as-rel.bz2` rather than
`20260901.as-rel.txt.bz2`. Nothing is wrong with the contents, but the parser
must not key off the exact filename — take a path as an argument and let the
caller name it. Anything that pattern-matches `*.as-rel.txt.bz2` will silently
find no input.

Deliberately not downloaded: `all-paths` (2.4 GB of raw BGP paths) and
`as-rel.v6-stable` (IPv6 only). Both are out of scope.

### File format, per the CAIDA README

Two kinds of relationship line, pipe-separated:

```
<provider-as>|<customer-as>|-1     customer-provider
<peer-as>|<peer-as>|0              peer-to-peer
```

**The first field is the provider.** That is the whole of what the file states:
which AS is the provider and which is the customer. It says nothing about which
way a PageRank vote should run, and that direction is a modelling choice of mine,
exactly as it is for peering links.

The choice taken here is customer to provider, on the reading that a customer
paying for transit is endorsing the provider it depends on. The opposite
direction is defensible: it is the provider that supplies the customer with
global reachability, so an argument can be made that the value flows downwards
and the vote with it. The dataset settles neither reading, so this decision has
to be justified in the write-up alongside the peering one, and not presented as
a property of the data.

Given that choice, the parser has to reverse `-1` lines explicitly, since the
file lists the provider first. It needs a test on a two-node graph: a ranking
built on the reversed direction still looks entirely plausible — a hierarchy,
with a small set of ASes on top — while being upside down, so a careful read
will not catch the mistake.

Since December 2022 the file also carries `# step N:` comments interleaved
between blocks of relationships, marking which of the eight inference steps
produced each block. Two consequences:

- The parser must skip **every** line beginning with `#`, not just a header
  block. A parser that reads a fixed number of header lines and then trusts the
  rest will silently mis-parse a comment as a relationship.
- The step tags are a per-relationship confidence signal, since earlier steps
  rest on stronger evidence than later ones. Worth keeping rather than
  discarding at parse time: it gives the data-quality section of the write-up
  something concrete to say instead of generic caveats.

### What is in the file

| Quantity | Count | Share |
| --- | --- | --- |
| Comment lines | 175 | — |
| Relationship lines | 525,848 | 100% |
| Customer-provider (`-1`) | 166,906 | 31.7% |
| Peer-to-peer (`0`) | 358,942 | 68.3% |

The two relationship counts add up to the total exactly, which establishes
something useful: the third field takes no values other than `-1` and `0`. The
parser does not need a branch for sibling links or anything else in this
snapshot, and a test asserting that the counts reconcile will catch a future
snapshot that introduces one.

Peering dominates at better than two to one. That is consistent with the
documented flattening of the Internet topology and the weight of internet
exchange points, so it reads as a property of the graph rather than a parsing
mistake.

It also has an uncomfortable design consequence. For 68.3% of the links the data
does not supply a direction: a `p2p` line records that two ASes exchange traffic
as equals, not that either confers importance on the other. So on the majority
of edges the direction is my modelling choice, not a fact from the dataset. That
has to be stated as such in the write-up, and the ranking should be recomputed
under the alternative choice to show how much of the result the choice is
responsible for.

### Header: input clique and IXPs

The header declares an **input clique of 18 ASNs** that CAIDA forces as the
transit-free clique at the top of the hierarchy:

```
174 209 286 701 1299 2828 2914 3257 3320 3356 3491
5511 6453 6461 6762 6830 7018 12956
```

This is the sanity check for the ranking. These are the ASes the inference
method itself treats as the top of the topology, so if PageRank does not put
them near the top, the bug is in my code and not an interesting finding. Useful
precisely because it is available before any result exists.

It also declares **28 ASNs belonging to internet exchange points**. IXP ASes
accumulate very large numbers of peering links by construction, which is a
structural artefact of how exchanges work rather than evidence of commercial
centrality, so they are a plausible source of distortion in the ranking.

The `source:` lines confirm the five-day observation window, 20260901 to
20260905, and name the RouteViews collectors used. That is the provenance detail
that makes the snapshot citable, so it goes in the write-up alongside the
citation.

### Pending measurements

Not measured yet, and deliberately left blank rather than estimated:

- Number of distinct ASes in the file.
- Maximum ASN, which decides whether an explicit ASN-to-index dictionary is
  needed or whether ASNs can be used directly as matrix indices. The tradeoff is
  memory against a layer of indirection: dense indexing over a sparse ASN space
  wastes rows, while a dictionary costs a lookup on every edge insertion.

### Open items

- **Direction of peering links.** Count them in both directions, or exclude them
  from the graph entirely? This governs 68.3% of the edges, so it is the single
  most consequential modelling decision in the project. Whichever way it goes,
  report the ranking under both.
- **Inversion of customer-provider links.** Confirm against the `serial-1`
  README and document that the vote runs from customer to provider, given that
  the file lists the provider first. Cover it with a test on a two-node graph
  where the two directions give visibly different answers.
- **The 28 IXP ASes.** Keep them or exclude them, and justify the choice. They
  are structurally unlike ordinary ASes, so leaving them in without comment is
  itself an unstated decision.
- **Dangling nodes** (ASes with no out-links), already open from the setup entry.
  Whether any exist in this snapshot depends on how the two decisions above
  resolve — excluding peering links would leave every AS with no providers
  dangling — so it cannot be settled independently of them.
---

## 2026-10-15 — Project 3.4: the assignment becomes concrete

The course changed the rules. When this repository was set up, the programming
project was free-topic and proposed by email. On 1 October the lecturer
published new guidelines with a **closed list of fifteen projects**, one of
which has to be chosen. Reserved:

> **Project 3.4 — PageRank on a Non-Web Graph (e.g., Social Network)**

The topic survives intact, which is the good news and the reason none of the
scaffolding is wasted. The AS-level Internet topology is a non-web graph by any
reading. But the brief attaches four specific requirements, and the repository
as written did not reflect them, which is a worse problem than it sounds: a
document that describes a different project than the one being graded is a
document that misleads the person writing the code.

### What the brief requires

1. Obtain a dataset of a network that is not the web, and model it as a
   directed graph.
2. Apply an **own implementation** of PageRank **with dangling-node
   treatment**. Project 3.2 (*Handling Dangling Nodes*) is folded into 3.4 as
   standard, so this is not a refinement one may skip.
3. Interpret the top-ranked nodes: what does a high PageRank mean here?
4. Compare the ranking against **in-degree** and discuss the differences.

And a line in the rubric that is worth more than it looks: **complexity is not
rewarded for its own sake.** A simple, correct, well-evaluated implementation
scores higher than an elaborate one that is badly explained. That sentence
reorders the whole project. The instinct behind the original three research
questions was to show range — Gauss-Seidel, local push, a convergence study —
and under this rubric range bought on the quality of the explanation is a net
loss.

### Promoted from open item to requirement

**Dangling nodes.** Previously an open item, deferred on the reasoning that it
could not be settled until the peering-direction and IXP decisions were made.
That reasoning is still technically right and now irrelevant: the dependency
does not excuse the delay, because the requirement has a name in the brief and
has to be answered whichever way the other two go. It is being promoted to a
core research question in its own right, and the write-up will have to show the
ranking under more than one convention rather than quietly pick the convenient
one.

There is an awkward consequence for the test suite. The toy graph was built
*deliberately* without dangling nodes, so that the convention would be chosen
explicitly and tested separately instead of being absorbed into the solver. That
was the right call, and it means the current ground truth does not cover what is
now a graded requirement. A second fixture is needed — a small graph with at
least one node having no out-links — added in the same working session as this
entry.

**In-degree.** Did not appear anywhere in the repository and is now the
mandatory baseline. This is the most useful thing the new brief contributes,
and that deserves saying plainly rather than being treated as an imposition:
in-degree is the correct comparison to be forced into. PageRank's whole claim
is that *who* points at you matters, not only *how many*. If the two orderings
agree on the AS graph, the eigenvector bought nothing and that is a real finding;
if they disagree, every disagreement is a concrete case to explain, and
explaining cases is exactly what question 3 asks for. Cheap to compute, too —
it is a row-sum of the matrix the solver already needs, so the baseline costs
essentially nothing beyond the discussion it enables.

It is now the central core question, and notebook `02` has been renamed from
`02_ranking_vs_asrank.ipynb` to `02_ranking_and_indegree.ipynb` to say so.
Naming a notebook after the optional comparison while the required one is
absent was the clearest symptom of the drift.

### Demoted from requirement to extension

**The customer cone comparison.** It was research question 1. It is now
extension 5, in a new notebook `05_customer_cone_comparison.ipynb`.

Nothing about it got worse — it is still the most interesting comparison in the
project, because the cone is an independent expert ranking built by people who
know the topology, whereas in-degree is a one-line baseline. But it is not what
the brief asks for, and it carries the one unresolved external dependency in
the project: whether AS Rank is computed from `serial-1` or `serial-2` (still
open, flagged in `data/README.md` since 18 September). A required result must
not rest on a question that might be answered by an email that never arrives.
In-degree has no such exposure: same file, same graph, same code.

So the demotion is really a de-risking. The cone comparison keeps its place as
the extension most worth doing, and it is the last to be cut, precisely because
it feeds back into core question 3 — it is evidence for or against my
interpretation of the top of the ranking.

### The extensions, and the order they get cut in

Convergence and local push are now extensions 6 and 7. Marked cuttable in the
README, with the order written down: **7 first, then 6, then 5.**

Writing the order down now, while there is no schedule pressure, is the point of
the exercise. Decided under pressure in late November, the thing that gets cut
is whatever happens to be unfinished, which is not the same as the thing that
matters least.

- **Local push (7) goes first.** Largest implementation effort, most delicate to
  get right, and the furthest from anything the brief mentions. It also answers
  a question about approximation algorithms rather than about the Internet,
  which makes it the hardest to fit into the narrative the submission is graded
  on.
- **Convergence (6) goes second.** Much cheaper, since iteration counts and
  residual histories fall out of the solver the core already requires, so it may
  well survive on economics alone. Still a question about the method, not the
  graph.
- **Customer cone (5) goes last**, for the reason given above.

Note that none of this removes the `exp/gauss-seidel-ordering` branch idea from
the September entry, but it does reclassify it: that is an extension of an
extension, and it only gets written if everything else is already finished and
written up.

### What changed in the repository

Documentation only. No algorithm code was written in this session, deliberately
— the implementation is the graded work and stays mine.

- `README.md`: Project 3.4 and its four requirements recorded under *Academic
  context*; research questions split into **Core** (the four requirements, with
  PageRank vs in-degree as the central one) and **Extensions** (cone,
  convergence, push) with the cut order stated; a new subsection justifying
  CAIDA against the SNAP datasets the brief suggests; layout tree updated.
- `notebooks/`: `02` renamed with `git mv` so the history follows the file,
  `05_customer_cone_comparison.ipynb` created. Both still empty.
- `data/README.md`: notes that in-degree is the required baseline and is derived
  from `as-rel` itself, so the open AS Rank provenance question now affects only
  the optional extension.
- `tests/test_toy_graph.py`: second fixture with a dangling node (see below).

### Why CAIDA and not SNAP, written down at last

The brief suggests SNAP as the dataset source, and a social network as the
example graph. This justification existed in the Plan of Attack but had never
made it into the repository, which is exactly the kind of reasoning that is
impossible to reconstruct in December. Four grounds, now in the README:

- **Size.** ~75k nodes, ~526k relationships: big enough that sparse storage and
  iteration counts matter, small enough to solve precisely and inspect by hand.
- **Natural direction.** The graph is directed by its economics rather than by a
  convention imposed afterwards. A customer-provider link is asymmetric because
  dependency and money run one way. Many SNAP social graphs are undirected and
  have to be directed artificially, which makes the modelling step a formality
  instead of a finding.
- **Real dangling nodes.** Now a requirement, and they occur naturally here
  rather than having to be manufactured.
- **An independent expert ranking exists.** The customer cones ship with the
  same snapshot. Most SNAP graphs come with no reference ordering at all, which
  would leave core question 3 with nothing external to check against.

Added the caveat that matters for the grade: the code is dataset-agnostic once
the edge list is loaded. Everything in `src/pagerank/` will work on a node-index
map plus an edge list, so a SNAP graph is a different loader and nothing else.
CAIDA supplies the data; it does not shape the implementation.

### A fixture for dangling nodes

`tests/test_toy_graph.py` gains `DANGLING_EDGES` — `0->1, 0->2, 1->2, 1->3,
2->3` — a 4-node graph where node 3 has no out-links but two in-links. The
in-links are the point: an isolated node would also be dangling, but it would
never accumulate the mass that makes the leak observable, so it would not
exercise the treatment at all.

Node 0 is unreachable, which gives a second useful handle. It receives nothing
through any edge, so whatever rank it ends up with is exactly the floor its
convention imposes — teleport mass plus whatever share of the leaked mass the
convention hands it. That makes node 0 a direct read-out of which convention is
in force, and it should come last in the ranking under either of them.

The exact PageRank constants are **left as `TODO`** on purpose. Deriving them by
hand is part of the graded work and the whole value of the first fixture is that
its expected values were proved, twice, before any solver existed. A fixture
whose constants were produced by something other than that derivation would look
identical and be worth nothing. The tests that would consume the constants are
skipped until they are filled in, and the ones that can already run — structural
assertions about which nodes dangle — do run.

Worth stating the subtlety the fixture exists to expose: with a dangling node
present the raw transition matrix is only **substochastic** — node 3's row is
all zeros, so mass leaks and the iteration converges to something that does not
sum to 1. Each convention repairs that differently, so there is **no
convention-independent ground truth** here. Whatever gets derived has to name
its convention, and the obvious trap is deriving the vector under one convention
and testing an implementation that uses another.

One correction to what I assumed while writing this. *(Half right. Superseded
by the second correction at the end of this entry — left standing because the
way it was half right is the instructive part.)* The three conventions do
not give three different answers: with the **uniform** teleport vector this
project uses, redistributing the leaked mass uniformly and redistributing it
along the teleport vector are *the same operation*. They only diverge under
personalised PageRank, which is extension 7. So the fixture distinguishes two
answers — redistribute versus renormalise — and the write-up should say that
rather than implying three independent choices. Noting it here because it is the
kind of thing that reads as sloppiness in a presentation if the count is wrong.

### Open items

Carried forward, with status under the new brief:

- **Dangling-node convention.** Core question 2, and mostly *resolved* by the
  equivalence proved in the correction below: under uniform teleportation the
  standard treatments all land on the same vector, so there is **one** answer
  to derive and one ranking to report, not several. What is left is
  presentation rather than decision — derive `DANGLING_PAGERANK_EXACT` by hand
  for the repaired matrix, implement that repair, and make the equivalence
  itself part of the written answer instead of comparing rankings that cannot
  differ. The one thing still worth measuring on the real graph: how many ASes
  actually dangle, and how much mass they hold, since that is what decides
  whether the treatment matters at all here.
- **Direction of peering links** (68.3% of edges) and **inversion of
  customer-provider links**: unchanged, and now explicitly part of core
  question 1, which asks which choices are facts from the data and which are
  mine.
- **The 28 IXP ASes.** Unchanged, and relevant to core question 4: an IXP AS
  accumulates peering links by construction, so it is a prime candidate for a
  large PageRank/in-degree disagreement. Useful rather than merely awkward.
- **Which AS Relationships variant AS Rank uses**, and the citation wording.
  Downgraded: affects extension 5 only. Partly resolved already — the shipped
  `caida_serial-1_README.txt` gives Form A's wording *and* points at
  `catalog.caida.org/dataset/as_relationships_serial_1` for citation
  information, which is the Form B domain. Citing both costs nothing and the
  file in `data/raw/` is the authority, so this is close to closed.
- **Distinct AS count and maximum ASN**, still unmeasured, still blocking the
  dense-versus-dictionary indexing decision. First thing to do with real data.
- Schedule unchanged: presentable by **25 November 2026**, presentation
  **10 December**. The new brief does not move either date, but it does mean the core is now a smaller,
  better-defined target than the three original research questions were — which
  is the first genuinely reassuring thing about the change.

### Correction, same day — the dangling conventions are all the same convention

The note earlier in this entry got half of it right and stated the other half
with more confidence than it deserved. Recording the whole thing rather than
quietly fixing the files, because the mistake is more useful to the
presentation than the fix is.

**What I assumed.** That after collapsing "redistribute uniformly" and
"redistribute along the teleport vector" into one operation, two genuinely
different answers were left: repair the matrix, or leave it substochastic and
renormalise. I wrote two `TODO` constants on that basis, and — worse — a
docstring instructing that the two derived vectors **must come out different**,
with a note that agreement would mean one of the derivations had used the wrong
convention.

**Why it is false.** Let `P` be the row-normalised transition matrix, with zero
rows at the dangling nodes; `v` the teleport vector; `alpha` the damping
factor; and `d` the 0/1 indicator vector of dangling rows. The repaired matrix
is `P' = P + d v^T`, so `P'^T = P^T + v d^T`, and its fixed point satisfies

```
y = (1-alpha) v + alpha P'^T y
  = (1-alpha) v + alpha P^T y + alpha v (d^T y)
  = [(1-alpha) + alpha m] v + alpha P^T y        with m = d^T y, a scalar
```

while the substochastic fixed point satisfies

```
x = (1-alpha) v + alpha P^T x
```

Both are of the form `z = k v + alpha P^T z`, whose solution is
`z = k (I - alpha P^T)^-1 v`. Same operator, same direction `v`; the two
differ **only** in the scalar `k`. So `y` and `x` are parallel, and since
`||y||_1 = 1` by construction, `y` is exactly `x` normalised — which is what
the renormalisation convention computes.

Note what the proof does *not* assume: nothing about `v`. The equivalence
`repair == renormalise` holds for any teleport vector, biased or not. The
uniform assumption is only needed for the other half, `redistribute uniformly
== redistribute along v`, which is the part I had already got right. Put
together: with the uniform teleportation this project uses, **all three options
collapse to one answer, not two.** Under a biased `v` there would be two, and
they would split along a different line than the one I drew.

**How it was verified.** Exact rational arithmetic over 300 random graphs with
dangling nodes, varying `n`, `alpha` and `v` across uniform and biased: zero
discrepancies. On `DANGLING_EDGES` at `alpha = 17/20` the two constructions
agree bit for bit. That check now lives in the suite as
`test_dangling_conventions_agree_exactly`, built both ways in `Fraction`
arithmetic with an exact equality assertion — no tolerance. It depends on no
hand-derived constant, so it runs today, and it is a better test than the one
it replaces: it demonstrates *why* the conventions coincide rather than
asserting in a comment that they do not.

**What actually does give a different vector:** renormalising at *every
iteration* instead of once at the end. Rescaling inside the recursion makes it
non-linear, so the argument above does not apply and the iteration converges
somewhere else. That is generally regarded as an implementation slip rather
than a named convention, and it is worth knowing mainly as a diagnosis: if the
solver ever disagrees with `DANGLING_PAGERANK_EXACT`, this is the first thing
to check, and "it implements a different convention" is not an available
excuse.

**Why this was not a wording problem.** Three concrete things were wrong, not
one sentence:

- Two `TODO` constants that would have been filled with identical numbers, so
  the second carried no information and the work of deriving it twice would
  have produced nothing.
- A proof obligation pointing the wrong way. A correct hand derivation would
  have produced two equal vectors, the docstring would have called that a
  convention error, and the most likely response to that would have been to go
  looking for a bug in a correct derivation. A test that fires on correct work
  is worse than no test.
- The claim that node 0, being unreachable, reads out which convention is in
  force. It does not: it holds the same value under treatments that produce the
  same vector. Node 0 is still useful — it sits at exactly the floor the model
  imposes and must rank last, which is the by-eye check at the bottom end, the
  mirror of node 3 in the first fixture — but it discriminates nothing.

**What it changes for the write-up, which is the real point.** Core question 2
was framed as "what changes in the ranking between conventions", and the
answer is now known in advance: nothing does, and that is provable in four
lines. That is a better answer than a comparison table. It turns a requirement
that would have been discharged by running the solver twice into a short
argument about why running it twice is unnecessary — which is squarely what the
rubric means by preferring a simple, correct, well-explained implementation to
an elaborate one. `README.md` has been reworded so the question does not
presuppose that something changes.

Filed under things to be less sure about: the two halves of that collapse
argument needed different assumptions, and I checked one of them and assumed
the other. The half I checked was the one I had just found; the half I assumed
was the one that felt like common knowledge.

---

## 2026-10-15 — Environment check before the solver work starts

No code yet. Before writing the first module that has to be imported, the
environment was checked end to end, and it was not what the previous entries
describe.

### What broke

**The editable install stopped resolving when the repository moved.** The
suite reported `12 passed, 3 skipped` and looked healthy; the skip reasons said
`could not import 'pagerank': No module named 'pagerank'`. Cause: the
repository was moved from `~/Development/pagerank-internet-as` to
`~/Development/INE/pagerank-internet-as`, and an editable install is an
absolute path. The finder in
`.venv/lib/python3.11/site-packages/__editable__.pagerank_internet_as-0.1.0.pth`
still contains `/Users/<user>/Development/pagerank-internet-as/src`.
Fix: `pip install -e ".[dev]"` again, at the start of the next session and
before any module that imports the package. No lock regeneration: the
installed versions do not change, only the project's own path.

Two lessons. First, this is the same class of problem as the September
lock-file gotcha: anything that embeds an absolute path breaks on a move or a
clone, and an editable install is machine-local state. Second, **a skip is not
a pass.** The suite was green for weeks while the package it exists to test
could not be imported. From now on the acceptance check names the one skip
that is allowed (`dangling ground truth not derived yet`) and its exact reason,
so an import failure cannot hide behind a skip count again.

### Open items

- The author's surname is now written without its accent in `LICENSE`,
  `pyproject.toml` and `README.md`, and the university is named in English.
  Deliberate: every file a reader or a packaging tool touches stays plain
  ASCII, so nothing depends on how a terminal, a form or a citation generator
  handles accented characters. The accent is still in the history.
- Eleven commits ahead of `origin/main`, unpushed. Push before the solver work
  starts, so the history stays reviewable in small steps.
- Next, in this order: the sparse graph structure and the power method with
  dangling-node redistribution, validated on the two fixtures in notebook 00;
  then the CAIDA parser and the dangling census on the real graph; then the
  core ranking and the in-degree comparison.
