# PageRank on the AS-level Internet graph

**Irene Sarrias Perez**

Internet and Networked Economy, MATT, ETSETB, UPC. Programming assignment, Project 3.4: PageRank on a non-web graph.

<!-- cell -->

## 0. Summary

- The CAIDA AS Relationships snapshot of 1 September 2026 gives 80,510 autonomous systems (ASes) and 884,790 directed edges. An edge points from a customer to its provider, and each peering link becomes one edge in each direction.
- PageRank is implemented from scratch with the power method on a sparse matrix. Dangling nodes (ASes with no out-link) redistribute their mass uniformly. The code is checked against exact fractions, a hand-derived fixture, and the lecture examples.
- At damping 0.85 the power method converges in 69 iterations to a residual below 1e-10. Of the top 20 ASes, 16 have more peers than customers and 4 belong to the clique. In all, 14 of the 18 clique members are in the top 100.
- PageRank and in-degree: the two rankings agree on 9 of the top 10, 44 of the top 50 and 86 of the top 100 ASes. They disagree where an AS has many in-links from networks with thousands of out-links, each of which passes on little rank.
- With peering links removed, 494 ASes have no out-link, all 18 clique members among them, and they hold 22.5% of the PageRank mass. The same solver handles it.

<!-- cell -->

The notebook needs numpy, scipy, pandas and matplotlib, and nothing else outside the Python standard library.

```python
from __future__ import annotations

import bz2
import hashlib
import os
import shutil
import time
import urllib.request
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Final, Iterable, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats
from scipy.sparse.csgraph import connected_components

pd.set_option("display.float_format", "{:.6g}".format)
pd.set_option("display.width", 120)
T0 = time.perf_counter()
```

<!-- cell -->

## 1. Problem and data

An autonomous system (AS) is a network under one administrative control: an Internet service provider, a university, a content company. ASes exchange traffic under commercial agreements of two kinds. In a **customer-provider** link the customer pays the provider for transit. In a **peer-to-peer** link two ASes exchange traffic between themselves and their customers, usually without payment.

The data is the CAIDA *AS Relationships* dataset (serial-1), snapshot 2026-09-01. It lists one relationship per line, inferred from BGP routing paths seen at many vantage points (Luckie et al., *AS Relationships, Customer Cones, and Validation*, IMC 2013). This is a non-web graph: the nodes are network operators, and the links are commercial agreements instead of hyperlinks between pages.

The cell below downloads the two files of the snapshot into `caida_data/`, next to this notebook, and checks their md5 sums. A file that is already there with the right sum is not downloaded again. A wrong sum stops the notebook with an error.

```python
DATA_DIR = Path("caida_data")
BASE_URL = "https://publicdata.caida.org/datasets/as-relationships/serial-1/"
SNAPSHOT_MD5 = {
    "20260901.as-rel.txt.bz2": "f4cba16613623eb1bc67355d1dc25213",
    "20260901.ppdc-ases.txt.bz2": "1c6b0c4c62a16d9bfa6c8fb75e6cd03d",
}


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    """md5 of a file, read in chunks so the whole file is never in memory."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def fetch_caida(data_dir: Path = DATA_DIR) -> dict[str, Path]:
    """Download the snapshot files that are missing and verify every md5.

    The download goes to a temporary name and is renamed only after the md5
    matches, so an interrupted or corrupted download never passes for a
    good file on the next run.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, expected in SNAPSHOT_MD5.items():
        path = data_dir / name
        if path.exists() and md5_of(path) == expected:
            print(f"{name}: already present, md5 OK")
        else:
            print(f"{name}: downloading")
            partial = path.with_name(name + ".part")
            with urllib.request.urlopen(BASE_URL + name, timeout=120) as response:
                with open(partial, "wb") as out:
                    shutil.copyfileobj(response, out)
            found = md5_of(partial)
            if found != expected:
                partial.unlink()
                raise RuntimeError(
                    f"{name}: md5 is {found}, expected {expected}. "
                    "The file at CAIDA has changed or the download is corrupt."
                )
            partial.replace(path)
            print(f"{name}: downloaded, md5 OK")
        paths[name] = path
    return paths


paths = fetch_caida()
AS_REL = paths["20260901.as-rel.txt.bz2"]
PPDC = paths["20260901.ppdc-ases.txt.bz2"]
for name, path in paths.items():
    print(f"{name}: {path.stat().st_size:,} bytes, md5 {md5_of(path)}")
```

The header of the file states what the data is and what was inferred. The first lines:

```python
header, data = [], []
with bz2.open(AS_REL, "rt", encoding="ascii") as handle:
    for line in handle:
        (header if line.startswith("#") else data).append(line.rstrip())
        if len(data) == 3:
            break
for line in header[:4]:
    print(line[:100])
print(f"... {len(header)} comment lines before the data, then one relationship per line:")
for line in data:
    print(line)
```

A relationship line is `provider|customer|-1` for a customer-provider link and `peer|peer|0` for a peering link.

<!-- cell -->

### Modelling the data as a directed graph

Some choices are facts from the data and some are decisions.

**Facts.** The nodes are the ASNs that appear in a relationship. A customer-provider line says who is the customer and who is the provider. A peer-to-peer line is symmetric. The header names an input clique of 18 transit-free ASes that the inference treats as the top of the hierarchy.

**Decisions.**

1. *Vote direction.* PageRank needs an edge to mean "endorses". We let a customer endorse its provider: the edge runs customer to provider, so rank flows up the hierarchy towards the networks that many others pay for transit.
2. *Peering links.* A peering link says nothing about who endorses whom. We either add an edge in each direction (`both`) or leave peering out (`none`). The counts in section 5 show why this matters: most edges are peering edges, and without them 494 ASes have no out-link, the whole clique included. The decision is argued in section 5.
3. *Duplicates and self-loops* are removed and counted. Every AS is kept, including the 28 IXP ASes (internet exchange points) that the header lists.

<!-- cell -->

## 2. Method

**From the graph to the Google matrix.** Number the $N$ ASes $1,\dots,N$. Let $A$ be the adjacency matrix, with $A_{ij}=1$ when $i$ links to $j$. The course builds PageRank in four steps, $A \to H \to S \to G$:

- $H$ divides each row of $A$ by its sum, so row $i$ spreads the vote of $i$ equally over its out-links. A dangling AS has a zero row.
- $S = H + \frac{1}{N}\, w\,\mathbf 1^T$, where $w_i = 1$ for dangling rows and 0 otherwise. A dangling AS now jumps to any AS with probability $1/N$, so every row of $S$ sums to 1.
- $G = \alpha S + (1-\alpha)\frac{1}{N}\mathbf 1\mathbf 1^T$. With probability $1-\alpha$ the surfer jumps to a uniformly random AS (teleportation).

PageRank is the row vector $\pi$ with $\pi^T = \pi^T G$ and $\pi^T\mathbf 1 = 1$. For $0<\alpha<1$ every entry of $G$ is positive, so $G$ is primitive. By the Perron-Frobenius theorem, $\pi$ exists, is unique and is strictly positive. The graph itself is not irreducible (section 5), so teleportation is what makes the answer unique.

**The update that is actually computed.** $G$ would have $80{,}510^2 \approx 6.5\times10^9$ entries, against 884,790 non-zeros in $H$. So $G$ is never built. One step of the power method is

$$\pi^{(k+1)T} = \alpha\,\pi^{(k)T}H + \big(\alpha\,\pi^{(k)T}w + 1-\alpha\big)\frac{\mathbf 1^T}{N},$$

which costs one sparse product. In the code the matrix is stored transposed, because the step is $x \leftarrow \alpha P^T x + \dots$ and a CSR matrix-vector product is fastest on the stored matrix.

**Dangling treatment.** The term $\alpha\,\pi^T w$ is the mass sitting on dangling ASes, and it is spread uniformly. With uniform teleportation this gives the same vector as leaving $H$ substochastic, solving the linear system, and normalising the result to sum 1. Section 4 checks this on a fixture.

**Stopping rule.** Iterate until $\|\pi^{(k+1)}-\pi^{(k)}\|_1 < 10^{-10}$. Each step shrinks the error by at least a factor $\alpha$, so about $\lceil \ln(10^{-10}) / \ln \alpha \rceil$ steps are enough, 142 for $\alpha=0.85$. Section 9.2 shows that the actual counts are lower.

<!-- cell -->

## 3. Implementation

The cells in this section define everything the later sections use. Each is a single block of code, shown here in full. Nothing outside the Python standard library, NumPy, SciPy, pandas and matplotlib is used, and no PageRank routine comes from a library.

**Graph.** The graph is stored as the transposed, row-normalised matrix $P^T$ in CSR form, because the power step multiplies by $P^T$. Out-degree and in-degree are kept as arrays, and a dangling node is one with out-degree 0. Duplicate edges are dropped, not summed into weights, and self-loops are removed. Both counts are kept.

```inline
pagerank.graph:DirectedGraph
pagerank.graph:from_edge_list
```

**Power iteration.** The state is a float64 vector, because the residual is compared with $10^{-10}$ and the vector holds values near $10^{-6}$. After each step the code checks that the vector still sums to 1 and raises if it does not. A wrong dangling term loses or creates mass, and this check turns that into an error instead of a quietly wrong ranking.

```inline
pagerank.power:PageRankResult
pagerank.power:_as_distribution
pagerank.power:pagerank_power
```

**Reading the data.** The parser keeps what the file says (relationship rows, the clique, the IXP list) and makes no modelling choice. `build_edge_list` is where the two decisions of section 1 live, as the parameters `peering` and `vote`. ASNs are mapped to node indices with a binary search on the sorted ASN array.

```inline
pagerank.caida:C2P
pagerank.caida:P2P
pagerank.caida:AsRelSnapshot
pagerank.caida:_open_text
pagerank.caida:read_as_rel
pagerank.caida:EdgeListBuild
pagerank.caida:index_of
pagerank.caida:build_edge_list
```

**Comparing rankings.** Ties are broken by node index, so a ranking is always a strict order and repeatable. Kendall's tau-b is used because in-degree is a small integer with many ties, and tau-b corrects for them.

```inline
pagerank.metrics:ranking
pagerank.metrics:rank_positions
pagerank.metrics:top_k_overlap
pagerank.metrics:kendall_tau
```

**Structure.** Strongly connected components tell which part of the graph is closed, meaning no edge leaves it, and which ASes are transient. This is used to describe the real graph and does not enter the PageRank computation.

```inline
pagerank.structure:GraphStructure
pagerank.structure:graph_structure
```

**The course construction in dense form.** These five functions build $A$, $H$, $S$ and $G$ as dense matrices and take $\pi$ as the dominant left eigenvector of $G$. They are for small graphs only. They are used in section 4 to check the sparse solver against the definition.

```inline
pagerank.dense:adjacency_matrix
pagerank.dense:hyperlink_matrix
pagerank.dense:stochastic_matrix
pagerank.dense:google_matrix
pagerank.dense:pagerank_eigenvector
```

<!-- cell -->

## 4. Validation

A solver that returns a plausible vector can still be wrong. Four checks, each against an answer that does not come from the solver.

### 4.1 Toy graph against exact fractions

Four nodes, five edges, $\alpha = 17/20$, with no dangling node. The exact PageRank was solved in rational arithmetic, so each entry is a fraction. `ALPHA` is kept as a `Fraction` in this section so that the exact answer and the solver use the same damping.

```constants
ALPHA
TOY_EDGES
TOY_N
TOY_PAGERANK_EXACT
```

```python
toy = from_edge_list(TOY_EDGES, TOY_N)
toy_result = pagerank_power(toy, alpha=float(ALPHA), tol=1e-14, max_iter=5000)
toy_exact = np.array([float(p) for p in TOY_PAGERANK_EXACT])
print(f"iterations                  {toy_result.iterations}")
print(f"exact fractions sum to 1:   {sum(TOY_PAGERANK_EXACT) == 1}")
print(f"max |solver - exact|        {np.abs(toy_result.vector - toy_exact).max():.2e}")
assert np.abs(toy_result.vector - toy_exact).max() < 1e-12
```

### 4.2 A graph with a dangling node

The second fixture has four nodes and edges $0\to1$, $0\to2$, $1\to2$, $1\to3$, $2\to3$. Node 3 has no out-link, so it is dangling. Node 0 has no in-link, so it receives only the share that every node gets: teleportation plus its part of node 3's redistributed mass. Its PageRank is derived by hand from the repaired matrix $S$ at $\alpha=17/20$ with uniform teleportation.

**Derivation by hand.** In the repaired matrix $S$, node 3 sends a quarter of its rank to each of the four nodes. Every node therefore gets the same base share $k$: its teleportation share plus its quarter of node 3's rank.

$$k = \frac{1-\alpha}{4} + \alpha\,\frac{p_3}{4} = \frac{3}{80} + \frac{17}{80}\,p_3 .$$

On top of $k$, a node receives the rank of each in-neighbour divided by that neighbour's out-degree:

$$p_0 = k, \qquad p_1 = k + \alpha\,\frac{p_0}{2}, \qquad p_2 = k + \alpha\left(\frac{p_0}{2} + \frac{p_1}{2}\right), \qquad p_3 = k + \alpha\left(\frac{p_1}{2} + p_2\right).$$

I did not solve for $k$ first. Going down the chain, I wrote each $p_i$ as a multiple of $k$:

$$p_0 = k, \qquad p_1 = \tfrac{57}{40}\,k, \qquad p_2 = \tfrac{3249}{1600}\,k, \qquad p_3 = \tfrac{106613}{32000}\,k .$$

The four add up to 1, so $k = 32000/249193$ and

$$p = \frac{(32000,\ 45600,\ 64980,\ 106613)}{249193} \approx (0.128,\ 0.183,\ 0.261,\ 0.428).$$

The equation for $k$ played no part in finding it, which makes it a free check: putting $p_3$ back in gives exactly $32000/249193$. Node 0 has no in-link and gets only the base share, so it ranks last. Node 3 ranks first because it collects all of node 2's rank and half of node 1's.

The cell below runs the solver on the fixture and compares it with the hand-derived exact vector. It also checks the claim of section 2 numerically: repairing the dangling row, and leaving the matrix substochastic and normalising the solution at the end, give the same vector.

```constants
DANGLING_EDGES
DANGLING_N
DANGLING_PAGERANK_EXACT
```

```python
dangling_graph = from_edge_list(DANGLING_EDGES, DANGLING_N)
print(f"dangling nodes in the fixture: {np.flatnonzero(dangling_graph.dangling).tolist()}")
dangling_result = pagerank_power(dangling_graph, alpha=float(ALPHA), tol=1e-14, max_iter=5000)
v_sparse = dangling_result.vector

# Convention 1: repair the matrix (S), build G, take the eigenvector.
h = hyperlink_matrix(adjacency_matrix(DANGLING_EDGES, DANGLING_N))
v_repaired = pagerank_eigenvector(google_matrix(stochastic_matrix(h), float(ALPHA)))

# Convention 2: keep H substochastic, solve z = (1-a) v + a H^T z, normalise.
n_fix = DANGLING_N
z = np.linalg.solve(
    np.eye(n_fix) - float(ALPHA) * h.T, (1 - float(ALPHA)) * np.full(n_fix, 1 / n_fix)
)
v_normalised = z / z.sum()

print(f"vector from the solver           {np.round(v_sparse, 6)}")
print(f"sum                              {v_sparse.sum():.12f}")
print(f"mass lost before normalising     {1 - z.sum():.4f}")
print(f"max |solver - repaired matrix|   {np.abs(v_sparse - v_repaired).max():.2e}")
print(f"max |solver - normalised solve|  {np.abs(v_sparse - v_normalised).max():.2e}")
assert np.abs(v_sparse - v_repaired).max() < 1e-12
assert np.abs(v_sparse - v_normalised).max() < 1e-12

if DANGLING_PAGERANK_EXACT is None:
    print("hand-derived exact vector: not filled in yet")
else:
    exact = np.array([float(p) for p in DANGLING_PAGERANK_EXACT])
    print(f"max |solver - hand-derived|      {np.abs(v_sparse - exact).max():.2e}")
    assert np.abs(v_sparse - exact).max() < 1e-12
```

### 4.3 Examples from the lecture

The lecture gives a five-node graph at $\alpha=0.85$ with the printed result $(0.1219, 0.1219, 0.1737, 0.2475, 0.3350)$. Node V (index 0) is dangling. The table shows the printed values, the sparse solver, and the dense eigenvector of $G$.

```python
FIVE_EDGES = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 0), (4, 1), (4, 2), (4, 3)]
FIVE_N = 5
FIVE_PRINTED = np.array([0.1219, 0.1219, 0.1737, 0.2475, 0.3350])

five_sparse = pagerank_power(
    from_edge_list(FIVE_EDGES, FIVE_N), alpha=0.85, tol=1e-14, max_iter=5000
).vector
five_dense = pagerank_eigenvector(
    google_matrix(
        stochastic_matrix(hyperlink_matrix(adjacency_matrix(FIVE_EDGES, FIVE_N))), 0.85
    )
)
five_table = pd.DataFrame(
    {"printed": FIVE_PRINTED, "sparse solver": five_sparse, "dense eigenvector": five_dense},
    index=list("VWXYZ"),
)
print(five_table.round(4))
print(f"max |sparse - dense|   {np.abs(five_sparse - five_dense).max():.2e}")
print(f"max |sparse - printed| {np.abs(five_sparse - FIVE_PRINTED).max():.2e} (the slide prints 4 decimals)")
assert np.abs(five_sparse - five_dense).max() < 1e-12
assert np.abs(five_sparse - FIVE_PRINTED).max() < 5e-5
```

The lecture's second example has no teleportation and four nodes W, X, Y, Z. The graph is strongly connected, so with $\alpha=1$ the matrix $S$ has a unique stationary vector, here exactly $(4,6,9,12)/31$. The sparse solver rejects $\alpha=1$, so the dense construction is used. The example also shows why in-degree is not PageRank: Y has the largest in-degree, Z has the largest PageRank.

```python
FOUR_EDGES = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 0), (3, 1), (3, 2)]
FOUR_EXACT = (Fraction(4, 31), Fraction(6, 31), Fraction(9, 31), Fraction(12, 31))

four_dense = pagerank_eigenvector(
    google_matrix(stochastic_matrix(hyperlink_matrix(adjacency_matrix(FOUR_EDGES, 4))), 1.0)
)
four_in_degree = from_edge_list(FOUR_EDGES, 4).in_degree
print(f"max |dense - exact|   {np.abs(four_dense - np.array([float(p) for p in FOUR_EXACT])).max():.2e}")
print(f"in-degree             {dict(zip('WXYZ', four_in_degree.tolist()))}  -> leader {'WXYZ'[int(four_in_degree.argmax())]}")
print(f"PageRank              {dict(zip('WXYZ', np.round(four_dense, 4).tolist()))}  -> leader {'WXYZ'[int(four_dense.argmax())]}")
assert np.abs(four_dense - np.array([float(p) for p in FOUR_EXACT])).max() < 1e-12
```

<!-- cell -->

### What the validation established

The sparse power method matches exact fractions on the toy graph, matches the two dangling conventions to rounding error on the fixture, and reproduces the lecture's printed values. The dense construction of the course and the sparse solver agree. The solver can now be applied to the real graph.

<!-- cell -->

## 5. The real graph

The snapshot is read and turned into the primary graph: peering `both`, vote customer to provider, all ASes kept.

```python
snapshot = read_as_rel(AS_REL)
print(f"relationship lines     {len(snapshot.relationships):,}")
print(f"  customer-provider    {snapshot.n_c2p:,}")
print(f"  peer-to-peer         {snapshot.n_p2p:,}")
print(f"input clique           {len(snapshot.clique)} ASes: {list(snapshot.clique)}")
print(f"IXP ASes in header     {len(snapshot.ixp_asns)}")

build = build_edge_list(snapshot, peering="both", vote="customer_to_provider")
graph = from_edge_list(build.edges, build.n)
peering_share = 2 * build.n_p2p_used / graph.n_edges
print()
print(f"primary graph: {graph.n:,} nodes, {graph.n_edges:,} edges")
print(f"duplicate edges dropped {graph.n_duplicates_dropped}, self-loops dropped {graph.n_self_loops_dropped}")
print(f"dangling nodes          {int(graph.dangling.sum())}")
print(f"share of edges that come from peering links: {peering_share:.1%}")
assert graph.n == 80510 and graph.n_edges == 884790
assert int(graph.dangling.sum()) == 0
```

The peering decision matters because peering supplies most of the edges. Under `none` the graph keeps only the customer-provider edges, and the ASes that have no provider have nowhere to send their vote:

```python
build_none = build_edge_list(snapshot, peering="none", vote="customer_to_provider")
graph_none = from_edge_list(build_none.edges, build_none.n)
assert np.array_equal(build_none.asns, build.asns)  # same node indices

clique_nodes = index_of(build.asns, np.array(snapshot.clique))
print(f"peering none: {graph_none.n_edges:,} edges, {int(graph_none.dangling.sum())} dangling nodes")
print(f"clique members among the dangling nodes: {int(graph_none.dangling[clique_nodes].sum())} of {len(clique_nodes)}")
assert int(graph_none.dangling.sum()) == 494
```

**Modelling decision: peering in both directions.** The main results use `peering="both"`, and `none` is kept as a sensitivity check in section 8. I considered `none`, because "who pays whom" is the cleanest economic reading of a vote. Two things in the data made me drop it. First, peering is most of the data: 358,942 of the 525,848 relationships in the file. A peering link is an agreement between equals and carries traffic both ways, so an edge in each direction is the closest I can get to reading it as a vote. Second, the counts above show what happens without it: all 18 clique members are left with no out-link. They are the networks at the top of the hierarchy, and their rank, 22.5% of the total (section 8), would be spread evenly over every AS instead of passed along any link. Without peering the ranking also ends up close to CAIDA's customer cone (section 9.1), which already exists. With peering, PageRank says something the cone does not: how central an AS is in the interconnection graph as a whole.

**Damping factor.** I use $\alpha = 0.85$, the value of the original PageRank paper and of the lecture. At that value a surfer follows $\alpha/(1-\alpha) \approx 5.7$ links on average before jumping, which is enough for rank to move up and across the hierarchy. Both ends of the range are worse for this graph. At $\alpha = 0.5$ every step is a random jump half of the time, so the links carry less weight. At $\alpha = 0.99$ the closed class described below holds 99% of the mass, the other 60,000 ASes end up close to zero, and the power method needs three times as many iterations (section 9.2). The value still matters: compared with 0.85, the order of the top ASes changes at both ends (Kendall tau-b about 0.55 and 0.64), so I report that sensitivity in section 9.2 instead of treating 0.85 as neutral.

**Reducibility.** A random surfer without teleportation would eventually be stuck in a closed class, a set of ASes with no edge leaving it. The strongly connected components show how the primary graph is built.

```python
structure = graph_structure(graph)
transient = structure.transient_mask
print(f"weakly connected components      {structure.n_weak_components}")
print(f"strongly connected components    {structure.n_strong_components:,}")
print(f"closed classes                   {structure.n_closed_classes}")
print(f"size of the largest closed class {structure.largest_closed_class_size:,}")
print(f"transient ASes                   {int(transient.sum()):,} of {graph.n:,}")
assert structure.n_closed_classes == 1 and structure.largest_closed_class_size == 20401
```

The graph is connected as an undirected graph, but it is not strongly connected: there is one closed class, and the other ASes are transient. Without teleportation, all the mass would end up inside the closed class and every transient AS would get 0. Teleportation is what gives them rank.

**Degree distribution.** In-degree and out-degree, as complementary cumulative distributions on log-log axes. Both are heavy-tailed: the median in-degree is 0 and the median out-degree is 2, while a few ASes have thousands of links.

```python
def ccdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Points (x, P(X >= x)) of the empirical survival function."""
    ordered = np.sort(values)
    unique, first = np.unique(ordered, return_index=True)
    return unique, 1.0 - first / ordered.size


fig, ax = plt.subplots(figsize=(6.5, 4.5))
for degree, label in ((graph.in_degree, "in-degree"), (graph.out_degree, "out-degree")):
    x, y = ccdf(degree[degree > 0])
    ax.loglog(x, y, ".", markersize=4, label=label)
ax.set_xlabel("degree d")
ax.set_ylabel("share of ASes (degree >= 1) with degree >= d")
ax.set_title("Degree distribution of the AS graph (peering both)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()
print(f"max in-degree {int(graph.in_degree.max()):,}, max out-degree {int(graph.out_degree.max()):,}")
print(f"median in-degree {int(np.median(graph.in_degree))}, median out-degree {int(np.median(graph.out_degree))}")
```

<!-- cell -->

## 6. Results: the ranking

PageRank at $\alpha=0.85$, to a residual below $10^{-10}$.

```python
ALPHA_MAIN = 0.85
TOL = 1e-10

result = pagerank_power(graph, alpha=ALPHA_MAIN, tol=TOL)
pr = result.vector
print(f"iterations      {result.iterations}")
print(f"converged       {result.converged}")
print(f"final residual  {result.residuals[-1]:.2e}")
print(f"sum of vector   {pr.sum():.15f}")
print(f"min / max       {pr.min():.2e} / {pr.max():.2e}  (uniform would be {1 / graph.n:.2e})")
assert result.converged and abs(pr.sum() - 1) < 1e-12
```

The top 20, with the facts about each AS taken from the snapshot: how many customers, providers and peers it has, and whether it is in the clique.

```python
rel = snapshot.relationships
c2p_rows = rel[rel[:, 2] == C2P]
p2p_rows = rel[rel[:, 2] == P2P]


def count_per_node(asn_column: np.ndarray) -> np.ndarray:
    """How many times each node's ASN appears in a column of ASNs."""
    counts = pd.Series(asn_column).value_counts()
    return counts.reindex(build.asns, fill_value=0).to_numpy()


n_customers = count_per_node(c2p_rows[:, 0])  # as1 is the provider
n_providers = count_per_node(c2p_rows[:, 1])  # as2 is the customer
n_peers = count_per_node(np.concatenate([p2p_rows[:, 0], p2p_rows[:, 1]]))
is_clique = np.zeros(graph.n, dtype=bool)
is_clique[clique_nodes] = True

in_degree = graph.in_degree.astype(np.float64)
pr_rank = rank_positions(pr)
in_rank = rank_positions(in_degree)
order_pr = ranking(pr)


def as_row(node: int) -> dict:
    return {
        "rank": int(pr_rank[node]),
        "ASN": int(build.asns[node]),
        "PageRank": float(pr[node]),
        "in-degree": int(graph.in_degree[node]),
        "in-degree rank": int(in_rank[node]),
        "out-degree": int(graph.out_degree[node]),
        "customers": int(n_customers[node]),
        "providers": int(n_providers[node]),
        "peers": int(n_peers[node]),
        "clique": "yes" if is_clique[node] else "no",
    }


top20 = pd.DataFrame([as_row(int(node)) for node in order_pr[:20]])
top20
```

The PageRank values are spread over more than three orders of magnitude. The smallest value, 1.9e-6, equals $(1-\alpha)/N$, the share an AS gets from teleportation alone, which is what an AS with no in-link receives.

```python
fig, ax = plt.subplots(figsize=(6.5, 4.5))
x, y = ccdf(pr)
ax.loglog(x, y, ".", markersize=4)
ax.axvline(1 / graph.n, color="gray", linestyle=":", label="uniform value 1/N")
ax.set_xlabel("PageRank p")
ax.set_ylabel("share of ASes with PageRank >= p")
ax.set_title("Distribution of PageRank (alpha 0.85)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()
```

Facts about the top of the ranking, as numbers only:

```python
top20_nodes = order_pr[:20]
hubs = int((n_peers[top20_nodes] > n_customers[top20_nodes]).sum())
print(f"top 20: {int(is_clique[top20_nodes].sum())} are clique members")
print(f"top 20: {hubs} have more peers than customers")
print(f"top 20: median customers {int(np.median(n_customers[top20_nodes]))}, median peers {int(np.median(n_peers[top20_nodes]))}, median providers {int(np.median(n_providers[top20_nodes]))}")
print(f"top 20: hold {pr[top20_nodes].sum():.1%} of the total PageRank")
clique_ranks = np.sort(pr_rank[clique_nodes])
print(f"clique members in the top 100: {int((clique_ranks <= 100).sum())} of {len(clique_nodes)}")
print(f"worst-placed clique member: rank {int(clique_ranks[-1])}")
```

**What the top of the ranking says.** I expected the top 20 to be the clique, the 18 networks at the top of the transit hierarchy. Only 4 of them made it: Lumen (AS3356), Cogent (AS174), Arelion (AS1299) and Zayo (AS6461). These four look like large transit providers. Each has between 2,400 and 6,500 customers and few peers (under 100, except Zayo with 409), and in PeeringDB their peering policy is restrictive or "No". Their rank comes from their customers, one vote per customer.

The other 16 have more peers than customers, most of them by a lot. SG.GS (AS24482, rank 2) has 59 customers and 9,779 peers. Valve (AS32590) is a game company with no customers at all, and it still ranks 11th on 5,159 peers, which surprised me more than anything else in the table. PeeringDB lists three of the 16 as content networks (Valve, F5 and Gcore) and most of the rest as network service providers. They peer widely: 9 declare an open policy and 6 a selective one. Since I model a peering session as a link in each direction, each session is a vote, and 5,000 peers mean 5,000 votes.

Hurricane Electric (AS6939) is first because it has both: 2,243 customers and 7,733 peers, with an open policy.

The clique is not far behind, with 14 of its 18 members in the top 100. AS2828 is the odd one, at rank 231. In this snapshot it has only 2 customers (section 8 looks at it under both conventions).

What PageRank measures here is how well connected an AS is through its agreements, with each link weighted by the rank of the AS it comes from and split among that AS's out-links. Every link counts the same, whether it carries a terabit or almost nothing, so this is not a traffic measure. The customer cone counts the networks below an AS, which is a different thing, and section 9.1 shows the two barely agree once peering is in the graph (tau-b 0.095 on the top-1000 union). If I wanted to know who sells transit to the most networks, I would look at the cone and the clique. PageRank answers another question, which networks sit at the centre of the interconnection graph, and its answer is the networks that peer most widely, Hurricane Electric first.

<!-- cell -->

## 7. PageRank versus in-degree

In-degree counts the in-links of an AS. PageRank counts them too, but weights each by the rank of the sender and divides it by the sender's out-degree. The two are close on this graph, so the disagreements are the informative part.

```python
union_top = np.union1d(order_pr[:1000], ranking(in_degree)[:1000])
comparison = pd.DataFrame(
    {
        "value": {
            "top-10 overlap": top_k_overlap(pr, in_degree, 10),
            "top-50 overlap": top_k_overlap(pr, in_degree, 50),
            "top-100 overlap": top_k_overlap(pr, in_degree, 100),
            "Kendall tau-b, all ASes": kendall_tau(pr, in_degree),
            "Kendall tau-b, top-1000 union": kendall_tau(pr, in_degree, subset=union_top),
        }
    }
)
comparison.round(3)
```

```python
mask = in_degree >= 1
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.scatter(in_degree[mask], pr[mask], s=4, alpha=0.25, label="AS")
ax.scatter(in_degree[is_clique], pr[is_clique], s=28, color="crimson", label="clique member")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("in-degree")
ax.set_ylabel("PageRank")
ax.set_title("PageRank against in-degree (peering both)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()
```

**Where they disagree.** The comparison is restricted to the 1000 best ASes by either measure, so that it is between networks that matter to at least one of them. For each AS the table gives the shift in rank and what its in-neighbours look like. The last column is the average amount that one in-link delivers, the term $\pi_j / d_j^{\text{out}}$ of the update averaged over the in-neighbours $j$.

```python
transpose = graph.transpose.tocsr()  # row `node` lists the in-neighbours of `node`


def in_link_contribution(node: int) -> float:
    """Average of pr[j] / out_degree[j] over the in-neighbours j of `node`."""
    tails = transpose.indices[transpose.indptr[node] : transpose.indptr[node + 1]]
    return float((pr[tails] / graph.out_degree[tails]).mean())


shift = pr_rank - in_rank  # negative: PageRank ranks the AS better than in-degree does


def disagreements(candidates: np.ndarray, direction: str, k: int = 10) -> pd.DataFrame:
    chosen = sorted(candidates.tolist(), key=lambda node: (-abs(int(shift[node])), node))[:k]
    rows = []
    for node in chosen:
        row = as_row(node)
        row["rank shift"] = abs(int(shift[node]))
        row["per in-link"] = in_link_contribution(node)
        row["better by"] = direction
        rows.append(row)
    return pd.DataFrame(rows)


higher_pr = disagreements(union_top[shift[union_top] < 0], "PageRank")
higher_in = disagreements(union_top[shift[union_top] > 0], "in-degree")
pd.concat([higher_pr, higher_in], ignore_index=True)[
    ["better by", "ASN", "rank", "in-degree rank", "rank shift", "in-degree", "peers", "customers", "per in-link"]
]
```

**Vote dilution.** An in-link from a large AS with many out-links delivers little, because the sender splits its rank over all its out-links. An AS can have a high in-degree and still receive little rank if its in-links come from senders that are small or that link to many others.

```python
print(f"median rank per in-link, ASes that PageRank ranks higher: {higher_pr['per in-link'].median():.2e}")
print(f"median rank per in-link, ASes that in-degree ranks higher: {higher_in['per in-link'].median():.2e}")
```

<!-- cell -->

### What the comparison established

On this graph PageRank and in-degree give almost the same top of the ranking, and they differ on a small set of ASes. The ASes that PageRank puts higher get more rank from each in-link than the ASes that in-degree puts higher.

<!-- cell -->

## 8. Dangling nodes on the real graph

Under the primary graph no AS is dangling, so the dangling term of the solver is inactive there. It becomes active when peering links are left out. The same solver is run on that graph without change.

```python
result_none = pagerank_power(graph_none, alpha=ALPHA_MAIN, tol=TOL)
pr_none = result_none.vector
dangling_mass = float(pr_none[graph_none.dangling].sum())
print(f"iterations                      {result_none.iterations}")
print(f"final residual                  {result_none.residuals[-1]:.2e}")
print(f"sum of vector                   {pr_none.sum():.15f}")
print(f"dangling ASes                   {int(graph_none.dangling.sum())}")
print(f"PageRank mass on dangling ASes  {dangling_mass:.4f}")
print(f"clique members that dangle      {int(graph_none.dangling[clique_nodes].sum())} of {len(clique_nodes)}")
print()
for k in (10, 20, 50, 100):
    print(f"top-{k:<3d} overlap, both vs none: {top_k_overlap(pr, pr_none, k):.2f}")
print(f"Kendall tau-b, both vs none: {kendall_tau(pr, pr_none):.3f}")
assert result_none.converged and abs(pr_none.sum() - 1) < 1e-12
```

Without peering, the clique is among the ASes with no out-link, and the top of the ranking changes a lot: the top-20 overlap between the two conventions is only 0.25. This is where the dangling treatment matters on real data. The mass of the dangling ASes is not lost: it is spread uniformly over all ASes at every step, and the vector still sums to 1.

```python
rank_none = rank_positions(pr_none)
pd.DataFrame(
    {
        "ASN": build.asns[clique_nodes],
        "rank, peering both": pr_rank[clique_nodes],
        "rank, peering none": rank_none[clique_nodes],
    }
).sort_values("rank, peering none").reset_index(drop=True)
```

```python
node_2828 = int(index_of(build.asns, np.array([2828]))[0])
print(
    f"AS 2828: {n_customers[node_2828]} customers, {n_providers[node_2828]} providers, "
    f"{n_peers[node_2828]} peers; rank {pr_rank[node_2828]} with peering both, "
    f"{rank_none[node_2828]} with peering none"
)
```

Under `none` an AS receives rank along links only from its customers, and AS 2828 has 2.

<!-- cell -->

## 9. Extensions

Two short extensions, after the core answers above.

### 9.1 Customer cone as an external reference

CAIDA also ships, for the same snapshot, the *customer cone* of every AS: the ASes reachable from it by going down customer links. It is an independent size measure, computed from paths and not from this graph's edges, so it can serve as a reference ranking. The code below reads the cone file and, as a check, recomputes cones directly from the relationships.

```inline
pagerank.caida:CustomerCones
pagerank.caida:read_ppdc_ases
pagerank.caida:customer_cone_sizes
```

```python
cones = read_ppdc_ases(PPDC)
cone_size = np.ones(graph.n)  # an AS with no cone line has no customers: size 1
position = np.clip(np.searchsorted(cones.asns, build.asns), 0, cones.asns.shape[0] - 1)
listed = cones.asns[position] == build.asns
cone_size[listed] = cones.cone_size[position[listed]]
print(f"ASes with a cone line: {int(listed.sum()):,} of {graph.n:,}")

cone_top1000 = ranking(cone_size)[:1000]


def versus_cone(scores: np.ndarray, name: str) -> dict:
    union = np.union1d(cone_top1000, ranking(scores)[:1000])
    return {
        "measure": name,
        "top-10 overlap": top_k_overlap(cone_size, scores, 10),
        "top-100 overlap": top_k_overlap(cone_size, scores, 100),
        "tau-b, top-1000 union": kendall_tau(cone_size, scores, subset=union),
    }


pd.DataFrame(
    [
        versus_cone(pr, "PageRank, peering both"),
        versus_cone(pr_none, "PageRank, peering none"),
        versus_cone(in_degree, "in-degree"),
    ]
).round(3)
```

```python
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, scores, title in (
    (axes[0], pr, "peering both"),
    (axes[1], pr_none, "peering none"),
):
    ax.scatter(cone_size, scores, s=4, alpha=0.25)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("customer cone size")
    ax.set_title(f"PageRank, {title}")
    ax.grid(True, which="both", alpha=0.3)
axes[0].set_ylabel("PageRank")
fig.tight_layout()
plt.show()
```

The cone measures how many networks sit below an AS in the hierarchy. PageRank with peering removed agrees with it most. There the rank flows only up customer-provider links, which is the relation the cone counts. With peering included the agreement drops, which fits peering edges giving rank to ASes whose cone is small. In-degree agrees least.

```python
union_cone = np.union1d(cone_top1000, order_pr[:1000])
own_cone = customer_cone_sizes(snapshot, build.asns[union_cone].tolist())
print(f"recursive cone >= CAIDA cone for all {union_cone.size:,} ASes checked: {bool((own_cone >= cone_size[union_cone]).all())}")
```

The two cone definitions are not the same object. CAIDA's cone is the set seen through observed paths, and the recursive one follows every inferred customer link, so it can only be larger. The comparison above uses CAIDA's.

### 9.2 The damping factor

The damping factor $\alpha$ sets how far a surfer follows links before teleporting. For $\alpha = 0.5,\ 0.85,\ 0.99$ the table gives the iterations needed, the estimate $\lceil\ln(10^{-10})/\ln\alpha\rceil$, the observed rate (median ratio of successive residuals over the second half of the run) and how much the ranking moves relative to $\alpha=0.85$.

```python
rows = []
for alpha in (0.50, 0.85, 0.99):
    run = pagerank_power(graph, alpha=alpha, tol=TOL, max_iter=5000)
    residuals = np.array(run.residuals)
    tail = residuals[residuals.size // 2 :]
    union_alpha = np.union1d(ranking(run.vector)[:1000], order_pr[:1000])
    rows.append(
        {
            "alpha": alpha,
            "iterations": run.iterations,
            "alpha^k estimate": int(np.ceil(np.log(TOL) / np.log(alpha))),
            "observed rate": float(np.median(tail[1:] / tail[:-1])),
            "top-10 overlap with 0.85": top_k_overlap(run.vector, pr, 10),
            "tau-b with 0.85 (top-1000 union)": kendall_tau(run.vector, pr, subset=union_alpha),
        }
    )
    assert rows[-1]["observed rate"] < alpha
pd.DataFrame(rows).round(3)
```

The estimate holds and is loose: the observed rate is below $\alpha$ because the second eigenvalue of the matrix is smaller than $\alpha$. The top of the ranking changes with $\alpha$: a smaller $\alpha$ puts more weight on teleportation and a larger one on the link structure.

<!-- cell -->

## 10. Limitations and conclusions

**Limitations.**

- The relationships are inferred from BGP paths, not observed. Some of them are wrong.
- The snapshot is one month. The graph changes, and a ranking for another month would differ in its details.
- Peering links are under-observed in BGP data, because they are only announced to the peer and its customers. So the graph under-counts them, and the true share of peering edges is probably above the 81% measured here.
- PageRank here measures position in the graph of agreements. It does not measure traffic, revenue or the reach of an AS in any physical sense.
- A single damping factor, 0.85, is used for the main results, from the web setting. Section 9.2 shows how far the ranking moves for other values.

[AUTHOR: conclusions]

```python
print(f"total run time of this notebook: {time.perf_counter() - T0:.0f} s")
```
