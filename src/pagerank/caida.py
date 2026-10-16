"""Read and model CAIDA ``serial-1`` AS-relationship snapshots.

The project ranks the CAIDA AS-level topology, distributed as an ``as-rel``
file: a header of comment lines followed by one relationship per line, in the
format documented in ``data/raw/caida_serial-1_README.txt``::

    <provider-as>|<customer-as>|-1      a customer-provider (c2p) link
    <peer-as>|<peer-as>|0               a peer-to-peer (p2p) link

From December 2022 the inference is annotated with ``# step N`` comment lines
before each block, N running 1..8, where lower N means the relationship rests
on stronger evidence. That tag is a per-link confidence signal worth keeping,
so the reader records which step each relationship came from.

This module does two separable jobs, kept apart on purpose. :func:`read_as_rel`
turns the file into an :class:`AsRelSnapshot`: a faithful, choice-free copy of
what the file says (header plus the raw ``(as1, as2, rel)`` triples and their
step tags). Turning that into a directed graph -- which way a customer-provider
link votes, whether peering contributes edges, which ASes to drop -- is a set
of modelling decisions and lives in the edge-list builder, so the same snapshot
can be rebuilt under every convention without re-reading the file.
"""

from __future__ import annotations

import bz2
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Final, Iterable, Literal

import numpy as np

#: Relationship code for a customer-provider link: the first field is the
#: provider, the second its customer. From the file format in the README.
C2P: Final = -1

#: Relationship code for a peer-to-peer link: the two fields are peers.
P2P: Final = 0


@dataclass(frozen=True)
class AsRelSnapshot:
    """One ``as-rel`` snapshot, parsed but not yet modelled.

    Holds exactly what the file states, with no edge-direction or peering
    choice applied. The three parallel arrays ``relationships`` and ``step``
    describe the same relationship lines in file order.

    Attributes:
        path: The path the snapshot was read from.
        clique: ASNs on the ``# input clique:`` line, in file order. These are
            the transit-free core the inference forces (see the README).
        ixp_asns: ASNs on the ``# IXP ASes:`` line, in file order.
        sources: The text after ``# source:`` on each source line, verbatim and
            in file order.
        relationships: ``(m, 3)`` int64 array; columns are ``as1``, ``as2`` and
            ``rel`` (``C2P`` or ``P2P``). For a ``C2P`` row ``as1`` is the
            provider and ``as2`` the customer; for a ``P2P`` row they are peers.
        step: ``(m,)`` int64 array; the ``N`` of the last ``# step N`` comment
            seen before each relationship, or 0 for any relationship appearing
            before the first step comment.
        n_comment_lines: Total count of lines beginning with ``#``, including
            the ones not matched by any known prefix.
    """

    path: str
    clique: tuple[int, ...]
    ixp_asns: tuple[int, ...]
    sources: tuple[str, ...]
    relationships: np.ndarray
    step: np.ndarray
    n_comment_lines: int

    @property
    def n_c2p(self) -> int:
        """Number of customer-provider relationships."""
        return int(np.count_nonzero(self.relationships[:, 2] == C2P))

    @property
    def n_p2p(self) -> int:
        """Number of peer-to-peer relationships."""
        return int(np.count_nonzero(self.relationships[:, 2] == P2P))


def _open_text(path: str | os.PathLike):
    """Open ``path`` as a text stream, transparently handling bz2.

    The snapshots ship bz2-compressed, but tests and small fixtures use plain
    text, so the suffix picks the opener. The file is ASCII.
    """
    name = os.fspath(path)
    if name.endswith(".bz2"):
        return bz2.open(name, "rt", encoding="ascii")
    return open(name, "rt", encoding="ascii")


def read_as_rel(path: str | os.PathLike) -> AsRelSnapshot:
    """Parse an ``as-rel`` file into an :class:`AsRelSnapshot`.

    Reads a ``serial-1`` relationship file (bz2 when the suffix is ``.bz2``,
    plain text otherwise). Comment lines are classified by their known
    prefixes and counted; every other ``#`` line is counted but ignored. Blank
    lines are ignored and not counted. Relationship lines are collected into
    integer arrays in file order, each tagged with the step it appeared under.

    Args:
        path: Path to the ``as-rel`` file.

    Returns:
        The parsed snapshot. ``relationships`` and ``step`` are parallel arrays
        in file order.

    Raises:
        ValueError: On a relationship value other than ``C2P`` or ``P2P``, a
            self-relationship (``as1 == as2``), or a malformed line (not three
            ``|``-separated integers). The message names the 1-based line
            number.
    """
    clique: tuple[int, ...] = ()
    ixp_asns: tuple[int, ...] = ()
    sources: list[str] = []
    rows: list[tuple[int, int, int]] = []
    steps: list[int] = []

    n_comment_lines = 0
    current_step = 0

    with _open_text(path) as handle:
        for k, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")

            if line.startswith("#"):
                n_comment_lines += 1
                if line.startswith("# input clique:"):
                    clique = tuple(int(t) for t in line[len("# input clique:"):].split())
                elif line.startswith("# IXP ASes:"):
                    ixp_asns = tuple(int(t) for t in line[len("# IXP ASes:"):].split())
                elif line.startswith("# source:"):
                    sources.append(line[len("# source:"):])
                elif line.startswith("# step "):
                    # "# step N: description" -- N is the token before the colon.
                    tag = line[len("# step "):].split(":", 1)[0].strip()
                    current_step = int(tag)
                # Any other comment line is counted above and otherwise ignored.
                continue

            if not line.strip():
                continue

            fields = line.split("|")
            if len(fields) != 3:
                raise ValueError(
                    f"line {k}: malformed relationship, expected three "
                    f"'|'-separated fields, got {len(fields)}"
                )
            try:
                as1, as2, rel = (int(f) for f in fields)
            except ValueError:
                raise ValueError(
                    f"line {k}: malformed relationship, non-integer field in {line!r}"
                ) from None

            if rel not in (C2P, P2P):
                raise ValueError(f"line {k}: unexpected relationship {rel!r}")
            if as1 == as2:
                raise ValueError(f"line {k}: self-relationship for AS {as1}")

            rows.append((as1, as2, rel))
            steps.append(current_step)

    relationships = np.array(rows, dtype=np.int64).reshape(-1, 3)
    step = np.array(steps, dtype=np.int64)

    return AsRelSnapshot(
        path=os.fspath(path),
        clique=clique,
        ixp_asns=ixp_asns,
        sources=tuple(sources),
        relationships=relationships,
        step=step,
        n_comment_lines=n_comment_lines,
    )


# --------------------------------------------------------------------------
# Customer cones: the CAIDA ppdc-ases file.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CustomerCones:
    """Per-AS customer-cone sizes read from a CAIDA ``ppdc-ases`` file.

    The ``ppdc-ases`` file ships with the same snapshot as the ``as-rel`` file
    and lists, one line per AS, the AS followed by every AS in its
    provider-peer-customer cone -- the set reachable by following
    provider-to-customer links downwards (see the ``serial-1`` README and
    Luckie et al., IMC 2013). It is CAIDA's own "how big is this AS" ordering,
    independent of anything this project computes, so it serves as an external
    reference for the ranking.

    Only the cone *size* per AS is kept: the member lists run to tens of
    thousands of ASNs for the clique and are not needed downstream, so they are
    counted per line and discarded during the parse.

    ``cone_size`` counts the distinct ASNs on a line including the leading AS
    itself. On the shipped snapshots every line lists its own AS among the
    members (``self_listed_share == 1.0``), but defining the size over the
    distinct set makes it independent of that convention: an AS with no
    customers has cone size 1 (itself) either way.

    Attributes:
        path: The path the cones were read from.
        clique: ASNs on the ``# inferred clique:`` header line, in file order.
            Empty when the header leaves it blank (as on this snapshot).
        total_size: The value on the ``# total size:`` header line, i.e. the
            number of ASes the file claims to cover.
        asns: ``(m,)`` int64, the ASNs with a cone, sorted ascending. Index
            ``i`` is ASN ``asns[i]``.
        cone_size: ``(m,)`` int64 aligned with ``asns``; distinct ASNs on each
            AS's line including the AS itself.
        n_comment_lines: Count of lines beginning with ``#``.
        self_listed_share: Share of data lines whose leading ASN also appears
            among its listed members.
    """

    path: str
    clique: tuple[int, ...]
    total_size: int
    asns: np.ndarray
    cone_size: np.ndarray
    n_comment_lines: int
    self_listed_share: float


def read_ppdc_ases(path: str | os.PathLike) -> CustomerCones:
    """Parse a CAIDA ``ppdc-ases`` file into per-AS cone sizes.

    Reads the provider-peer-customer cone file (bz2 when the suffix is
    ``.bz2``, plain text otherwise). The header carries two comment lines,
    ``# inferred clique:`` (optionally followed by the clique ASNs; blank on
    this snapshot) and ``# total size:``. Each remaining line is an AS followed
    by the ASNs in its cone, space separated.

    The cone size stored is the number of *distinct* ASNs on the line including
    the leading AS, so that an AS with no customers has size 1 regardless of
    whether the file repeats the leading AS in the member list. The share of
    lines that do repeat it is reported in ``self_listed_share`` so the
    convention on a given file is measurable rather than assumed.

    Member lists are counted and discarded per line; they are never all held in
    memory at once. ``str.split`` on each line is adequate (a few seconds on the
    80k-line snapshot even though the clique lines carry tens of thousands of
    members).

    Args:
        path: Path to the ``ppdc-ases`` file.

    Returns:
        The parsed :class:`CustomerCones`. ``asns`` is sorted ascending and
        ``cone_size`` is aligned with it.

    Raises:
        ValueError: On a line with a non-integer token. The message names the
            1-based line number.
    """
    clique: tuple[int, ...] = ()
    total_size = 0
    n_comment_lines = 0
    n_self_listed = 0

    asn_list: list[int] = []
    size_list: list[int] = []

    with _open_text(path) as handle:
        for k, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")

            if line.startswith("#"):
                n_comment_lines += 1
                if line.startswith("# inferred clique:"):
                    body = line[len("# inferred clique:"):].strip()
                    if body:
                        clique = tuple(int(t) for t in body.split())
                elif line.startswith("# total size:"):
                    total_size = int(line[len("# total size:"):].strip())
                continue

            if not line.strip():
                continue

            try:
                tokens = [int(t) for t in line.split()]
            except ValueError:
                raise ValueError(
                    f"line {k}: malformed cone line, non-integer token in "
                    f"{line!r}"
                ) from None

            lead = tokens[0]
            members = tokens[1:]
            # Distinct ASNs on the line, including the leading AS. Using a set
            # absorbs both the self-listing and any duplicate member, so the
            # size is "distinct ASNs in the cone" under either convention.
            distinct = set(tokens)
            asn_list.append(lead)
            size_list.append(len(distinct))
            if lead in members:
                n_self_listed += 1

    asns = np.array(asn_list, dtype=np.int64)
    cone_size = np.array(size_list, dtype=np.int64)
    # Sort by ASN so the array is a sorted index like EdgeListBuild.asns.
    order = np.argsort(asns, kind="stable")
    asns = asns[order]
    cone_size = cone_size[order]

    n_lines = len(asn_list)
    self_listed_share = (n_self_listed / n_lines) if n_lines else 0.0

    return CustomerCones(
        path=os.fspath(path),
        clique=clique,
        total_size=total_size,
        asns=asns,
        cone_size=cone_size,
        n_comment_lines=n_comment_lines,
        self_listed_share=self_listed_share,
    )


def customer_cone_sizes(
    snapshot: AsRelSnapshot, asns: Iterable[int]
) -> np.ndarray:
    """Compute customer-cone sizes directly from the relationship file.

    The customer cone of an AS is the set of ASes reachable from it by
    following provider-to-customer links downwards (the ``-1`` rows, where
    ``as1`` is the provider and ``as2`` its customer). Its size is the number
    of *distinct* ASes in that set including the AS itself, so an AS with no
    customers has cone size 1.

    This recomputes from the ``as-rel`` snapshot what the shipped ``ppdc-ases``
    file states, so the two can be cross-checked: CAIDA's inference also folds
    peer and sibling relationships into the "provider-peer-customer" cone, so
    the two need not agree exactly, but the direct-customer reachability
    computed here is a lower bound a consistent file cannot fall below.

    A provider-to-customer adjacency is built once from the ``-1`` rows; each
    requested ASN is then expanded by breadth-first search. The visited set
    makes the traversal count every reached AS once and terminates on the
    cycles that AS relationships can contain.

    Args:
        snapshot: The parsed relationship snapshot.
        asns: The ASNs to compute cone sizes for, in the order the result
            should follow.

    Returns:
        An int64 array aligned with ``asns``; entry ``i`` is the customer-cone
        size of ``asns[i]`` (distinct ASes reachable downwards, including
        itself, minimum 1).
    """
    rel = snapshot.relationships
    c2p = rel[rel[:, 2] == C2P]
    providers = c2p[:, 0]
    customers = c2p[:, 1]

    # provider -> list of direct customers, built once. Group by provider via a
    # sort so the adjacency is a slice per provider rather than a Python dict of
    # lists rebuilt per edge.
    order = np.argsort(providers, kind="stable")
    sorted_prov = providers[order]
    sorted_cust = customers[order]
    unique_prov, starts = np.unique(sorted_prov, return_index=True)
    ends = np.append(starts[1:], sorted_prov.shape[0])
    adjacency: dict[int, np.ndarray] = {
        int(p): sorted_cust[s:e]
        for p, s, e in zip(unique_prov, starts, ends)
    }

    from collections import deque

    sizes: list[int] = []
    for asn in asns:
        root = int(asn)
        visited = {root}
        queue = deque([root])
        while queue:
            node = queue.popleft()
            children = adjacency.get(node)
            if children is None:
                continue
            for child in children:
                c = int(child)
                if c not in visited:
                    visited.add(c)
                    queue.append(c)
        sizes.append(len(visited))

    return np.array(sizes, dtype=np.int64)


# --------------------------------------------------------------------------
# From snapshot to edge list: this is where the modelling choices live.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeListBuild:
    """A snapshot turned into node indices and a directed edge list.

    The three parameters that distinguish one build from another are recorded
    alongside the result so the notebook can report what each one did, and the
    counts make the effect of exclusion and of dropping peering auditable.

    Attributes:
        asns: ``(n,)`` int64, the ASNs surviving exclusion, sorted ascending.
            Node index ``i`` is ASN ``asns[i]``; the sorted order is what makes
            :func:`index_of` a binary search.
        edges: ``(k, 2)`` int64 of node indices, ``tail -> head`` in the vote
            direction. Duplicates and self-loops are *not* removed here; that is
            :func:`pagerank.graph.from_edge_list`'s job, which also counts them.
        peering: ``"both"`` or ``"none"``, the peering convention used.
        vote: ``"customer_to_provider"`` or ``"provider_to_customer"``, which
            way a customer-provider link points.
        excluded_asns: The ASNs excluded, sorted ascending.
        n_c2p_used: Customer-provider relationships that contributed an edge
            (i.e. survived exclusion).
        n_p2p_used: Peer-to-peer relationships that survived exclusion. Each
            contributes two edges under ``both`` and none under ``none``.
        n_relationships_excluded: Relationships dropped because an endpoint was
            in ``excluded_asns``.
        n_asns_dropped: ASNs present in the snapshot but absent from ``asns``,
            i.e. those that only ever appeared in excluded relationships.
        snapshot_path: The path of the snapshot this build came from.
    """

    asns: np.ndarray
    edges: np.ndarray
    peering: str
    vote: str
    excluded_asns: tuple[int, ...]
    n_c2p_used: int
    n_p2p_used: int
    n_relationships_excluded: int
    n_asns_dropped: int
    snapshot_path: str

    @property
    def n(self) -> int:
        """Number of nodes (distinct surviving ASNs)."""
        return int(self.asns.shape[0])


def index_of(asns: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Map ASNs to node indices against a sorted ASN array.

    Args:
        asns: Sorted-ascending int64 array of ASNs (an :class:`EdgeListBuild`'s
            ``asns``).
        query: ASNs to look up.

    Returns:
        An int64 array ``pos`` with ``asns[pos] == query`` elementwise.

    Raises:
        KeyError: If any ASN in ``query`` is not present, naming the first one.
    """
    query = np.asarray(query, dtype=np.int64)
    pos = np.searchsorted(asns, query)
    # searchsorted can return n for values past the end; clip before indexing.
    clipped = np.clip(pos, 0, asns.shape[0] - 1)
    bad = asns[clipped] != query
    if np.any(bad):
        first = int(query[np.argmax(bad)])
        raise KeyError(f"ASN {first} is not in the index")
    return pos


def build_edge_list(
    snapshot: AsRelSnapshot,
    *,
    peering: Literal["both", "none"] = "both",
    vote: Literal[
        "customer_to_provider", "provider_to_customer"
    ] = "customer_to_provider",
    exclude_asns: Iterable[int] = (),
) -> EdgeListBuild:
    """Turn a snapshot into an index map and a directed edge list.

    Three modelling choices are parameters rather than defaults baked into the
    data, so notebook 01 can build and compare every variant from one read:

    * ``vote`` -- which endpoint of a customer-provider link the edge points
      to. The default ``customer_to_provider`` is the reading the lab notebook
      argued for: a customer paying for transit endorses its provider, so the
      edge runs customer -> provider and mass flows up towards the core.
    * ``peering`` -- whether a peer-to-peer link contributes edges at all.
      ``both`` adds an edge each way (peering is symmetric); ``none`` adds
      nothing, which makes the transit-free clique dangle.
    * ``exclude_asns`` -- ASNs to drop entirely, with every relationship that
      touches them. Used to study the effect of the IXP ASes.

    Exclusion happens *before* the ASN set is built, so an AS that only ever
    appeared next to an excluded one disappears from the graph and is counted
    in ``n_asns_dropped``.

    Args:
        snapshot: The parsed snapshot.
        peering: Peering convention, ``"both"`` or ``"none"``.
        vote: Customer-provider edge direction.
        exclude_asns: ASNs to drop, with all their relationships.

    Returns:
        An :class:`EdgeListBuild`. Edges are node indices into ``asns``, not
        ASNs. Duplicates and self-loops are left in for ``from_edge_list`` to
        remove and count.

    Raises:
        ValueError: If ``peering`` or ``vote`` is not one of the allowed values.
    """
    if peering not in ("both", "none"):
        raise ValueError(f"peering must be 'both' or 'none', got {peering!r}")
    if vote not in ("customer_to_provider", "provider_to_customer"):
        raise ValueError(
            "vote must be 'customer_to_provider' or 'provider_to_customer', "
            f"got {vote!r}"
        )

    rel = snapshot.relationships
    as1, as2, kind = rel[:, 0], rel[:, 1], rel[:, 2]

    excluded = np.array(sorted(set(int(a) for a in exclude_asns)), dtype=np.int64)
    excluded_tuple = tuple(int(a) for a in excluded)

    # Distinct ASNs in the whole snapshot, for the dropped-count below.
    all_asns = np.unique(rel[:, :2])

    # Drop relationships touching an excluded ASN, before anything else.
    if excluded.size:
        touched = np.isin(as1, excluded) | np.isin(as2, excluded)
    else:
        touched = np.zeros(rel.shape[0], dtype=bool)
    n_relationships_excluded = int(touched.sum())
    keep = ~touched

    as1, as2, kind = as1[keep], as2[keep], kind[keep]

    # The surviving ASN set defines the node indices. Sorted so index_of works.
    if as1.size:
        asns = np.unique(np.concatenate([as1, as2]))
    else:
        asns = np.array([], dtype=np.int64)
    n_asns_dropped = int(all_asns.shape[0] - asns.shape[0])

    is_c2p = kind == C2P
    is_p2p = kind == P2P
    n_c2p_used = int(is_c2p.sum())
    n_p2p_used = int(is_p2p.sum())

    # Build edges as ASN pairs first, then map once to node indices.
    tails: list[np.ndarray] = []
    heads: list[np.ndarray] = []

    # Customer-provider rows are stored (provider, customer) = (as1, as2).
    prov, cust = as1[is_c2p], as2[is_c2p]
    if vote == "customer_to_provider":
        tails.append(cust)
        heads.append(prov)
    else:  # provider_to_customer
        tails.append(prov)
        heads.append(cust)

    # Peer-to-peer rows: symmetric under "both", absent under "none".
    if peering == "both":
        pa, pb = as1[is_p2p], as2[is_p2p]
        tails.append(pa)
        heads.append(pb)
        tails.append(pb)
        heads.append(pa)

    tail_asns = (
        np.concatenate(tails) if tails else np.array([], dtype=np.int64)
    )
    head_asns = (
        np.concatenate(heads) if heads else np.array([], dtype=np.int64)
    )

    if tail_asns.size:
        edges = np.column_stack(
            [index_of(asns, tail_asns), index_of(asns, head_asns)]
        ).astype(np.int64)
    else:
        edges = np.empty((0, 2), dtype=np.int64)

    return EdgeListBuild(
        asns=asns.astype(np.int64),
        edges=edges,
        peering=peering,
        vote=vote,
        excluded_asns=excluded_tuple,
        n_c2p_used=n_c2p_used,
        n_p2p_used=n_p2p_used,
        n_relationships_excluded=n_relationships_excluded,
        n_asns_dropped=n_asns_dropped,
        snapshot_path=snapshot.path,
    )


# --------------------------------------------------------------------------
# Persisting a build under data/processed/
# --------------------------------------------------------------------------


def _md5_of_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    """Return the hex md5 of a file, read in 1 MiB chunks.

    Stored in the sidecar so a loaded build can be traced back to the exact raw
    snapshot it was built from, without copying the raw data into the repo.
    """
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_root() -> str:
    """The repository root, taken as the parent of the ``src/`` directory.

    This module lives at ``src/pagerank/caida.py``, so the root is three levels
    up. Used only to turn an absolute snapshot path into a repository-relative
    one for the sidecar.
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _portable_snapshot_path(path: str) -> str:
    """Return a path safe to write into a sidecar: no machine-specific prefix.

    A path under the repository root is stored relative to it
    (``data/raw/20260901.as-rel.bz2``); anything else is reduced to its
    basename. The rule is that nothing the project writes carries an absolute
    local path, even into untracked files.
    """
    absolute = os.path.abspath(path)
    root = _repo_root()
    if absolute == root or absolute.startswith(root + os.sep):
        return os.path.relpath(absolute, root)
    return os.path.basename(path)


def save_build(build: EdgeListBuild, path: str | os.PathLike) -> None:
    """Write a build to ``path`` as an ``.npz`` plus a ``.json`` sidecar.

    The arrays go into ``np.savez_compressed`` under ``path``; the scalar
    fields and the md5 of the raw snapshot go into ``<path>.json`` beside it.
    The split keeps the large arrays compressed while the metadata stays
    human-readable.

    Args:
        build: The build to persist.
        path: Target path for the ``.npz`` (the sidecar is ``<path>.json``).
    """
    name = os.fspath(path)
    np.savez_compressed(name, asns=build.asns, edges=build.edges)
    # savez_compressed appends .npz if absent; mirror that for the sidecar.
    npz_name = name if name.endswith(".npz") else name + ".npz"

    # Compute the md5 from the original (possibly absolute) path, which is where
    # the file actually is, *before* reducing the path to a portable form.
    try:
        snapshot_md5 = _md5_of_file(build.snapshot_path)
    except OSError:
        # The raw snapshot may be absent (it is never in the repo); record the
        # absence rather than failing the save.
        snapshot_md5 = None

    sidecar = {
        "peering": build.peering,
        "vote": build.vote,
        "excluded_asns": list(build.excluded_asns),
        "n_c2p_used": build.n_c2p_used,
        "n_p2p_used": build.n_p2p_used,
        "n_relationships_excluded": build.n_relationships_excluded,
        "n_asns_dropped": build.n_asns_dropped,
        "snapshot_path": _portable_snapshot_path(build.snapshot_path),
        "snapshot_md5": snapshot_md5,
    }
    with open(npz_name + ".json", "w", encoding="ascii") as handle:
        json.dump(sidecar, handle, indent=2, sort_keys=True)


def load_build(path: str | os.PathLike) -> EdgeListBuild:
    """Read back a build saved by :func:`save_build`.

    Args:
        path: The ``.npz`` path passed to :func:`save_build` (with or without
            the ``.npz`` suffix).

    Returns:
        An :class:`EdgeListBuild` equal to the one saved: arrays identical,
        scalar fields restored from the sidecar.
    """
    name = os.fspath(path)
    npz_name = name if name.endswith(".npz") else name + ".npz"
    with np.load(npz_name) as data:
        asns = data["asns"].astype(np.int64)
        edges = data["edges"].astype(np.int64)
    with open(npz_name + ".json", "r", encoding="ascii") as handle:
        sidecar = json.load(handle)

    return EdgeListBuild(
        asns=asns,
        edges=edges,
        peering=sidecar["peering"],
        vote=sidecar["vote"],
        excluded_asns=tuple(sidecar["excluded_asns"]),
        n_c2p_used=sidecar["n_c2p_used"],
        n_p2p_used=sidecar["n_p2p_used"],
        n_relationships_excluded=sidecar["n_relationships_excluded"],
        n_asns_dropped=sidecar["n_asns_dropped"],
        snapshot_path=sidecar["snapshot_path"],
    )
