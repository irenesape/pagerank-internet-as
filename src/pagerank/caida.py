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
import os
from dataclasses import dataclass
from typing import Final

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
