"""Tests for :mod:`pagerank.caida`.

The reader is checked against a tiny synthetic ``as-rel`` file written by the
test, exercising header parsing, the four relationship counts, the per-line
step tag, and the three error cases, plus the invariant that reading the bz2
and plain-text forms of the same bytes yields equal arrays. The builder is
checked on the same fixture for edge direction, peering, exclusion and the
save/load round-trip.

Writing the fixture inline keeps the expected numbers next to the input, so a
change to one forces a change to the other. The real snapshot is reconciled in
notebook 01, not here: the suite must run without the raw data present.
"""

from __future__ import annotations

import bz2
from pathlib import Path

import numpy as np
import pytest

from pagerank.caida import C2P, P2P, AsRelSnapshot, read_as_rel

# --------------------------------------------------------------------------
# Synthetic fixture
#
# Mirrors the design's worked example. Five relationships, two steps, a clique
# of two, one IXP AS, one source line. Chosen so that every count below is
# small enough to read off by eye.
# --------------------------------------------------------------------------

SYNTHETIC = """\
# input clique: 1 2
# IXP ASes: 9
# source:topology|BGP|20260901|routeviews|x
# step 1: clique
1|3|-1
2|3|-1
# step 2: peering
1|2|0
3|9|0
9|4|-1
"""


@pytest.fixture()
def plain_file(tmp_path: Path) -> Path:
    """Write the synthetic snapshot as plain text and return its path."""
    path = tmp_path / "toy.as-rel.txt"
    path.write_text(SYNTHETIC, encoding="ascii")
    return path


@pytest.fixture()
def bz2_file(tmp_path: Path) -> Path:
    """Write the same bytes bz2-compressed, so the suffix selects the opener."""
    path = tmp_path / "toy.as-rel.bz2"
    with bz2.open(path, "wt", encoding="ascii") as handle:
        handle.write(SYNTHETIC)
    return path


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------


def test_reader_parses_the_header(plain_file: Path) -> None:
    """Clique, IXP list and source text come through in file order."""
    snap = read_as_rel(plain_file)
    assert snap.clique == (1, 2)
    assert snap.ixp_asns == (9,)
    assert snap.sources == ("topology|BGP|20260901|routeviews|x",)


def test_reader_counts_comment_lines(plain_file: Path) -> None:
    """Every ``#`` line counts: clique, IXP, source and the two step lines."""
    snap = read_as_rel(plain_file)
    assert snap.n_comment_lines == 5


def test_reader_counts_relationships_by_type(plain_file: Path) -> None:
    """Three c2p, two p2p, five in total."""
    snap = read_as_rel(plain_file)
    assert snap.relationships.shape == (5, 3)
    assert snap.n_c2p == 3
    assert snap.n_p2p == 2


def test_reader_records_the_relationship_triples(plain_file: Path) -> None:
    """Rows are kept verbatim and in file order."""
    snap = read_as_rel(plain_file)
    expected = np.array(
        [
            [1, 3, C2P],
            [2, 3, C2P],
            [1, 2, P2P],
            [3, 9, P2P],
            [9, 4, C2P],
        ],
        dtype=np.int64,
    )
    assert np.array_equal(snap.relationships, expected)


def test_reader_tags_each_relationship_with_its_step(plain_file: Path) -> None:
    """The first two rows fall under step 1, the last three under step 2."""
    snap = read_as_rel(plain_file)
    assert np.array_equal(snap.step, np.array([1, 1, 2, 2, 2], dtype=np.int64))


def test_bz2_and_plain_text_agree(plain_file: Path, bz2_file: Path) -> None:
    """The suffix must be the only difference between the two reads."""
    plain = read_as_rel(plain_file)
    compressed = read_as_rel(bz2_file)
    assert plain.clique == compressed.clique
    assert plain.ixp_asns == compressed.ixp_asns
    assert plain.sources == compressed.sources
    assert plain.n_comment_lines == compressed.n_comment_lines
    assert np.array_equal(plain.relationships, compressed.relationships)
    assert np.array_equal(plain.step, compressed.step)


def test_reader_ignores_blank_lines_without_counting_them(tmp_path: Path) -> None:
    """Blank lines are neither relationships nor comments."""
    text = "# input clique: 1\n\n1|2|-1\n\n"
    path = tmp_path / "blanks.as-rel.txt"
    path.write_text(text, encoding="ascii")
    snap = read_as_rel(path)
    assert snap.n_comment_lines == 1
    assert snap.relationships.shape == (1, 3)


def test_relationships_before_the_first_step_are_tagged_zero(tmp_path: Path) -> None:
    """A relationship appearing before any ``# step`` comment gets step 0."""
    text = "# input clique: 1\n1|2|-1\n# step 1: x\n2|3|-1\n"
    path = tmp_path / "earlystep.as-rel.txt"
    path.write_text(text, encoding="ascii")
    snap = read_as_rel(path)
    assert np.array_equal(snap.step, np.array([0, 1], dtype=np.int64))


# --------------------------------------------------------------------------
# Reader error cases: each raises ValueError naming the 1-based line number.
# --------------------------------------------------------------------------


def test_reader_rejects_an_unknown_relationship_value(tmp_path: Path) -> None:
    text = "# input clique: 1\n1|2|-1\n1|2|2\n"
    path = tmp_path / "badrel.as-rel.txt"
    path.write_text(text, encoding="ascii")
    with pytest.raises(ValueError, match=r"line 3:.*2"):
        read_as_rel(path)


def test_reader_rejects_a_self_relationship(tmp_path: Path) -> None:
    text = "# input clique: 1\n5|5|-1\n"
    path = tmp_path / "selfrel.as-rel.txt"
    path.write_text(text, encoding="ascii")
    with pytest.raises(ValueError, match=r"line 2:.*self-relationship.*5"):
        read_as_rel(path)


def test_reader_rejects_a_line_with_the_wrong_field_count(tmp_path: Path) -> None:
    text = "# input clique: 1\n1|2\n"
    path = tmp_path / "twofields.as-rel.txt"
    path.write_text(text, encoding="ascii")
    with pytest.raises(ValueError, match=r"line 2:.*malformed"):
        read_as_rel(path)


def test_reader_returns_an_as_rel_snapshot(plain_file: Path) -> None:
    """Sanity: the public return type is the documented dataclass."""
    assert isinstance(read_as_rel(plain_file), AsRelSnapshot)


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------

from pagerank.caida import (  # noqa: E402  (grouped with the builder tests)
    EdgeListBuild,
    build_edge_list,
    index_of,
    load_build,
    save_build,
)
from pagerank.graph import from_edge_list  # noqa: E402
from pagerank.power import pagerank_power  # noqa: E402


def _asn_edges(build: EdgeListBuild) -> set[tuple[int, int]]:
    """Map a build's index edges back to ASN pairs, as an unordered set."""
    return {
        (int(build.asns[t]), int(build.asns[h]))
        for t, h in build.edges
    }


def test_builder_default_direction_and_peering(plain_file: Path) -> None:
    """Defaults: c2p votes customer -> provider, peering both ways.

    The worked example in the design: the five relationships produce the seven
    ASN edges below, over the five ASNs 1,2,3,4,9.
    """
    snap = read_as_rel(plain_file)
    build = build_edge_list(snap)

    assert np.array_equal(build.asns, np.array([1, 2, 3, 4, 9], dtype=np.int64))
    assert build.n == 5
    assert _asn_edges(build) == {
        (3, 1),  # 1|3|-1 -> customer 3 to provider 1
        (3, 2),  # 2|3|-1
        (1, 2),  # 1|2|0  both directions
        (2, 1),
        (3, 9),  # 3|9|0  both directions
        (9, 3),
        (4, 9),  # 9|4|-1 -> customer 4 to provider 9
    }
    assert build.n_c2p_used == 3
    assert build.n_p2p_used == 2
    assert build.edges.shape[0] == 7


def test_builder_peering_none_drops_peer_edges(plain_file: Path) -> None:
    """``peering='none'`` keeps only the three customer-provider edges."""
    snap = read_as_rel(plain_file)
    build = build_edge_list(snap, peering="none")
    assert build.edges.shape[0] == 3
    assert _asn_edges(build) == {(3, 1), (3, 2), (4, 9)}


def test_builder_vote_reversal_flips_the_ranking(tmp_path: Path) -> None:
    """One c2p link ranks the endpoint the edge points to above the other.

    On a two-AS file ``1|2|-1`` (1 is provider, 2 is customer): under
    ``customer_to_provider`` the edge is 2 -> 1, so the provider 1 ranks first;
    under ``provider_to_customer`` it is 1 -> 2, so the customer 2 ranks first.
    This is the whole point of making the vote a parameter.
    """
    path = tmp_path / "twoas.as-rel.txt"
    path.write_text("# input clique: 1\n1|2|-1\n", encoding="ascii")
    snap = read_as_rel(path)

    for vote, winner_asn in (
        ("customer_to_provider", 1),
        ("provider_to_customer", 2),
    ):
        build = build_edge_list(snap, vote=vote)
        graph = from_edge_list(
            [(int(t), int(h)) for t, h in build.edges], build.n
        )
        result = pagerank_power(graph, alpha=0.85)
        top_index = int(result.vector.argmax())
        assert int(build.asns[top_index]) == winner_asn, vote


def test_builder_exclusion_drops_relationships_and_orphaned_asns(
    plain_file: Path,
) -> None:
    """Excluding AS 9 drops its two relationships and AS 4 with them.

    AS 4 appears only in ``9|4|-1``; once that relationship is dropped, AS 4
    has no surviving relationship and must disappear from the node set. AS 9's
    other relationship ``3|9|0`` is dropped too, but AS 3 survives via its c2p
    links, so only 4 and 9 are gone.
    """
    snap = read_as_rel(plain_file)
    build = build_edge_list(snap, exclude_asns=[9])

    assert np.array_equal(build.asns, np.array([1, 2, 3], dtype=np.int64))
    assert build.excluded_asns == (9,)
    assert build.n_relationships_excluded == 2
    assert build.n_asns_dropped == 2  # ASNs 4 and 9
    assert _asn_edges(build) == {(3, 1), (3, 2), (1, 2), (2, 1)}


def test_builder_rejects_bad_parameters(plain_file: Path) -> None:
    """``peering`` and ``vote`` are validated."""
    snap = read_as_rel(plain_file)
    with pytest.raises(ValueError, match="peering"):
        build_edge_list(snap, peering="sometimes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="vote"):
        build_edge_list(snap, vote="upwards")  # type: ignore[arg-type]


def test_index_of_raises_on_an_unknown_asn() -> None:
    """A query ASN absent from the sorted set names itself in the error."""
    asns = np.array([1, 2, 3, 9], dtype=np.int64)
    assert np.array_equal(index_of(asns, np.array([9, 1])), np.array([3, 0]))
    with pytest.raises(KeyError, match="7"):
        index_of(asns, np.array([1, 7]))


def test_save_load_round_trips_exactly(plain_file: Path) -> None:
    """A saved build loads back equal: arrays identical, scalars restored."""
    snap = read_as_rel(plain_file)
    build = build_edge_list(snap, exclude_asns=[9])

    target = plain_file.parent / "build.npz"
    save_build(build, target)
    loaded = load_build(target)

    assert np.array_equal(loaded.asns, build.asns)
    assert np.array_equal(loaded.edges, build.edges)
    assert loaded.peering == build.peering
    assert loaded.vote == build.vote
    assert loaded.excluded_asns == build.excluded_asns
    assert loaded.n_c2p_used == build.n_c2p_used
    assert loaded.n_p2p_used == build.n_p2p_used
    assert loaded.n_relationships_excluded == build.n_relationships_excluded
    assert loaded.n_asns_dropped == build.n_asns_dropped
    # The fixture snapshot lives in tmp_path, outside the repository, so its
    # path is reduced to a basename on save and that is what loads back.
    assert loaded.snapshot_path == plain_file.name


def test_save_writes_the_snapshot_md5_into_the_sidecar(plain_file: Path) -> None:
    """The sidecar records the md5 of the raw snapshot, and no machine path.

    The md5 is computed from the file as it actually sits on disk (an absolute
    path here), but the ``snapshot_path`` written into the sidecar must carry no
    absolute local path: a tmp_path file outside the repository is reduced to
    its basename.
    """
    import hashlib
    import json

    snap = read_as_rel(plain_file)
    build = build_edge_list(snap)
    target = plain_file.parent / "build2.npz"
    save_build(build, target)

    sidecar = json.loads((plain_file.parent / "build2.npz.json").read_text())
    expected = hashlib.md5(plain_file.read_bytes()).hexdigest()
    assert sidecar["snapshot_md5"] == expected
    assert sidecar["snapshot_path"] == plain_file.name
    # No value anywhere in the sidecar may be an absolute local path.
    for value in sidecar.values():
        if isinstance(value, str):
            assert not value.startswith("/"), value


def test_portable_snapshot_path_relativises_paths_under_the_repo() -> None:
    """A path under the repo is stored relative; anything else is a basename."""
    from pagerank.caida import _portable_snapshot_path, _repo_root

    root = _repo_root()
    under_repo = str(Path(root) / "data" / "raw" / "20260901.as-rel.bz2")
    assert _portable_snapshot_path(under_repo) == "data/raw/20260901.as-rel.bz2"

    outside = "/somewhere/else/20260901.as-rel.bz2"
    assert _portable_snapshot_path(outside) == "20260901.as-rel.bz2"


# --------------------------------------------------------------------------
# Customer cones (ppdc-ases)
#
# A synthetic cone file mirroring the real one's shape: two header comment
# lines (an empty inferred-clique line and a total-size line), then three ASes.
# AS 10 lists itself among its members and has a duplicate member (20 twice),
# so its distinct cone is {10, 20, 30} -> size 3. AS 20 lists itself and one
# customer -> {20, 30} -> size 2. AS 30 does NOT list itself and has no
# customers, so the line is just "30" -> {30} -> size 1. Two of the three lines
# self-list, so the share is 2/3.
# --------------------------------------------------------------------------

CONES = """\
# inferred clique:
# total size: 3
10 10 20 20 30
20 20 30
30
"""


@pytest.fixture()
def cone_plain_file(tmp_path: Path) -> Path:
    """Write the synthetic cone file as plain text and return its path."""
    path = tmp_path / "toy.ppdc-ases.txt"
    path.write_text(CONES, encoding="ascii")
    return path


@pytest.fixture()
def cone_bz2_file(tmp_path: Path) -> Path:
    """Write the same cone bytes bz2-compressed, so the suffix selects the opener."""
    path = tmp_path / "toy.ppdc-ases.bz2"
    with bz2.open(path, "wt", encoding="ascii") as handle:
        handle.write(CONES)
    return path


def test_cone_reader_parses_the_header(cone_plain_file: Path) -> None:
    """Empty inferred-clique line gives an empty tuple; total size is read."""
    from pagerank.caida import CustomerCones, read_ppdc_ases

    cones = read_ppdc_ases(cone_plain_file)
    assert isinstance(cones, CustomerCones)
    assert cones.clique == ()
    assert cones.total_size == 3
    assert cones.n_comment_lines == 2


def test_cone_reader_sizes_count_distinct_asns_including_self(
    cone_plain_file: Path,
) -> None:
    """Cone size is the distinct ASNs on the line including the leading AS.

    AS 10: {10, 20, 30} -> 3 (the duplicate 20 is absorbed).
    AS 20: {20, 30}     -> 2.
    AS 30: {30}         -> 1 (no customers, not self-listed, still size 1).
    """
    from pagerank.caida import read_ppdc_ases

    cones = read_ppdc_ases(cone_plain_file)
    assert np.array_equal(cones.asns, np.array([10, 20, 30], dtype=np.int64))
    assert np.array_equal(cones.cone_size, np.array([3, 2, 1], dtype=np.int64))


def test_cone_reader_measures_the_self_listed_share(cone_plain_file: Path) -> None:
    """Two of the three lines repeat their leading AS among the members."""
    from pagerank.caida import read_ppdc_ases

    cones = read_ppdc_ases(cone_plain_file)
    assert cones.self_listed_share == pytest.approx(2 / 3)


def test_cone_reader_returns_asns_sorted_ascending(tmp_path: Path) -> None:
    """ASNs come back sorted even when the file lists them out of order."""
    from pagerank.caida import read_ppdc_ases

    text = "# inferred clique:\n# total size: 3\n30 30\n10 10 20\n20 20\n"
    path = tmp_path / "unsorted.ppdc-ases.txt"
    path.write_text(text, encoding="ascii")
    cones = read_ppdc_ases(path)
    assert np.array_equal(cones.asns, np.array([10, 20, 30], dtype=np.int64))
    # Aligned: 10 -> {10,20} = 2, 20 -> {20} = 1, 30 -> {30} = 1.
    assert np.array_equal(cones.cone_size, np.array([2, 1, 1], dtype=np.int64))


def test_cone_reader_bz2_and_plain_text_agree(
    cone_plain_file: Path, cone_bz2_file: Path
) -> None:
    """The suffix must be the only difference between the two reads."""
    from pagerank.caida import read_ppdc_ases

    plain = read_ppdc_ases(cone_plain_file)
    compressed = read_ppdc_ases(cone_bz2_file)
    assert plain.clique == compressed.clique
    assert plain.total_size == compressed.total_size
    assert plain.n_comment_lines == compressed.n_comment_lines
    assert plain.self_listed_share == compressed.self_listed_share
    assert np.array_equal(plain.asns, compressed.asns)
    assert np.array_equal(plain.cone_size, compressed.cone_size)


def test_cone_reader_reads_a_non_empty_clique_line(tmp_path: Path) -> None:
    """When the header lists clique ASNs after the colon, they are captured."""
    from pagerank.caida import read_ppdc_ases

    text = "# inferred clique: 3356 174\n# total size: 1\n10 10\n"
    path = tmp_path / "clique.ppdc-ases.txt"
    path.write_text(text, encoding="ascii")
    cones = read_ppdc_ases(path)
    assert cones.clique == (3356, 174)


def test_cone_reader_rejects_a_non_integer_token(tmp_path: Path) -> None:
    """A malformed data line raises ValueError naming the 1-based line number."""
    from pagerank.caida import read_ppdc_ases

    text = "# inferred clique:\n# total size: 1\n10 20 x\n"
    path = tmp_path / "bad.ppdc-ases.txt"
    path.write_text(text, encoding="ascii")
    with pytest.raises(ValueError, match=r"line 3:.*malformed"):
        read_ppdc_ases(path)


# --------------------------------------------------------------------------
# Customer cones recomputed from the relationship file
# --------------------------------------------------------------------------


def test_customer_cone_sizes_on_the_synthetic_fixture(plain_file: Path) -> None:
    """Cones by hand on the shared fixture's c2p rows (1->3, 2->3, 9->4).

    AS1 reaches {1, 3}; AS2 reaches {2, 3}; AS3 has no customers, so {3};
    AS9 reaches {9, 4}. Sizes 2, 2, 1, 2.
    """
    from pagerank.caida import customer_cone_sizes

    snap = read_as_rel(plain_file)
    sizes = customer_cone_sizes(snap, [1, 2, 3, 9])
    assert np.array_equal(sizes, np.array([2, 2, 1, 2], dtype=np.int64))


def test_customer_cone_sizes_counts_shared_customers_once(tmp_path: Path) -> None:
    """A diamond shares a customer: distinct reachable ASes, not path count.

    Chain ``1|2|-1, 1|3|-1, 2|4|-1, 3|4|-1``: AS1 reaches 2 and 3, both of
    which reach 4. AS1's cone is {1, 2, 3, 4} = 4, not 5: AS4 is reached by
    two paths but counted once.
    """
    from pagerank.caida import customer_cone_sizes

    text = "# input clique: 1\n1|2|-1\n1|3|-1\n2|4|-1\n3|4|-1\n"
    path = tmp_path / "chain.as-rel.txt"
    path.write_text(text, encoding="ascii")
    snap = read_as_rel(path)
    sizes = customer_cone_sizes(snap, [1, 2, 3, 4])
    # AS1 -> {1,2,3,4}=4; AS2 -> {2,4}=2; AS3 -> {3,4}=2; AS4 -> {4}=1.
    assert np.array_equal(sizes, np.array([4, 2, 2, 1], dtype=np.int64))
