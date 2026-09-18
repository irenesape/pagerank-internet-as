# Data

Nothing in `raw/` or `processed/` is tracked by git. This file is the recipe for
rebuilding both from scratch.

## Source

**CAIDA AS Relationships**, produced by the Center for Applied Internet Data
Analysis at UC San Diego.
Dataset page: <https://www.caida.org/catalog/datasets/as-relationships/>

CAIDA infers business relationships between autonomous systems from BGP data
and publishes them as monthly snapshots. Two variants exist:

| Variant | Coverage | How links are inferred |
| --- | --- | --- |
| `serial-1` | 1998 – present, one file per month | Relationships inferred from RouteViews and RIPE RIS BGP table snapshots taken over a five-day window |
| `serial-2` | Oct 2015 – present, one file per month | The serial-1 graph plus additional peering links inferred from BGP communities at IXP looking glasses and from traceroute data |

**This project uses `serial-1`.** Reason: it is the variant the published
inference method and the AS Rank rankings are built on, so comparing our
PageRank ranking against [AS Rank](https://asrank.caida.org/) is an
apples-to-apples comparison. `serial-2` adds peering links from a different
measurement pipeline, which would confound research question 1.

Record the exact snapshot date used in `LAB_NOTEBOOK.md`. The graph changes
month to month, so a result is only reproducible if the snapshot is pinned.

## How to obtain it

1. Go to the access page and complete the short form:
   <https://www.caida.org/catalog/datasets/request_user_info_forms/as_relationships>
2. Download one `serial-1` monthly file. Filenames follow the pattern
   `YYYYMMDD.as-rel.txt.bz2`.
3. Put it in `data/raw/` unmodified, keeping the original filename. Do not
   decompress in place and do not rename: the filename *is* the provenance
   record.

Every `serial-1` data directory ships its own `README` describing the file
format. Read it before writing the parser and confirm the field semantics
there rather than trusting second-hand descriptions, including this one.

At the time of writing, `serial-1` files are plain text, one relationship per
line, with `#` introducing comment lines. Relationship lines are
pipe-separated, where the trailing field distinguishes a provider-to-customer
link from a peer-to-peer one. Directionality is the part worth being careful
about: it determines the transpose you feed to PageRank, and getting it
backwards produces a plausible-looking but wrong ranking.

## Terms of use

The publicly accessible CAIDA datasets are distributed under the
**CAIDA Acceptable Use Agreement for Publicly Accessible Datasets**:
<https://www.caida.org/about/legal/aua/public_aua/>

What that means in practice for this repository:

- The agreement grants a limited, non-transferable licence to copy, modify and
  use the data. It is *not* an open-data licence, and it does not grant
  redistribution rights. Hence `data/raw/` is git-ignored: publishing the
  snapshots here would be redistribution.
- Any publication, web page or public presentation using the data **must** cite
  it (see below).
- Publications using the dataset are expected to be reported back to CAIDA:
  <https://www.caida.org/catalog/datasets/publications/report-publication>

Derived aggregates that cannot reconstruct the source data — the ranking tables
and figures in `results/` — are fine to publish with the citation attached.

## Required citation

The form of citation required by the agreement:

> The CAIDA AS Relationships Dataset, \<date range used\>,
> https://www.caida.org/catalog/datasets/as-relationships/

Fill in the actual snapshot date before submitting. For the methodology behind
the inference, cite the accompanying paper as well:

> M. Luckie, B. Huffaker, k. claffy, A. Dhamdhere, V. Giotsas. *AS
> Relationships, Customer Cones, and Validation.* Proceedings of the Internet
> Measurement Conference (IMC), pp. 243–256, October 2013.

If the AS Rank comparison in research question 1 uses the AS Rank web
application or its API, cite that as a separate source.

## Layout

```
data/
├── raw/         downloaded snapshots, byte-identical to the source
├── processed/   derived artefacts: edge lists, AS-number-to-index maps, sparse matrices
└── README.md    this file
```

Keep `raw/` immutable and write every derived artefact to `processed/`. If a
processing bug appears you re-derive `processed/` and never have to re-download,
and you can always tell which files came from CAIDA and which came from your
code.

---

*Content on this page describing the dataset and its terms is summarised in our
own words from the [CAIDA AS Relationships dataset
page](https://www.caida.org/catalog/datasets/as-relationships/) and the
[Public AUA](https://www.caida.org/about/legal/aua/public_aua/); consult those
pages for the authoritative text.*
