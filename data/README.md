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

**This project uses `serial-1`.** The defensible reasons: it is the longer
series (1998 onwards, against October 2015 for `serial-2`), and it is derived
from a single measurement pipeline — BGP table snapshots — whereas `serial-2`
merges in peering links inferred from BGP communities and traceroute. Keeping to
one pipeline means an unexpected result can be attributed to the AS graph rather
than to the mixture of methods that produced it.

> **To verify before submission.** The CAIDA dataset page states that the
> `serial-1` inference method is similar to the one described in *AS
> Relationships, Customer Cones, and Validation* — the AS Rank paper. It does
> **not** state which variant the [AS Rank web application](https://asrank.caida.org/)
> is actually computed from. Research question 1 compares our ranking against AS
> Rank, so this matters: if AS Rank is built on `serial-2`, the comparison is
> between graphs with different edge sets and the write-up must say so instead of
> calling it apples-to-apples. Check the AS Rank documentation and its API
> metadata, and if it remains unclear, ask `topology-info@caida.org` rather than
> assuming.

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

Two different wordings are in circulation, and they are not equivalent. Both are
recorded here rather than picking one silently.

**Form A — as given on the dataset page:**

> The CAIDA AS Relationships Dataset, \<date range used\>,
> https://www.caida.org/catalog/datasets/as-relationships/

**Form B — as given in the Public AUA, the governing legal text:**

> The CAIDA UCSD \<Dataset Name\> - \<dates used\>,
> https://catalog.caida.org/dataset/\<dataset-URL\>

They differ in the dataset name (`CAIDA UCSD ...` versus `CAIDA ...`) and in the
domain (`catalog.caida.org/dataset/...` versus
`caida.org/catalog/datasets/...`).

> **To verify before submission.** Determine which wording applies. The Public
> AUA is the agreement actually accepted at download time, so its template
> (Form B) is the safer default, and the specific text shown during the download
> process takes precedence over both of the above. Resolve the exact dataset name
> and catalogue URL then, and record the decision in `LAB_NOTEBOOK.md`. If it
> stays ambiguous, citing both forms is acceptable and costs nothing;
> under-citing is the only failure mode with consequences.

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
