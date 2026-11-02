# Data guide

The release uses four aligned CSV tables in [`data/processed/`](../data/processed/).
They contain the numerical inputs used by the paper, without forum text, user
identifiers, paper titles, author/contact information, or original paper IDs.
UTF-8 encoding, comma delimiters, and a header row are used throughout.

## At a glance

| Table | Unit | Rows |
| :--- | :--- | ---: |
| `datasets.csv` | Conference edition or ARR cycle | 9 |
| `self_reports.csv` | Quantitative self-report record | 1,885 |
| `score_histograms.csv` | Score/count bin for a dataset, source, and score type | 469 |
| `iclr_population.csv` | ICLR population submission record | 19,084 |

The study's **2,467 self-report records** comprise 1,885 individual rows and
**582 Paper Copilot reports supplied as aggregate histograms**. The latter
remain aggregate counts; this release does not invent 582 individual identities.
Of the ICLR population rows, 18,783 have scores and 301 have no supplied rating.

## Dataset identifiers

`event_id` links the tables. The identifiers are `arr_2024_02`, `arr_2024_06`,
`arr_2025_02`, `iclr_2024`, `iclr_2025`, `www_2025`, `kdd_2025`, `cvpr_2025`,
and `aaai_2025`. ARR cycles map to ACL 2024, EMNLP 2024, and ACL 2025 for
acceptance-rate comparisons. A cycle and its associated conference are related
cohorts, not individually linked submissions.

## Column dictionary

### `datasets.csv`

| Columns | Meaning |
| :--- | :--- |
| `event_id`, `conference`, `year`, `cycle`, `label` | Stable key, family, calendar year, ARR cycle where applicable, and display label |
| `associated_conference` | Conference edition used for the acceptance benchmark |
| `population_acceptance_rate` | Published main-conference acceptance fraction used in descriptive comparisons |
| `main_plus_findings_acceptance_rate` | ARR combined main-plus-Findings fraction; **not** Findings alone |
| `calibration_reference_acceptance_rate` | Reference fraction used for decision calibration; ICLR 2024 uses 2,260/7,262, distinct from its published rate |
| `report_records_zhihu`, `report_records_reddit`, `report_records_papercopilot` | Original dataset-level collection counts, including records with incomplete fields |
| `platform_mapping` | `per_record` or `pooled_only`; see the WWW/KDD limitation below |
| `score_population_available` | 1 for ARR/ICLR, 0 for the other acceptance-only population benchmarks |

### `self_reports.csv`

| Columns | Meaning |
| :--- | :--- |
| `report_id`, `event_id`, `platform` | Generated row identifier, dataset key, and source (`zhihu`, `reddit`, or `pooled`) |
| `review_score_raw` | Supplied mean review score before the analysis-specific rounding |
| `review_score_processed` | ARR mean rounded to the nearest 0.5; other review means rounded to two decimals |
| `meta_review_score` | ARR meta-review score, kept separate from the review score |
| `novelty_score`, `quality_score` | Supplied WWW/KDD mean novelty and quality scores; not substituted for review scores |
| `decision` | `main`, `findings`, `accepted`, `rejected`, `withdrawn`, or an empty field |
| `decision_raw` | Original decision code: `M`, `F`, `A`, `R`, `W`, or empty |
| `source_row` | One-based original quantitative-row position; not a person or submission identifier |

An **empty field means unavailable**, not zero or rejection. A blank decision
does not establish that a report preceded decision release. ARR acceptance-rate
comparisons count `main` as accepted; Findings is a distinct outcome. Reports
may repeat the same submission across platforms or ARR cycles; IDs identify
records and cannot be used to claim unique papers.

### `score_histograms.csv`

| Columns | Meaning |
| :--- | :--- |
| `event_id`, `group`, `platform`, `score_type` | Dataset, `population`/`sample`, source, and `review`/`meta` score type |
| `score`, `count` | Preserved source score/bin value and integer frequency |

Population rows use `platform=population`. Samples use `zhihu`, `reddit`, or
`papercopilot`. Each histogram is normalized separately during analysis. ARR
histograms retain published 0.5-point review categories and the supplied
Paper Copilot bin midpoints; these representations must not be silently merged
by extra rounding. Zero-valued ARR categories excluded by the original
preprocessing are absent. Review and meta-review population counts
overlap and must not be added as independent submissions. Population review
counts are 16,677 for ARR and 18,783 for ICLR; ARR meta-review counts total 16,348.

### `iclr_population.csv`

`population_id` is a generated row ID; `event_id` identifies the year.
`review_score` is the mean of supplied reviewer ratings rounded to two decimals.
`status_raw` retains the public status label, while `status` maps presentation
categories to `accepted`, Reject to `rejected`, Withdraw to `withdrawn`, and
remaining statuses to `unknown`. `source_row` preserves original JSON order.
Score-based analyses exclude blank scores. Decision-stratified CDFs additionally
exclude unknown statuses; F1 treats score-bearing unknown statuses as negatives.

## Ordering, scope, and sources

Histogram bin order and within-source record order are preserved. This matters
for reproducing seeded bootstrap draws. Full-sample calibration uses processed
histograms; no-decision calibration uses the raw mean-score fields. The plotting
code applies ARR's 0.5-point categorization only where the figure requires it.

WWW/KDD source files combine platforms without a reliable row-level mapping.
Their report rows therefore use `platform=pooled`; dataset-level Zhihu/Reddit
counts are retained as metadata and must not be assigned to individual rows.

[`data/provenance.json`](../data/provenance.json) lists original filenames and
SHA-256 hashes, all public discussion/benchmark URLs from the paper's Table S2,
and the migration's representation choices. Population sources include
[ARR statistics](https://stats.aclrollingreview.org/),
[ARR health](https://github.com/acl-org/arr-health), and
[Paper Copilot's ICLR archive](https://github.com/papercopilot/paperlists/tree/main/iclr).
