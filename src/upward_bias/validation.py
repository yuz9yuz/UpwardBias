"""Cross-table checks for the quantitative dataset."""

from __future__ import annotations

from collections import Counter
import csv
import math

from .data import Dataset


def validate_data(data_dir=None) -> dict:
    """Validate keys, missing values, frequencies, and overlapping data views."""
    store = Dataset(data_dir)
    identifiers = [row["event_id"] for row in store.datasets]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate event identifiers in datasets.csv")
    events = set(identifiers)
    reports = store.reports()
    report_ids = [row["report_id"] for row in reports]
    if len(report_ids) != len(set(report_ids)):
        raise ValueError("Duplicate report identifiers")
    decisions = {"", "accepted", "main", "findings", "rejected", "withdrawn"}
    for row in reports:
        if row["event_id"] not in events or row["decision"] not in decisions:
            raise ValueError(f"Invalid event or decision in {row['report_id']}")
        for field in ("review_score_raw", "review_score_processed", "meta_review_score",
                      "novelty_score", "quality_score"):
            if row[field] is not None and not math.isfinite(row[field]):
                raise ValueError(f"Nonfinite score in {row['report_id']}")

    # A histogram is an alternative view of the same records, not extra papers.
    histograms = {}
    with (store.data_dir / "score_histograms.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        key = tuple(row[k] for k in ("event_id", "group", "platform", "score_type", "score"))
        if key in histograms or row["event_id"] not in events or int(row["count"]) < 0:
            raise ValueError(f"Invalid histogram row: {key}")
        if not math.isfinite(float(row["score"])):
            raise ValueError(f"Nonfinite histogram score: {key}")
        histograms[key] = int(row["count"])
    combinations = {key[:4] for key in histograms}
    fields = {"review": "review_score_processed", "meta": "meta_review_score"}
    for event, group, platform, score_type in sorted(combinations):
        if group != "sample" or platform == "papercopilot":
            continue
        actual = Counter(r[fields[score_type]] for r in store.reports(event, platform)
                         if r[fields[score_type]] is not None)
        expected = Counter({float(k[-1]): value for k, value in histograms.items()
                            if k[:4] == (event, group, platform, score_type)})
        if actual != expected:
            raise ValueError(f"Report/histogram mismatch: {event}/{platform}/{score_type}")

    population_ids = []
    population_rows = []
    for event in identifiers:
        rows = store.population_rows(event)
        population_rows.extend(rows)
        population_ids.extend(r["population_id"] for r in rows)
        if not rows:
            continue
        actual = Counter(r["review_score"] for r in rows if r["review_score"] is not None)
        expected = Counter(store.histogram(event))
        if actual != expected:
            raise ValueError(f"Population/histogram mismatch: {event}")
    if len(population_ids) != len(set(population_ids)):
        raise ValueError("Duplicate population identifiers")

    for metadata in store.datasets:
        event = metadata["event_id"]
        expected_n = metadata["report_records_zhihu"] + metadata["report_records_reddit"]
        if len(store.reports(event)) != expected_n:
            raise ValueError(f"Report count mismatch: {event}")
        if metadata["platform_mapping"] == "per_record":
            for platform in ("zhihu", "reddit"):
                if len(store.reports(event, platform)) != metadata[f"report_records_{platform}"]:
                    raise ValueError(f"Platform count mismatch: {event}/{platform}")
        copilot_n = metadata["report_records_papercopilot"]
        if copilot_n and len(store.histogram(event, "sample", "papercopilot")) != copilot_n:
            raise ValueError(f"Paper Copilot histogram count mismatch: {event}")

    aggregate_n = sum(r["report_records_papercopilot"] for r in store.datasets)
    return {
        "events": len(events),
        "individual_self_report_rows": len(reports),
        "aggregate_papercopilot_reports": aggregate_n,
        "total_reports_represented": len(reports) + aggregate_n,
        "histogram_rows": len(histograms),
        "iclr_population_rows": len(population_rows),
        "iclr_score_bearing_population_rows": sum(r["review_score"] is not None for r in population_rows),
        "cross_table_checks": "passed",
    }
