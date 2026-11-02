"""Read the four normalized CSV tables used by every released analysis.

Aggregate histogram rows are expanded only in memory. Paper Copilot counts
never become invented report identities, and missing decisions remain missing.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

ARR_CYCLES = ("2402", "2406", "2502")
ICLR_YEARS = ("24", "25")
PLATFORMS = ("zhihu", "reddit")
ARR_LABELS = {"2402": "ARR 2024 February", "2406": "ARR 2024 June", "2502": "ARR 2025 February"}
ICLR_LABELS = {"24": "ICLR 2024", "25": "ICLR 2025"}

_EVENT_IDS = {("ARR", "2402"): "arr_2024_02", ("ARR", "2406"): "arr_2024_06",
              ("ARR", "2502"): "arr_2025_02", ("ICLR", "24"): "iclr_2024",
              ("ICLR", "25"): "iclr_2025"}
_NUMERIC = {"review_score_raw", "review_score_processed", "meta_review_score",
            "novelty_score", "quality_score", "review_score", "score",
            "population_acceptance_rate", "main_plus_findings_acceptance_rate",
            "calibration_reference_acceptance_rate"}
_INTEGER = {"source_row", "count", "year", "report_records_zhihu",
            "report_records_reddit", "report_records_papercopilot", "score_population_available"}


def event_id(family: str, cycle: str | int | None = None) -> str:
    """Accept canonical IDs or the concise family/cycle labels used in the paper."""
    if cycle is None:
        value = str(family)
        if value.startswith("iclr_") or value.startswith("arr_"):
            return value
        if value in {"24", "25", "2024", "2025"}:
            return f"iclr_{value if len(value) == 4 else '20' + value}"
        return value
    family, cycle = family.upper(), str(cycle)
    if family == "ICLR":
        cycle = cycle[-2:]
    try:
        return _EVENT_IDS[(family, cycle)]
    except KeyError as exc:
        raise ValueError(f"Unknown score dataset: {family} {cycle}") from exc


def _table(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in row.keys() & _NUMERIC:
            row[key] = float(row[key]) if row[key] != "" else None
        for key in row.keys() & _INTEGER:
            row[key] = int(row[key]) if row[key] != "" else None
    return rows


class Dataset:
    """A small in-memory store; ``data_dir`` may be the repo, data, or processed folder."""

    def __init__(self, data_dir: str | Path | None = None):
        folder = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parents[2] / "data" / "processed"
        if not (folder / "datasets.csv").is_file():
            folder = folder / "processed" if (folder / "processed" / "datasets.csv").is_file() else folder / "data" / "processed"
        if not (folder / "datasets.csv").is_file():
            raise FileNotFoundError("Expected datasets.csv, self_reports.csv, score_histograms.csv, and iclr_population.csv in data/processed")
        self.data_dir = folder
        self.datasets = _table(folder / "datasets.csv")
        self._metadata = {row["event_id"]: row for row in self.datasets}
        self._reports = _table(folder / "self_reports.csv")
        self._histograms = _table(folder / "score_histograms.csv")
        self._population = _table(folder / "iclr_population.csv")

    def metadata(self, identifier: str) -> dict:
        return dict(self._metadata[event_id(identifier)])

    def reports(self, event_id: str | None = None, platform: str | None = None) -> list[dict]:
        """Return quantitative rows in original source order; blank decisions are ``''``."""
        return [dict(row) for row in self._reports
                if (event_id is None or row["event_id"] == event_id)
                and (platform is None or row["platform"] == platform)]

    def population_rows(self, identifier: str) -> list[dict]:
        """ICLR rows in original order, including rows with an unreported score (None)."""
        identifier = event_id(identifier)
        return [dict(row) for row in self._population if row["event_id"] == identifier]

    def histogram(self, event_id: str, group: str = "population", platform: str = "population",
                  score_type: str = "review") -> np.ndarray:
        rows = [r for r in self._histograms if r["event_id"] == event_id and r["group"] == group
                and r["platform"] == platform and r["score_type"] == score_type]
        if not rows:
            raise ValueError(f"No {score_type} histogram for {event_id}/{group}/{platform}")
        return np.repeat(np.array([r["score"] for r in rows], dtype=float),
                         np.array([r["count"] for r in rows], dtype=int))

    def sample(self, event_id: str, platform: str, score_type: str = "review", group: str = "all") -> np.ndarray:
        if group == "all":
            return self.histogram(event_id, "sample", platform, score_type)
        if group != "nd":
            raise ValueError("Sample group must be 'all' or 'nd'")
        field = "review_score_raw" if score_type == "review" else "meta_review_score"
        return np.asarray([r[field] for r in self.reports(event_id, platform)
                           if r["decision"] == "" and r[field] is not None], dtype=float)

    def load_population(self, family, cycle, score_type="review"):
        return self.histogram(event_id(family, cycle), score_type=score_type)

    def load_processed_sample(self, family, cycle, platform, score_type="review"):
        return self.sample(event_id(family, cycle), platform, score_type)

    def load_paper_copilot(self, cycle):
        return self.sample(event_id("ARR", cycle), "papercopilot")

    def load_no_decision_sample(self, family, cycle, platform):
        return self.sample(event_id(family, cycle), platform, group="nd")

    def load_iclr_decision_sample(self, year, platform, decision):
        decision = {"A": "accepted", "R": "rejected", "Accept": "accepted", "Reject": "rejected"}.get(decision, decision)
        return np.asarray([r["review_score_raw"] for r in self.reports(event_id("ICLR", year), platform)
                           if r["decision"] == decision and r["review_score_raw"] is not None], dtype=float)

    def load_iclr_population_rows(self, year):
        rows = [r for r in self.population_rows(event_id(year)) if r["review_score"] is not None]
        return (np.asarray([r["review_score"] for r in rows], dtype=float),
                np.asarray([r["status"] for r in rows], dtype=str))

    def load_iclr_population_by_decision(self, year):
        scores, statuses = self.load_iclr_population_rows(year)
        return scores[statuses == "accepted"], scores[np.isin(statuses, ["rejected", "withdrawn"])]


Store = Dataset


@lru_cache(maxsize=8)
def get_dataset(data_dir: str | Path | None = None) -> Dataset:
    return Dataset(data_dir)


def load_population(family, cycle, score_type="review", data_dir=None):
    return get_dataset(data_dir).load_population(family, cycle, score_type)


def load_processed_sample(family, cycle, platform, score_type="review", data_dir=None):
    return get_dataset(data_dir).load_processed_sample(family, cycle, platform, score_type)


def load_paper_copilot(cycle, data_dir=None):
    return get_dataset(data_dir).load_paper_copilot(cycle)


def load_no_decision_sample(family, cycle, platform, data_dir=None):
    return get_dataset(data_dir).load_no_decision_sample(family, cycle, platform)


def load_iclr_decision_sample(year, platform, decision, data_dir=None):
    return get_dataset(data_dir).load_iclr_decision_sample(year, platform, decision)


def load_iclr_population_rows(year, data_dir=None):
    return get_dataset(data_dir).load_iclr_population_rows(year)


def load_iclr_population_by_decision(year, data_dir=None):
    return get_dataset(data_dir).load_iclr_population_by_decision(year)
