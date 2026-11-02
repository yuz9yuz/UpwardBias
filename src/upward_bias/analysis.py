"""Descriptive comparisons and reproducible manuscript results from tidy data."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, ks_2samp

from .calibration import DEFAULT_BOOTSTRAP_SEED, pooled_results, run_bootstrap
from .data import get_dataset, load_iclr_decision_sample, load_iclr_population_by_decision


CYCLES = {"ARR": ("2402", "2406", "2502"), "ICLR": ("24", "25")}
PLATFORMS = ("zhihu", "reddit")
EVENTS = ("arr_2024_02", "arr_2024_06", "arr_2025_02", "iclr_2024", "iclr_2025",
          "www_2025", "kdd_2025", "cvpr_2025", "aaai_2025")


def event_id(family: str, cycle: str) -> str:
    return (f"arr_20{cycle[:2]}_{cycle[2:]}" if family == "ARR"
            else f"iclr_20{cycle}")


def summarize(values: np.ndarray) -> dict:
    """Inclusive empirical CDF and population-divisor standard deviation."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot summarize an empty score array")
    support, counts = np.unique(values, return_counts=True)
    return {"n": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
            "support": support.tolist(), "counts": counts.tolist(),
            "cdf": (np.cumsum(counts) / values.size).tolist()}


def compare_scores(population: np.ndarray, sample: np.ndarray) -> dict:
    test = ks_2samp(population, sample, alternative="two-sided")
    pop, sam = summarize(population), summarize(sample)
    gap = sam["mean"] - pop["mean"]
    return {"population": pop, "sample": sam, "mean_gap": gap,
            "standardized_mean_gap": gap / pop["std"],
            "ks_statistic": float(test.statistic), "p_value": float(test.pvalue)}


def build_panel(family: str, score_type: str = "review", data_dir=None) -> dict:
    """Pooled raw-score comparison for Figure 1A-C; score types never mix."""
    store = get_dataset(data_dir)
    platforms = (*PLATFORMS, "papercopilot") if family == "ARR" and score_type == "review" else PLATFORMS
    populations, samples = [], []
    for cycle in CYCLES[family]:
        event = event_id(family, cycle)
        populations.append(store.histogram(event, score_type=score_type))
        samples.extend(store.sample(event, p, score_type=score_type) for p in platforms)
    return {"family": family, "score_type": score_type, "cycles": list(CYCLES[family]),
            "platforms": list(platforms),
            **compare_scores(np.concatenate(populations), np.concatenate(samples))}


def build_results(include_meta: bool = True, data_dir=None) -> dict:
    groups = [("ARR", "review"), ("ARR", "meta"), ("ICLR", "review")]
    return {"panels": [build_panel(family, score_type, data_dir)
                       for family, score_type in groups if include_meta or score_type != "meta"]}


def stratified_scores(data_dir=None) -> list[dict]:
    store, rows = get_dataset(data_dir), []
    for family in CYCLES:
        for score_type in (("review", "meta") if family == "ARR" else ("review",)):
            for cycle in CYCLES[family]:
                event = event_id(family, cycle)
                population = store.histogram(event, score_type=score_type)
                platforms = (*PLATFORMS, "papercopilot") if family == "ARR" and score_type == "review" else PLATFORMS
                for platform in platforms:
                    sample = store.sample(event, platform, score_type=score_type)
                    rows.append({"event_id": event, "family": family, "cycle": cycle,
                                 "score_type": score_type, "platform": platform,
                                 **compare_scores(population, sample)})
    return rows


def standardized_platform_scores(data_dir=None) -> list[dict]:
    store, rows = get_dataset(data_dir), []
    for platform in (*PLATFORMS, "papercopilot"):
        populations, samples = [], []
        groups = [("ARR", "review")] if platform == "papercopilot" else [
            ("ARR", "review"), ("ARR", "meta"), ("ICLR", "review")]
        for family, score_type in groups:
            for cycle in CYCLES[family]:
                event = event_id(family, cycle)
                population = store.histogram(event, score_type=score_type)
                sample = store.sample(event, platform, score_type=score_type)
                mean, std = population.mean(), population.std()
                populations.append((population - mean) / std)
                samples.append((sample - mean) / std)
        rows.append({"platform": platform,
                     **compare_scores(np.concatenate(populations), np.concatenate(samples))})
    return rows


def acceptance_rates(data_dir=None) -> list[dict]:
    store, rows = get_dataset(data_dir), []
    for event in EVENTS:
        records = [r for r in store.reports(event_id=event) if r["decision"]]
        accepted = sum(r["decision"] in {"accepted", "main"} for r in records)
        n = len(records)
        population_rate = float(store.metadata(event)["population_acceptance_rate"])
        rows.append({"event_id": event, "population_acceptance_rate": population_rate,
                     "sample_n": n, "sample_accepted": accepted,
                     "sample_acceptance_rate": accepted / n,
                     "inflation_ratio": accepted / n / population_rate,
                     "p_value": float(binomtest(accepted, n, population_rate).pvalue)})
    return rows


def decision_stratified_scores(data_dir=None) -> list[dict]:
    rows = []
    for decision, index in (("accepted", 0), ("rejected", 1)):
        populations, samples = [], []
        for cycle in CYCLES["ICLR"]:
            populations.append(load_iclr_population_by_decision(cycle, data_dir=data_dir)[index])
            samples.extend(load_iclr_decision_sample(cycle, p, decision, data_dir=data_dir)
                           for p in PLATFORMS)
        population, sample = np.concatenate(populations), np.concatenate(samples)
        rows.append({"decision": decision,
                     "sample_mean_population_percentile": float(np.mean(population <= sample.mean())),
                     **compare_scores(population, sample)})
    return rows


def no_decision_scores(data_dir=None) -> list[dict]:
    """Figure 2E: ARR review reports use the available 0.5-point categories."""
    store, rows = get_dataset(data_dir), []
    for family, score_type in (("ARR", "review"), ("ARR", "meta"), ("ICLR", "review")):
        populations, samples = [], []
        for cycle in CYCLES[family]:
            event = event_id(family, cycle)
            populations.append(store.histogram(event, score_type=score_type))
            for platform in PLATFORMS:
                sample = store.sample(event, platform, score_type=score_type, group="nd")
                if family == "ARR" and score_type == "review":
                    sample = np.rint(sample * 2) / 2
                samples.append(sample)
        rows.append({"family": family, "score_type": score_type,
                     **compare_scores(np.concatenate(populations), np.concatenate(samples))})
    return rows


def wilson_interval(successes: np.ndarray, trials: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = 1.959963984540054
    proportions = successes / trials
    denominator = 1 + z ** 2 / trials
    center = (proportions + z ** 2 / (2 * trials)) / denominator
    half = z / denominator * np.sqrt(proportions * (1 - proportions) / trials
                                    + z ** 2 / (4 * trials ** 2))
    return np.clip(center - half, 0, 1), np.clip(center + half, 0, 1)


def reporting_proportions(family: str, data_dir=None) -> dict:
    store, populations, samples, rates = get_dataset(data_dir), [], [], []
    for cycle in CYCLES[family]:
        event = event_id(family, cycle)
        population = store.histogram(event)
        populations.append(population)
        rates.append((len(population), float(store.metadata(event)["population_acceptance_rate"])))
        samples.extend(r["review_score_raw"] for r in store.reports(event_id=event)
                       if r["platform"] in PLATFORMS and r["review_score_raw"] is not None)
    population, sample = np.concatenate(populations), np.asarray(samples)
    fraction = sum(n * rate for n, rate in rates) / sum(n for n, _ in rates)
    threshold = float(np.percentile(population, (1 - fraction) * 100))
    bins = np.arange(.75, 4.75, .5) if family == "ARR" else np.arange(1, 10, .3)
    n_population, _ = np.histogram(population, bins)
    n_sample, _ = np.histogram(sample, bins)
    centers = (bins[:-1] + bins[1:]) / 2
    visible = ((centers >= 1) & (centers <= 4) if family == "ARR" else
               (centers >= threshold - 2) & (centers <= threshold + 1)) & (n_population > 0)
    lower, upper = wilson_interval(n_sample[visible], n_population[visible])
    return {"family": family, "threshold": threshold, "population_acceptance_rate": fraction,
            "bin_width": .5 if family == "ARR" else .3,
            "x": centers[visible].tolist(), "y": (n_sample[visible] / n_population[visible]).tolist(),
            "lower": lower.tolist(), "upper": upper.tolist(),
            "population_counts": n_population[visible].tolist(), "sample_counts": n_sample[visible].tolist()}


def boundary_distances(family: str, data_dir=None) -> dict:
    """Compute distances within iterations before pooling ND-only records."""
    store, population_distances, sample_distances, details = get_dataset(data_dir), [], [], []
    for cycle in CYCLES[family]:
        event = event_id(family, cycle)
        population = store.histogram(event)
        parts = [store.sample(event, p, group="nd") for p in PLATFORMS]
        sample = np.concatenate(parts)
        metadata = store.metadata(event)
        bars = {"Main" if family == "ARR" else "Accepted": float(
            np.quantile(population, 1 - float(metadata["population_acceptance_rate"])))}
        if family == "ARR":
            bars["Main + Findings"] = float(np.quantile(population, 1 - float(metadata["main_plus_findings_acceptance_rate"])))
            sample = np.rint(sample * 2) / 2
        bar_values = np.asarray(list(bars.values()))
        population_distances.append(np.round(np.min(np.abs(population[:, None] - bar_values), axis=1), 8))
        sample_distances.append(np.round(np.min(np.abs(sample[:, None] - bar_values), axis=1), 8))
        details.append({"cycle": cycle, "population_n": len(population), "sample_n": len(sample),
                        "sample_n_by_platform": dict(zip(PLATFORMS, [len(p) for p in parts])),
                        "acceptance_bar_scores": bars})
    population, sample = np.concatenate(population_distances), np.concatenate(sample_distances)
    maximum = max(population.max(), sample.max())
    if family == "ARR":
        x = np.arange(int(round(maximum / .5)) + 1) / 2
        counts = [np.asarray([np.count_nonzero(v == point) for point in x]) for v in (population, sample)]
        edges = None
    else:
        edges = np.round(np.arange(int(np.ceil(maximum / .5)) + 1) * .5, 8)
        x = (edges[:-1] + edges[1:]) / 2
        counts = [np.histogram(v, bins=edges)[0] for v in (population, sample)]
    assert counts[0].sum() == len(population) and counts[1].sum() == len(sample)
    return {"family": family, "population_n": len(population), "sample_n": len(sample),
            "cycles": details, "x": x.tolist(), "bin_width": .5,
            "bin_edges": edges.tolist() if edges is not None else None,
            "population_counts": counts[0].tolist(), "sample_counts": counts[1].tolist(),
            "population_proportions": (counts[0] / len(population)).tolist(),
            "sample_proportions": (counts[1] / len(sample)).tolist()}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run_analysis(data_dir, out_dir, bootstrap_iterations: int = 1000,
                 seed: int = DEFAULT_BOOTSTRAP_SEED) -> dict:
    """Write all numerical results from the normalized CSV data."""
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    store = get_dataset(data_dir)
    all_reports, nd_only = pooled_results(data_dir), pooled_results(data_dir, nd_only=True)
    counts = Counter(r["platform"] for r in store.reports())
    counts["papercopilot"] = sum(int(row["report_records_papercopilot"]) for row in store.datasets)
    result = {
        "record_counts_by_platform": dict(counts),
        "record_level_reports_n": len(store.reports()),
        "histogram_only_reports_n": counts["papercopilot"],
        "pooled_score_comparisons": build_results(data_dir=data_dir)["panels"],
        "stratified_score_comparisons": stratified_scores(data_dir),
        "standardized_platform_scores": standardized_platform_scores(data_dir),
        "acceptance_rates": acceptance_rates(data_dir),
        "decision_stratified_scores": decision_stratified_scores(data_dir),
        "no_decision_scores": no_decision_scores(data_dir),
        "reporting_proportions": [reporting_proportions(f, data_dir) for f in CYCLES],
        "boundary_distances": [boundary_distances(f, data_dir) for f in CYCLES],
        "calibration": {"all_reports": all_reports, "nd_only": nd_only},
    }
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_csv(output / "acceptance_rates.csv", result["acceptance_rates"])
    score_rows = []
    for row in result["stratified_score_comparisons"]:
        score_rows.append({k: row[k] for k in ("event_id", "score_type", "platform", "mean_gap", "standardized_mean_gap", "ks_statistic", "p_value")} | {
            "population_n": row["population"]["n"], "sample_n": row["sample"]["n"],
            "population_mean": row["population"]["mean"], "sample_mean": row["sample"]["mean"]})
    _write_csv(output / "score_comparisons.csv", score_rows)
    metric_rows = []
    for group, data in result["calibration"].items():
        for row in data["comparisons"]:
            for metric in ("wasserstein", "f1", "withdrawals"):
                if metric in row:
                    metric_rows.append({"sample_group": group, "target": row["target_title"],
                                        "reference": row["reference_title"], "metric": metric,
                                        **{k: row[metric][k] for k in ("bias_agnostic", "bias_aware", "relative_change_percent")}})
    _write_csv(output / "calibration_metrics.csv", metric_rows)
    if bootstrap_iterations:
        bootstrap = run_bootstrap(data_dir, iterations=bootstrap_iterations, seed=seed)
        (output / "bootstrap.json").write_text(json.dumps(bootstrap, indent=2) + "\n", encoding="utf-8")
        result["bootstrap"] = bootstrap
    else:
        (output / "bootstrap.json").unlink(missing_ok=True)
    return result
