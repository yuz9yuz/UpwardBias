"""Historical quantile-delta calibration and its retrospective evaluation.

Only the estimator used in the manuscript is implemented here. Reference and
target reports are pooled before calibration; target populations are used for
evaluation, never as correction inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .data import (
    load_iclr_population_rows,
    load_no_decision_sample,
    load_population,
    load_processed_sample,
)


W1_GRID_SIZE = 20_000
DEFAULT_BOOTSTRAP_SEED = 20261101
PLATFORMS = ("zhihu", "reddit")


@dataclass(frozen=True)
class Comparison:
    family: str
    target: str
    reference: str
    reference_acceptance_rate: float

    @property
    def target_title(self) -> str:
        return dataset_title(self.family, self.target)

    @property
    def reference_title(self) -> str:
        return dataset_title(self.family, self.reference)


COMPARISONS = (
    Comparison("ICLR", "25", "24", 2260 / 7262),
    Comparison("ARR", "2406", "2402", 0.213),
    Comparison("ARR", "2502", "2406", 0.208),
)


@dataclass
class DataBundle:
    target_population: np.ndarray
    target_sample: np.ndarray
    reference_population: np.ndarray
    reference_sample: np.ndarray


def dataset_title(family: str, cycle: str) -> str:
    if family == "ICLR":
        return f"ICLR 20{cycle}"
    return {"2402": "ARR 2024 Feb", "2406": "ARR 2024 Jun",
            "2502": "ARR 2025 Feb"}[cycle]


def pool_bundle(comparison: Comparison, data_dir: Path | str | None = None,
                nd_only: bool = False) -> DataBundle:
    """Concatenate Zhihu then Reddit, preserving reproducible record order."""
    loader = load_no_decision_sample if nd_only else load_processed_sample
    family, target, reference = comparison.family, comparison.target, comparison.reference
    return DataBundle(
        load_population(family, target, data_dir=data_dir),
        np.concatenate([loader(family, target, p, data_dir=data_dir) for p in PLATFORMS]),
        load_population(family, reference, data_dir=data_dir),
        np.concatenate([loader(family, reference, p, data_dir=data_dir) for p in PLATFORMS]),
    )


def quantile_delta_population(bundle: DataBundle) -> np.ndarray:
    """Return one adjusted score per target report, at midpoint sample ranks.

    The pointwise adjustment need not be monotone. Its evaluated scores define
    an empirical distribution; no clipping or monotonicity fit is applied.
    """
    n = len(bundle.target_sample)
    if n == 0 or len(bundle.reference_sample) == 0:
        raise ValueError("Calibration requires nonempty target and reference samples")
    ranks = (np.arange(n, dtype=float) + 0.5) / n
    return (np.quantile(bundle.target_sample, ranks)
            + np.quantile(bundle.reference_population, ranks)
            - np.quantile(bundle.reference_sample, ranks))


def upper_cdf(observations: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Inclusive empirical upper-tail proportions, P(X >= query)."""
    values = np.sort(observations)
    return (len(values) - np.searchsorted(values, query, side="left")) / len(values)


def wasserstein_1d(first: np.ndarray, second: np.ndarray,
                   grid_size: int = W1_GRID_SIZE) -> float:
    """Score-space distance using the paper's interpolated-quantile convention."""
    ranks = (np.arange(grid_size, dtype=float) + 0.5) / grid_size
    return float(np.mean(np.abs(np.quantile(first, ranks) - np.quantile(second, ranks))))


def decision_metrics(scores: np.ndarray, labels: np.ndarray, truth: np.ndarray,
                     prediction: np.ndarray, acceptance_rate: float) -> tuple[float, int]:
    """F1 and above-bar withdrawals placed below the estimated score bar.

    Actual acceptance is positive for F1; all other statuses are negative.
    Withdrawal counts describe score-based standing, not counterfactual decisions.
    """
    predicted_accept = prediction <= acceptance_rate
    actual_accept = np.isin(labels, ["accepted", "Accept"])
    true_positive = int(np.sum(predicted_accept & actual_accept))
    predicted_positive, actual_positive = int(predicted_accept.sum()), int(actual_accept.sum())
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    wrong_withdrawals = np.sum((truth <= acceptance_rate) & (prediction > acceptance_rate)
                               & np.isin(labels, ["withdrawn", "Withdraw"]))
    return f1, int(wrong_withdrawals)


def _metric(before: float, after: float, lower_is_better: bool) -> dict:
    change = 100 * (after / before - 1)
    return {"bias_agnostic": before, "bias_aware": after,
            "relative_change_percent": change,
            "relative_improvement_percent": 100 * (1 - after / before) if lower_is_better else change,
            "lower_is_better": lower_is_better}


def evaluate_bundle(comparison: Comparison, bundle: DataBundle,
                    data_dir: Path | str | None = None, nd_only: bool = False) -> dict:
    adjusted = quantile_delta_population(bundle)
    result = {
        "comparison": {"family": comparison.family, "target": comparison.target,
                       "reference": comparison.reference, "platform": "pooled",
                       "reference_acceptance_rate": comparison.reference_acceptance_rate},
        "target_title": comparison.target_title, "reference_title": comparison.reference_title,
        "sample_group": "Pooled Zhihu + Reddit; " + ("no decision reported (ND-only)" if nd_only else "all reports"),
        "target_population_n": len(bundle.target_population),
        "reference_population_n": len(bundle.reference_population),
        "target_sample_n": len(bundle.target_sample),
        "reference_sample_n": len(bundle.reference_sample),
        "wasserstein": _metric(wasserstein_1d(bundle.target_population, bundle.target_sample),
                               wasserstein_1d(bundle.target_population, adjusted), True),
    }
    if comparison.family == "ICLR":
        scores, labels = load_iclr_population_rows(comparison.target, data_dir=data_dir)
        np.testing.assert_array_equal(np.sort(scores), np.sort(bundle.target_population))
        truth = upper_cdf(scores, scores)
        before = decision_metrics(scores, labels, truth, upper_cdf(bundle.target_sample, scores),
                                  comparison.reference_acceptance_rate)
        after = decision_metrics(scores, labels, truth, upper_cdf(adjusted, scores),
                                 comparison.reference_acceptance_rate)
        result["f1"] = _metric(before[0], after[0], False)
        result["f1"]["absolute_change"] = after[0] - before[0]
        result["withdrawals"] = _metric(before[1], after[1], True)
        result["withdrawals"]["above_bar_withdrawals_n"] = int(np.sum(
            (truth <= comparison.reference_acceptance_rate)
            & np.isin(labels, ["withdrawn", "Withdraw"])))
    return result


def pooled_results(data_dir: Path | str | None = None, nd_only: bool = False) -> dict:
    return {
        "method": "additive quantile-delta calibration",
        "pooling": "concatenate Zhihu and Reddit in target and reference before calibration",
        "reports": "ND-only reports" if nd_only else "all scored reports",
        "wasserstein_method": "linear empirical quantiles at 20,000 midpoint ranks; original score units",
        "comparisons": [evaluate_bundle(c, pool_bundle(c, data_dir, nd_only), data_dir, nd_only)
                        for c in COMPARISONS],
    }


run_calibration = pooled_results


@lru_cache(maxsize=None)
def _quantile_plan(n: int, grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = (n - 1) * ((np.arange(grid_size, dtype=float) + 0.5) / grid_size)
    lower = np.floor(index).astype(np.intp)
    return lower, np.minimum(lower + 1, n - 1), index - lower


def _interpolate(sorted_values: np.ndarray, plan: tuple) -> np.ndarray:
    """Match NumPy linear-quantile arithmetic, including its upper-end formula."""
    lower, upper, fraction = plan
    a, b = sorted_values[..., lower], sorted_values[..., upper]
    difference = b - a
    result = a + difference * fraction
    np.subtract(b, difference * (1 - fraction), out=result, where=fraction >= 0.5)
    return result


def _summary(values: np.ndarray) -> dict:
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "percentile_95_interval": np.percentile(values, [2.5, 97.5]).tolist()}


def bootstrap_case(bundle: DataBundle, seed: int, iterations: int = 1000,
                   batch_size: int = 32) -> dict:
    """Resample both report sets independently, keeping populations fixed.

    Batches cache interpolation indices only. Every resample and adjusted score
    array is sorted; the estimator and numerical distance match point estimates.
    """
    if iterations < 1 or batch_size < 1:
        raise ValueError("iterations and batch_size must be positive")
    nt, nr = len(bundle.target_sample), len(bundle.reference_sample)
    ranks = (np.arange(nt, dtype=float) + 0.5) / nt
    reference_q = np.quantile(bundle.reference_population, ranks)
    grid = (np.arange(W1_GRID_SIZE, dtype=float) + 0.5) / W1_GRID_SIZE
    truth_q = np.quantile(bundle.target_population, grid)
    target_plan, reference_plan = _quantile_plan(nt, nt), _quantile_plan(nr, nt)
    distance_plan = _quantile_plan(nt, W1_GRID_SIZE)

    def evaluate(target: np.ndarray, reference: np.ndarray):
        target, reference = np.sort(target, axis=-1), np.sort(reference, axis=-1)
        adjusted = (_interpolate(target, target_plan) + reference_q
                    - _interpolate(reference, reference_plan))
        baseline_q = np.ascontiguousarray(_interpolate(target, distance_plan))
        adjusted_q = np.ascontiguousarray(_interpolate(np.sort(adjusted, axis=-1), distance_plan))
        return (np.mean(np.abs(baseline_q - truth_q), axis=-1),
                np.mean(np.abs(adjusted_q - truth_q), axis=-1), adjusted)

    point_before, point_after, adjusted = evaluate(bundle.target_sample, bundle.reference_sample)
    np.testing.assert_array_equal(adjusted, quantile_delta_population(bundle))
    np.testing.assert_equal(point_before, wasserstein_1d(bundle.target_population, bundle.target_sample))
    np.testing.assert_equal(point_after, wasserstein_1d(bundle.target_population, adjusted))
    rng = np.random.default_rng(seed)
    baseline, aware = np.empty(iterations), np.empty(iterations)
    for start in range(0, iterations, batch_size):
        stop = min(start + batch_size, iterations)
        targets, references = [], []
        for _ in range(start, stop):
            targets.append(rng.choice(bundle.target_sample, nt, replace=True))
            references.append(rng.choice(bundle.reference_sample, nr, replace=True))
        baseline[start:stop], aware[start:stop], _ = evaluate(np.asarray(targets), np.asarray(references))
    reduction = baseline - aware
    wins = int(np.count_nonzero(aware < baseline))
    return {
        "seed": seed, "iterations": iterations, "target_sample_n": nt, "reference_sample_n": nr,
        "target_population_n": len(bundle.target_population),
        "reference_population_n": len(bundle.reference_population),
        "point_estimates": {"bias_agnostic_w1": float(point_before), "bias_aware_w1": float(point_after),
                            "relative_reduction_percent": float(100 * (1 - point_after / point_before))},
        "strict_improvement_count": wins, "tie_count": int(np.count_nonzero(aware == baseline)),
        "probability_improved": wins / iterations,
        "bias_agnostic_w1": _summary(baseline), "bias_aware_w1": _summary(aware),
        "absolute_w1_reduction": _summary(reduction),
        "relative_reduction_percent": _summary(100 * reduction / baseline),
    }


def run_bootstrap(data_dir: Path | str | None = None, iterations: int = 1000,
                  seed: int = DEFAULT_BOOTSTRAP_SEED) -> dict:
    rows = []
    for nd_only in (False, True):
        for index, comparison in enumerate(COMPARISONS):
            case_seed = seed + (100 if nd_only else 0) + index
            row = bootstrap_case(pool_bundle(comparison, data_dir, nd_only), case_seed, iterations)
            row.update({"sample_group": "nd_only" if nd_only else "all_reports",
                        "family": comparison.family, "target": comparison.target,
                        "reference": comparison.reference, "target_title": comparison.target_title,
                        "reference_title": comparison.reference_title})
            rows.append(row)
    return {"iterations_per_comparison": iterations, "base_seed": seed,
            "random_generator": "numpy.random.default_rng (PCG64)",
            "w1_grid_size": W1_GRID_SIZE,
            "resampling": "independent target/reference report resamples; populations fixed",
            "improvement_definition": "strictly smaller calibrated distance in the same replicate",
            "interval_definition": "2.5th and 97.5th percentiles of replicate statistics",
            "comparisons": rows}
