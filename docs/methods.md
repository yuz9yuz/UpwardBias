# Analysis and reproduction

The release implements the analyses used in the manuscript. All calculations
start from the four CSV tables in `data/processed/`; no external working files
or preassembled figure PDFs are required.

## Distribution and reporting comparisons

Scores are compared within their original conference scales. ARR review and
meta-review distributions are separate, overlapping cohorts. Pooled CDFs combine
records across iterations; platform comparisons standardize scores using each
iteration's population mean and standard deviation. Distribution tests are
two-sided Kolmogorov–Smirnov tests. Acceptance-rate tests are two-sided binomial
tests against published population rates, using reports with disclosed outcomes.
The reported p values are unadjusted for multiple comparisons.

Reporting proportions divide self-report counts by population counts within
score bins and use 95% Wilson intervals. ARR uses published 0.5-point categories;
ICLR uses 0.3-point bins. These are descriptive comparisons, without fitted
regressions. A score-defined acceptance bar is the population quantile implied
by the acceptance rate, not the actual conference decision rule.

Boundary distances are calculated separately within each iteration and then
pooled. ARR uses the closer of its main and main-plus-Findings bars; ICLR
distances use 0.5-point bins. No-decision-reported samples retain records with
empty decision fields. This restriction does not establish when a report was
posted.

## Historical calibration

For target iteration *t* and reference iteration *r*, the adjustment is

```text
adjusted_score(u) = sample_target_quantile(u)
                  + population_reference_quantile(u)
                  - sample_reference_quantile(u)
```

Zhihu and Reddit records are pooled before computing quantiles. The reference
population is an input; the target population is used only for retrospective
evaluation. The correction is evaluated at the target sample's midpoint ranks.
The empirical distribution of the resulting adjusted scores is the estimated
population, even if the unsorted adjustment curve crosses. Scores are not
clipped and no monotonicity fit is applied.

Three transfers are evaluated: ICLR 2024 to 2025, ARR February to June 2024,
and ARR June 2024 to February 2025. The full-report variant uses processed
score histograms. The restricted variant uses raw scores from records without
disclosed decisions in both target and reference samples. Both retain the full
reference population.

## Evaluation and bootstrap

Distributional error uses the absolute difference between linearly interpolated
empirical quantiles at 20,000 evenly spaced midpoint levels, in original score
units. This numerical convention is retained for both point estimates and
resampling; displayed CDFs remain empirical step functions.

ICLR decision evaluation uses the historical score-bearing acceptance fraction
`2260 / 7262`. A paper is above the score-defined bar when its inclusive
upper-tail score proportion is no greater than that fraction. F1 treats accepted
submissions as positive and other statuses as negative. The withdrawal metric
counts submissions above the population-defined bar but placed below it by
an estimate; it does not estimate counterfactual acceptance.

Each bootstrap replicate independently resamples target and reference reports
at their original sizes, with both populations fixed. The default uses 1,000
replicates per comparison and NumPy's `default_rng` generator. For the three
full-report pairs, seeds are `20261101`, `20261102`, and `20261103`; the three
restricted-sample seeds add `100`. A win means a strictly smaller calibrated
distance in the same replicate. Percentile ranges summarize replicate
statistics, not uncertainty in the bootstrap win probability.

## Outputs and portability

`outputs/tables/` contains human-readable CSV summaries and complete JSON panel
data. `outputs/figures/` contains eight vector PDFs and their PNG previews.
`outputs/run_info.json` records package versions, seeds, and input checksums.

The plotting code draws every panel directly. It uses the manuscript's physical
figure width, shared sizes and palette, including purple (`#75518E`) for the
bias-agnostic estimator and blue (`#356F8F`) for the bias-aware estimator.
Different installed fonts can change glyph widths slightly; Arial is preferred
and DejaVu Sans is the fallback. The repository includes no font downloads.
