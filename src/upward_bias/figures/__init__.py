"""Recreate the four main and four supplementary figures from normalized data."""
from pathlib import Path

from .descriptive import figure1, figure2
from .calibration import metric_figure, calibration_cdfs
from .supplementary import stratified_means, decision_cdfs, no_decision_cdfs


def generate_figures(data_dir=None, output_dir="outputs/figures", results=None):
    """Save eight vector PDFs and matching PNGs; return the PDF paths.

    ``results`` is the dictionary returned by ``analysis.run_analysis``.
    When omitted, descriptive and point-estimate results are recomputed without
    bootstrap resampling and saved beside the figure directory.
    """
    if results is None:
        from ..analysis import run_analysis
        results = run_analysis(data_dir, Path(output_dir).parent / "results",
                               bootstrap_iterations=0)
    result = results
    paths = [figure1(result, output_dir), figure2(result, output_dir),
             metric_figure(result["calibration"]["all_reports"], output_dir, "figure3", "Figure 3"),
             calibration_cdfs(data_dir, output_dir),
             stratified_means(data_dir, output_dir), decision_cdfs(data_dir, output_dir),
             no_decision_cdfs(data_dir, output_dir),
             metric_figure(result["calibration"]["nd_only"], output_dir, "figure_s4", "Supplementary Figure 4")]
    return paths
