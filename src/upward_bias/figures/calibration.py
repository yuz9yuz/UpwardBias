"""Main calibration figures and the ND-only supplementary comparison."""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.path import Path as MplPath

from ..calibration import COMPARISONS, pool_bundle, quantile_delta_population
from ..style import (
    AGNOSTIC, AXES_HEIGHT, AXIS_SIZE, CDF_WIDTH, INK, NUMBER_SIZE,
    POPULATION, SAMPLE, axes, canvas, cdf, cdf_axes, panel_label, save, title,
)


def _arrow(ax, metric, offset, top, bar_width):
    before, after = metric["bias_agnostic"], metric["bias_aware"]
    start = (offset + bar_width / 2 + .035, before)
    end = (offset + 1 - bar_width / 4, after + top * (.13 if after == 0 else .025))
    options = {"arrowstyle": "-|>", "mutation_scale": 5.4, "linewidth": .65,
               "color": "#3D555D", "shrinkA": 0, "shrinkB": 0, "clip_on": False,
               "zorder": 4}
    is_f1 = top == 1
    if is_f1:
        path = MplPath([start, (offset + .5, before), (offset + .75, .99 * top), end],
                       [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        arrow = FancyArrowPatch(path=path, **options)
    else:
        arrow = FancyArrowPatch(start, end, connectionstyle="angle3,angleA=0,angleB=-90",
                                **options)
    ax.add_patch(arrow)
    change = metric["relative_change_percent"]
    text = f"{change:+.0f}%" if top == 25 else f"{change:+.1f}%"
    position = ((offset + 1.02, .99 * top) if is_f1 else
                (offset + 1.18, after + (before - after) * .47 + top * .03))
    ax.text(*position, text, ha="center", va="top" if is_f1 else "center",
            fontsize=NUMBER_SIZE, fontweight="bold", color=INK, zorder=5)


def _metric_group(ax, metric, offset, top, bar_width):
    values = [metric["bias_agnostic"], metric["bias_aware"]]
    ax.bar([offset, offset + 1], values, width=bar_width,
           color=[AGNOSTIC, SAMPLE], edgecolor="none", zorder=2)
    unit = top / AXES_HEIGHT
    for index, (value, color) in enumerate(zip(values, (AGNOSTIC, SAMPLE))):
        text = str(int(value)) if top == 25 else f"{value:.4f}"
        if value:
            ax.text(offset + index, value - 1.4 * unit, text, ha="center", va="top",
                    color="white", fontsize=NUMBER_SIZE, zorder=3)
        else:
            ax.plot([offset + index - bar_width / 2, offset + index + bar_width / 2],
                    [0, 0], linewidth=CDF_WIDTH, color=color, clip_on=False, zorder=3)
            ax.text(offset + index, unit, text, ha="center", va="bottom", color=color,
                    fontsize=NUMBER_SIZE, zorder=3)
    _arrow(ax, metric, offset, top, bar_width)


def metric_figure(results, output_dir, stem="figure3", name="Figure 3"):
    fig = canvas(145)
    rows = results["comparisons"]
    specs = [
        ("A", 0, 104, [(rows[0], "f1")], "Acceptance prediction F1", 1, "F1 Score"),
        ("B", 112, 104, [(rows[0], "withdrawals")],
         "Above-bar withdrawals\nplaced below bar", 25, "# of Submissions"),
        ("C", 224, 244, [(r, "wasserstein") for r in rows],
         "Wasserstein distance", .65, "Distance"),
    ]
    for letter, left, width, comparisons, header, top, ylabel in specs:
        axes_width = width - 25.5
        ax = axes(fig, left + 22.5, 40, axes_width)
        xlim = (-.55, 1.55) if len(comparisons) == 1 else (-.55, 7.3)
        ax.set_xlim(*xlim)
        ax.set_ylim(0, top)
        ax.set_yticks(np.arange(0, 1.01, .2) if top == 1 else
                      np.arange(0, 26, 5) if top == 25 else [0, .2, .4, .6])
        ax.set_ylabel(ylabel)
        bar_width = 19.05 / axes_width * (xlim[1] - xlim[0])
        centers, contexts = [], []
        for index, (row, metric_name) in enumerate(comparisons):
            offset = index * 2.8
            _metric_group(ax, row[metric_name], offset, top, bar_width)
            centers.append(offset + .5)
            contexts.append(f"{row['target_title']},\nreference: {row['reference_title']}")
        ax.set_xticks(centers, contexts, fontsize=AXIS_SIZE)
        ax.tick_params(axis="x", length=0, pad=2.15)
        panel_label(fig, letter, left + .4, 25.75)
        title(fig, header, left + 22.5 + axes_width / 2, 21)
    fig.legend(handles=[Patch(facecolor=AGNOSTIC, label="Bias-agnostic estimator"),
                        Patch(facecolor=SAMPLE, label="Bias-aware estimator")],
               loc="upper center", bbox_to_anchor=(.5, 1 - 5 / 145), ncol=2,
               frameon=False, borderpad=0, borderaxespad=0, fontsize=NUMBER_SIZE,
               handlelength=1.5, handletextpad=.35, columnspacing=1.5)
    return save(fig, output_dir, stem, name)


def calibration_cdfs(data_dir, output_dir):
    fig = canvas(150)
    width = (468 - 16) / 3
    for index, comparison in enumerate(COMPARISONS):
        bundle = pool_bundle(comparison, data_dir)
        adjusted = quantile_delta_population(bundle)
        left = index * (width + 8)
        ax = axes(fig, left + 24.5, 40, width - 28.7)
        limits = (0, 5.1) if comparison.family == "ARR" else (.9, 10.1)
        for values, color, linestyle in ((bundle.target_population, POPULATION, "-"),
                                        (bundle.target_sample, AGNOSTIC, ":"),
                                        (adjusted, SAMPLE, "--")):
            cdf(ax, values, color, linestyle, limits)
        cdf_axes(ax, comparison.family)
        title(fig, f"{comparison.target_title},\nreference: {comparison.reference_title}",
              left + 24.5 + (width - 28.7) / 2, 21)
        panel_label(fig, "ABC"[index], left + .4, 25.75)
    handles = [Line2D([], [], color=color, linestyle=line, linewidth=CDF_WIDTH, label=label)
               for color, line, label in ((POPULATION, "-", "Population"),
                   (AGNOSTIC, ":", "Bias-agnostic estimator"),
                   (SAMPLE, "--", "Bias-aware estimator"))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1 - 5 / 150),
               ncol=3, frameon=False, borderpad=0, borderaxespad=0, fontsize=NUMBER_SIZE,
               handlelength=2.2, handletextpad=.45, columnspacing=1.5)
    return save(fig, output_dir, "figure4", "Figure 4")
