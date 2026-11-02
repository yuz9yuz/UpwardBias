"""Supplementary Figures S1-S3, stratified by cycle and reporting platform."""
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import ks_2samp

from ..data import Dataset, event_id
from ..style import (
    AXES_HEIGHT, CDF_WIDTH, INK, NUMBER_SIZE, PAPER_COPILOT, POPULATION, REDDIT,
    SAMPLE, SCALE, axes, canvas, cdf, cdf_axes, panel_label, save, significance, title,
)


SERIES = (("population", "Population", POPULATION, "o"),
          ("zhihu", "Zhihu", SAMPLE, "s"),
          ("reddit", "Reddit", REDDIT, "^"),
          ("papercopilot", "Paper Copilot", PAPER_COPILOT, "D"))


def _values(store, family, cycle, score_type, platform):
    if platform == "population":
        return store.load_population(family, cycle, score_type)
    return store.load_processed_sample(family, cycle, platform, score_type)


def stratified_means(data_dir, output_dir):
    store, fig = Dataset(data_dir), canvas(145)
    specs = (("A", 184, "ARR", ("2402", "2406", "2502"), "review", (1.75, 5.1), True),
             ("B", 156, "ARR", ("2402", "2406", "2502"), "meta", (1.65, 5.35), False),
             ("C", 112, "ICLR", ("24", "25"), "review", (3.5, 8.55), False))
    left = 0
    for letter, width, family, cycles, score_type, limits, copilot in specs:
        ax = axes(fig, left + 24.5, 30, width - 28.7)
        selected = SERIES if copilot else SERIES[:3]
        offsets = [-.27, -.09, .09, .27] if copilot else [-.18, 0, .18]
        values = {(cycle, platform): _values(store, family, cycle, score_type, platform)
                  for cycle in cycles for platform, _, _, _ in selected}
        for (platform, _, color, marker), offset in zip(selected, offsets):
            means = [values[cycle, platform].mean() for cycle in cycles]
            stds = [values[cycle, platform].std() for cycle in cycles]
            bar = ax.errorbar(np.arange(len(cycles)) + offset, means, yerr=stds,
                              fmt=marker, color=color, markerfacecolor=color,
                              markeredgecolor="white", markeredgewidth=.7 * SCALE,
                              markersize=8.5 * SCALE, elinewidth=2.6 * SCALE,
                              capsize=4.2 * SCALE, capthick=1.8 * SCALE, zorder=3)
            for cap in bar.lines[1]:
                cap.set_markeredgewidth(1.8 * SCALE)
        units = (limits[1] - limits[0]) / AXES_HEIGHT
        for cycle_index, cycle in enumerate(cycles):
            maximum = max(values[cycle, p].mean() + values[cycle, p].std()
                          for p, _, _, _ in selected)
            for level, ((platform, _, _, _), offset) in enumerate(zip(selected[1:], offsets[1:])):
                text = significance(ks_2samp(values[cycle, "population"], values[cycle, platform]).pvalue)
                x1, x2 = cycle_index + offsets[0], cycle_index + offset
                base = maximum + (2.5 + 9.5 * level) * units
                top = base + 1.8 * units
                ax.plot([x1, x1, x2, x2], [base, top, top, base], color=INK,
                        linewidth=1.4 * SCALE, zorder=4)
                ax.text((x1 + x2) / 2, top + .35 * units, text, ha="center", va="bottom",
                        fontsize=NUMBER_SIZE, zorder=5)
        labels = [store.metadata(event_id(family, cycle))["label"].replace(" Feb", "\nFeb").replace(" Jun", "\nJun")
                  for cycle in cycles]
        ax.set_xticks(np.arange(len(cycles)), labels)
        ax.tick_params(axis="x", length=0, pad=2.15)
        ax.set_xlim(-.5, len(cycles) - .5)
        ax.set_ylim(*limits)
        kind = "Meta-Review" if score_type == "meta" else "Review"
        ax.set_ylabel(f"{family} {kind} Score")
        ax.set_yticks([4, 5, 6, 7, 8] if family == "ICLR" else
                     [2, 3, 4, 5] if score_type == "meta" else [2, 2.5, 3, 3.5, 4, 4.5, 5])
        panel_label(fig, letter, left + .4, 22.5)
        left += width + 8
    handles = [Line2D([], [], color=color, marker=marker, linestyle="none",
                      markerfacecolor=color, markeredgecolor="white", markeredgewidth=.7 * SCALE,
                      markersize=8.5 * SCALE, label=label) for _, label, color, marker in SERIES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1 - 5 / 145),
               ncol=4, frameon=False, borderpad=0, borderaxespad=0, fontsize=NUMBER_SIZE,
               handlelength=1.1, handletextpad=.3, columnspacing=1.5)
    return save(fig, output_dir, "figure_s1", "Supplementary Figure 1")


def _decision_legend(fig, ax, decision):
    curves = list(ax.lines)
    labels = [f"{p} ({decision})" for p in ("Population", "Zhihu", "Reddit")]
    for location, wrapped in (("upper left", False), ("upper left", True),
                              ("lower right", False), ("lower right", True)):
        display = [label.replace(" (", "\n(") for label in labels] if wrapped else labels
        legend = ax.legend(handles=curves, labels=display, loc=location, frameon=False,
                           fontsize=NUMBER_SIZE, handlelength=1.3, handletextpad=.3,
                           borderaxespad=.4, borderpad=0, labelspacing=.35)
        fig.canvas.draw()
        box = legend.get_window_extent(fig.canvas.get_renderer())
        plot = ax.get_window_extent(fig.canvas.get_renderer())
        inside = box.x0 >= plot.x0 and box.x1 <= plot.x1 and box.y0 >= plot.y0 and box.y1 <= plot.y1
        intersects = any(line.get_path().transformed(line.get_transform()).intersects_bbox(
            box.padded(.5), filled=False) for line in curves)
        if inside and not intersects:
            return
    # A portable font fallback can alter the optimal position slightly.
    ax.legend(handles=curves, labels=[label.replace(" (", "\n(") for label in labels],
              loc="lower right", frameon=False, fontsize=NUMBER_SIZE,
              handlelength=1.3, handletextpad=.3, borderpad=0, labelspacing=.35)


def decision_cdfs(data_dir, output_dir):
    store, fig = Dataset(data_dir), canvas(274)
    width = 230
    for index, (cycle, decision) in enumerate((("24", "A"), ("24", "R"), ("25", "A"), ("25", "R"))):
        row, column = divmod(index, 2)
        left, top = column * (width + 8), row * 132
        ax = axes(fig, left + 24.5, top + 40, width - 28.7)
        population = store.load_iclr_population_by_decision(cycle)[0 if decision == "A" else 1]
        values = (population, store.load_iclr_decision_sample(cycle, "zhihu", decision),
                  store.load_iclr_decision_sample(cycle, "reddit", decision))
        for sample, color, line in zip(values, (POPULATION, SAMPLE, REDDIT), ("-", "--", ":")):
            cdf(ax, sample, color, line, (.9, 10.1))
        cdf_axes(ax, "ICLR")
        _decision_legend(fig, ax, "Accepted" if decision == "A" else "Rejected")
        title(fig, f"ICLR 20{cycle}", left + 24.5 + (width - 28.7) / 2, top + 21)
        panel_label(fig, "ABCD"[index], left + .4, top + 25.75)
    return save(fig, output_dir, "figure_s2", "Supplementary Figure 2")


def no_decision_cdfs(data_dir, output_dir):
    store, fig = Dataset(data_dir), canvas(290)
    width = (468 - 16) / 3
    handles = [Line2D([], [], color=color, linewidth=CDF_WIDTH, linestyle=line, label=label)
               for color, line, label in ((POPULATION, "-", "Population"),
                   (SAMPLE, "--", "Zhihu (No decision reported)"),
                   (REDDIT, ":", "Reddit (No decision reported)"))]
    handles += [Line2D([], [], color=INK, linewidth=1.8 * SCALE, linestyle=(0, (3.7, 1.6)),
                       label="Acceptance bar"),
                Line2D([], [], color=SAMPLE, linestyle="none", marker="s", markersize=7 * SCALE,
                       markeredgewidth=0, label="Zhihu median"),
                Line2D([], [], color=REDDIT, linestyle="none", marker="^", markersize=8 * SCALE,
                       markeredgewidth=0, label="Reddit median")]
    for top_y, subset in ((4, handles[:3]), (16, handles[3:])):
        fig.legend(handles=subset, loc="upper center", bbox_to_anchor=(.5, 1 - top_y / 290),
                   ncol=3, frameon=False, borderaxespad=0, handlelength=1.7,
                   handletextpad=.45, columnspacing=1.6)
    specs = (("ARR", "2402"), ("ARR", "2406"), ("ARR", "2502"),
             ("ICLR", "24"), ("ICLR", "25"))
    for index, (family, cycle) in enumerate(specs):
        row, column = (0, index) if index < 3 else (1, index - 3)
        left = column * (width + 8) if row == 0 else (468 - 2 * width - 8) / 2 + column * (width + 8)
        top = 48 if row == 0 else 180
        ax = axes(fig, left + 24.5, top, width - 28.7)
        population = store.load_population(family, cycle)
        samples = {p: store.load_no_decision_sample(family, cycle, p) for p in ("zhihu", "reddit")}
        limits = (0, 5.1) if family == "ARR" else (.9, 10.1)
        for values, color, line in ((population, POPULATION, "-"),
                                   (samples["zhihu"], SAMPLE, "--"), (samples["reddit"], REDDIT, ":")):
            cdf(ax, values, color, line, limits)
        rate = store.metadata(event_id(family, cycle))["population_acceptance_rate"]
        threshold = np.quantile(population, 1 - rate)
        ax.axvline(threshold, color=INK, linewidth=1.8 * SCALE,
                   linestyle=(0, (3.7, 1.6)), zorder=2)
        medians = {p: np.median(values) for p, values in samples.items()}
        tied = np.isclose(medians["zhihu"], medians["reddit"], atol=1e-6)
        for platform, color, marker, size in (("zhihu", SAMPLE, "s", 7), ("reddit", REDDIT, "^", 8)):
            marker_y = (.025 if platform == "zhihu" else .095) if tied else .045
            ax.scatter([medians[platform]], [marker_y], marker=marker, color=color,
                       s=(size * SCALE) ** 2, linewidths=0, zorder=4)
        cdf_axes(ax, family)
        title(fig, store.metadata(event_id(family, cycle))["label"],
              left + 24.5 + (width - 28.7) / 2, top - 13)
        panel_label(fig, "ABCDE"[index], left + .4, top - 7.25)
    return save(fig, output_dir, "figure_s3", "Supplementary Figure 3")
