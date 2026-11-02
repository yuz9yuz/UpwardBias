"""Main Figures 1 and 2: population comparisons and reporting patterns."""
import numpy as np
from matplotlib.lines import Line2D

from ..style import (
    AXES_HEIGHT, CDF_WIDTH, INK, POPULATION, SAMPLE, BELOW, NUMBER_SIZE, SCALE,
    axes, canvas, cdf_axes, cdf_summary, panel_label, save, significance,
)


SCORE_LABELS = ["ARR\nReview", "ARR\nMeta-Review", "ICLR\nReview"]
PLATFORM_LABELS = ["Zhihu", "Reddit", "Paper Copilot"]


def _bracket(ax, left, right, base, height, text):
    top = base + height
    ax.plot([left, left, right, right], [base, top, top, base], color=INK,
            linewidth=1.65 * SCALE)
    ax.text((left + right) / 2, top, text, ha="center", va="bottom",
            fontweight="bold", fontsize=11.5 * SCALE)


def _score_bars(ax, comparisons, no_decision=False):
    x = np.arange(3) * (.75 if no_decision else .53)
    width = .22 if no_decision else .16
    for index, (group, color) in enumerate((("population", POPULATION), ("sample", SAMPLE))):
        ax.bar(x + index * width, [r[group]["mean"] for r in comparisons], width=width,
               yerr=[r[group]["std"] for r in comparisons], color=color,
               edgecolor="white", linewidth=.65 * SCALE,
               capsize=4 * SCALE, label=("Sample (No decision reported)" if
                   no_decision and group == "sample" else group.title()),
               error_kw={"ecolor": INK, "elinewidth": (2.5 if no_decision else 1.5) * SCALE,
                         "capthick": (1.8 if no_decision else 1.2) * SCALE})
    for index, row in enumerate(comparisons):
        base = max(row[g]["mean"] + row[g]["std"] for g in ("population", "sample")) + 8 / 30
        _bracket(ax, x[index], x[index] + width, base, 8 / 24, significance(row["p_value"]))
    ax.set_xticks(x + width / 2, SCORE_LABELS)
    ax.set_ylim(0, 8)
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.set_ylabel("Score")
    ax.legend(frameon=False, loc="upper left", handlelength=1.1, handletextpad=.3)


def _platform_scores(ax, rows):
    x, offset = np.arange(3) * .35, .16
    handles = []
    for index, (group, color) in enumerate((("population", POPULATION), ("sample", SAMPLE))):
        errors = ax.errorbar(x + index * offset, [r[group]["mean"] for r in rows],
                            yerr=[r[group]["std"] for r in rows], fmt="o", color=color,
                            elinewidth=2.6 * SCALE, capsize=5.5 * SCALE,
                            capthick=1.8 * SCALE, markersize=8.5 * SCALE,
                            markeredgecolor="white", markeredgewidth=.7 * SCALE,
                            label=group.title())
        for cap in errors.lines[1]:
            cap.set_markeredgewidth(1.8 * SCALE)
        handles.append(errors)
    for index, row in enumerate(rows):
        base = max(row[g]["mean"] + row[g]["std"] for g in ("population", "sample")) + 3.8 / 30
        _bracket(ax, x[index], x[index] + offset, base, 3.8 / 24, significance(row["p_value"]))
    ax.set_xticks(x + offset / 2, PLATFORM_LABELS)
    ax.set_ylim(-1.2, 2.6)
    ax.set_yticks([-1, 0, 1, 2])
    ax.set_ylabel("Normalized Score")
    ax.legend(handles=handles, ncol=2, frameon=False, loc="upper center",
              columnspacing=.8, handlelength=1.4, handletextpad=.3, borderaxespad=.4)


def figure1(results, output_dir):
    fig = canvas(245)
    panels = results["pooled_score_comparisons"]
    d_left = 346.9660671490301
    cdf_slot = d_left / 3 - 8
    for index, row in enumerate(panels):
        left = index * (cdf_slot + 8)
        ax = axes(fig, left + 22.5, 18, cdf_slot - 22.5 - 4.2)
        limits = (0, 5.1) if row["family"] == "ARR" else (.9, 10.1)
        cdf_summary(ax, row["population"], POPULATION, "-", limits, "Population")
        cdf_summary(ax, row["sample"], SAMPLE, "--", limits, "Sample")
        cdf_axes(ax, row["family"], row["score_type"])
        ax.legend(frameon=False, loc="upper left")
        panel_label(fig, "ABC"[index], left + .4, 10.75)
    ax = axes(fig, d_left + 13, 18, 104.8)
    _score_bars(ax, panels)
    panel_label(fig, "D", d_left + .4, 10.75)

    ax = axes(fig, 25, 150, 304.0)
    rates = results["acceptance_rates"]
    x, width = np.arange(9) * .6, .22
    for index, (key, color, label) in enumerate((
        ("population_acceptance_rate", POPULATION, "Population"),
        ("sample_acceptance_rate", SAMPLE, "Sample"),
    )):
        ax.bar(x + index * width, [r[key] for r in rates], width=width, color=color,
               edgecolor="white", linewidth=.65 * SCALE, label=label)
    for index, row in enumerate(rates):
        base = max(row["population_acceptance_rate"], row["sample_acceptance_rate"]) + 1 / 30
        _bracket(ax, x[index], x[index] + width, base, 1 / 24, significance(row["p_value"]))
    labels = ["ACL 2024", "EMNLP 2024", "ACL 2025", "ICLR 2024", "ICLR 2025",
              "WWW 2025", "KDD 2025", "CVPR 2025", "AAAI 2025"]
    ax.set_xticks(x + width / 2, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Acceptance Rate")
    ax.legend(frameon=False, loc="upper left")
    panel_label(fig, "E", .4, 139)
    ax = axes(fig, 359.6, 150, 104.9)
    _platform_scores(ax, results["standardized_platform_scores"])
    panel_label(fig, "F", 341.37, 139)
    return save(fig, output_dir, "figure1", "Figure 1")


def _reporting(ax, row):
    family = row["family"]
    x, y = np.asarray(row["x"]), np.asarray(row["y"])
    low, high = np.asarray(row["lower"]), np.asarray(row["upper"])
    for selected, color in ((x < row["threshold"], BELOW), (x >= row["threshold"], SAMPLE)):
        ax.errorbar(x[selected], y[selected],
                    yerr=np.vstack([y[selected] - low[selected], high[selected] - y[selected]]),
                    fmt="o", linestyle="none", markersize=7.5 * SCALE,
                    markerfacecolor=color, markeredgecolor="white", markeredgewidth=.9 * SCALE,
                    ecolor=color, elinewidth=2.5 * SCALE, capsize=4.3 * SCALE,
                    capthick=1.8 * SCALE, zorder=5)
    ax.axvline(row["threshold"], color=INK, linewidth=1.8 * SCALE,
               linestyle=(0, (3.7, 1.6)), dash_capstyle="butt")
    ax.set_xlabel(f"{family} Review Score")
    ax.set_ylabel("P(Report)")
    if family == "ARR":
        ax.set_xlim(.85, 4.15)
        ax.set_xticks(np.arange(1, 4.1, .5))
        ax.set_ylim(0, .18)
        ax.set_yticks(np.arange(0, .16, .05))
    else:
        ax.set_xticks(np.arange(4, 7, .5))
        ax.set_ylim(0, .07)
        ax.set_yticks(np.arange(0, .061, .02))
    handle = Line2D([], [], color=INK, linewidth=1.8 * SCALE,
                    linestyle=(0, (3.7, 1.6)), dash_capstyle="butt")
    ax.legend(handles=[handle], labels=["Acceptance bar"], frameon=False,
              loc="upper left", handlelength=1.3, handletextpad=.3, borderaxespad=.4)


def _decision_cdf(ax, row):
    decision = row["decision"].title()
    cdf_summary(ax, row["population"], POPULATION, "-", (.9, 10.1))
    cdf_summary(ax, row["sample"], SAMPLE, "--", (.9, 10.1))
    cdf_axes(ax, "ICLR")
    # Two-line labels fit the same physical font inside narrow main panels.
    ax.legend(labels=[f"Population\n({decision})", f"Sample\n({decision})"],
              frameon=False, loc="upper left", handlelength=1.3,
              handletextpad=.3, borderaxespad=.4, borderpad=0, labelspacing=.35)


def _distance(ax, row):
    x = np.asarray(row["x"])
    for key, color, line, face, label in (
        ("population_proportions", POPULATION, "-", "white", "Population"),
        ("sample_proportions", SAMPLE, "--", SAMPLE, "Sample (No decision reported)"),
    ):
        ax.plot(x, 100 * np.asarray(row[key]), color=color, linestyle=line,
                linewidth=CDF_WIDTH, marker="o", markersize=7.5 * SCALE,
                markerfacecolor=face, markeredgecolor=color,
                markeredgewidth=(1.6 if key == "population_proportions" else 1) * SCALE,
                label=label)
    if row["family"] == "ARR":
        ax.set_xlim(-.07, x[-1] + .07)
        ax.set_xticks(x)
    else:
        ax.set_xlim(0, row["bin_edges"][-1])
        ax.set_xticks(np.arange(int(row["bin_edges"][-1]) + 1))
    ax.set_ylim(0, 50)
    ax.set_yticks(np.arange(0, 51, 10))
    ax.set_xlabel(f"{row['family']} Distance to Acceptance Bar\n(Review-Score Points)")
    ax.set_ylabel("Proportion (%)")
    ax.legend(loc="upper right", frameon=False, handlelength=1.5, handletextpad=.3)


def figure2(results, output_dir):
    fig = canvas(264)
    top_width, bottom_width = 111.0, (468 - 16) / 3
    for index in range(7):
        top = index < 4
        column = index if top else index - 4
        width = top_width if top else bottom_width
        left, y = column * (width + 8), 0 if top else 132
        ax = axes(fig, left + 24.5, y + 18, width - 24.5 - 4.2)
        if index < 2:
            _reporting(ax, results["reporting_proportions"][index])
        elif index < 4:
            _decision_cdf(ax, results["decision_stratified_scores"][index - 2])
        elif index == 4:
            _score_bars(ax, results["no_decision_scores"], no_decision=True)
        else:
            _distance(ax, results["boundary_distances"][index - 5])
        panel_label(fig, "ABCDEFG"[index], left + .4, y + 10.75)
    return save(fig, output_dir, "figure2", "Figure 2")
