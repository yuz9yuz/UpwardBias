"""Shared physical typography, colors and vector export for all eight figures."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

PAGE_WIDTH = 468.0
AXES_HEIGHT = 81.19005964708155
SCALE = .43054616076066536
NUMBER_SIZE = 5.381827009508317
AXIS_SIZE = 6.027646250649315
LABEL_SIZE = 10.335159438617402
CDF_WIDTH = 1.0763654019016633
INK = "#202124"
POPULATION = "#B65A57"
SAMPLE = "#356F8F"
REDDIT = "#267563"
PAPER_COPILOT = "#75518E"
AGNOSTIC = PAPER_COPILOT
BELOW = "#82AFC7"
YMAX = 1.025


def font_family():
    """Prefer the manuscript font; use Matplotlib's bundled font portably."""
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
        return "Arial"
    except ValueError:
        return "DejaVu Sans"


def canvas(height):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [font_family()],
        "font.size": 13 * SCALE, "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": INK,
        "axes.labelsize": AXIS_SIZE, "axes.labelpad": 5 * SCALE,
        "axes.linewidth": 1.15 * SCALE, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.labelsize": NUMBER_SIZE, "ytick.labelsize": NUMBER_SIZE,
        "xtick.major.size": 3.5 * SCALE, "ytick.major.size": 3.5 * SCALE,
        "xtick.major.width": SCALE, "ytick.major.width": SCALE,
        "xtick.major.pad": 3.5 * SCALE, "ytick.major.pad": 3.5 * SCALE,
        "legend.fontsize": NUMBER_SIZE, "legend.handlelength": 1.7,
        "legend.handletextpad": .45, "legend.labelspacing": .35,
        "lines.linewidth": 1.9 * SCALE, "lines.solid_capstyle": "round",
        "lines.markersize": 6 * SCALE, "lines.markeredgewidth": SCALE,
        "patch.linewidth": SCALE, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    return plt.figure(figsize=(PAGE_WIDTH / 72, height / 72))


def axes(fig, left, top, width, height=AXES_HEIGHT):
    page_height = fig.get_figheight() * 72
    return fig.add_axes([left / PAGE_WIDTH, (page_height - top - height) / page_height,
                         width / PAGE_WIDTH, height / page_height])


def panel_label(fig, letter, left, baseline):
    fig.text(left / PAGE_WIDTH, 1 - baseline / (fig.get_figheight() * 72), letter,
             ha="left", va="baseline", fontsize=LABEL_SIZE, fontweight="bold")


def title(fig, text, center, top):
    fig.text(center / PAGE_WIDTH, 1 - top / (fig.get_figheight() * 72), text,
             ha="center", va="top", fontsize=AXIS_SIZE, linespacing=1.08)


def significance(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def cdf(ax, values, color, linestyle, limits, label=None):
    support, counts = np.unique(np.asarray(values, dtype=float), return_counts=True)
    ax.step(np.r_[limits[0], support, limits[1]],
            np.r_[0, np.cumsum(counts) / counts.sum(), 1], where="post",
            color=color, linestyle=linestyle, linewidth=CDF_WIDTH, label=label)


def cdf_summary(ax, summary, color, linestyle, limits, label=None):
    ax.step(np.r_[limits[0], summary["support"], limits[1]],
            np.r_[0, summary["cdf"], 1], where="post", color=color,
            linestyle=linestyle, linewidth=CDF_WIDTH, label=label)


def cdf_axes(ax, family, score_type="review"):
    ax.set_xlim((0, 5.1) if family == "ARR" else (.9, 10.1))
    ax.set_ylim(0, YMAX)
    ax.set_xticks([0, 1, 2, 3, 4, 5] if family == "ARR" else [2, 4, 6, 8, 10])
    ax.set_yticks(np.linspace(0, 1, 6))
    kind = "Meta-Review" if score_type == "meta" else "Review"
    ax.set_xlabel(f"{family} {kind} Score")
    ax.set_ylabel("Cumulative Probability")


def save(fig, output_dir, stem, name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(pdf, bbox_inches=None, pad_inches=0,
                metadata={"Title": name, "Creator": "upward-bias: reproducible analysis"})
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"Saved {pdf.name}")
    return pdf
