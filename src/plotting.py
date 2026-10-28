"""One consistent chart style for every figure in the repo."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

from .config import FIGURES

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CONTROL, TREATMENT = "#9a9993", BLUE
SEGMENT_COLORS = {"new": BLUE, "returning": ORANGE, "all": INK_2}


def setup() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
        "font.family": "DejaVu Sans", "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png")


def pct(ax, axis: str = "y", decimals: int = 0) -> None:
    fmt = mpl.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def zero_line(ax, axis: str = "y") -> None:
    (ax.axhline if axis == "y" else ax.axvline)(0, color=INK_2, lw=1, zorder=1)


__all__ = ["setup", "save", "pct", "zero_line", "plt", "INK", "INK_2", "GRID", "BLUE", "ORANGE", "AQUA",
           "YELLOW", "CONTROL", "TREATMENT", "SEGMENT_COLORS"]
