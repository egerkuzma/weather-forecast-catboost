"""
Render the evaluation charts from artifacts/ into docs/img/.

Every figure is rendered twice -- light and dark -- so the README can serve the
one matching the reader's GitHub theme via <picture>/prefers-color-scheme.

Colours come from a palette validated for colour-vision deficiency (adjacent
CVD dE 26.5 light / 27.3 dark, both well above the 8.0 target). Series identity
is never carried by colour alone: every multi-series chart also has a legend and
direct labels.

Run: python plots.py   (after train_weather.py and, optionally, ablation.py)
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

# Non-interactive backend + writable caches, before pyplot is imported.
MATPLOTLIB_DIR = Path(".matplotlib_cache")
MATPLOTLIB_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MATPLOTLIB_DIR.resolve())

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ARTIFACT_DIR = Path("artifacts")
IMG_DIR = Path("docs/img")
PLOT_DAYS = 30  # trailing window of the test period for the time-series chart
TOP_FEATURES = 16


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    axis: str
    series_1: str  # blue  -- actual
    series_2: str  # green -- predicted
    accent: str  # highlighted bar
    ramp: tuple[str, ...]  # sequential blue, light -> dark


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series_1="#2a78d6",
    series_2="#008300",
    accent="#2a78d6",
    ramp=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series_1="#3987e5",
    series_2="#008300",
    accent="#3987e5",
    ramp=("#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"),
)


def style(theme: Theme) -> None:
    """Recessive chrome: hairline solid grid, no top/right spines, muted ticks."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": theme.surface,
        "axes.facecolor": theme.surface,
        "savefig.facecolor": theme.surface,
        "text.color": theme.ink,
        "axes.labelcolor": theme.ink_secondary,
        "axes.edgecolor": theme.axis,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": theme.grid,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",  # never dashed -- dashes read as "threshold"
        "xtick.color": theme.muted,
        "ytick.color": theme.muted,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "figure.dpi": 140,
    })


def finish(fig, ax_list, theme: Theme, name: str) -> None:
    for ax in np.atleast_1d(ax_list).ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(theme.axis)
        ax.spines["bottom"].set_color(theme.axis)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if theme.name == "light" else "-dark"
    fig.savefig(IMG_DIR / f"{name}{suffix}.png", bbox_inches="tight")
    plt.close(fig)


def title(ax, text: str, subtitle: str, theme: Theme) -> None:
    ax.set_title(text, loc="left", fontsize=13, fontweight="600",
                 color=theme.ink, pad=18 if subtitle else 10)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 8),
                    textcoords="offset points", fontsize=9.5, color=theme.ink_secondary,
                    va="bottom", ha="left")


# --------------------------------------------------------------------------- #
# charts
# --------------------------------------------------------------------------- #

def chart_actual_vs_pred(pred: pd.DataFrame, theme: Theme) -> None:
    """Actual vs predicted over the last PLOT_DAYS of the test period.

    Both series are stamped at valid_time (the hour being forecast), never at
    feature time. Gaps are re-inserted as NaN so the line breaks over the hours
    the station did not report, instead of drawing a straight lie across them.
    """
    window = pred[pred["valid_time"] >= pred["valid_time"].max() - pd.Timedelta(days=PLOT_DAYS)]
    grid = pd.date_range(window["valid_time"].min(), window["valid_time"].max(), freq="1h")
    w = window.set_index("valid_time").reindex(grid)

    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(w.index, w["actual"], color=theme.series_1, linewidth=2, label="Actual", zorder=3)
    ax.plot(w.index, w["predicted"], color=theme.series_2, linewidth=2, label="Predicted", zorder=2)

    # Direct labels at the line ends: the coloured line itself sits beside the
    # ink text, so identity never rests on colour alone.
    for col, colour, label in (("actual", theme.series_1, "Actual"),
                               ("predicted", theme.series_2, "Predicted")):
        last = w[col].last_valid_index()
        if last is not None:
            ax.annotate(label, xy=(last, w[col].loc[last]), xytext=(8, 0),
                        textcoords="offset points", fontsize=9.5,
                        color=theme.ink_secondary, va="center", fontweight="600")
            ax.plot([last], [w[col].loc[last]], "o", color=colour, markersize=5,
                    markeredgecolor=theme.surface, markeredgewidth=2, zorder=4)

    title(ax, "24-hour temperature forecast vs reality",
          f"Last {PLOT_DAYS} days of the test period · line breaks = station not reporting", theme)
    ax.set_ylabel("Temperature, °C")
    ax.legend(loc="upper right", labelcolor=theme.ink_secondary, fontsize=9.5)
    ax.margins(x=0.08)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    finish(fig, ax, theme, "actual_vs_pred")


def chart_skill(metrics: dict, theme: Theme) -> None:
    """The model against the trivial forecasts it must beat. Emphasis, not eight hues."""
    names = ["Model (CatBoost)"] + list(metrics["baselines"])
    values = [metrics["test_mae"]] + list(metrics["baselines"].values())
    order = np.argsort(values)[::-1]
    names = [names[i] for i in order]
    values = [values[i] for i in order]
    colours = [theme.accent if n.startswith("Model") else theme.muted for n in names]

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    bars = ax.barh(names, values, color=colours, height=0.62)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.2f}", xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=10, color=theme.ink_secondary, fontweight="600")

    title(ax, "Model error vs trivial baselines",
          "Mean absolute error on 2024–2025 test data, °C · lower is better", theme)
    ax.set_xlabel("MAE, °C")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.margins(x=0.12)
    finish(fig, ax, theme, "skill_vs_baselines")


def chart_ablation(ablation: dict, theme: Theme) -> None:
    """Where the skill comes from -- and where it only appears to."""
    names = [k.replace("Station only", "Station only\n(no ERA5)").replace("+ERA5 ", "+ ERA5 at ")
             .replace("t-", "t−") for k in ablation]
    values = [v["mean"] for v in ablation.values()]
    errors = [v["sd"] for v in ablation.values()]
    station_only = values[0]
    # The headline bar is the only one that is not honestly deployable.
    colours = [theme.accent if k == "+ERA5 t-0h" else theme.muted for k in ablation]

    fig, ax = plt.subplots(figsize=(8.5, 4))
    bars = ax.bar(names, values, color=colours, width=0.6,
                  yerr=errors, capsize=4, ecolor=theme.ink_secondary)
    for bar, value, err in zip(bars, values, errors):
        ax.annotate(f"{value:.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + err),
                    xytext=(0, 6), textcoords="offset points", ha="center",
                    fontsize=10, color=theme.ink_secondary, fontweight="600")

    # Reference line at the station-only level; the leftmost bar already carries
    # the label, so a text annotation here would only collide with the bar values.
    ax.axhline(station_only, color=theme.axis, linewidth=1, zorder=1)

    survived = (values[0] - values[2]) / (values[0] - values[1]) * 100
    title(ax, f"{100 - survived:.0f}% of the ERA5 gain does not survive a lag",
          "Test MAE over 3 seeds (error bars = sd) · ERA5 assimilates observations up to ~9h after t",
          theme)
    ax.set_ylabel("Test MAE, °C")
    ax.set_ylim(0, max(values) * 1.28)
    ax.xaxis.grid(False)
    finish(fig, ax, theme, "era5_ablation")


def chart_error_analysis(pred: pd.DataFrame, theme: Theme) -> None:
    """Four views of the residuals: shape, daily cycle, seasonality, calibration."""
    err = pred["predicted"] - pred["actual"]
    valid = pd.to_datetime(pred["valid_time"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    (ax_hist, ax_hour), (ax_month, ax_scatter) = axes

    # 1. residual distribution
    ax_hist.hist(err, bins=70, color=theme.series_1, alpha=0.9)
    ax_hist.axvline(0, color=theme.axis, linewidth=1)
    title(ax_hist, "Residual distribution", f"Bias {err.mean():+.2f} °C · σ {err.std():.2f} °C", theme)
    ax_hist.set_xlabel("Predicted − actual, °C")
    ax_hist.set_ylabel("Hours")
    ax_hist.xaxis.grid(False)

    # 2. MAE by hour of day
    by_hour = err.abs().groupby(valid.dt.hour).mean()
    ax_hour.bar(by_hour.index, by_hour.values, color=theme.series_1, width=0.7)
    title(ax_hour, "Error by hour of day", "Mean absolute error, °C", theme)
    ax_hour.set_xlabel("Hour (local)")
    ax_hour.set_xticks(range(0, 24, 3))
    ax_hour.xaxis.grid(False)

    # 3. MAE by month -- the subtitle is computed, never asserted
    by_month = err.abs().groupby(valid.dt.month).mean()
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax_month.bar([labels[m - 1] for m in by_month.index], by_month.values,
                 color=theme.series_1, width=0.7)
    worst, best = by_month.idxmax(), by_month.idxmin()
    title(ax_month, "Error by month",
          f"Mean absolute error, °C · worst {labels[worst - 1]} {by_month[worst]:.2f}, "
          f"best {labels[best - 1]} {by_month[best]:.2f}", theme)
    ax_month.xaxis.grid(False)
    ax_month.tick_params(axis="x", labelsize=8)

    # 4. calibration -- density, not 16k overplotted dots
    cmap = LinearSegmentedColormap.from_list("seq_blue", theme.ramp)
    lim = [min(pred["actual"].min(), pred["predicted"].min()) - 1,
           max(pred["actual"].max(), pred["predicted"].max()) + 1]
    ax_scatter.hexbin(pred["actual"], pred["predicted"], gridsize=45, cmap=cmap,
                      mincnt=1, linewidths=0, extent=(*lim, *lim))
    ax_scatter.plot(lim, lim, color=theme.muted, linewidth=1.2, zorder=3)
    ax_scatter.annotate("perfect", xy=(lim[1], lim[1]), xytext=(-6, -14),
                        textcoords="offset points", ha="right", fontsize=9, color=theme.muted)
    title(ax_scatter, "Calibration", "Density of forecasts vs outcomes, °C", theme)
    ax_scatter.set_xlabel("Actual")
    ax_scatter.set_ylabel("Predicted")
    ax_scatter.set_xlim(lim)
    ax_scatter.set_ylim(lim)

    fig.tight_layout(h_pad=3.2, w_pad=2.6)
    finish(fig, axes, theme, "error_analysis")


def chart_feature_importance(importance: pd.DataFrame, theme: Theme) -> None:
    top = importance.head(TOP_FEATURES).iloc[::-1]
    # ERA5 features are the ones that do not survive the honesty check -- mark
    # them so the chart cannot be read as "these are all fair game".
    is_era5 = top["feature"].str.startswith("era5_")
    colours = [theme.muted if e else theme.accent for e in is_era5]

    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.barh(top["feature"], top["importance"], color=colours, height=0.68)
    title(ax, f"Top {TOP_FEATURES} features",
          "CatBoost importance · grey = ERA5 (not available in real time)", theme)
    ax.set_xlabel("Importance")
    ax.yaxis.grid(False)
    ax.tick_params(axis="y", labelsize=8.5)
    finish(fig, ax, theme, "feature_importance")


def main() -> None:
    pred = pd.read_csv(ARTIFACT_DIR / "predictions.csv", parse_dates=["valid_time"])
    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text())
    importance = pd.read_csv(ARTIFACT_DIR / "feature_importance.csv")
    ablation_path = ARTIFACT_DIR / "ablation.json"
    ablation = json.loads(ablation_path.read_text()) if ablation_path.exists() else None

    for theme in (LIGHT, DARK):
        style(theme)
        chart_actual_vs_pred(pred, theme)
        chart_skill(metrics, theme)
        chart_error_analysis(pred, theme)
        chart_feature_importance(importance, theme)
        if ablation:
            chart_ablation(ablation, theme)

    rendered = sorted(p.name for p in IMG_DIR.glob("*.png"))
    print(f"Rendered {len(rendered)} images to {IMG_DIR}/:")
    for name in rendered:
        print(f"  {name}")
    if not ablation:
        print("\nNo artifacts/ablation.json -- run `python ablation.py` for the ERA5 chart.")


if __name__ == "__main__":
    main()
