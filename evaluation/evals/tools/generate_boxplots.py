#!/usr/bin/env python3
"""
Generate boxplot diagrams for evaluation metrics.
Scientific / LaTeX-style plots using matplotlib pgf-like rendering.
"""

import json
import argparse
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import numpy as np

# ============================================================================
# Configuration
# ============================================================================
LOG_DIR = Path("evaluation_logs")
OUTPUT_DIR = Path("evaluation_plots")

SETUP_LABELS = {
    "mcp": "MCP Server",
    "code_interpreter": "Code Interpreter",
    "no_tools": "No Tools",
}

SETUP_ORDER = ["MCP Server", "Code Interpreter", "No Tools"]

TEST_CASE_ORDER = [
    "list_buildings",
    "highest_measured_height",
    "roof_volume_sum",
    "raise_building_by_id",
    "add_building",
]

TEST_CASE_LABELS = {
    "list_buildings": "List Buildings",
    "highest_measured_height": "Highest Measured Height",
    "roof_volume_sum": "Roof Volume Sum",
    "raise_building_by_id": "Raise Building By ID",
    "add_building": "Add Building",
}

# Colors – muted, publication-quality
PALETTE = {
    "MCP Server": "#4C72B0",
    "Code Interpreter": "#55A868",
    "No Tools": "#C44E52",
}


# ============================================================================
# LaTeX-style matplotlib configuration
# ============================================================================
def _setup_latex_style():
    """Configure matplotlib for scientific / LaTeX-style output."""
    plt.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        # Sizes
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        # Axes
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
        "grid.linestyle": "--",
        # Ticks
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


# ============================================================================
# Data loading
# ============================================================================
def _find_latest_date_folder() -> Path | None:
    if not LOG_DIR.exists():
        return None
    date_folders = sorted([d for d in LOG_DIR.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
    return date_folders[0] if date_folders else None


def load_evaluation_data(date_folder: Path) -> dict:
    data = {}
    setup_dirs = [d for d in sorted(date_folder.iterdir())
                  if d.is_dir() and d.name not in ("datasets", "workspaces")]
    for setup_dir in setup_dirs:
        setup_key = setup_dir.name
        if setup_key not in SETUP_LABELS:
            continue
        for json_file in setup_dir.glob("evaluation_*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                log_data = json.load(f)
            test_case_id = log_data.get("test_case", {}).get("id")
            if not test_case_id:
                continue
            data.setdefault(test_case_id, {})[setup_key] = log_data
    return data


def extract_metrics(data: dict) -> pd.DataFrame:
    records = []
    for test_case_id, setups in data.items():
        for setup_key, log_data in setups.items():
            setup_label = SETUP_LABELS.get(setup_key, setup_key)
            for run in log_data.get("runs", []):
                if run.get("error") or run.get("retry_exhausted"):
                    continue
                tokens = run.get("tokens") or {}
                records.append({
                    "test_case": test_case_id,
                    "test_label": TEST_CASE_LABELS.get(test_case_id, test_case_id),
                    "setup": setup_label,
                    "duration_s": (run.get("duration_ms") or 0) / 1000.0,
                    "total_tokens": tokens.get("total_tokens", 0),
                    "input_tokens": tokens.get("input_tokens", 0),
                    "output_tokens": tokens.get("output_tokens", 0),
                })
    return pd.DataFrame(records)


# ============================================================================
# Plotting helpers
# ============================================================================
def _add_thousand_sep(ax, axis="y"):
    """Add thousand separator to axis tick labels."""
    fmt = ticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)


def _auto_ylim(ax, df, y_col, margin=0.15):
    """Set y-axis limits based on IQR to avoid outlier distortion."""
    vals = df[y_col].dropna()
    if vals.empty:
        return
    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 2.5 * iqr  # slightly more generous than default 1.5
    lower = max(0, q1 - 1.5 * iqr)
    # Ensure we show at least up to the max whisker
    data_max = vals[vals <= upper].max() if (vals <= upper).any() else vals.max()
    ymax = data_max * (1 + margin)
    ax.set_ylim(bottom=lower, top=ymax)


def _draw_boxplot(ax, df, y_col, y_label, show_legend=False, use_log=False):
    """Draw a single grouped boxplot on ax."""
    present_setups = [s for s in SETUP_ORDER if s in df["setup"].unique()]
    pal = [PALETTE[s] for s in present_setups]

    sns.boxplot(
        data=df, x="setup", y=y_col, hue="setup",
        order=present_setups, hue_order=present_setups,
        palette=pal, width=0.5, linewidth=0.8,
        showfliers=False,  # no outlier dots
        boxprops=dict(edgecolor="black", linewidth=0.6),
        whiskerprops=dict(color="black", linewidth=0.6),
        capprops=dict(color="black", linewidth=0.6),
        medianprops=dict(color="black", linewidth=1.0),
        ax=ax, legend=show_legend,
    )
    # Add individual data points as jitter strip
    sns.stripplot(
        data=df, x="setup", y=y_col, hue="setup",
        order=present_setups, hue_order=present_setups,
        palette=pal, size=3, alpha=0.4, jitter=0.12,
        ax=ax, legend=False, edgecolor="none",
    )
    ax.set_ylabel(y_label)
    ax.set_xlabel("")
    # Rotate x-labels to avoid overlap
    ax.set_xticks(range(len(present_setups)))
    ax.set_xticklabels(present_setups, rotation=25, ha="right")
    if use_log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    else:
        _add_thousand_sep(ax)
        _auto_ylim(ax, df, y_col)


def _draw_grouped_boxplot(ax, df, y_col, y_label, show_legend=True, use_log=False):
    """Draw a grouped boxplot with test cases on x and setups as hue."""
    existing = [tc for tc in TEST_CASE_ORDER if tc in df["test_case"].unique()]
    labels = [TEST_CASE_LABELS.get(tc, tc) for tc in existing]
    present_setups = [s for s in SETUP_ORDER if s in df["setup"].unique()]
    pal = [PALETTE[s] for s in present_setups]

    # Map test_case -> label for ordering
    df = df.copy()
    df["test_label_ord"] = pd.Categorical(df["test_label"],
                                           categories=labels, ordered=True)

    sns.boxplot(
        data=df, x="test_label_ord", y=y_col,
        hue="setup", hue_order=present_setups,
        palette=pal, width=0.6, linewidth=0.7,
        showfliers=False,  # no outlier dots
        boxprops=dict(linewidth=0.6),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
        medianprops=dict(color="black", linewidth=0.9),
        ax=ax, legend=show_legend,
    )
    # Add jitter strip for individual data points
    sns.stripplot(
        data=df, x="test_label_ord", y=y_col,
        hue="setup", hue_order=present_setups,
        palette=pal, size=2.5, alpha=0.35, jitter=0.08,
        dodge=True, ax=ax, legend=False, edgecolor="none",
    )
    ax.set_ylabel(y_label)
    ax.set_xlabel("")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    if use_log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    else:
        _add_thousand_sep(ax)
        _auto_ylim(ax, df, y_col)
    if show_legend:
        ax.legend(title="Setup", loc="upper right", fontsize=8, title_fontsize=9)


# ============================================================================
# Plot functions
# ============================================================================
def create_per_testcase_plots(df: pd.DataFrame, date_str: str):
    """One figure per test case: separate Duration and Token plots."""
    for tc_id in TEST_CASE_ORDER:
        df_tc = df[df["test_case"] == tc_id]
        if df_tc.empty:
            continue

        label = TEST_CASE_LABELS.get(tc_id, tc_id)

        # Duration plot
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        fig.suptitle(f"{label} — Duration", fontsize=11, fontweight="bold", y=1.02)
        _draw_boxplot(ax, df_tc, "duration_s", "Duration (s)")
        fig.tight_layout()
        out = OUTPUT_DIR / f"boxplot_{tc_id}_duration_{date_str}.pdf"
        fig.savefig(out)
        fig.savefig(out.with_suffix(".png"))
        print(f"  Saved: {out.stem}  (.pdf + .png)")
        plt.close(fig)

        # Tokens plot (total, input, output side by side)
        fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.2))
        fig.suptitle(f"{label} — Tokens", fontsize=11, fontweight="bold", y=1.02)
        _draw_boxplot(axes[0], df_tc, "total_tokens", "Total Tokens", use_log=True)
        _draw_boxplot(axes[1], df_tc, "input_tokens", "Input Tokens", use_log=True)
        _draw_boxplot(axes[2], df_tc, "output_tokens", "Output Tokens", use_log=True)
        fig.tight_layout()
        out = OUTPUT_DIR / f"boxplot_{tc_id}_tokens_{date_str}.pdf"
        fig.savefig(out)
        fig.savefig(out.with_suffix(".png"))
        print(f"  Saved: {out.stem}  (.pdf + .png)")
        plt.close(fig)


def create_overview_plot(df: pd.DataFrame, date_str: str):
    """Combined overview: Duration + Tokens for all test cases."""
    existing = [tc for tc in TEST_CASE_ORDER if tc in df["test_case"].unique()]
    if not existing:
        return
    df_filt = df[df["test_case"].isin(existing)]

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6.0))

    _draw_grouped_boxplot(axes[0], df_filt, "duration_s", "Duration (s)", show_legend=True)
    _draw_grouped_boxplot(axes[1], df_filt, "total_tokens", "Total Tokens", show_legend=False, use_log=True)

    fig.tight_layout()
    out = OUTPUT_DIR / f"boxplot_overview_{date_str}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"  Saved: {out.stem}  (.pdf + .png)")
    plt.close(fig)


def create_token_breakdown_plot(df: pd.DataFrame, date_str: str):
    """Input vs Output tokens breakdown."""
    existing = [tc for tc in TEST_CASE_ORDER if tc in df["test_case"].unique()]
    if not existing:
        return
    df_filt = df[df["test_case"].isin(existing)]

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6.0))

    _draw_grouped_boxplot(axes[0], df_filt, "input_tokens", "Input Tokens", show_legend=True, use_log=True)
    _draw_grouped_boxplot(axes[1], df_filt, "output_tokens", "Output Tokens", show_legend=False, use_log=True)

    fig.tight_layout()
    out = OUTPUT_DIR / f"boxplot_token_breakdown_{date_str}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"  Saved: {out.stem}  (.pdf + .png)")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Generate boxplot diagrams")
    parser.add_argument("--date", help="Date folder (YYYYMMDD_HHMMSS)")
    args = parser.parse_args()

    if args.date:
        date_folder = LOG_DIR / args.date
        if not date_folder.exists():
            print(f"Error: {date_folder} not found"); return 1
    else:
        date_folder = _find_latest_date_folder()
        if not date_folder:
            print(f"Error: No logs in {LOG_DIR}"); return 1

    print(f"Using logs from: {date_folder}")
    date_str = date_folder.name

    data = load_evaluation_data(date_folder)
    if not data:
        print("Error: No data found"); return 1

    df = extract_metrics(data)
    if df.empty:
        print("Error: No metrics"); return 1

    print(f"  {len(df)} data points, {len(df['test_case'].unique())} test cases\n")

    OUTPUT_DIR.mkdir(exist_ok=True)
    _setup_latex_style()

    print("Per test case:")
    create_per_testcase_plots(df, date_str)
    print("\nOverview:")
    create_overview_plot(df, date_str)
    print("\nToken breakdown:")
    create_token_breakdown_plot(df, date_str)

    print(f"\nAll plots saved to: {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
