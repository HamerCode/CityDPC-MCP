from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

from .compare_tool_usage_runs import TEST_ORDER


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "evaluation_logs"
OUT_DIR = ROOT / "evaluation_plots"

TEST_LABELS = {
    "list_buildings": "list_buildings",
    "highest_measured_height": "highest_measured_height",
    "roof_volume_sum": "roof_volume_sum",
    "raise_building_by_id": "raise_building_by_id",
    "add_building": "add_building",
}


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_metric_by_testcase(run_folder: str, setup: str, metric: str) -> dict[str, list[float]]:
    base = LOG_DIR / run_folder / setup
    out: dict[str, list[float]] = {tc: [] for tc in TEST_ORDER}
    if not base.exists():
        return out

    for p in sorted(base.glob("*.json")):
        try:
            data = _read_json(p)
        except Exception:
            continue
        tc = ((data.get("meta") or {}).get("test_case_id")) or ((data.get("test_case") or {}).get("id"))
        if tc not in out:
            continue
        for run in (data.get("runs") or []):
            if metric == "duration_ms":
                v = run.get("duration_ms")
            else:
                v = ((run.get("tokens") or {}).get("total_tokens"))
            if v is not None:
                out[tc].append(float(v))
    return out


def _boxplot_two_configs(
    vals_a: dict[str, list[float]],
    vals_b: dict[str, list[float]],
    ylabel: str,
    title: str,
    label_a: str,
    label_b: str,
    out_path: Path,
) -> None:
    line_w = 1.2
    median_w = 0.9
    x = np.arange(len(TEST_ORDER))
    width = 0.34

    fig, ax = plt.subplots(figsize=(14, 5))

    data_a = [vals_a.get(tc, []) for tc in TEST_ORDER]
    data_b = [vals_b.get(tc, []) for tc in TEST_ORDER]

    bp_a = ax.boxplot(
        data_a,
        positions=x - width / 2,
        widths=width,
        patch_artist=True,
        showfliers=False,
        showmeans=False,
        boxprops=dict(edgecolor="black", linewidth=line_w),
        whiskerprops=dict(color="black", linewidth=line_w),
        capprops=dict(color="black", linewidth=line_w),
        medianprops=dict(color="black", linewidth=median_w, linestyle="-"),
    )
    bp_b = ax.boxplot(
        data_b,
        positions=x + width / 2,
        widths=width,
        patch_artist=True,
        showfliers=False,
        showmeans=False,
        boxprops=dict(edgecolor="black", linewidth=line_w),
        whiskerprops=dict(color="black", linewidth=line_w),
        capprops=dict(color="black", linewidth=line_w),
        medianprops=dict(color="black", linewidth=median_w, linestyle="-"),
    )

    for patch in bp_a["boxes"]:
        patch.set_facecolor(mcolors.to_rgba("#d62728", alpha=0.65))  # red = medium
        patch.set_edgecolor("black")
        patch.set_linewidth(line_w)
    for patch in bp_b["boxes"]:
        patch.set_facecolor(mcolors.to_rgba("#1f77b4", alpha=0.65))  # blue = minimal
        patch.set_edgecolor("black")
        patch.set_linewidth(line_w)

    ax.set_xticks(x)
    ax.set_xticklabels([TEST_LABELS[tc] for tc in TEST_ORDER], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    # No in-plot title; caption is handled in LaTeX.
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor="#d62728", alpha=0.65, label=label_a),
            Patch(facecolor="#1f77b4", alpha=0.65, label=label_b),
        ],
        loc="best",
        frameon=True,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--setup", default="mcp")
    ap.add_argument("--label-a", default="medium")
    ap.add_argument("--label-b", default="minimal")
    args = ap.parse_args()

    dur_a = _collect_metric_by_testcase(args.run_a, args.setup, "duration_ms")
    dur_b = _collect_metric_by_testcase(args.run_b, args.setup, "duration_ms")
    tok_a = _collect_metric_by_testcase(args.run_a, args.setup, "total_tokens")
    tok_b = _collect_metric_by_testcase(args.run_b, args.setup, "total_tokens")

    stem = f"{args.run_a}_vs_{args.run_b}_{args.setup}"
    out_dur = OUT_DIR / f"boxplot_duration_{stem}.png"
    out_tok = OUT_DIR / f"boxplot_tokens_{stem}.png"

    _boxplot_two_configs(
        dur_a,
        dur_b,
        ylabel="Duration (ms)",
        title="Duration Distribution by Test Case",
        label_a=args.label_a,
        label_b=args.label_b,
        out_path=out_dur,
    )
    _boxplot_two_configs(
        tok_a,
        tok_b,
        ylabel="Total Tokens",
        title="Token Distribution by Test Case",
        label_a=args.label_a,
        label_b=args.label_b,
        out_path=out_tok,
    )

    print(f"Generated: {out_dur}")
    print(f"Generated: {out_tok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
