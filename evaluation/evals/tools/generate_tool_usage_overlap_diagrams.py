"""
Generate per-testcase overlap diagrams for tool-call correctness.

Outputs:
- evaluation_tables/tool_usage_order_overlap_<runA>_vs_<runB>.png
- evaluation_tables/tool_usage_params_overlap_<runA>_vs_<runB>.png

Bars per testcase:
- model A correctness rate
- model B correctness rate
- overlap rate (both models correct on the same run index)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
TOOL_CALLS_DIR = ROOT / "evaluation_tool_calls"
OUT_DIR = ROOT / "evaluation_tables"


def load_compliance(run_folder: str, setup: str = "mcp") -> dict:
    p = TOOL_CALLS_DIR / run_folder / setup / "tool_call_compliance.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_flags(comp: dict, test_case: str, key: str) -> list[bool]:
    run_details = ((comp.get("compliance") or {}).get(test_case, {}) or {}).get("run_details", [])
    # Sort by run number so overlap is index-aligned.
    run_details = sorted(run_details, key=lambda x: x.get("run", 0))
    return [bool(r.get(key)) for r in run_details]


def rates_and_overlap(flags_a: list[bool], flags_b: list[bool]) -> tuple[float, float, float]:
    n = min(len(flags_a), len(flags_b))
    if n == 0:
        return 0.0, 0.0, 0.0
    a = flags_a[:n]
    b = flags_b[:n]
    rate_a = 100.0 * sum(a) / n
    rate_b = 100.0 * sum(b) / n
    overlap = 100.0 * sum(1 for i in range(n) if a[i] and b[i]) / n
    return rate_a, rate_b, overlap


def plot_metric(
    title: str,
    metric_key: str,
    run_a: str,
    run_b: str,
    label_a: str,
    label_b: str,
    out_png: Path,
) -> None:
    comp_a = load_compliance(run_a)
    comp_b = load_compliance(run_b)

    test_cases = [
        "list_buildings",
        "highest_measured_height",
        "roof_volume_sum",
        "raise_building_by_id",
        "add_building",
    ]

    a_rates: list[float] = []
    b_rates: list[float] = []
    overlap_rates: list[float] = []

    for tc in test_cases:
        a_flags = extract_flags(comp_a, tc, metric_key)
        b_flags = extract_flags(comp_b, tc, metric_key)
        ra, rb, ro = rates_and_overlap(a_flags, b_flags)
        a_rates.append(ra)
        b_rates.append(rb)
        overlap_rates.append(ro)

    x = np.arange(len(test_cases))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.bar(x - w, a_rates, w, label=label_a)
    ax.bar(x, b_rates, w, label=label_b)
    ax.bar(x + w, overlap_rates, w, label="Overlap (beide korrekt)")

    ax.set_ylim(0, 105)
    ax.set_ylabel("Korrektheit (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(test_cases, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate overlap diagrams for tool-call correctness.")
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--label-a", default="gpt-5 medium")
    parser.add_argument("--label-b", default="gpt-5 minimal")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    order_png = OUT_DIR / f"tool_usage_order_overlap_{args.run_a}_vs_{args.run_b}.png"
    params_png = OUT_DIR / f"tool_usage_params_overlap_{args.run_a}_vs_{args.run_b}.png"

    plot_metric(
        title="Tool-Call Reihenfolge: Modellvergleich + Overlap je Testcase",
        metric_key="order_ok",
        run_a=args.run_a,
        run_b=args.run_b,
        label_a=args.label_a,
        label_b=args.label_b,
        out_png=order_png,
    )
    plot_metric(
        title="Tool-Call Parameter: Modellvergleich + Overlap je Testcase",
        metric_key="params_ok",
        run_a=args.run_a,
        run_b=args.run_b,
        label_a=args.label_a,
        label_b=args.label_b,
        out_png=params_png,
    )

    print(f"Generated: {order_png}")
    print(f"Generated: {params_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
