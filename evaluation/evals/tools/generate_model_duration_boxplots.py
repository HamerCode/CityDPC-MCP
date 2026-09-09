#!/usr/bin/env python3
"""Generate long runtime boxplots comparing two model evaluation folders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "evaluation_plots"

DEFAULT_MISTRAL_RUN = "04.06_00-43_mistralai-mistral-small-4-119b_high_completions"
DEFAULT_GPTOSS_RUN = "04.06_13-41_gpt-oss-120b_high_completions"

SETUP_ORDER = ["mcp", "no_tools", "code_interpreter"]
SETUP_LABELS = {
    "mcp": "MCP",
    "no_tools": "No Tools",
    "code_interpreter": "Sandbox",
}

TEST_CASE_ORDER = [
    "list_buildings",
    "highest_measured_height",
    "raise_building_by_id",
    "add_building",
    "count_party_walls",
    "roof_volume_sum",
]
TEST_CASE_LABELS = {
    "list_buildings": "List\nbuildings",
    "highest_measured_height": "Highest\nheight",
    "raise_building_by_id": "Raise\nbuilding",
    "add_building": "Add\nbuilding",
    "count_party_walls": "Party\nwalls",
    "roof_volume_sum": "Roof\nvolume",
}

MODEL_COLORS = {
    "Mistral": "#1f77b4",
    "GPT-OSS": "#d62728",
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Computer Modern Roman",
                "CMU Serif",
                "Times New Roman",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
            "axes.linewidth": 0.7,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.45,
            "grid.linestyle": "--",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _duration_seconds(run: dict[str, Any]) -> float | None:
    duration_ms = run.get("duration_ms")
    if duration_ms is not None and not isinstance(duration_ms, bool):
        try:
            return float(duration_ms) / 1000.0
        except (TypeError, ValueError):
            pass

    duration_seconds = run.get("duration_seconds")
    if duration_seconds is not None and not isinstance(duration_seconds, bool):
        try:
            return float(duration_seconds)
        except (TypeError, ValueError):
            pass

    return None


def _discover_test_cases(run_dirs: list[Path]) -> list[str]:
    found: set[str] = set()
    for run_dir in run_dirs:
        for setup_dir in run_dir.iterdir() if run_dir.exists() else []:
            if not setup_dir.is_dir():
                continue
            for log_path in setup_dir.glob("*.json"):
                data = _load_json(log_path)
                meta = data.get("meta") or {}
                test_case_id = meta.get("test_case_id") or (data.get("test_case") or {}).get("id")
                if test_case_id:
                    found.add(str(test_case_id))

    ordered = [test_case for test_case in TEST_CASE_ORDER if test_case in found]
    ordered.extend(sorted(found - set(ordered)))
    return ordered


def load_durations(run_dir: Path, model: str) -> list[dict[str, Any]]:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")

    records: list[dict[str, Any]] = []
    for setup in SETUP_ORDER:
        setup_dir = run_dir / setup
        if not setup_dir.exists():
            continue

        for log_path in sorted(setup_dir.glob("*.json")):
            data = _load_json(log_path)
            meta = data.get("meta") or {}
            test_case_id = meta.get("test_case_id") or (data.get("test_case") or {}).get("id")
            if not test_case_id:
                test_case_id = log_path.stem

            for run in data.get("runs") or []:
                duration_s = _duration_seconds(run)
                if duration_s is None:
                    continue
                records.append(
                    {
                        "model": model,
                        "setup": setup,
                        "test_case": str(test_case_id),
                        "run": run.get("run"),
                        "duration_s": duration_s,
                    }
                )

    return records


def _write_csv(records: list[dict[str, Any]], output_prefix: Path) -> Path:
    csv_path = output_prefix.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "setup", "test_case", "run", "duration_s"])
        writer.writeheader()
        writer.writerows(records)

    return csv_path


def _values(
    records: list[dict[str, Any]],
    *,
    model: str,
    setup: str,
    test_case: str,
) -> list[float]:
    return [
        float(record["duration_s"])
        for record in records
        if record["model"] == model
        and record["setup"] == setup
        and record["test_case"] == test_case
    ]


def create_duration_boxplot(
    records: list[dict[str, Any]],
    output_path: Path,
    test_cases: list[str],
    *,
    log_scale: bool = False,
) -> None:
    if not records:
        raise RuntimeError("No duration records found.")

    groups: list[tuple[str, str, float]] = []
    x = 1.0
    for setup_index, setup in enumerate(SETUP_ORDER):
        setup_records = [record for record in records if record["setup"] == setup]
        if not setup_records:
            continue
        if setup_index > 0:
            x += 1.0
        for test_case in test_cases:
            if any(
                record["setup"] == setup and record["test_case"] == test_case
                for record in records
            ):
                groups.append((setup, test_case, x))
                x += 1.0

    fig, ax = plt.subplots(figsize=(15.5, 5.8))
    offsets = {"Mistral": -0.18, "GPT-OSS": 0.18}
    width = 0.32

    for model in ["Mistral", "GPT-OSS"]:
        color = MODEL_COLORS[model]
        data: list[list[float]] = []
        positions: list[float] = []

        for setup, test_case, group_x in groups:
            vals = _values(records, model=model, setup=setup, test_case=test_case)
            if vals:
                data.append(vals)
                positions.append(group_x + offsets[model])

        box = ax.boxplot(
            data,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            whis=1.5,
            manage_ticks=False,
            medianprops={"color": "black", "linewidth": 1.1},
            boxprops={"edgecolor": "black", "linewidth": 0.9},
            whiskerprops={"color": "black", "linewidth": 0.8},
            capprops={"color": "black", "linewidth": 0.8},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor("black")

    centers = [group_x for _setup, _test_case, group_x in groups]
    labels = [TEST_CASE_LABELS.get(test_case, test_case.replace("_", "\n")) for _setup, test_case, _group_x in groups]
    ax.set_xticks(centers)
    ax.set_xticklabels(labels)

    setup_centers: dict[str, list[float]] = {}
    for setup, _test_case, group_x in groups:
        setup_centers.setdefault(setup, []).append(group_x)

    for setup, setup_positions in setup_centers.items():
        center = sum(setup_positions) / len(setup_positions)
        ax.text(
            center,
            -0.13,
            SETUP_LABELS.get(setup, setup),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
        )

    ordered_setups = [setup for setup in SETUP_ORDER if setup in setup_centers]
    for left_setup, right_setup in zip(ordered_setups, ordered_setups[1:]):
        separator = (max(setup_centers[left_setup]) + min(setup_centers[right_setup])) / 2.0
        ax.axvline(separator, color="0.75", linewidth=0.8)

    ax.set_ylabel("Runtime (s)")
    ax.set_xlabel("")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Patch(facecolor=MODEL_COLORS["Mistral"], edgecolor="black", alpha=0.35, label="Mistral"),
            Patch(facecolor=MODEL_COLORS["GPT-OSS"], edgecolor="black", alpha=0.35, label="GPT-OSS"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncols=2,
        frameon=False,
    )

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Runtime (s, log scale)")
    else:
        ax.set_ylim(bottom=0)

    fig.subplots_adjust(left=0.06, right=0.995, top=0.91, bottom=0.20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".pdf"))
    fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare model runtimes with long boxplots.")
    parser.add_argument("--mistral-run", default=DEFAULT_MISTRAL_RUN)
    parser.add_argument("--gptoss-run", default=DEFAULT_GPTOSS_RUN)
    parser.add_argument(
        "--output-prefix",
        default=str(OUTPUT_DIR / "boxplot_runtime_mistral_vs_gptoss_100runs_all_setups"),
    )
    parser.add_argument("--skip-log-variant", action="store_true")
    args = parser.parse_args()

    mistral_dir = ROOT / "evaluation_logs" / args.mistral_run
    gptoss_dir = ROOT / "evaluation_logs" / args.gptoss_run
    output_prefix = Path(args.output_prefix)
    if not output_prefix.is_absolute():
        output_prefix = ROOT / output_prefix

    records = []
    records.extend(load_durations(mistral_dir, "Mistral"))
    records.extend(load_durations(gptoss_dir, "GPT-OSS"))
    test_cases = _discover_test_cases([mistral_dir, gptoss_dir])

    _setup_style()
    csv_path = _write_csv(records, output_prefix)
    create_duration_boxplot(records, output_prefix, test_cases, log_scale=False)
    print(f"Saved: {output_prefix.with_suffix('.pdf')}")
    print(f"Saved: {output_prefix.with_suffix('.png')}")
    print(f"Saved: {csv_path}")

    if not args.skip_log_variant:
        log_prefix = output_prefix.with_name(output_prefix.name + "_logy")
        create_duration_boxplot(records, log_prefix, test_cases, log_scale=True)
        print(f"Saved: {log_prefix.with_suffix('.pdf')}")
        print(f"Saved: {log_prefix.with_suffix('.png')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
