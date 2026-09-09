"""
Generate a LaTeX comparison table for N (>=2) evaluation run folders.

This is the multi-model generalization of generate_model_comparison_table.py
(which is limited to two models). It reuses that module's metric collection and
success logic, so any data fix that updates `state_validation.checks` is picked
up automatically here too.

Per setup/test-case it shows, for each model: Avg. tokens, optionally tool calls,
runtime, and success rate. Setup-average rows and a final Total row are appended.

Usage:
  python3 -m evals.tools.generate_multi_model_comparison_table \
      --run 04.06_13-41_gpt-oss-120b_high_completions --label "GPT-OSS" \
      --run 04.06_00-43_mistralai-mistral-small-4-119b_high_completions --label "Mistral" \
      --run 07.06_20-45_gpt-5.4-mini_low_codex-oauth --label "GPT-5.4 Mini (07.06_20-45)" \
      --output evaluation_tables/compare_gptoss_vs_mistral_100runs_all_calls_time.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import evals.tools.generate_model_comparison_table as gm

CellMetrics = gm.CellMetrics


def _aggregate(cells: list[CellMetrics | None]) -> CellMetrics:
    """Mean of each metric over the available cells (mirrors the 2-model totals)."""
    toks = [c.avg_tokens for c in cells if c and c.available and c.avg_tokens is not None]
    calls = [c.avg_tool_calls for c in cells if c and c.available and c.avg_tool_calls is not None]
    durs = [c.avg_duration_ms for c in cells if c and c.available and c.avg_duration_ms is not None]
    srs = [c.success_rate for c in cells if c and c.available and c.success_rate is not None]
    run_count = sum(c.run_count for c in cells if c and c.available)
    return CellMetrics(
        avg_tokens=mean(toks) if toks else None,
        avg_tool_calls=mean(calls) if calls else None,
        avg_duration_ms=mean(durs) if durs else None,
        success_rate=mean(srs) if srs else None,
        run_count=run_count,
        available=run_count > 0,
    )


def _cells_for_model(metrics: dict, setup: str, test_cases: list[str]) -> list[CellMetrics | None]:
    return [metrics.get((setup, tc)) for tc in test_cases]


def build_table(
    models: list[dict],
    setups: list[str],
    test_cases: list[str],
    setup_labels: dict[str, str],
    include_tool_calls: bool = True,
) -> str:
    n = len(models)
    metrics_per_model = 4 if include_tool_calls else 3
    col_spec = "cl" + "|".join(["r" * metrics_per_model] * n)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\begingroup",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        f"\\begin{{tabular}}{{{col_spec}}}",
    ]

    header = " &  & " + " & ".join(
        f"\\multicolumn{{{metrics_per_model}}}{{c}}{{\\textbf{{{gm._escape_latex(m['label'])}}}}}" for m in models
    ) + r" \\"
    lines.append(header)

    cmid = "".join(
        f"\\cmidrule(lr){{{3 + metrics_per_model * i}-{2 + metrics_per_model * (i + 1)}}}"
        for i in range(n)
    )
    lines.append(cmid)

    per_model_headers = [r"\textbf{\shortstack{Avg.\\Tokens}}"]
    if include_tool_calls:
        per_model_headers.append(r"\textbf{\shortstack{Tool\\Calls}}")
    per_model_headers.extend(
        [
            r"\textbf{\shortstack{Time\\(s)}}",
            r"\textbf{\shortstack{Success\\Rate}}",
        ]
    )
    per_model_head = " & ".join(per_model_headers)
    lines.append(
        r"\textbf{Setup} & \textbf{Test Case} & "
        + " & ".join([per_model_head] * n)
        + r" \\"
    )
    lines.append(r"\midrule")

    def metric_cells(m: CellMetrics | None) -> list[str]:
        tok, calls, time, sr = gm._format_metric_cells(m)
        return [tok, calls, time, sr] if include_tool_calls else [tok, time, sr]

    def data_row(setup_cell: str, label: str, cells: list[CellMetrics | None], bold: bool = False) -> str:
        parts = []
        for c in cells:
            values = metric_cells(c)
            if bold:
                values = [f"\\textbf{{{v}}}" for v in values]
            parts.extend(values)
        label_tex = f"\\textbf{{{gm._escape_latex(label)}}}" if bold else gm._escape_latex(label)
        return f"{setup_cell} & {label_tex} & " + " & ".join(parts) + r" \\"

    first = True
    for setup in setups:
        if not first:
            lines.append(r"\midrule")
        first = False
        setup_label = gm._display_setup_label(setup, setup_labels)
        n_rows = len(test_cases) + 1  # + setup avg
        for row_index, tc in enumerate(test_cases):
            setup_cell = ""
            if row_index == 0:
                setup_cell = (
                    f"\\multirow{{{n_rows}}}{{*}}"
                    f"{{\\rotatebox[origin=c]{{90}}{{\\textbf{{{gm._escape_latex(setup_label)}}}}}}}"
                )
            cells = [m["metrics"].get((setup, tc)) for m in models]
            lines.append(data_row(setup_cell, gm._display_test_case_label(tc), cells))

        # Setup average per model
        avg_cells = [_aggregate(_cells_for_model(m["metrics"], setup, test_cases)) for m in models]
        lines.append(data_row("", "Setup avg.", avg_cells, bold=True))

    # Total over all setups/test cases
    lines.append(r"\midrule")
    total_cells = [
        _aggregate([m["metrics"].get((s, tc)) for s in setups for tc in test_cases])
        for m in models
    ]
    lines.append(data_row(r"\textbf{Total}", "All Test Cases", total_cells, bold=True))

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an N-model LaTeX comparison table.")
    parser.add_argument("--run", action="append", required=True, help="Run folder (repeatable).")
    parser.add_argument("--label", action="append", default=None, help="Display label (repeatable, matches --run order).")
    parser.add_argument("--setup", action="append", default=None, help="Optional setup filter.")
    parser.add_argument("--test-case", action="append", default=None, help="Optional test-case filter.")
    parser.add_argument("--ground-truth", default=None, help="Optional alternative ground truth file.")
    parser.add_argument("--omit-tool-calls", action="store_true", help="Do not include Tool Calls columns.")
    parser.add_argument("--output", default=None, help="Write table to this path.")
    args = parser.parse_args()

    if args.ground_truth:
        gm.GROUND_TRUTH_PATH = Path(args.ground_truth)
    gt = gm._load_ground_truth()

    include_setups = set(args.setup) if args.setup else None
    include_tests = set(args.test_case) if args.test_case else None

    labels = args.label or []
    models = []
    for i, run_name in enumerate(args.run):
        run_dir = gm.LOG_DIR / run_name if not Path(run_name).is_absolute() else Path(run_name)
        if not run_dir.exists():
            run_dir = Path(run_name)
        metrics = gm._collect_run_metrics(run_dir, run_name, gt, include_setups, include_tests)
        label = labels[i] if i < len(labels) and labels[i] else gm._infer_run_label(run_dir, fallback=run_name)
        models.append({"name": run_name, "dir": run_dir, "metrics": metrics, "label": label})

    setup_keys = set()
    test_case_keys = set()
    for m in models:
        setup_keys |= {s for s, _ in m["metrics"].keys()}
        test_case_keys |= {tc for _, tc in m["metrics"].keys()}

    preferred_setups = ["mcp", "no_tools", "code_interpreter"]
    preferred_tests = [
        "list_buildings",
        "highest_measured_height",
        "roof_volume_sum",
        "raise_building_by_id",
        "add_building",
        "count_party_walls",
    ]
    setups = gm._ordered_keys(setup_keys, preferred_setups)
    test_cases = gm._ordered_keys(test_case_keys, preferred_tests)
    if include_setups is not None:
        setups = [s for s in setups if s in include_setups]
    if include_tests is not None:
        test_cases = [tc for tc in test_cases if tc in include_tests]

    setup_labels: dict[str, str] = {}
    for m in models:
        setup_labels.update(gm._collect_setup_labels(m["dir"]))

    latex = build_table(
        models,
        setups,
        test_cases,
        setup_labels,
        include_tool_calls=not args.omit_tool_calls,
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(latex + "\n", encoding="utf-8")
        print(f"Generated: {out}")
    else:
        print(latex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
