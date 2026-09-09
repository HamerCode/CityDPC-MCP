from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from .compare_tool_usage_runs import (
    IGNORED_TOOLS,
    OUT_DIR,
    TEST_ORDER,
    count_dependency_success,
    esc,
    infer_label,
    load_compliance,
    load_runtime_metrics,
    load_success_rates,
    summarize_tool_usage,
)


def _compute_call_medians(compliance_json: dict) -> dict[str, dict[str, float]]:
    comp = (compliance_json.get("compliance") or {})
    out: dict[str, dict[str, float]] = {}
    all_actual: list[float] = []
    all_success: list[float] = []

    for tc in TEST_ORDER:
        tc_data = comp.get(tc) or {}
        runs = tc_data.get("run_details") or []
        actual_vals: list[float] = []
        success_vals: list[float] = []

        for r in runs:
            calls = r.get("tool_calls") or []
            actual = sum(1 for c in calls if c.get("name") not in IGNORED_TOOLS)
            successful, _ = count_dependency_success(tc, calls)
            actual_vals.append(float(actual))
            success_vals.append(float(successful))

        if actual_vals:
            actual_median = float(statistics.median(actual_vals))
            success_median = float(statistics.median(success_vals))
            all_actual.extend(actual_vals)
            all_success.extend(success_vals)
        else:
            actual_median = 0.0
            success_median = 0.0

        out[tc] = {
            "actual_median": actual_median,
            "successful_median": success_median,
        }

    out["_total"] = {
        "actual_median": float(statistics.median(all_actual)) if all_actual else 0.0,
        "successful_median": float(statistics.median(all_success)) if all_success else 0.0,
    }
    return out


def build_two_group_table(
    run_a: str,
    run_b: str,
    setup: str,
    label_a: str,
    label_b: str,
) -> str:
    summary_a = summarize_tool_usage(load_compliance(run_a, setup))
    summary_b = summarize_tool_usage(load_compliance(run_b, setup))
    success_a = load_success_rates(run_a, setup)
    success_b = load_success_rates(run_b, setup)
    runtime_a = load_runtime_metrics(run_a, setup)
    runtime_b = load_runtime_metrics(run_b, setup)
    med_a = _compute_call_medians(load_compliance(run_a, setup))
    med_b = _compute_call_medians(load_compliance(run_b, setup))

    lines = [
        r"\section*{Reasoning-Effort Vergleich (MCP)}",
        r"\textbf{Run A:} " + esc(run_a) + " (" + esc(label_a) + r")"
        + r" \quad "
        + r"\textbf{Run B:} " + esc(run_b) + " (" + esc(label_b) + r")"
        + r" \quad "
        + r"\textbf{Setup:} " + esc(setup),
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{l|rrrr|rrrr}",
        r"\toprule",
        r"& \multicolumn{4}{c|}{" + esc(label_a) + r"} & \multicolumn{4}{c}{" + esc(label_b) + r"} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"Test Case & Successful Tool Calls (\O) & Actual Tool Calls (\O) & Tool Usage Eff. (\%) & Success Rate (\%) & Successful Tool Calls (\O) & Actual Tool Calls (\O) & Tool Usage Eff. (\%) & Success Rate (\%) \\",
        r"\midrule",
    ]

    for tc in TEST_ORDER:
        sr_a = float(((success_a.get(tc) or {}).get("rate", 0.0) or 0.0))
        sr_b = float(((success_b.get(tc) or {}).get("rate", 0.0) or 0.0))
        succ_a = float(((med_a.get(tc) or {}).get("successful_median", 0.0) or 0.0))
        succ_b = float(((med_b.get(tc) or {}).get("successful_median", 0.0) or 0.0))
        act_a = float(((med_a.get(tc) or {}).get("actual_median", 0.0) or 0.0))
        act_b = float(((med_b.get(tc) or {}).get("actual_median", 0.0) or 0.0))
        tue_a = 100.0 * float(((summary_a.get(tc) or {}).get("tue", 0.0) or 0.0))
        tue_b = 100.0 * float(((summary_b.get(tc) or {}).get("tue", 0.0) or 0.0))

        lines.append(
            f"{esc(tc)} & "
            f"{succ_a:.2f} & {act_a:.2f} & \\textbf{{{tue_a:.1f}}} & \\textbf{{{sr_a:.1f}}} & "
            f"{succ_b:.2f} & {act_b:.2f} & \\textbf{{{tue_b:.1f}}} & \\textbf{{{sr_b:.1f}}} \\\\"
        )

    total_sr_a = float(((success_a.get("_total") or {}).get("rate", 0.0) or 0.0))
    total_sr_b = float(((success_b.get("_total") or {}).get("rate", 0.0) or 0.0))
    total_succ_a = float(((med_a.get("_total") or {}).get("successful_median", 0.0) or 0.0))
    total_succ_b = float(((med_b.get("_total") or {}).get("successful_median", 0.0) or 0.0))
    total_act_a = float(((med_a.get("_total") or {}).get("actual_median", 0.0) or 0.0))
    total_act_b = float(((med_b.get("_total") or {}).get("actual_median", 0.0) or 0.0))
    total_tue_a = 100.0 * float(((summary_a.get("_total") or {}).get("tue", 0.0) or 0.0))
    total_tue_b = 100.0 * float(((summary_b.get("_total") or {}).get("tue", 0.0) or 0.0))

    lines.extend(
        [
            r"\midrule",
            r"\textbf{Total} & "
            + f"\\textbf{{{total_succ_a:.2f}}} & \\textbf{{{total_act_a:.2f}}} & \\textbf{{{total_tue_a:.1f}}} & \\textbf{{{total_sr_a:.1f}}} & "
            + f"\\textbf{{{total_succ_b:.2f}}} & \\textbf{{{total_act_b:.2f}}} & \\textbf{{{total_tue_b:.1f}}} & \\textbf{{{total_sr_b:.1f}}} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\caption{Vergleich der Aufgabenerfuellung nach Reasoning-Effort (je Test Case) mit zwei Hauptspalten fuer beide Modellkonfigurationen. Successful/Actual Tool Calls sind als Median ueber Runs ausgewiesen. Success Rate steht jeweils als letzte Metrik im Spaltenblock.}",
            r"\end{table}",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True, help="z.B. 07.02_15-30_gpt-5-mini_medium")
    ap.add_argument("--run-b", required=True, help="z.B. 07.02_19-32_gpt-5-mini_minimal")
    ap.add_argument("--setup", default="mcp")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--output", default=None, help="Optionaler Dateiname .tex")
    args = ap.parse_args()

    label_a = args.label_a or infer_label(args.run_a, args.setup)
    label_b = args.label_b or infer_label(args.run_b, args.setup)

    tex = build_two_group_table(
        run_a=args.run_a,
        run_b=args.run_b,
        setup=args.setup,
        label_a=label_a,
        label_b=label_b,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = args.output or f"reasoning_effort_compare_{args.run_a}_vs_{args.run_b}_{args.setup}.tex"
    out_path = OUT_DIR / out_name
    out_path.write_text(tex, encoding="utf-8")
    print(f"Generated: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
