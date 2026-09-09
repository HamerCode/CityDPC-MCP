from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
TOOL_CALLS_DIR = ROOT / "evaluation_tool_calls"
EVAL_LOGS_DIR = ROOT / "evaluation_logs"
OUT_DIR = ROOT / "evaluation_tables"

TEST_ORDER = [
    "list_buildings",
    "highest_measured_height",
    "roof_volume_sum",
    "raise_building_by_id",
    "add_building",
]

IGNORED_TOOLS = {"list_datasets", "take_snapshot"}

# Tool dependency graph per test case.
# A tool call is "successful" only if:
# 1) call status == completed
# 2) tool belongs to test-case graph
# 3) all prerequisite tools have already completed earlier in same run
TOOL_DEPENDENCIES: dict[str, dict[str, list[str]]] = {
    "list_buildings": {
        "load_dataset": [],
        "get_buiding_Id_list": ["load_dataset"],
    },
    "highest_measured_height": {
        "load_dataset": [],
        "get_all_buildings": ["load_dataset"],
        "number_of_buildings": ["get_all_buildings"],
        "get_buiding_Id_list": ["get_all_buildings"],
        "get_building_by_id": ["get_all_buildings"],
    },
    "roof_volume_sum": {
        "load_dataset": [],
        "get_buiding_Id_list": ["load_dataset"],
        "calculate_roof_volume_by_id": ["get_buiding_Id_list"],
    },
    "raise_building_by_id": {
        "load_dataset": [],
        "get_building_by_id": ["load_dataset"],
        "enrich_building": ["get_building_by_id"],
        "save_dataset": ["enrich_building"],
    },
    "add_building": {
        "load_dataset": [],
        "create_building": ["load_dataset"],
        "enrich_building": ["create_building"],
        "save_dataset": ["enrich_building"],
    },
}


def esc(text: str) -> str:
    text = text.replace("\\", r"\textbackslash{}")
    repl = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return text


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_compliance(run_folder: str, setup: str) -> dict[str, Any]:
    path = TOOL_CALLS_DIR / run_folder / setup / "tool_call_compliance.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing compliance: {path}")
    return read_json(path)


def load_ideal_sequences(run_folder: str, setup: str) -> dict[str, Any]:
    path = TOOL_CALLS_DIR / run_folder / setup / "ideal_tool_call_sequences.json"
    if not path.exists():
        return {}
    return read_json(path)


def load_success_rates(run_folder: str, setup: str) -> dict[str, dict[str, float]]:
    path = EVAL_LOGS_DIR / run_folder / "revalidation_report.json"
    if not path.exists():
        return {}
    data = read_json(path)
    setup_summary = ((data.get("summary") or {}).get(setup) or {})

    out: dict[str, dict[str, float]] = {}
    total_runs = 0
    total_correct = 0
    for tc, v in setup_summary.items():
        runs = int(v.get("runs", 0) or 0)
        correct = int(v.get("correct", 0) or 0)
        rate = (100.0 * correct / runs) if runs else 0.0
        out[tc] = {"runs": runs, "correct": correct, "rate": rate}
        total_runs += runs
        total_correct += correct
    out["_total"] = {
        "runs": total_runs,
        "correct": total_correct,
        "rate": (100.0 * total_correct / total_runs) if total_runs else 0.0,
    }
    return out


def load_runtime_metrics(run_folder: str, setup: str) -> dict[str, dict[str, float]]:
    run_dir = EVAL_LOGS_DIR / run_folder / setup
    if not run_dir.exists():
        return {}

    out: dict[str, dict[str, float]] = {}
    total_runs = 0
    total_duration_sum = 0.0
    total_tokens = 0

    for p in sorted(run_dir.glob("*.json")):
        try:
            data = read_json(p)
        except Exception:
            continue

        tc = ((data.get("meta") or {}).get("test_case_id")) or ((data.get("test_case") or {}).get("id"))
        runs = data.get("runs") or []
        if not tc or not runs:
            continue

        durations = [r.get("duration_ms") for r in runs if r.get("duration_ms") is not None]
        token_totals = [
            ((r.get("tokens") or {}).get("total_tokens"))
            for r in runs
            if ((r.get("tokens") or {}).get("total_tokens")) is not None
        ]

        run_count = len(runs)
        dur_sum = float(sum(durations)) if durations else 0.0
        tok_sum = int(sum(token_totals)) if token_totals else 0
        avg_dur = (dur_sum / len(durations)) if durations else 0.0

        out[tc] = {
            "runs": run_count,
            "avg_duration_ms": avg_dur,
            "total_tokens": tok_sum,
        }

        total_runs += run_count
        total_duration_sum += dur_sum
        total_tokens += tok_sum

    out["_total"] = {
        "runs": total_runs,
        "avg_duration_ms": (total_duration_sum / total_runs) if total_runs else 0.0,
        "total_tokens": total_tokens,
    }
    return out


def infer_label(run_folder: str, setup: str) -> str:
    run_dir = EVAL_LOGS_DIR / run_folder / setup
    if not run_dir.exists():
        return run_folder
    for p in sorted(run_dir.glob("*.json")):
        try:
            data = read_json(p)
        except Exception:
            continue
        meta = data.get("meta") or {}
        model = meta.get("model")
        effort = meta.get("reasoning_effort")
        if model and effort:
            return f"{model} ({effort})"
        if model:
            return str(model)
    return run_folder


def count_dependency_success(test_case: str, calls: list[dict[str, Any]]) -> tuple[int, int]:
    deps = TOOL_DEPENDENCIES.get(test_case, {})
    seen_completed: set[str] = set()
    successful = 0
    turns = 0

    for c in calls:
        name = c.get("name")
        if not name or name in IGNORED_TOOLS:
            continue

        turns += 1

        if name not in deps:
            continue
        if c.get("status") != "completed":
            continue

        needed = deps.get(name, [])
        if all(req in seen_completed for req in needed):
            successful += 1
            seen_completed.add(name)

    return successful, turns


def summarize_tool_usage(compliance_json: dict[str, Any]) -> dict[str, dict[str, float]]:
    comp = compliance_json.get("compliance", {}) or {}
    out: dict[str, dict[str, float]] = {}

    for tc in TEST_ORDER:
        tc_data = comp.get(tc, {})
        runs = tc_data.get("run_details", []) or []
        total_runs = len(runs)

        if total_runs == 0:
            out[tc] = {
                "runs": 0,
                "actual_calls_avg": 0.0,
                "successful_calls_total": 0.0,
                "successful_calls_avg": 0.0,
                "interaction_turns_total": 0.0,
                "tue": 0.0,
            }
            continue

        actual_calls_total = 0
        successful_total = 0
        turns_total = 0

        for r in runs:
            calls = r.get("tool_calls") or []
            actual_calls_total += sum(
                1 for c in calls if (c.get("name") not in IGNORED_TOOLS)
            )
            succ, turns = count_dependency_success(tc, calls)
            successful_total += succ
            turns_total += turns

        out[tc] = {
            "runs": float(total_runs),
            "actual_calls_avg": actual_calls_total / total_runs,
            "successful_calls_total": float(successful_total),
            "successful_calls_avg": successful_total / total_runs,
            "interaction_turns_total": float(turns_total),
            "tue": (successful_total / turns_total) if turns_total else 0.0,
        }

    total_runs = int(sum(int(v["runs"]) for v in out.values()))
    weighted_actual_sum = sum(v["actual_calls_avg"] * int(v["runs"]) for v in out.values())
    successful_total = sum(v["successful_calls_total"] for v in out.values())
    turns_total = sum(v["interaction_turns_total"] for v in out.values())

    out["_total"] = {
        "runs": float(total_runs),
        "actual_calls_avg": (weighted_actual_sum / total_runs) if total_runs else 0.0,
        "successful_calls_total": successful_total,
        "successful_calls_avg": (successful_total / total_runs) if total_runs else 0.0,
        "interaction_turns_total": turns_total,
        "tue": (successful_total / turns_total) if turns_total else 0.0,
    }

    return out


def render_args(args: dict[str, Any]) -> str:
    if not args:
        return "(keine Parameter)"
    parts = []
    for k in sorted(args.keys()):
        v = args[k]
        if isinstance(v, list):
            val = f"{len(v)} Elemente"
        else:
            val = str(v)
        parts.append(f"{k}={val}")
    return "(" + ", ".join(parts) + ")"


def build_model_table(label: str, summary: dict[str, dict[str, float]], success: dict[str, dict[str, float]], runtime: dict[str, dict[str, float]]) -> list[str]:
    total = summary.get("_total", {})
    total_success = success.get("_total", {"rate": 0.0})
    total_runtime = runtime.get("_total", {"avg_duration_ms": 0.0, "total_tokens": 0})

    lines = [
        rf"\subsubsection*{{{esc(label)}}}",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Test Case & Runs & Avg Dur. (ms) & Total Tok. & Successful Tool Calls (\O) & Actual Calls (\O) & Tool Usage Efficiency (\%) & Success Rate (\%) \\",
        r"\midrule",
    ]

    for tc in TEST_ORDER:
        s = summary.get(tc, {})
        sr = success.get(tc, {"rate": 0.0})
        rm = runtime.get(tc, {"avg_duration_ms": 0.0, "total_tokens": 0})

        lines.append(
            f"{esc(tc)} & {int(s.get('runs', 0) or 0)} & "
            f"{float(rm.get('avg_duration_ms', 0.0) or 0.0):.0f} & "
            f"{int(rm.get('total_tokens', 0) or 0)} & "
            f"{float(s.get('successful_calls_avg', 0.0) or 0.0):.2f} & "
            f"{float(s.get('actual_calls_avg', 0.0) or 0.0):.2f} & "
            f"\\textbf{{{(100.0 * float(s.get('tue', 0.0) or 0.0)):.1f}}} & "
            f"\\textbf{{{float(sr.get('rate', 0.0) or 0.0):.1f}}} \\\\" 
        )

    lines.extend(
        [
            r"\midrule",
            r"\textbf{Total} & "
            + f"\\textbf{{{int(total.get('runs', 0) or 0)}}} & "
            + f"\\textbf{{{float(total_runtime.get('avg_duration_ms', 0.0) or 0.0):.0f}}} & "
            + f"\\textbf{{{int(total_runtime.get('total_tokens', 0) or 0)}}} & "
            + f"\\textbf{{{float(total.get('successful_calls_avg', 0.0) or 0.0):.2f}}} & "
            + f"\\textbf{{{float(total.get('actual_calls_avg', 0.0) or 0.0):.2f}}} & "
            + f"\\textbf{{{(100.0 * float(total.get('tue', 0.0) or 0.0)):.1f}}} & "
            + f"\\textbf{{{float(total_success.get('rate', 0.0) or 0.0):.1f}}} \\\\" ,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]
    )

    return lines


def build_ground_truth_table(ideal_spec: dict[str, Any]) -> list[str]:
    lines = [
        r"\subsection*{Ideale Tool-Calls (Ground Truth)}",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"Test Case & Ideale Tool Calls (untereinander) \\",
        r"\midrule",
    ]

    tc_specs = (ideal_spec.get("test_cases") or {}) if isinstance(ideal_spec, dict) else {}

    for tc in TEST_ORDER:
        seqs = (tc_specs.get(tc, {}) or {}).get("ideal_sequences") or []
        if not seqs:
            lines.append(f"{esc(tc)} & -- \\\\")
            continue

        cell_parts = []
        for i, seq in enumerate(seqs, 1):
            cell_parts.append(f"Sequenz {i}:\\\\")
            cell_parts.append("0. (optional) list\\_datasets\\\\")
            for j, step in enumerate(seq, 1):
                tool = step.get("tool", "")
                args = render_args(step.get("arguments") or {})
                suffix = ""
                if step.get("optional"):
                    suffix += " [optional]"
                if step.get("repeat"):
                    suffix += f" [wiederholen: {step.get('repeat')}]"
                cell_parts.append(f"{j}. {esc(tool)} {esc(args)}{esc(suffix)}\\\\")
            cell_parts.append("\\\\")

        lines.append(
            f"{esc(tc)} & "
            + r"\begin{minipage}{0.74\linewidth}\raggedright\footnotesize "
            + " ".join(cell_parts).strip()
            + r"\end{minipage} \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
        ]
    )
    return lines


def sanity_assert_different(sum_a: dict[str, dict[str, float]], sum_b: dict[str, dict[str, float]]) -> None:
    diffs = []
    for tc in TEST_ORDER:
        a = sum_a.get(tc, {})
        b = sum_b.get(tc, {})
        keyset = ["actual_calls_avg", "successful_calls_avg", "tue"]
        if any(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) > 1e-9 for k in keyset):
            diffs.append(tc)
    if not diffs:
        raise RuntimeError("Both model tables are numerically identical. Check input run folders.")


def build_doc(
    run_a: str,
    run_b: str,
    label_a: str,
    label_b: str,
    sum_a: dict[str, dict[str, float]],
    sum_b: dict[str, dict[str, float]],
    success_a: dict[str, dict[str, float]],
    success_b: dict[str, dict[str, float]],
    runtime_a: dict[str, dict[str, float]],
    runtime_b: dict[str, dict[str, float]],
    ideal_spec: dict[str, Any],
) -> str:
    lines = [
        r"\section*{MCP Tool-Call Analyse (Vergleich)}",
        r"\textbf{Run A:} " + esc(run_a) + " (" + esc(label_a) + r")"
        + r" \quad "
        + r"\textbf{Run B:} " + esc(run_b) + " (" + esc(label_b) + r")"
        + r" \quad "
        + r"\textbf{Setup:} mcp",
        "",
        r"\subsection*{Befolgungsquote je Test Case}",
    ]

    lines.extend(build_model_table(label_a, sum_a, success_a, runtime_a))
    lines.extend(build_model_table(label_b, sum_b, success_b, runtime_b))
    lines.extend(build_ground_truth_table(ideal_spec))

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--setup", default="mcp")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    args = ap.parse_args()

    comp_a = load_compliance(args.run_a, args.setup)
    comp_b = load_compliance(args.run_b, args.setup)

    ideal = load_ideal_sequences(args.run_a, args.setup)
    success_a = load_success_rates(args.run_a, args.setup)
    success_b = load_success_rates(args.run_b, args.setup)
    runtime_a = load_runtime_metrics(args.run_a, args.setup)
    runtime_b = load_runtime_metrics(args.run_b, args.setup)

    sum_a = summarize_tool_usage(comp_a)
    sum_b = summarize_tool_usage(comp_b)

    sanity_assert_different(sum_a, sum_b)

    label_a = args.label_a or infer_label(args.run_a, args.setup)
    label_b = args.label_b or infer_label(args.run_b, args.setup)

    tex = build_doc(
        run_a=args.run_a,
        run_b=args.run_b,
        label_a=label_a,
        label_b=label_b,
        sum_a=sum_a,
        sum_b=sum_b,
        success_a=success_a,
        success_b=success_b,
        runtime_a=runtime_a,
        runtime_b=runtime_b,
        ideal_spec=ideal,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"tool_usage_compare_{args.run_a}_vs_{args.run_b}_{args.setup}.tex"
    out_path.write_text(tex, encoding="utf-8")
    print(f"Generated: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
