#!/usr/bin/env python3
"""Generate an MCP tool-usage-efficiency comparison table for model runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "evaluation_logs"
OUTPUT_DIR = ROOT / "evaluation_tables"

DEFAULT_MISTRAL_RUN = "04.06_00-43_mistralai-mistral-small-4-119b_high_completions"
DEFAULT_GPTOSS_RUN = "04.06_13-41_gpt-oss-120b_high_completions"
DEFAULT_GPT54_RUN = "11.06_14-51_gpt-5.4-mini_high_codex-oauth"
DEFAULT_SETUP = "mcp"

TEST_CASE_ORDER = [
    "list_buildings",
    "highest_measured_height",
    "raise_building_by_id",
    "add_building",
    "count_party_walls",
]

TEST_CASE_LABELS = {
    "list_buildings": "List buildings",
    "highest_measured_height": "Highest height",
    "raise_building_by_id": "Raise building",
    "add_building": "Add building",
    "count_party_walls": "Party walls",
}

IGNORED_TOOLS = {"list_datasets", "take_snapshot"}

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
    "count_party_walls": {
        "load_dataset": [],
        "get_party_walls": ["load_dataset"],
    },
}


def _escape_latex(value: str) -> str:
    value = value.replace("\\", r"\textbackslash{}")
    replacements = {
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
    for key, replacement in replacements.items():
        value = value.replace(key, replacement)
    return value


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _tool_name(call: dict[str, Any]) -> str | None:
    return call.get("name") or ((call.get("function") or {}).get("name"))


def _tool_arguments(call: dict[str, Any]) -> Any:
    return call.get("arguments") or ((call.get("function") or {}).get("arguments"))


def _normalised_arguments(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    try:
        return json.dumps(json.loads(str(value)), sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(value).replace(" ", "")


def _has_output_payload(call: dict[str, Any]) -> bool:
    return bool(call.get("output") or call.get("result") or call.get("output_preview"))


def _dedupe_adjacent_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse request/result duplicate log entries from Codex OAuth runs."""
    out: list[dict[str, Any]] = []
    index = 0

    while index < len(calls):
        current = calls[index]
        if index + 1 < len(calls):
            following = calls[index + 1]
            same_call = (
                _tool_name(current) == _tool_name(following)
                and _normalised_arguments(_tool_arguments(current))
                == _normalised_arguments(_tool_arguments(following))
            )
            if same_call and not _has_output_payload(current) and _has_output_payload(following):
                out.append(following)
                index += 2
                continue

        out.append(current)
        index += 1

    return out


def _normalized_calls(run: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    raw_calls = [call for call in (run.get("tool_calls") or []) if isinstance(call, dict)]
    for call in _dedupe_adjacent_tool_calls(raw_calls):
        if not isinstance(call, dict):
            continue
        name = _tool_name(call)
        if not name:
            continue
        out.append(
            {
                "name": name,
                "status": call.get("status"),
                "arguments": _tool_arguments(call),
            }
        )
    return out


def _count_successful_tool_calls(test_case: str, calls: list[dict[str, Any]]) -> tuple[int, int]:
    deps = TOOL_DEPENDENCIES.get(test_case, {})
    seen_completed: set[str] = set()
    successful = 0
    actual = 0

    for call in calls:
        name = call.get("name")
        if not name or name in IGNORED_TOOLS:
            continue

        actual += 1

        if name not in deps:
            continue
        if call.get("status") != "completed":
            continue

        required = deps.get(name, [])
        if all(req in seen_completed for req in required):
            successful += 1
            seen_completed.add(name)

    return successful, actual


def _run_success(test_case: str, run: dict[str, Any]) -> bool:
    if run.get("error") or run.get("retry_exhausted"):
        return False

    checks = (run.get("state_validation") or {}).get("checks") or {}
    if test_case == "raise_building_by_id":
        if checks:
            return bool(checks.get("building_found") and checks.get("height_correct"))
    elif test_case == "add_building":
        if checks:
            return bool(
                checks.get("building_found")
                and checks.get("id_correct")
                and checks.get("height_correct")
                and checks.get("cityjson_valid")
                and checks.get("citydpc_importable")
                and checks.get("geometry_valid")
                and checks.get("building_displayable")
            )

    validation = run.get("validation") or {}
    accuracy = validation.get("accuracy")
    try:
        return float(accuracy) >= 1.0
    except (TypeError, ValueError):
        return False


def _collect_metrics(run_folder: str, setup: str) -> dict[str, dict[str, float]]:
    setup_dir = LOG_DIR / run_folder / setup
    if not setup_dir.exists():
        raise FileNotFoundError(f"Setup folder not found: {setup_dir}")

    metrics: dict[str, dict[str, float]] = {}
    total_runs = 0
    total_successful_calls = 0
    total_actual_calls = 0
    total_successful_runs = 0

    for test_case in TEST_CASE_ORDER:
        log_path = setup_dir / f"{test_case}.json"
        if not log_path.exists():
            continue

        data = _load_json(log_path)
        runs = data.get("runs") or []
        successful_calls = 0
        actual_calls = 0
        successful_runs = 0

        for run in runs:
            if not isinstance(run, dict):
                continue
            succ, actual = _count_successful_tool_calls(test_case, _normalized_calls(run))
            successful_calls += succ
            actual_calls += actual
            if _run_success(test_case, run):
                successful_runs += 1

        run_count = len(runs)
        metrics[test_case] = {
            "runs": float(run_count),
            "successful_calls_avg": (successful_calls / run_count) if run_count else 0.0,
            "actual_calls_avg": (actual_calls / run_count) if run_count else 0.0,
            "tue": (successful_calls / actual_calls) if actual_calls else 0.0,
            "success_rate": (successful_runs / run_count) if run_count else 0.0,
        }

        total_runs += run_count
        total_successful_calls += successful_calls
        total_actual_calls += actual_calls
        total_successful_runs += successful_runs

    metrics["_total"] = {
        "runs": float(total_runs),
        "successful_calls_avg": (total_successful_calls / total_runs) if total_runs else 0.0,
        "actual_calls_avg": (total_actual_calls / total_runs) if total_runs else 0.0,
        "tue": (total_successful_calls / total_actual_calls) if total_actual_calls else 0.0,
        "success_rate": (total_successful_runs / total_runs) if total_runs else 0.0,
    }
    return metrics


def _fmt(value: float, digits: int = 2) -> str:
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _fmt_pct(value: float) -> str:
    return f"{value * 100.0:.1f}"


def _row_cells(metrics: dict[str, float], bold: bool = False) -> str:
    values = [
        _fmt(metrics.get("successful_calls_avg", 0.0), 2),
        _fmt(metrics.get("actual_calls_avg", 0.0), 2),
        _fmt_pct(metrics.get("tue", 0.0)),
        _fmt_pct(metrics.get("success_rate", 0.0)),
    ]
    if bold:
        values = [rf"\textbf{{{value}}}" for value in values]
    return " & ".join(values)


def _cmidrules(group_count: int) -> str:
    rules = []
    for index in range(group_count):
        start = 2 + index * 4
        end = start + 3
        rules.append(rf"\cmidrule(lr){{{start}-{end}}}")
    return "".join(rules)


def _metric_header(group_count: int) -> list[str]:
    lines: list[str] = ["Test Case"]
    for index in range(group_count):
        is_last_group = index == group_count - 1
        lines.extend(
            [
                r"& \multicolumn{1}{c}{STC (\O)}",
                r"& \multicolumn{1}{c}{ATC (\O)}",
                r"& \multicolumn{1}{c}{TUE (\%)}",
                r"& \multicolumn{1}{c" + ("" if is_last_group else "|") + r"}{SR (\%)}",
            ]
        )
    lines[-1] += r" \\"
    return lines


def build_table(groups: list[tuple[str, dict[str, dict[str, float]]]]) -> str:
    group_count = len(groups)
    if group_count < 2:
        raise ValueError("At least two model groups are required")

    column_spec = "l" + "|".join("rrrr" for _ in groups)
    header_groups = " & ".join(
        rf"\multicolumn{{4}}{{c{'|' if index < group_count - 1 else ''}}}{{\textbf{{{_escape_latex(label)}}}}}"
        for index, (label, _) in enumerate(groups)
    )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        "& " + header_groups + r" \\",
        _cmidrules(group_count),
        *_metric_header(group_count),
        r"\midrule",
    ]

    for test_case in TEST_CASE_ORDER:
        if not any(test_case in metrics for _, metrics in groups):
            continue
        lines.append(
            f"{_escape_latex(TEST_CASE_LABELS.get(test_case, test_case))} & "
            + " & ".join(_row_cells(metrics.get(test_case, {})) for _, metrics in groups)
            + r" \\"
        )

    lines.extend(
        [
            r"\midrule",
            r"\textbf{Total} & "
            + " & ".join(_row_cells(metrics.get("_total", {}), bold=True) for _, metrics in groups)
            + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\caption{Comparison of tool usage efficiency in the MCP setup between the high-reasoning model configurations. "
            r"\textit{STC} = Successful Tool Calls (\O), "
            r"\textit{ATC} = Actual Tool Calls (\O), "
            r"\textit{TUE} = Tool Usage Efficiency (\%), "
            r"\textit{SR} = Success Rate (\%).}",
            r"\label{tab:gptoss_mistral_mcp_tool_usage}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create MCP TUE comparison table for model runs.")
    parser.add_argument("--mistral-run", default=DEFAULT_MISTRAL_RUN)
    parser.add_argument("--gptoss-run", default=DEFAULT_GPTOSS_RUN)
    parser.add_argument("--gpt54-run", default=DEFAULT_GPT54_RUN)
    parser.add_argument("--setup", default=DEFAULT_SETUP)
    parser.add_argument("--label-mistral", default="Mistral (high)")
    parser.add_argument("--label-gptoss", default="GPT-OSS (high)")
    parser.add_argument("--label-gpt54", default="GPT-5.4 Mini (high)")
    parser.add_argument("--no-gpt54", action="store_true", help="Only compare GPT-OSS and Mistral.")
    parser.add_argument(
        "--output",
        default="compare_gptoss_vs_mistral_mcp_tool_usage_efficiency.tex",
    )
    args = parser.parse_args()

    mistral_metrics = _collect_metrics(args.mistral_run, args.setup)
    gptoss_metrics = _collect_metrics(args.gptoss_run, args.setup)
    groups = [
        (args.label_gptoss, gptoss_metrics),
        (args.label_mistral, mistral_metrics),
    ]
    if not args.no_gpt54:
        groups.append((args.label_gpt54, _collect_metrics(args.gpt54_run, args.setup)))

    tex = build_table(groups)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / args.output
    output_path.write_text(tex, encoding="utf-8")
    print(f"Generated: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
