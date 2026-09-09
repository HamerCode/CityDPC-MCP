"""
Generate a LaTeX table comparing two evaluation run folders.

For each setup/test-case combination the table shows:

* Avg. total tokens
* Avg. tool calls
* Avg. runtime
* Success rate

Cost is intentionally omitted.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "evaluation_logs"
GROUND_TRUTH_PATH = ROOT / "ground_truth.json"

SKIP_DIRS = {"datasets", "workspaces"}
IGNORED_MCP_ATC_TOOLS = {"list_datasets", "take_snapshot"}

SETUP_DISPLAY_LABELS = {
    "code_interpreter": "Sandbox",
    "mcp": "MCP",
    "no_tools": "No Tools",
}

TEST_CASE_DISPLAY_LABELS = {
    "list_buildings": "List buildings",
    "highest_measured_height": "Highest height",
    "roof_volume_sum": "Roof volume sum",
    "raise_building_by_id": "Raise building",
    "add_building": "Add building",
    "count_party_walls": "Party walls",
}


def _escape_latex(value: str) -> str:
    value = value.replace("\\", r"\textbackslash{}")
    value = value.replace("&", r"\&")
    value = value.replace("%", r"\%")
    value = value.replace("$", r"\$")
    value = value.replace("#", r"\#")
    value = value.replace("_", r"\_")
    value = value.replace("{", r"\{")
    value = value.replace("}", r"\}")
    value = value.replace("~", r"\textasciitilde{}")
    value = value.replace("^", r"\textasciicircum{}")
    return value


def _display_setup_label(setup: str, setup_labels: dict[str, str]) -> str:
    return SETUP_DISPLAY_LABELS.get(setup, setup_labels.get(setup, setup))


def _display_test_case_label(test_case: str) -> str:
    return TEST_CASE_DISPLAY_LABELS.get(test_case, test_case.replace("_", " "))


def _format_number(value: int | float | None) -> str:
    if value is None:
        return "--"
    return f"{int(round(value)):,}".replace(",", r"\,")


def _format_decimal(value: int | float | None, digits: int = 1) -> str:
    if value is None:
        return "--"
    text = f"{float(value):.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _format_percentage(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value * 100:.0f}\\%"


def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _extract_total_tokens(run: dict[str, Any]) -> int | None:
    tokens = run.get("tokens") or {}

    total_tokens = _to_int(tokens.get("total_tokens"))
    if total_tokens is not None:
        return total_tokens

    input_tokens = _to_int(tokens.get("input_tokens"))
    output_tokens = _to_int(tokens.get("output_tokens"))
    if input_tokens is not None and output_tokens is not None:
        return input_tokens + output_tokens
    if input_tokens is not None:
        return input_tokens
    if output_tokens is not None:
        return output_tokens

    usage = run.get("usage") or {}
    usage_total = _to_int(usage.get("total_tokens"))
    if usage_total is not None:
        return usage_total

    prompt_tokens = _to_int(usage.get("prompt_tokens"))
    completion_tokens = _to_int(usage.get("completion_tokens"))
    if prompt_tokens is not None and completion_tokens is not None:
        return prompt_tokens + completion_tokens
    if prompt_tokens is not None:
        return prompt_tokens
    if completion_tokens is not None:
        return completion_tokens

    return None


def _tool_name(call: dict[str, Any]) -> str | None:
    return call.get("name") or ((call.get("function") or {}).get("name"))


def _tool_arguments(call: dict[str, Any]) -> Any:
    return call.get("arguments") or ((call.get("function") or {}).get("arguments"))


def _normalised_tool_arguments(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    try:
        return json.dumps(json.loads(str(value)), sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(value).replace(" ", "")


def _has_tool_output_payload(call: dict[str, Any]) -> bool:
    return bool(call.get("output") or call.get("result") or call.get("output_preview"))


def _dedupe_adjacent_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse request/result duplicate log entries while keeping real repeats."""
    out: list[dict[str, Any]] = []
    index = 0

    while index < len(calls):
        current = calls[index]
        if index + 1 < len(calls):
            following = calls[index + 1]
            same_call = (
                _tool_name(current) == _tool_name(following)
                and _normalised_tool_arguments(_tool_arguments(current))
                == _normalised_tool_arguments(_tool_arguments(following))
            )
            if same_call and not _has_tool_output_payload(current) and _has_tool_output_payload(following):
                out.append(following)
                index += 2
                continue

        out.append(current)
        index += 1

    return out


def _extract_tool_calls_count(run: dict[str, Any], setup_key: str | None = None) -> int | None:
    tool_calls = run.get("tool_calls")
    if isinstance(tool_calls, list):
        calls = _dedupe_adjacent_tool_calls([call for call in tool_calls if isinstance(call, dict)])
        if setup_key == "mcp":
            calls = [call for call in calls if _tool_name(call) not in IGNORED_MCP_ATC_TOOLS]
        return len(calls)

    tool_calls_count = _to_int(run.get("tool_calls_count"))
    if tool_calls_count is not None:
        return tool_calls_count

    return None


def _extract_duration_ms(run: dict[str, Any]) -> float | None:
    duration_ms = _to_float(run.get("duration_ms"))
    if duration_ms is not None:
        return duration_ms

    duration_seconds = _to_float(run.get("duration_seconds"))
    if duration_seconds is not None:
        return duration_seconds * 1000

    return None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_ground_truth() -> dict[str, Any]:
    if not GROUND_TRUTH_PATH.exists():
        return {}
    try:
        data = _load_json(GROUND_TRUTH_PATH)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _run_success(test_case_id: str, run: dict[str, Any], gt: dict[str, Any]) -> bool:
    if run.get("error") or run.get("retry_exhausted"):
        return False

    checks = (run.get("state_validation") or {}).get("checks") or {}

    if test_case_id == "raise_building_by_id":
        if checks:
            return bool(checks.get("building_found") and checks.get("height_correct"))

        validation = run.get("validation") or {}
        accuracy = validation.get("accuracy")
        return bool((accuracy or 0) >= 1.0)

    if test_case_id == "add_building":
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
        return bool((accuracy or 0) >= 1.0)

    validation = run.get("validation") or {}
    accuracy = validation.get("accuracy")
    if accuracy is None:
        return False
    return bool(accuracy >= 1.0) if not gt else bool(accuracy >= 1.0)


@dataclass
class CellMetrics:
    avg_tokens: float | None
    avg_tool_calls: float | None
    avg_duration_ms: float | None
    success_rate: float | None
    run_count: int
    available: bool


def _collect_run_metrics(
    run_dir: Path,
    run_id: str,
    gt: dict[str, Any],
    include_setups: set[str] | None = None,
    include_tests: set[str] | None = None,
) -> dict[tuple[str, str], CellMetrics]:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")

    metrics: dict[tuple[str, str], CellMetrics] = {}

    for setup_dir in sorted(run_dir.iterdir(), key=lambda p: p.name):
        if not setup_dir.is_dir() or setup_dir.name in SKIP_DIRS:
            continue

        setup_key = setup_dir.name
        if include_setups is not None and setup_key not in include_setups:
            continue

        for log_path in sorted(setup_dir.glob("*.json")):
            log_data = _load_json(log_path)
            meta = log_data.get("meta") or {}
            test_case_id = meta.get("test_case_id") or (log_data.get("test_case") or {}).get("id")

            if not test_case_id:
                continue
            if include_tests is not None and test_case_id not in include_tests:
                continue

            runs = log_data.get("runs") or []
            token_values: list[int] = []
            tool_call_values: list[int] = []
            duration_values: list[float] = []
            success_count = 0

            for run in runs:
                total_tokens = _extract_total_tokens(run)
                if total_tokens is not None:
                    token_values.append(total_tokens)

                tool_calls_count = _extract_tool_calls_count(run, setup_key=setup_key)
                if tool_calls_count is not None:
                    tool_call_values.append(tool_calls_count)

                duration_ms = _extract_duration_ms(run)
                if duration_ms is not None:
                    duration_values.append(duration_ms)

                if _run_success(test_case_id, run, gt):
                    success_count += 1

            run_count = len(runs)
            if run_count == 0:
                metrics[(setup_key, test_case_id)] = CellMetrics(
                    avg_tokens=None,
                    avg_tool_calls=None,
                    avg_duration_ms=None,
                    success_rate=0.0,
                    run_count=0,
                    available=False,
                )
                continue

            avg_tokens = mean(token_values) if token_values else None
            avg_tool_calls = mean(tool_call_values) if tool_call_values else None
            avg_duration_ms = mean(duration_values) if duration_values else None
            success_rate = success_count / run_count
            metrics[(setup_key, test_case_id)] = CellMetrics(
                avg_tokens=avg_tokens,
                avg_tool_calls=avg_tool_calls,
                avg_duration_ms=avg_duration_ms,
                success_rate=success_rate,
                run_count=run_count,
                available=True,
            )

    if not metrics:
        raise RuntimeError(f"No metric data found for run {run_id}. Check run folder and filters.")

    return metrics


def _collect_setup_labels(run_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}

    for setup_dir in sorted(run_dir.iterdir(), key=lambda p: p.name):
        if not setup_dir.is_dir() or setup_dir.name in SKIP_DIRS:
            continue

        setup_key = setup_dir.name
        for log_path in sorted(setup_dir.glob("*.json")):
            try:
                data = _load_json(log_path)
            except Exception:
                continue
            meta = data.get("meta") or {}
            out[setup_key] = meta.get("setup") or setup_key
            break

    return out


def _infer_run_label(run_dir: Path, fallback: str | None = None) -> str:
    model = None
    reasoning = None

    for setup_dir in sorted(run_dir.iterdir(), key=lambda p: p.name):
        if not setup_dir.is_dir() or setup_dir.name in SKIP_DIRS:
            continue
        for log_path in sorted(setup_dir.glob("*.json")):
            try:
                data = _load_json(log_path)
            except Exception:
                continue
            meta = data.get("meta") or {}
            if model is None:
                model = meta.get("model")
            if reasoning is None:
                reasoning = meta.get("reasoning_effort")
            if model:
                return f"{model} ({reasoning})" if reasoning else str(model)

    return fallback or run_dir.name


def _ordered_keys(values: set[str], preferred: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    for value in preferred:
        if value in values and value not in seen:
            ordered.append(value)
            seen.add(value)

    for value in sorted(values):
        if value not in seen:
            ordered.append(value)
            seen.add(value)

    return ordered


def _collect_cell_matrix(
    metrics_a: dict[tuple[str, str], CellMetrics],
    metrics_b: dict[tuple[str, str], CellMetrics],
    setups: list[str],
    test_cases: list[str],
) -> list[tuple[str, str, CellMetrics | None, CellMetrics | None]]:
    return [
        (setup, test_case, metrics_a.get((setup, test_case)), metrics_b.get((setup, test_case)))
        for setup in setups
        for test_case in test_cases
    ]


def _total_from_rows(
    rows: list[tuple[str, str, CellMetrics | None, CellMetrics | None]],
) -> tuple[
    tuple[float | None, float | None, float | None, float | None],
    tuple[float | None, float | None, float | None, float | None],
]:
    tokens_a: list[float] = []
    tokens_b: list[float] = []
    tool_calls_a: list[float] = []
    tool_calls_b: list[float] = []
    duration_a: list[float] = []
    duration_b: list[float] = []
    success_a: list[float] = []
    success_b: list[float] = []

    for _setup, _test_case, a, b in rows:
        if a is not None and a.available and a.avg_tokens is not None:
            tokens_a.append(a.avg_tokens)
        if b is not None and b.available and b.avg_tokens is not None:
            tokens_b.append(b.avg_tokens)
        if a is not None and a.available and a.avg_tool_calls is not None:
            tool_calls_a.append(a.avg_tool_calls)
        if b is not None and b.available and b.avg_tool_calls is not None:
            tool_calls_b.append(b.avg_tool_calls)
        if a is not None and a.available and a.avg_duration_ms is not None:
            duration_a.append(a.avg_duration_ms)
        if b is not None and b.available and b.avg_duration_ms is not None:
            duration_b.append(b.avg_duration_ms)
        if a is not None and a.available and a.success_rate is not None:
            success_a.append(a.success_rate)
        if b is not None and b.available and b.success_rate is not None:
            success_b.append(b.success_rate)

    avg_tok_a = mean(tokens_a) if tokens_a else None
    avg_calls_a = mean(tool_calls_a) if tool_calls_a else None
    avg_duration_a = mean(duration_a) if duration_a else None
    avg_sr_a = mean(success_a) if success_a else None
    avg_tok_b = mean(tokens_b) if tokens_b else None
    avg_calls_b = mean(tool_calls_b) if tool_calls_b else None
    avg_duration_b = mean(duration_b) if duration_b else None
    avg_sr_b = mean(success_b) if success_b else None

    return (avg_tok_a, avg_calls_a, avg_duration_a, avg_sr_a), (
        avg_tok_b,
        avg_calls_b,
        avg_duration_b,
        avg_sr_b,
    )


def _format_metric_cells(metrics: CellMetrics | None) -> tuple[str, str, str, str]:
    if metrics is None or not metrics.available:
        return "--", "--", "--", "--"

    tokens = _format_number(metrics.avg_tokens)
    calls = _format_decimal(metrics.avg_tool_calls, digits=2)
    duration = _format_decimal(
        (metrics.avg_duration_ms / 1000) if metrics.avg_duration_ms is not None else None
    )
    success = _format_percentage(metrics.success_rate)
    return tokens, calls, duration, success


def _summary_metrics(
    rows: list[tuple[str, str, CellMetrics | None, CellMetrics | None]],
) -> tuple[CellMetrics, CellMetrics]:
    (tok_a, calls_a, duration_a, sr_a), (tok_b, calls_b, duration_b, sr_b) = _total_from_rows(rows)
    run_count_a = sum((a.run_count if a is not None and a.available else 0) for _s, _tc, a, _b in rows)
    run_count_b = sum((b.run_count if b is not None and b.available else 0) for _s, _tc, _a, b in rows)
    return (
        CellMetrics(tok_a, calls_a, duration_a, sr_a, run_count_a, run_count_a > 0),
        CellMetrics(tok_b, calls_b, duration_b, sr_b, run_count_b, run_count_b > 0),
    )


def build_latex_table(
    run_a: str,
    run_b: str,
    rows: list[tuple[str, str, CellMetrics | None, CellMetrics | None]],
    label_a: str,
    label_b: str,
    setup_labels: dict[str, str],
    fit_width: bool = False,
    vertical_setups: bool = True,
    include_setup_summaries: bool = True,
) -> str:
    (total_tok_a, total_calls_a, total_duration_a, total_sr_a), (
        total_tok_b,
        total_calls_b,
        total_duration_b,
        total_sr_b,
    ) = _total_from_rows(rows)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\begingroup",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.05}",
    ]

    if fit_width:
        lines.append(r"\makebox[\textwidth][c]{%")

    lines.extend(
        [
            r"\begin{tabular}{clrrrr|rrrr}",
            f" &  & \\multicolumn{{4}}{{c}}{{\\textbf{{{_escape_latex(label_a)}}}}} & \\multicolumn{{4}}{{c}}{{\\textbf{{{_escape_latex(label_b)}}}}} \\\\",
            r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
            (
                r"\textbf{Setup} & \textbf{Test Case} & \textbf{\shortstack{Avg.\\Tokens}} "
                r"& \textbf{\shortstack{Tool\\Calls}} & \textbf{\shortstack{Time\\(s)}} "
                r"& \textbf{\shortstack{Success\\Rate}} & \textbf{\shortstack{Avg.\\Tokens}} "
                r"& \textbf{\shortstack{Tool\\Calls}} & \textbf{\shortstack{Time\\(s)}} "
                r"& \textbf{\shortstack{Success\\Rate}} \\"
            ),
            r"\midrule",
        ]
    )

    setup_row_counts: dict[str, int] = {}
    for setup, _test_case, _ma, _mb in rows:
        setup_row_counts[setup] = setup_row_counts.get(setup, 0) + 1
    if include_setup_summaries:
        setup_row_counts = {setup: row_count + 1 for setup, row_count in setup_row_counts.items()}

    rows_by_setup: dict[str, list[tuple[str, str, CellMetrics | None, CellMetrics | None]]] = {}
    setup_order: list[str] = []
    for row in rows:
        setup = row[0]
        if setup not in rows_by_setup:
            rows_by_setup[setup] = []
            setup_order.append(setup)
        rows_by_setup[setup].append(row)

    def append_data_row(
        setup: str,
        test_case_label: str,
        ma: CellMetrics | None,
        mb: CellMetrics | None,
        setup_cell: str = "",
        bold: bool = False,
    ) -> None:
        a_tok, a_calls, a_time, a_sr = _format_metric_cells(ma)
        b_tok, b_calls, b_time, b_sr = _format_metric_cells(mb)
        if bold:
            test_case_label = f"\\textbf{{{_escape_latex(test_case_label)}}}"
            a_tok, a_calls, a_time, a_sr = (f"\\textbf{{{v}}}" for v in (a_tok, a_calls, a_time, a_sr))
            b_tok, b_calls, b_time, b_sr = (f"\\textbf{{{v}}}" for v in (b_tok, b_calls, b_time, b_sr))
        else:
            test_case_label = _escape_latex(test_case_label)

        lines.append(
            f"{setup_cell} & {test_case_label} & "
            f"{a_tok} & {a_calls} & {a_time} & {a_sr} & "
            f"{b_tok} & {b_calls} & {b_time} & {b_sr} \\\\"
        )

    previous_setup = False
    for setup in setup_order:
        if previous_setup:
            lines.append(r"\midrule")
        previous_setup = True

        setup_label = _display_setup_label(setup, setup_labels)
        setup_rows = rows_by_setup[setup]
        for row_index, (_setup, test_case, ma, mb) in enumerate(setup_rows):
            if vertical_setups:
                setup_cell = ""
                if row_index == 0:
                    setup_cell = (
                        f"\\multirow{{{setup_row_counts[setup]}}}{{*}}"
                        f"{{\\rotatebox[origin=c]{{90}}{{\\textbf{{{_escape_latex(setup_label)}}}}}}}"
                    )
            else:
                setup_cell = f"\\textbf{{{_escape_latex(setup_label)}}}"

            append_data_row(setup, _display_test_case_label(test_case), ma, mb, setup_cell=setup_cell)

        if include_setup_summaries:
            ma_summary, mb_summary = _summary_metrics(setup_rows)
            if vertical_setups:
                setup_cell = ""
            else:
                setup_cell = (
                    f"\\textbf{{{_escape_latex(setup_label)}}}"
                )
            append_data_row(setup, "Setup avg.", ma_summary, mb_summary, setup_cell=setup_cell, bold=True)

    lines.extend(
        [
            r"\midrule",
            f"\\textbf{{Total}} & \\textbf{{All Test Cases}} & {_format_number(total_tok_a)} "
            f"& {_format_decimal(total_calls_a)} & {_format_decimal((total_duration_a / 1000) if total_duration_a is not None else None)} "
            f"& {_format_percentage(total_sr_a)} & {_format_number(total_tok_b)} & {_format_decimal(total_calls_b)} "
            f"& {_format_decimal((total_duration_b / 1000) if total_duration_b is not None else None)} "
            f"& {_format_percentage(total_sr_b)} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )

    if fit_width:
        lines.append(r"}")

    lines.extend(
        [
            r"\endgroup",
            r"\end{table}",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a two-model LaTeX comparison table from evaluation runs.")
    parser.add_argument("--run-a", required=True, help="Left model run folder")
    parser.add_argument("--run-b", required=True, help="Right model run folder")
    parser.add_argument("--label-a", default=None, help="Display label for run A")
    parser.add_argument("--label-b", default=None, help="Display label for run B")
    parser.add_argument("--setup", action="append", default=None, help="Optional setup filter")
    parser.add_argument("--test-case", action="append", default=None, help="Optional test-case filter")
    parser.add_argument("--ground-truth", default=None, help="Optional alternative ground truth file")
    parser.add_argument("--output", default=None, help="Write table to this path")
    parser.add_argument(
        "--fit-width",
        action="store_true",
        help="Center the tabular in a \\makebox[\\textwidth][c]{...} wrapper. Does not scale.",
    )
    parser.add_argument(
        "--no-fit-width",
        action="store_true",
        help="Deprecated; tables are unwrapped by default.",
    )
    parser.add_argument(
        "--horizontal-setups",
        action="store_true",
        help="Print setup labels horizontally in every row instead of one rotated multirow label per setup.",
    )

    args = parser.parse_args()

    include_setups = set(args.setup) if args.setup else None
    include_tests = set(args.test_case) if args.test_case else None

    global GROUND_TRUTH_PATH
    if args.ground_truth:
        GROUND_TRUTH_PATH = Path(args.ground_truth)

    gt = _load_ground_truth()

    run_a_dir = LOG_DIR / args.run_a if not Path(args.run_a).is_absolute() else Path(args.run_a)
    if not run_a_dir.exists():
        run_a_dir = Path(args.run_a)
    run_b_dir = LOG_DIR / args.run_b if not Path(args.run_b).is_absolute() else Path(args.run_b)
    if not run_b_dir.exists():
        run_b_dir = Path(args.run_b)

    metrics_a = _collect_run_metrics(
        run_a_dir,
        args.run_a,
        gt,
        include_setups=include_setups,
        include_tests=include_tests,
    )
    metrics_b = _collect_run_metrics(
        run_b_dir,
        args.run_b,
        gt,
        include_setups=include_setups,
        include_tests=include_tests,
    )

    setup_keys = {key for key, _ in metrics_a.keys()} | {key for key, _ in metrics_b.keys()}
    test_case_keys = {tc for _, tc in metrics_a.keys()} | {tc for _, tc in metrics_b.keys()}

    preferred_setups = ["mcp", "no_tools", "code_interpreter"]
    preferred_tests = [
        "list_buildings",
        "highest_measured_height",
        "roof_volume_sum",
        "raise_building_by_id",
        "add_building",
        "count_party_walls",
    ]

    setups = _ordered_keys(setup_keys, preferred_setups)
    test_cases = _ordered_keys(test_case_keys, preferred_tests)

    if include_setups is not None:
        setups = [s for s in setups if s in include_setups]
    if include_tests is not None:
        test_cases = [tc for tc in test_cases if tc in include_tests]

    rows = _collect_cell_matrix(metrics_a, metrics_b, setups, test_cases)

    label_a = args.label_a or _infer_run_label(run_a_dir, fallback=args.run_a)
    label_b = args.label_b or _infer_run_label(run_b_dir, fallback=args.run_b)
    setup_labels = _collect_setup_labels(run_b_dir)
    setup_labels.update(_collect_setup_labels(run_a_dir))

    latex = build_latex_table(
        run_a=args.run_a,
        run_b=args.run_b,
        rows=rows,
        label_a=label_a,
        label_b=label_b,
        setup_labels=setup_labels,
        fit_width=args.fit_width and not args.no_fit_width,
        vertical_setups=not args.horizontal_setups,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(latex + "\n", encoding="utf-8")
        print(f"Generated: {out_path}")
    else:
        print(latex)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
