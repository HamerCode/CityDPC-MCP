"""
Analyze MCP tool-call sequences and parameter correctness for the latest run folder.

Outputs:
1) ordered tool calls for the last run per test case
2) ground-truth tool-call definitions (including accepted variants)
3) compliance report: order correctness + parameter correctness frequency
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.cases import RAISE_BUILDING_SPEC, TEST_BUILDING_SPEC

LOG_DIR = Path("evaluation_logs")
OUT_DIR = Path("evaluation_tool_calls")
TABLES_DIR = Path("evaluation_tables")
DEFAULT_SETUP = "mcp"
DATASET_FILENAME = "evaluation.city.json"
GROUND_TRUTH_FILE = Path("ground_truth.json")


def _find_latest_date_folder() -> Path | None:
    if not LOG_DIR.exists():
        return None
    folders = [d for d in LOG_DIR.iterdir() if d.is_dir()]
    if not folders:
        return None
    return sorted(folders, key=lambda d: d.stat().st_mtime, reverse=True)[0]


def _safe_load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _strip_leading_list_datasets(calls: list[dict]) -> list[dict]:
    out = list(calls)
    while out and out[0].get("name") == "list_datasets":
        out = out[1:]
    return out


def _get_calls(run: dict) -> list[dict]:
    calls = run.get("tool_calls") or []
    # New compact format already has "name"/"arguments". Fallback for legacy.
    normalized = []
    for c in calls:
        if not isinstance(c, dict):
            continue
        name = c.get("name") or (c.get("function") or {}).get("name")
        # take_snapshot is a helper/safety step and should not count as a tool call in this analysis
        if name == "take_snapshot":
            continue
        normalized.append(
            {
                "name": name,
                "arguments": c.get("arguments") or (c.get("function") or {}).get("arguments") or {},
                "status": c.get("status"),
                "step": c.get("step"),
            }
        )
    return normalized


def _args_as_dict(args: Any) -> dict:
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _is_close(a: Any, b: Any, eps: float = 1e-6) -> bool:
    try:
        return abs(float(a) - float(b)) <= eps
    except Exception:
        return a == b


def _validate_load_dataset(call: dict) -> tuple[bool, str]:
    args = _args_as_dict(call.get("arguments"))
    filename = args.get("filename")
    if not filename:
        return False, "missing filename"
    ok = filename == DATASET_FILENAME or (filename.startswith("evaluation__") and filename.endswith(".city.json"))
    return ok, "filename must be evaluation.city.json or a run-specific variant"


def _validate_add_create_building(call: dict) -> tuple[bool, str]:
    args = _args_as_dict(call.get("arguments"))
    checks = [
        args.get("id") == TEST_BUILDING_SPEC["id"],
        _is_close(args.get("groundSurfaceHeight"), TEST_BUILDING_SPEC["groundSurfaceHeight"]),
        _is_close(args.get("geometryHeight"), TEST_BUILDING_SPEC["measuredHeight"]),
        str(args.get("roofType")) == str(TEST_BUILDING_SPEC["roofType"]),
    ]
    return all(checks), "create_building core params mismatch"


def _validate_add_enrich_building(call: dict) -> tuple[bool, str]:
    args = _args_as_dict(call.get("arguments"))
    checks = [
        args.get("building_id") == TEST_BUILDING_SPEC["id"],
        _is_close(args.get("measured_height"), TEST_BUILDING_SPEC["measuredHeight"]),
        str(args.get("roof_type")) == str(TEST_BUILDING_SPEC["roofType"]),
        _is_close(args.get("roof_height"), TEST_BUILDING_SPEC["roofHeight"]),
        str(args.get("function")) == str(TEST_BUILDING_SPEC["function"]),
        int(args.get("storeys_above_ground", -1)) == int(TEST_BUILDING_SPEC["storeysAboveGround"]),
    ]
    return all(checks), "enrich_building params mismatch for add_building"


def _validate_raise_get(call: dict) -> tuple[bool, str]:
    args = _args_as_dict(call.get("arguments"))
    ok = args.get("building_id") == RAISE_BUILDING_SPEC["target_building_id"]
    return ok, "get_building_by_id target mismatch"


def _validate_raise_enrich(call: dict) -> tuple[bool, str]:
    args = _args_as_dict(call.get("arguments"))
    checks = [
        args.get("building_id") == RAISE_BUILDING_SPEC["target_building_id"],
        _is_close(args.get("measured_height"), RAISE_BUILDING_SPEC["new_height"]),
    ]
    return all(checks), "enrich_building params mismatch for raise_building_by_id"


def _validate_roof_calc(call: dict) -> tuple[bool, str]:
    args = _args_as_dict(call.get("arguments"))
    bid = args.get("building_id")
    ok = isinstance(bid, str) and len(bid.strip()) > 0
    return ok, "calculate_roof_volume_by_id needs non-empty building_id"


def _all_names(calls: list[dict]) -> list[str]:
    return [c.get("name", "") for c in calls]


def _check_order_list_buildings(calls: list[dict]) -> tuple[bool, str]:
    names = _all_names(_strip_leading_list_datasets(calls))
    expected = ["load_dataset", "get_buiding_Id_list"]
    return names == expected, f"expected {expected}, got {names}"


def _check_order_highest(calls: list[dict]) -> tuple[bool, str]:
    names = _all_names(_strip_leading_list_datasets(calls))
    # Accepted:
    # A) load_dataset -> get_all_buildings
    # B) load_dataset -> get_all_buildings -> (number_of_buildings)? -> (get_buiding_Id_list)? -> get_building_by_id*
    if len(names) >= 2 and names[0] == "load_dataset" and names[1] == "get_all_buildings":
        rest = names[2:]
        allowed = {"number_of_buildings", "get_buiding_Id_list", "get_building_by_id"}
        if any(n not in allowed for n in rest):
            return False, "contains disallowed tools after get_all_buildings"
        # enforce relative order blocks
        seen_ids = False
        for n in rest:
            if n == "get_building_by_id":
                seen_ids = True
            elif seen_ids and n in {"number_of_buildings", "get_buiding_Id_list"}:
                return False, "number/get_id_list appears after get_building_by_id"
        return True, "accepted highest_measured_height order"
    return False, f"must start with ['load_dataset', 'get_all_buildings'], got {names[:2]}"


def _check_order_roof(calls: list[dict]) -> tuple[bool, str]:
    names = _all_names(_strip_leading_list_datasets(calls))
    if len(names) < 3:
        return False, "too short"
    if names[0] != "load_dataset" or names[1] != "get_buiding_Id_list":
        return False, "must start with load_dataset -> get_buiding_Id_list"
    if not all(n == "calculate_roof_volume_by_id" for n in names[2:]):
        return False, "all remaining calls must be calculate_roof_volume_by_id"
    return True, "accepted roof_volume_sum order"


def _check_order_raise(calls: list[dict]) -> tuple[bool, str]:
    names = _all_names(_strip_leading_list_datasets(calls))
    # Strict minimal expected flow
    expected = ["load_dataset", "get_building_by_id", "enrich_building", "save_dataset"]
    if names == expected:
        return True, "exact expected sequence"
    # Relaxed acceptance: must include expected steps in order, extras allowed between.
    idx = 0
    for n in names:
        if idx < len(expected) and n == expected[idx]:
            idx += 1
    if idx == len(expected):
        return True, "contains expected subsequence (extras allowed)"
    return False, f"missing expected subsequence {expected}"


def _check_order_add(calls: list[dict]) -> tuple[bool, str]:
    names = _all_names(_strip_leading_list_datasets(calls))
    expected = ["load_dataset", "create_building", "enrich_building", "save_dataset"]
    return names == expected, f"expected {expected}, got {names}"


def _check_params(test_case_id: str, calls: list[dict]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    stripped = _strip_leading_list_datasets(calls)
    if not stripped:
        return False, ["no calls"]

    # load_dataset check for all test cases
    load_calls = [c for c in stripped if c.get("name") == "load_dataset"]
    if not load_calls:
        issues.append("missing load_dataset")
    else:
        ok, msg = _validate_load_dataset(load_calls[0])
        if not ok:
            issues.append(msg)

    if test_case_id == "add_building":
        creates = [c for c in stripped if c.get("name") == "create_building"]
        enriches = [c for c in stripped if c.get("name") == "enrich_building"]
        if not creates:
            issues.append("missing create_building")
        else:
            ok, msg = _validate_add_create_building(creates[0])
            if not ok:
                issues.append(msg)
        if not enriches:
            issues.append("missing enrich_building")
        else:
            ok, msg = _validate_add_enrich_building(enriches[0])
            if not ok:
                issues.append(msg)
    elif test_case_id == "raise_building_by_id":
        gets = [c for c in stripped if c.get("name") == "get_building_by_id"]
        enriches = [c for c in stripped if c.get("name") == "enrich_building"]
        if not gets:
            issues.append("missing get_building_by_id")
        else:
            ok, msg = _validate_raise_get(gets[0])
            if not ok:
                issues.append(msg)
        if not enriches:
            issues.append("missing enrich_building")
        else:
            ok, msg = _validate_raise_enrich(enriches[0])
            if not ok:
                issues.append(msg)
    elif test_case_id == "roof_volume_sum":
        calcs = [c for c in stripped if c.get("name") == "calculate_roof_volume_by_id"]
        if not calcs:
            issues.append("missing calculate_roof_volume_by_id")
        else:
            for c in calcs:
                ok, msg = _validate_roof_calc(c)
                if not ok:
                    issues.append(msg)
                    break

    return len(issues) == 0, issues


def _order_checker(test_case_id: str):
    return {
        "list_buildings": _check_order_list_buildings,
        "highest_measured_height": _check_order_highest,
        "roof_volume_sum": _check_order_roof,
        "raise_building_by_id": _check_order_raise,
        "add_building": _check_order_add,
    }.get(test_case_id)


def _ground_truth_spec() -> dict[str, Any]:
    return {
        "notes": {
            "highest_measured_height_alt_allowed": (
                "After get_all_buildings, the max height can already be derived. "
                "Longer variants are accepted as long as they keep allowed order."
            ),
        },
        "test_cases": {
            "list_buildings": {
                "accepted_sequence": ["load_dataset", "get_buiding_Id_list"],
                "required_param_checks": {"load_dataset.filename": DATASET_FILENAME},
            },
            "highest_measured_height": {
                "accepted_start": ["load_dataset", "get_all_buildings"],
                "allowed_followup_tools": [
                    "number_of_buildings",
                    "get_buiding_Id_list",
                    "get_building_by_id",
                ],
                "required_param_checks": {"load_dataset.filename": DATASET_FILENAME},
            },
            "roof_volume_sum": {
                "accepted_prefix": ["load_dataset", "get_buiding_Id_list"],
                "required_repeated_tool": "calculate_roof_volume_by_id",
                "required_param_checks": {
                    "load_dataset.filename": DATASET_FILENAME,
                    "calculate_roof_volume_by_id.building_id": "non_empty_string",
                },
            },
            "raise_building_by_id": {
                "expected_core_sequence": ["load_dataset", "get_building_by_id", "enrich_building", "save_dataset"],
                "required_param_checks": {
                    "load_dataset.filename": DATASET_FILENAME,
                    "get_building_by_id.building_id": RAISE_BUILDING_SPEC["target_building_id"],
                    "enrich_building.building_id": RAISE_BUILDING_SPEC["target_building_id"],
                    "enrich_building.measured_height": RAISE_BUILDING_SPEC["new_height"],
                },
            },
            "add_building": {
                "expected_sequence": ["load_dataset", "create_building", "enrich_building", "save_dataset"],
                "required_param_checks": {
                    "load_dataset.filename": DATASET_FILENAME,
                    "create_building.id": TEST_BUILDING_SPEC["id"],
                    "create_building.groundSurfaceHeight": TEST_BUILDING_SPEC["groundSurfaceHeight"],
                    "create_building.geometryHeight": TEST_BUILDING_SPEC["measuredHeight"],
                    "create_building.roofType": TEST_BUILDING_SPEC["roofType"],
                    "enrich_building.building_id": TEST_BUILDING_SPEC["id"],
                    "enrich_building.measured_height": TEST_BUILDING_SPEC["measuredHeight"],
                    "enrich_building.roof_type": TEST_BUILDING_SPEC["roofType"],
                    "enrich_building.roof_height": TEST_BUILDING_SPEC["roofHeight"],
                    "enrich_building.function": TEST_BUILDING_SPEC["function"],
                    "enrich_building.storeys_above_ground": TEST_BUILDING_SPEC["storeysAboveGround"],
                },
            },
        },
    }


def _ideal_tool_sequences() -> dict[str, Any]:
    """Human-readable ideal tool-call sequences with concrete parameter expectations."""
    return {
        "notes": {
            "highest_measured_height_short_path_allowed": True,
            "roof_volume_sum_repeat_rule": (
                "calculate_roof_volume_by_id is called once per building_id returned by get_buiding_Id_list"
            ),
        },
        "test_cases": {
            "list_buildings": {
                "ideal_sequences": [
                    [
                        {"tool": "load_dataset", "arguments": {"filename": DATASET_FILENAME}},
                        {"tool": "get_buiding_Id_list", "arguments": {}},
                    ]
                ],
            },
            "highest_measured_height": {
                "ideal_sequences": [
                    [
                        {"tool": "load_dataset", "arguments": {"filename": DATASET_FILENAME}},
                        {"tool": "get_all_buildings", "arguments": {}},
                    ],
                    [
                        {"tool": "load_dataset", "arguments": {"filename": DATASET_FILENAME}},
                        {"tool": "get_all_buildings", "arguments": {}},
                        {"tool": "number_of_buildings", "arguments": {}, "optional": True},
                        {"tool": "get_buiding_Id_list", "arguments": {}, "optional": True},
                        {
                            "tool": "get_building_by_id",
                            "arguments": {"building_id": "<id_from_get_buiding_Id_list>"},
                            "repeat": "0..n",
                        },
                    ],
                ],
            },
            "roof_volume_sum": {
                "ideal_sequences": [
                    [
                        {"tool": "load_dataset", "arguments": {"filename": DATASET_FILENAME}},
                        {"tool": "get_buiding_Id_list", "arguments": {}},
                        {
                            "tool": "calculate_roof_volume_by_id",
                            "arguments": {"building_id": "<id_from_get_buiding_Id_list>"},
                            "repeat": "for_each_building_id",
                        },
                    ]
                ],
            },
            "raise_building_by_id": {
                "ideal_sequences": [
                    [
                        {"tool": "load_dataset", "arguments": {"filename": DATASET_FILENAME}},
                        {
                            "tool": "get_building_by_id",
                            "arguments": {"building_id": RAISE_BUILDING_SPEC["target_building_id"]},
                        },
                        {
                            "tool": "enrich_building",
                            "arguments": {
                                "building_id": RAISE_BUILDING_SPEC["target_building_id"],
                                "measured_height": RAISE_BUILDING_SPEC["new_height"],
                            },
                        },
                        {"tool": "save_dataset", "arguments": {}},
                    ]
                ],
            },
            "add_building": {
                "ideal_sequences": [
                    [
                        {"tool": "load_dataset", "arguments": {"filename": DATASET_FILENAME}},
                        {
                            "tool": "create_building",
                            "arguments": {
                                "id": TEST_BUILDING_SPEC["id"],
                                "groundsCoordinates": TEST_BUILDING_SPEC["groundCoordinates"],
                                "groundSurfaceHeight": TEST_BUILDING_SPEC["groundSurfaceHeight"],
                                "geometryHeight": TEST_BUILDING_SPEC["measuredHeight"],
                                "roofType": TEST_BUILDING_SPEC["roofType"],
                                "roofHeight": TEST_BUILDING_SPEC["roofHeight"],
                            },
                        },
                        {
                            "tool": "enrich_building",
                            "arguments": {
                                "building_id": TEST_BUILDING_SPEC["id"],
                                "measured_height": TEST_BUILDING_SPEC["measuredHeight"],
                                "roof_type": TEST_BUILDING_SPEC["roofType"],
                                "roof_height": TEST_BUILDING_SPEC["roofHeight"],
                                "function": TEST_BUILDING_SPEC["function"],
                                "storeys_above_ground": TEST_BUILDING_SPEC["storeysAboveGround"],
                            },
                        },
                        {"tool": "save_dataset", "arguments": {}},
                    ]
                ],
            },
        },
    }


def _render_ideal_sequences_markdown(spec: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Ideal Tool-Call Sequences (Ground Truth)")
    lines.append("")
    notes = spec.get("notes", {})
    if notes:
        lines.append("## Notes")
        for k, v in notes.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    for test_case, tc_spec in sorted((spec.get("test_cases") or {}).items()):
        lines.append(f"## {test_case}")
        seqs = tc_spec.get("ideal_sequences") or []
        for i, seq in enumerate(seqs, start=1):
            lines.append(f"- Sequenz {i}:")
            for idx, step in enumerate(seq, start=1):
                extra = []
                if step.get("optional"):
                    extra.append("optional")
                if step.get("repeat"):
                    extra.append(f"repeat={step['repeat']}")
                extra_str = f" ({', '.join(extra)})" if extra else ""
                args = json.dumps(step.get("arguments", {}), ensure_ascii=False, sort_keys=True)
                lines.append(f"  {idx}. `{step.get('tool')}`{extra_str} args={args}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _escape_latex(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
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
    return "".join(repl.get(ch, ch) for ch in text)


def _render_ideal_sequences_latex(spec: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(r"\section*{Ideal Tool-Call Sequences (Ground Truth)}")
    lines.append("")
    notes = spec.get("notes", {})
    if notes:
        lines.append(r"\textbf{Notes}")
        lines.append(r"\begin{itemize}")
        for k, v in notes.items():
            lines.append(r"\item " + _escape_latex(f"{k}: {v}"))
        lines.append(r"\end{itemize}")
        lines.append("")

    for test_case, tc_spec in sorted((spec.get("test_cases") or {}).items()):
        lines.append(r"\subsection*{" + _escape_latex(test_case) + "}")
        seqs = tc_spec.get("ideal_sequences") or []
        for i, seq in enumerate(seqs, start=1):
            lines.append(r"\textbf{Sequenz " + str(i) + "}")
            lines.append(r"\begin{enumerate}")
            for step in seq:
                extra = []
                if step.get("optional"):
                    extra.append("optional")
                if step.get("repeat"):
                    extra.append(f"repeat={step['repeat']}")
                extra_str = f" ({', '.join(extra)})" if extra else ""
                args = json.dumps(step.get("arguments", {}), ensure_ascii=False, sort_keys=True)
                item = (
                    r"\texttt{" + _escape_latex(str(step.get("tool"))) + r"}"
                    + _escape_latex(extra_str + " ")
                    + _escape_latex(f"args={args}")
                )
                lines.append(r"\item " + item)
            lines.append(r"\end{enumerate}")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def _latex_args_block(args: dict[str, Any]) -> str:
    pretty = json.dumps(args, ensure_ascii=False, sort_keys=True, indent=2)
    escaped_lines = [_escape_latex(line) for line in pretty.splitlines()]
    body = r"\\ ".join(escaped_lines) if escaped_lines else r"\{\}"
    return (
        r"\begin{minipage}{0.94\linewidth}"
        + "\n"
        + r"\ttfamily\footnotesize "
        + body
        + "\n"
        + r"\end{minipage}"
    )


def _render_adherence_latex(ideal_spec: dict[str, Any], compliance: dict[str, Any], date_folder: str, setup: str) -> str:
    def _load_building_count() -> int | None:
        if not GROUND_TRUTH_FILE.exists():
            return None
        try:
            with GROUND_TRUTH_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return int(
                (
                    ((data.get("solutions") or {}).get("list_buildings") or {}).get("output") or {}
                ).get("total")
            )
        except Exception:
            return None

    def _optimal_calls_label(test_case: str, building_count: int | None) -> str:
        if test_case == "list_buildings":
            return "2(3)"
        if test_case == "highest_measured_height":
            return "2(3)"
        if test_case == "raise_building_by_id":
            return "4(5)"
        if test_case == "add_building":
            return "4(5)"
        if test_case == "roof_volume_sum":
            if isinstance(building_count, int) and building_count >= 0:
                base = 2 + building_count
                return f"{base}({base + 1})"
            return "2+N(3+N)"
        return "-"

    def _format_arg_value(key: str, value: Any) -> str:
        if key == "groundsCoordinates" and isinstance(value, list):
            return f"{len(value)} Koordinatenpunkte"
        if isinstance(value, float):
            return f"{value:.3f}".rstrip("0").rstrip(".")
        if isinstance(value, (int, bool)):
            return str(value)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return f"Liste[{len(value)}]"
        if isinstance(value, dict):
            return "Objekt"
        return str(value)

    def _format_args_readable(args: dict[str, Any]) -> str:
        if not args:
            return "keine Parameter"
        parts = []
        for k in sorted(args.keys()):
            parts.append(f"{k}={_format_arg_value(k, args[k])}")
        return ", ".join(parts)

    def _ideal_calls_cell(test_case: str) -> str:
        tc_spec = (ideal_spec.get("test_cases") or {}).get(test_case) or {}
        seqs = tc_spec.get("ideal_sequences") or []
        seq_blocks: list[str] = []
        for idx, seq in enumerate(seqs, start=1):
            lines = [f"Sequenz {idx}:"]
            lines.append("0. (optional) list_datasets")
            for i, step in enumerate(seq, start=1):
                extra = []
                if step.get("optional"):
                    extra.append("optional")
                if step.get("repeat"):
                    extra.append(f"wiederholen: {step['repeat']}")
                note = f" [{'; '.join(extra)}]" if extra else ""
                args_txt = _format_args_readable(step.get("arguments", {}))
                lines.append(f"{i}. {step.get('tool')} ({args_txt}){note}")
            seq_blocks.append(r"\\ ".join(_escape_latex(l) for l in lines))
        content = r"\\ \\ ".join(seq_blocks) if seq_blocks else "-"
        return r"\begin{minipage}{0.74\linewidth}\raggedright\footnotesize " + content + r"\end{minipage}"

    building_count = _load_building_count()

    lines: list[str] = []
    lines.append(r"\section*{MCP Tool-Call Analyse}")
    lines.append(r"\textbf{Run:} " + _escape_latex(date_folder) + r" \quad \textbf{Setup:} " + _escape_latex(setup))
    lines.append("")
    lines.append(r"\subsection*{Befolgungsquote je Test Case}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\hline")
    lines.append(r"Test Case & Runs & Actual Calls (\O) & Optimal Calls & Order OK (\%) & Params OK (\%) \\")
    lines.append(r"\hline")
    for test_case in sorted(compliance.keys()):
        c = compliance[test_case]
        total = int(c.get("total_runs", 0))
        order = float(c.get("order_ok_rate", 0.0)) * 100.0
        params = float(c.get("params_ok_rate", 0.0)) * 100.0
        run_details = c.get("run_details") or []
        total_calls = sum(len((rd.get("tool_calls") or [])) for rd in run_details)
        avg_calls = (total_calls / len(run_details)) if run_details else 0.0
        optimal = _optimal_calls_label(test_case, building_count)
        lines.append(
            f"{_escape_latex(test_case)} & {total} & {avg_calls:.2f} & {_escape_latex(optimal)} & {order:.1f} & {params:.1f} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    lines.append("")
    lines.append(r"\subsection*{Ideale Tool-Calls (Ground Truth)}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{ll}")
    lines.append(r"\hline")
    lines.append(r"Test Case & Ideale Tool Calls (untereinander) \\")
    lines.append(r"\hline")
    for test_case in sorted(compliance.keys()):
        lines.append(
            _escape_latex(test_case) + " & " + _ideal_calls_cell(test_case) + r" \\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines).strip() + "\n"


def analyze(date_folder: Path, setup_key: str) -> dict[str, Any]:
    setup_dir = date_folder / setup_key
    if not setup_dir.exists():
        raise FileNotFoundError(f"Setup folder not found: {setup_dir}")

    log_files = sorted(setup_dir.glob("*.json"))
    if not log_files:
        raise FileNotFoundError(f"No JSON logs in: {setup_dir}")

    last_runs_by_test: dict[str, dict] = {}
    compliance_by_test: dict[str, Any] = {}

    for log_file in log_files:
        data = _safe_load_json(log_file)
        meta = data.get("meta", {}) or {}
        test_case_id = meta.get("test_case_id") or (data.get("test_case", {}) or {}).get("id")
        runs = data.get("runs", []) or []
        if not test_case_id or not runs:
            continue

        checker = _order_checker(test_case_id)
        total = len(runs)
        order_ok = 0
        params_ok = 0
        both_ok = 0
        details = []

        for r in runs:
            calls = _get_calls(r)
            if checker is None:
                ord_ok, ord_msg = False, "no checker for test case"
            else:
                ord_ok, ord_msg = checker(calls)
            par_ok, par_issues = _check_params(test_case_id, calls)

            if ord_ok:
                order_ok += 1
            if par_ok:
                params_ok += 1
            if ord_ok and par_ok:
                both_ok += 1

            details.append(
                {
                    "run": r.get("run_index", r.get("run")),
                    "order_ok": ord_ok,
                    "order_note": ord_msg,
                    "params_ok": par_ok,
                    "param_issues": par_issues,
                    "tool_calls": calls,
                }
            )

        # last run (by run_index/run)
        last = sorted(details, key=lambda d: d.get("run") or 0)[-1]
        last_runs_by_test[test_case_id] = {
            "run": last.get("run"),
            "tool_calls": last.get("tool_calls"),
        }

        compliance_by_test[test_case_id] = {
            "log_file": str(log_file),
            "total_runs": total,
            "order_ok_runs": order_ok,
            "params_ok_runs": params_ok,
            "both_ok_runs": both_ok,
            "order_ok_rate": round(order_ok / total, 4) if total else 0.0,
            "params_ok_rate": round(params_ok / total, 4) if total else 0.0,
            "both_ok_rate": round(both_ok / total, 4) if total else 0.0,
            "run_details": details,
        }

    return {
        "date_folder": date_folder.name,
        "setup": setup_key,
        "latest_run_tool_calls": last_runs_by_test,
        "ground_truth": _ground_truth_spec(),
        "ideal_sequences": _ideal_tool_sequences(),
        "compliance": compliance_by_test,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze MCP tool-call order and parameter correctness.")
    parser.add_argument("--date", help="Date folder (e.g. 07.02_16-11_gpt-5_medium). Defaults to latest.")
    parser.add_argument("--setup", default=DEFAULT_SETUP, help="Setup folder (default: mcp).")
    args = parser.parse_args()

    if args.date:
        date_folder = LOG_DIR / args.date
        if not date_folder.exists():
            print(f"Error: Date folder not found: {date_folder}")
            return 1
    else:
        date_folder = _find_latest_date_folder()
        if not date_folder:
            print("Error: No evaluation logs found.")
            return 1

    result = analyze(date_folder=date_folder, setup_key=args.setup)

    out_base = OUT_DIR / date_folder.name / args.setup
    out_base.mkdir(parents=True, exist_ok=True)

    ordered_file = out_base / "latest_run_tool_calls_ordered.json"
    gt_file = out_base / "tool_call_ground_truth.json"
    ideal_json_file = out_base / "ideal_tool_call_sequences.json"
    ideal_md_file = out_base / "ideal_tool_call_sequences.md"
    ideal_tex_file = out_base / "ideal_tool_call_sequences.tex"
    adherence_tex_file = out_base / "tool_call_adherence_report.tex"
    compliance_file = out_base / "tool_call_compliance.json"

    with ordered_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "date_folder": result["date_folder"],
                "setup": result["setup"],
                "latest_run_tool_calls": result["latest_run_tool_calls"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with gt_file.open("w", encoding="utf-8") as f:
        json.dump(result["ground_truth"], f, ensure_ascii=False, indent=2)
    with ideal_json_file.open("w", encoding="utf-8") as f:
        json.dump(result["ideal_sequences"], f, ensure_ascii=False, indent=2)
    with ideal_md_file.open("w", encoding="utf-8") as f:
        f.write(_render_ideal_sequences_markdown(result["ideal_sequences"]))
    with ideal_tex_file.open("w", encoding="utf-8") as f:
        f.write(_render_ideal_sequences_latex(result["ideal_sequences"]))
    with compliance_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "date_folder": result["date_folder"],
                "setup": result["setup"],
                "compliance": result["compliance"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with adherence_tex_file.open("w", encoding="utf-8") as f:
        f.write(
            _render_adherence_latex(
                ideal_spec=result["ideal_sequences"],
                compliance=result["compliance"],
                date_folder=result["date_folder"],
                setup=result["setup"],
            )
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    adherence_table_file = TABLES_DIR / f"tool_call_adherence_{date_folder.name}_{args.setup}.tex"
    with adherence_table_file.open("w", encoding="utf-8") as f:
        f.write(
            _render_adherence_latex(
                ideal_spec=result["ideal_sequences"],
                compliance=result["compliance"],
                date_folder=result["date_folder"],
                setup=result["setup"],
            )
        )

    print(f"Analyzed: {date_folder.name} / {args.setup}")
    print(f"- Ordered latest run calls: {ordered_file}")
    print(f"- Ground truth:             {gt_file}")
    print(f"- Ideal sequences (JSON):   {ideal_json_file}")
    print(f"- Ideal sequences (MD):     {ideal_md_file}")
    print(f"- Ideal sequences (TeX):    {ideal_tex_file}")
    print(f"- Compliance report:        {compliance_file}")
    print(f"- Adherence report (TeX):   {adherence_tex_file}")
    print(f"- Adherence (Tables TeX):   {adherence_table_file}")

    for tc, s in sorted(result["compliance"].items()):
        print(
            f"  {tc}: order {s['order_ok_runs']}/{s['total_runs']}, "
            f"params {s['params_ok_runs']}/{s['total_runs']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
