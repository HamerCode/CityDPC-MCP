"""
Generates LaTeX tables from evaluation logs and combines them into a single document.
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev

# Add project root to sys.path to allow running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.cases import RAISE_BUILDING_SPEC, TEST_BUILDING_SPEC, TEST_CASES


# ============================================================================
# CONFIGURATION
# ============================================================================
LOG_DIR = Path("evaluation_logs")
TABLE_DIR = Path("evaluation_tables")

# Pricing (USD per 1M tokens) - gpt-5-mini
PRICE_INPUT_1M = 0.250
PRICE_CACHED_INPUT_1M = 0.025
PRICE_OUTPUT_1M = 2.000
CENTRAL_TENDENCY = "mean"  # "mean" | "median"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def _calculate_cost(input_tokens: int, output_tokens: int, cached_input_tokens: int | None = None) -> float:
    """Calculate cost in USD."""
    if input_tokens is None or output_tokens is None:
        return 0.0
    cached_tokens = cached_input_tokens or 0
    billable_input = max(input_tokens - cached_tokens, 0)
    return (
        (billable_input / 1_000_000 * PRICE_INPUT_1M)
        + (cached_tokens / 1_000_000 * PRICE_CACHED_INPUT_1M)
        + (output_tokens / 1_000_000 * PRICE_OUTPUT_1M)
    )


def _format_cost(value: float) -> str:
    """Format cost as USD string."""
    if value is None:
        return "--"
    return f"\\${value:.4f}"


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters."""
    # Backslash MUSS zuerst ersetzt werden, sonst werden die \ von \_ etc. auch ersetzt
    text = text.replace('\\', r'\textbackslash')
    # Dann alle anderen Sonderzeichen
    for char, escaped in [
        ('&', r'\&'), ('%', r'\%'), ('$', r'\$'), ('#', r'\#'),
        ('_', r'\_'), ('{', r'\{'), ('}', r'\}'),
        ('~', r'\textasciitilde'), ('^', r'\textasciicircum')
    ]:
        text = text.replace(char, escaped)
    return text


def _format_number(value, decimals=0) -> str:
    """Format number with thousands separator."""
    if value is None:
        return "--"
    if decimals == 0:
        return f"{int(value):,}".replace(",", r"\,")
    return f"{value:,.{decimals}f}".replace(",", r"\,")


def _format_percentage(value) -> str:
    """Format value as percentage."""
    if value is None:
        return "N/A"
    return f"{value*100:.0f}\\%"


def _checkmark(value) -> str:
    """Render boolean-style validation marker for LaTeX tables."""
    if value is True:
        return r"\checkmark"
    if value is False:
        return r"\textcolor{red}{\(\times\)}"
    return "--"


def _central(values, decimals=0) -> str:
    """Calculate and format central tendency (mean/median)."""
    numeric = [v for v in values if v is not None]
    if not numeric:
        return "--"
    val = median(numeric) if CENTRAL_TENDENCY == "median" else mean(numeric)
    return _format_number(val, decimals)


def _central_cost(values) -> str:
    """Format mean/median cost based on configured central tendency."""
    numeric = [v for v in values if v is not None]
    if not numeric:
        return "--"
    val = median(numeric) if CENTRAL_TENDENCY == "median" else mean(numeric)
    return _format_cost(val)


def _metric_label() -> str:
    return "Median" if CENTRAL_TENDENCY == "median" else "Average"


def _stddev(values, decimals=2) -> str:
    """Calculate and format standard deviation."""
    numeric = [v for v in values if v is not None]
    if len(numeric) < 2:
        return "--"
    return _format_number(stdev(numeric), decimals)


def _find_latest_date_folder() -> Path | None:
    """Find the latest date folder in evaluation_logs."""
    if not LOG_DIR.exists():
        return None
    date_folders = sorted([d for d in LOG_DIR.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
    return date_folders[0] if date_folders else None


def _extract_eval_info(date_folder: Path) -> dict:
    """Extract model, reasoning effort, date and time from folder name and log meta.

    Folder name format: DD.MM_HH-MM_model_reasoning
    Example: 06.02_20-41_gpt-5-mini_medium
    """
    info = {
        "model": None,
        "reasoning_effort": None,
        "date": None,
        "time": None,
    }

    # Parse folder name: DD.MM_HH-MM_model[_reasoning]
    folder_name = date_folder.name
    m = re.match(r"(\d{2})\.(\d{2})_(\d{2})-(\d{2})_(.+)", folder_name)
    if m:
        day, month, hour, minute = m.group(1), m.group(2), m.group(3), m.group(4)
        rest = m.group(5)  # e.g. "gpt-5-mini_medium" or "gpt-5-mini"
        info["date"] = f"{day}.{month}.{datetime.now().year}"
        info["time"] = f"{hour}:{minute}"

        # Parse model and reasoning from folder name rest part
        known_efforts = {"minimal", "low", "medium", "high"}
        parts = rest.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in known_efforts:
            info["model"] = parts[0]
            info["reasoning_effort"] = parts[1]
        else:
            info["model"] = rest

        # Override with log meta if available (more reliable)
        first_meta = _read_first_log_meta(date_folder)
        if first_meta:
            if first_meta.get("model"):
                info["model"] = first_meta["model"]
            if first_meta.get("reasoning_effort"):
                info["reasoning_effort"] = first_meta["reasoning_effort"]

    return info


def _read_first_log_meta(date_folder: Path) -> dict | None:
    """Read meta from first available log file in any setup subdirectory."""
    skip_dirs = {"datasets", "workspaces"}
    for setup_dir in sorted(date_folder.iterdir()):
        if not setup_dir.is_dir() or setup_dir.name in skip_dirs:
            continue
        for log_file in setup_dir.glob("*.json"):
            try:
                with log_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("meta", {})
            except Exception:
                continue
    return None


def _get_comparison_values(
    test_case_id: str,
    validation: dict,
    state_validation: dict | None = None,
    ground_truth: dict | None = None,
) -> tuple[str, str]:
    """Get expected and actual values as short strings."""
    extracted = validation.get("extracted_data", {}) if validation else {}
    state_validation = state_validation or {}
    ground_truth = ground_truth or {}
    gt_output = ground_truth.get("solutions", {}).get(test_case_id, {}).get("output", {})

    if test_case_id == "list_buildings":
        expected_total = gt_output.get("total")
        if expected_total is None and isinstance(gt_output.get("building_ids"), list):
            expected_total = len(gt_output.get("building_ids", []))
        actual_total = extracted.get("total")
        if actual_total is None and isinstance(extracted.get("building_ids"), list):
            actual_total = len(extracted.get("building_ids", []))
        expected_str = f"{expected_total} buildings" if expected_total is not None else "N/A"
        actual_str = f"{actual_total} buildings" if actual_total is not None else "N/A"
        return expected_str, actual_str

    elif test_case_id == "highest_measured_height":
        expected_id = gt_output.get("building_id", "N/A")
        expected_height = f"{gt_output.get('measuredHeight')}m" if gt_output.get("measuredHeight") is not None else "N/A"
        actual_id = extracted.get("building_id", "N/A")
        actual_height = f"{extracted.get('measuredHeight')}m" if extracted.get("measuredHeight") is not None else "N/A"
        return f"{expected_id}, {expected_height}", f"{actual_id}, {actual_height}"

    elif test_case_id == "roof_volume_sum":
        expected = f"{gt_output.get('roof_volume_sum_m3')} m³" if gt_output.get("roof_volume_sum_m3") is not None else "N/A"
        actual = f"{extracted.get('roof_volume_sum_m3')} m³" if extracted.get("roof_volume_sum_m3") is not None else "N/A"
        return expected, actual

    elif test_case_id == "raise_building_by_id":
        expected = state_validation.get("expected", {})
        expected_id = expected.get("building_id") or gt_output.get("building_id", "N/A")
        expected_height = (
            expected.get("new_height")
            if expected.get("new_height") is not None
            else gt_output.get("new_measuredHeight")
        )
        checks = state_validation.get("checks", {})
        actual_height = checks.get("output_height") or checks.get("actual_height")
        expected_height_str = f"{expected_height}m" if expected_height is not None else "N/A"
        actual_height_str = f"{actual_height}m" if actual_height is not None else "N/A"
        return f"{expected_id}, {expected_height_str}", f"{expected_id}, {actual_height_str}"

    elif test_case_id == "add_building":
        expected = state_validation.get("expected", {})
        checks = state_validation.get("checks", {})
        expected_id = expected.get("building_id") or gt_output.get("building_id", "N/A")
        actual_id = checks.get("actual_id") or checks.get("output_building_id") or expected_id
        actual_height = checks.get("actual_height", "N/A")
        return f"{expected_id}", f"{actual_id}, height {actual_height}"

    return "N/A", "N/A"


# ============================================================================
# TABLE BUILDERS
# ============================================================================
def build_metrics_table(log_data: dict, ground_truth: dict = {}) -> str:
    """Build metrics table with accuracy and expected/actual values."""
    meta = log_data.get("meta", {})
    test_case = log_data.get("test_case", {})
    setup_key = meta.get("setup_key", "setup")
    setup_label = meta.get("setup", setup_key)
    date_value = meta.get("log_date", "")
    test_case_id = test_case.get("id", "test")

    runs = log_data.get("runs", [])
    rows = []
    durations = []
    total_tokens = []
    input_tokens = []
    output_tokens = []
    tool_calls = []
    accuracies = []
    costs = []

    for run in runs:
        duration_ms = run.get("duration_ms")
        tokens = run.get("tokens") or {}
        total = tokens.get("total_tokens")
        input_tok = tokens.get("input_tokens")
        output_tok = tokens.get("output_tokens")
        cached_tok = tokens.get("cached_input_tokens")
        calls = run.get("tool_calls_count")
        error = run.get("error")
        retry_exhausted = run.get("retry_exhausted", False)

        # Validation / Accuracy
        validation = run.get("validation", {}) or {}
        state_validation = run.get("state_validation", {}) or {}
        accuracy = state_validation.get("accuracy")
        if accuracy is None:
            accuracy = validation.get("accuracy")
        expected_state = _get_expected_state_values(test_case_id, ground_truth)
        actual_state = _get_actual_state_values(test_case_id, run, expected_state)
        if accuracy is None and expected_state and actual_state:
            accuracy = _compute_state_accuracy(test_case_id, expected_state, actual_state)
        expected, actual = _get_comparison_values(test_case_id, validation, state_validation, ground_truth)
        is_correct = _is_run_correct(test_case_id, run, ground_truth)
        success_str = "True" if is_correct is True else ("False" if is_correct is False else "N/A")

        cost = _calculate_cost(input_tok, output_tok, cached_tok)
        durations.append(duration_ms)
        total_tokens.append(total)
        input_tokens.append(input_tok)
        output_tokens.append(output_tok)
        tool_calls.append(calls)
        accuracies.append(accuracy)
        costs.append(cost)

        rows.append([
            str(run.get("run", "")),
            _format_number(duration_ms),
            _format_number(total),
            _format_number(input_tok),
            _format_number(output_tok),
            _format_cost(cost),
            _format_number(calls),
            _escape_latex(actual),
            success_str,
        ])

    # Calculate averages
    numeric_accuracies = [a for a in accuracies if a is not None]
    avg_accuracy = mean(numeric_accuracies) if numeric_accuracies else None

    # Get expected value for average row
    first_run = runs[0] if runs else {}
    first_validation = first_run.get("validation", {}) if first_run else {}
    first_state_validation = first_run.get("state_validation", {}) if first_run else {}
    expected_avg, _ = _get_comparison_values(test_case_id, first_validation, first_state_validation, ground_truth)

    caption = _escape_latex(f"{setup_label} - {test_case_id} - Metrics ({date_value})")
    label = _escape_latex(f"tab:{setup_key}_{test_case_id}_metrics_{date_value}")

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\scriptsize",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{r r r r r r r p{3cm} l}",
        r"\hline",
        "Run & Dur. & Tok. Total & Tok. In & Tok. Out & Cost & Calls & Result & Success \\\\",
        r"\hline",
    ]

    # Ground truth row
    gt_expected, _ = _get_comparison_values(test_case_id, first_validation, first_state_validation, ground_truth)
    lines.append(
        r"\textbf{Ground Truth} & -- & -- & -- & -- & -- & -- & "
        + _escape_latex(gt_expected)
        + " & -- \\\\"
    )
    lines.append(r"\hline")

    for row in rows:
        lines.append(" & ".join(row) + r" \\")

    total_cost = sum(costs) if costs else 0.0
    total_tokens_sum = sum([v for v in total_tokens if v is not None]) if total_tokens else 0
    total_input_sum = sum([v for v in input_tokens if v is not None]) if input_tokens else 0
    total_output_sum = sum([v for v in output_tokens if v is not None]) if output_tokens else 0
    total_calls_sum = sum([v for v in tool_calls if v is not None]) if tool_calls else 0

    lines.extend([
        r"\hline",
        f"{_metric_label()}. & {_central(durations)} & {_central(total_tokens)} & "
        f"{_central(input_tokens)} & {_central(output_tokens)} & "
        f"{_central_cost(costs)} & "
        f"{_central(tool_calls)} & "
        f"{_escape_latex(expected_avg)} & -- \\\\",
        f"Total & -- & {_format_number(total_tokens_sum)} & "
        f"{_format_number(total_input_sum)} & {_format_number(total_output_sum)} & "
        f"{_format_cost(total_cost)} & {_format_number(total_calls_sum)} & "
        f"-- & -- \\\\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def _format_extracted_value(test_case_id: str, extracted: dict, state_validation: dict | None = None) -> str:
    """Format extracted values for display based on test case."""
    state_validation = state_validation or {}
    if not extracted and not state_validation:
        return "N/A"
    
    # Handle case where extracted is a string
    if isinstance(extracted, str):
        return extracted[:100]

    if test_case_id == "list_buildings":
        total = extracted.get("total", len(extracted.get("building_ids", [])))
        return f"Total: {total} buildings"
    elif test_case_id == "highest_measured_height":
        bid = extracted.get("building_id", "N/A")
        height = extracted.get("measuredHeight", "N/A")
        return f"ID: {bid}, Height: {height}m"
    elif test_case_id == "roof_volume_sum":
        vol = extracted.get("roof_volume_sum_m3", "N/A")
        return f"Sum: {vol} m³"
    elif test_case_id == "raise_building_by_id":
        expected = state_validation.get("expected", {})
        checks = state_validation.get("checks", {})
        bid = expected.get("building_id", "N/A")
        new_h = checks.get("output_height") or checks.get("actual_height") or "N/A"
        return f"ID: {bid}, New Height: {new_h}m"
    elif test_case_id == "add_building":
        expected = state_validation.get("expected", {})
        checks = state_validation.get("checks", {})
        bid = checks.get("actual_id") or checks.get("output_building_id") or expected.get("building_id", "N/A")
        height = checks.get("actual_height", "N/A")
        return f"ID: {bid}, Height: {height}m"
    else:
        return str(extracted)[:100]


def _load_snapshot(snapshot_path: str | None) -> dict | None:
    if not snapshot_path:
        return None
    try:
        with Path(snapshot_path).open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _get_expected_state_values(test_case_id: str, ground_truth: dict) -> dict:
    expected = {}
    output = ground_truth.get("solutions", {}).get(test_case_id, {}).get("output", {})
    if test_case_id == "raise_building_by_id":
        expected["building_id"] = output.get("building_id") or RAISE_BUILDING_SPEC.get("target_building_id")
        expected["new_height"] = output.get("new_measuredHeight") or RAISE_BUILDING_SPEC.get("new_height")
    elif test_case_id == "add_building":
        expected["building_id"] = output.get("building_id") or TEST_BUILDING_SPEC.get("id")
        city_obj = output.get("cityjson_object", {})
        expected["measuredHeight"] = (
            city_obj.get("attributes", {}).get("measuredHeight") or TEST_BUILDING_SPEC.get("measuredHeight")
        )
    return expected


def _get_actual_state_values(test_case_id: str, run: dict, expected: dict) -> dict:
    actual = {}
    state_validation = run.get("state_validation", {}) or {}
    checks = state_validation.get("checks", {}) or {}

    if test_case_id == "raise_building_by_id":
        actual_height = checks.get("actual_height") or checks.get("output_height")
        if actual_height is not None:
            actual["new_height"] = actual_height
            return actual

        snapshot = _load_snapshot(run.get("dataset_state", {}).get("snapshot_path"))
        if snapshot:
            bid = expected.get("building_id")
            attrs = snapshot.get("CityObjects", {}).get(bid, {}).get("attributes", {})
            actual["new_height"] = attrs.get("measuredHeight")
        return actual

    if test_case_id == "add_building":
        # 1. Direkt aus state_validation.checks (File-basiert)
        if checks.get("id_correct"):
            # Building wurde mit erwarteter ID gefunden → ID aus expected nehmen
            actual["building_id"] = expected.get("building_id")
            actual["measuredHeight"] = checks.get("actual_height")
            return actual
        if checks.get("actual_id"):
            # Building hat andere ID
            actual["building_id"] = checks.get("actual_id")
            actual["measuredHeight"] = checks.get("actual_height")
            return actual
        if checks.get("output_building_id"):
            actual["building_id"] = checks.get("output_building_id")
            return actual

        # 2. Fallback: Snapshot laden
        snapshot = _load_snapshot(run.get("dataset_state", {}).get("snapshot_path"))
        if snapshot:
            bid = expected.get("building_id")
            if bid and bid in snapshot.get("CityObjects", {}):
                actual["building_id"] = bid
                attrs = snapshot["CityObjects"][bid].get("attributes", {})
                actual["measuredHeight"] = attrs.get("measuredHeight")
                return actual
            # Fallback: finde neue EVAL-Test-ID
            for cand_id, cand_obj in snapshot.get("CityObjects", {}).items():
                if cand_id.startswith("EVAL_TEST_BUILDING_"):
                    actual["building_id"] = cand_id
                    actual["measuredHeight"] = cand_obj.get("attributes", {}).get("measuredHeight")
                    break
        return actual

    return actual


def _compute_state_accuracy(test_case_id: str, expected: dict, actual: dict) -> float | None:
    if test_case_id == "raise_building_by_id":
        exp_h = expected.get("new_height")
        act_h = actual.get("new_height")
        if exp_h is None or act_h is None:
            return None
        return 1.0 if abs(float(exp_h) - float(act_h)) < 0.001 else 0.0

    if test_case_id == "add_building":
        exp_id = expected.get("building_id")
        act_id = actual.get("building_id")
        if not exp_id or not act_id:
            return None
        if exp_id != act_id:
            return 0.0
        exp_h = expected.get("measuredHeight")
        act_h = actual.get("measuredHeight")
        if exp_h is None or act_h is None:
            return 1.0
        return 1.0 if abs(float(exp_h) - float(act_h)) < 0.001 else 0.0

    return None


def _is_run_correct(test_case_id: str, run: dict, ground_truth: dict) -> bool | None:
    """
    Bestimmt ob ein Run korrekt war.
    Für state-change Tests: Prüft direkt die File-basierten Checks aus state_validation.
    Für read-only Tests: Nutzt die Validierungs-Accuracy.
    """
    state_validation = run.get("state_validation", {}) or {}
    checks = state_validation.get("checks", {}) or {}

    # ── State-Change Tests: direkt File-Checks verwenden ──
    if test_case_id == "raise_building_by_id":
        # Wurde das Gebäude im File gefunden UND die Höhe korrekt geändert?
        if checks:
            return bool(checks.get("building_found") and checks.get("height_correct"))
        # Fallback: Snapshot prüfen
        expected_state = _get_expected_state_values(test_case_id, ground_truth)
        actual_state = _get_actual_state_values(test_case_id, run, expected_state)
        computed = _compute_state_accuracy(test_case_id, expected_state, actual_state)
        if computed is not None:
            return computed >= 1.0
        return False

    if test_case_id == "add_building":
        # Strikte Bewertung: nur gültig, wenn alle 4 Integritäts-Haken gesetzt sind
        # (CJ, Imp, Geo, Disp) und die Kernattribute stimmen.
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
        # Fallback: Snapshot prüfen
        expected_state = _get_expected_state_values(test_case_id, ground_truth)
        actual_state = _get_actual_state_values(test_case_id, run, expected_state)
        computed = _compute_state_accuracy(test_case_id, expected_state, actual_state)
        if computed is not None:
            return computed >= 1.0
        return False

    # ── Read-Only Tests: bevorzugt direkte Re-Validierung aus extracted_data ──
    if test_case_id == "highest_measured_height":
        validation = run.get("validation", {}) or {}
        extracted = validation.get("extracted_data", {}) or {}
        gt = (
            (ground_truth.get("solutions", {}) or {})
            .get("highest_measured_height", {})
            .get("output", {})
        )
        exp_id = gt.get("building_id")
        exp_h = gt.get("measuredHeight")
        act_id = extracted.get("building_id")
        act_h = extracted.get("measuredHeight")
        if exp_id is not None and exp_h is not None and act_id is not None and act_h is not None:
            try:
                return (act_id == exp_id) and (abs(float(act_h) - float(exp_h)) < 0.001)
            except Exception:
                return act_id == exp_id

    # Fallback: gespeicherte Validierungs-Accuracy
    validation = run.get("validation", {}) or {}
    accuracy = validation.get("accuracy")
    if accuracy is None:
        return None
    return accuracy >= 1.0


def build_ground_truth_and_outputs_table(setup_logs: list, test_case_id: str, ground_truth: dict, caption_info: str = "") -> str:
    """Build table showing ground truth vs extracted values per setup."""
    gt = ground_truth.get("solutions", {}).get(test_case_id, {}).get("output", {})
    expected_state = _get_expected_state_values(test_case_id, ground_truth)

    if test_case_id == "raise_building_by_id":
        gt_str = f"ID: {expected_state.get('building_id', 'N/A')}, New Height: {expected_state.get('new_height', 'N/A')}m"
    elif test_case_id == "add_building":
        gt_str = f"ID: {expected_state.get('building_id', 'N/A')}, Height: {expected_state.get('measuredHeight', 'N/A')}m"
    else:
        gt_str = _format_extracted_value(test_case_id, gt)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\caption{" + _escape_latex(f"{test_case_id} - Expected vs Actual" + (f" ({caption_info})" if caption_info else "")) + "}",
        r"\begin{tabular}{l l l}",
        r"\toprule",
        r"Setup & Extracted Value & Accuracy \\",
        r"\midrule",
        r"\textbf{Ground Truth} & " + _escape_latex(gt_str) + r" & -- \\",
        r"\midrule",
    ]

    # Extrahierte Werte pro Setup — Durchschnitt über alle Runs
    for setup_name, log_data in setup_logs:
        runs = log_data.get("runs", [])
        if not runs:
            continue
        
        # Berechne Success Rate über alle Runs (file-basiert)
        correct_count = sum(1 for run in runs if _is_run_correct(test_case_id, run, ground_truth) is True)
        total_count = len(runs)
        success_rate = correct_count / total_count if total_count > 0 else 0.0
        
        # Extrahierten Wert vom ersten Run anzeigen
        first_run = runs[0]
        validation = first_run.get("validation", {}) or {}
        state_validation = first_run.get("state_validation", {}) or {}
        extracted = validation.get("extracted_data", {})
        
        actual_state = _get_actual_state_values(test_case_id, first_run, expected_state)
        if test_case_id == "raise_building_by_id" and actual_state.get("new_height") is not None:
            extracted_str = f"ID: {expected_state.get('building_id', 'N/A')}, New Height: {actual_state.get('new_height')}m"
        elif test_case_id == "add_building" and actual_state.get("building_id"):
            height = actual_state.get("measuredHeight", "N/A")
            extracted_str = f"ID: {actual_state.get('building_id')}, Height: {height}m"
        else:
            extracted_str = _format_extracted_value(test_case_id, extracted, state_validation)
        
        acc_str = _format_percentage(success_rate)

        lines.append(
            _escape_latex(setup_name) + " & " +
            _escape_latex(extracted_str) + " & " +
            acc_str + r" \\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def build_combined_metrics_table(setup_logs: list, test_case_id: str, ground_truth: dict, caption_info: str = "") -> str:
    """Build combined metrics table showing averages per setup (one row per setup)."""
    meta = setup_logs[0][1].get("meta", {})
    date_value = meta.get("updated_at", "")[:10].replace("-", "")
    expected_state = _get_expected_state_values(test_case_id, ground_truth)

    rows = []

    for setup_name, log_data in setup_logs:
        runs = log_data.get("runs", [])

        # Collect metrics for this setup
        durations = []
        total_tokens = []
        input_tokens = []
        output_tokens = []
        tool_calls = []
        accuracies = []

        for run in runs:
            durations.append(run.get("duration_ms"))
            tokens = run.get("tokens") or {}
            total_tokens.append(tokens.get("total_tokens"))
            input_tokens.append(tokens.get("input_tokens"))
            output_tokens.append(tokens.get("output_tokens"))
            tool_calls.append(run.get("tool_calls_count"))
            
            # File-based correctness check
            is_correct = _is_run_correct(test_case_id, run, ground_truth)
            accuracies.append(1.0 if is_correct else 0.0)

        # Calculate averages for this setup
        numeric_accuracies = [a for a in accuracies if a is not None]
        avg_accuracy = mean(numeric_accuracies) if numeric_accuracies else None

        rows.append([
            _escape_latex(setup_name),
            _central(durations),
            _central(total_tokens),
            _central(input_tokens),
            _central(output_tokens),
            _central(tool_calls),
            _format_percentage(avg_accuracy),
        ])

    if not rows:
        return "% No runs found"

    caption_suffix = f" ({caption_info})" if caption_info else f" ({date_value})"
    caption_base = f"{test_case_id} - {_metric_label()} Metrics per Setup{caption_suffix}"
    caption_extra = ""
    if test_case_id == "add_building":
        caption_extra = (
            f" **: Strenge Bewertungsregel fuer add_building: Ein Run zaehlt nur dann als erfolgreich, "
            f"wenn alle vier Integritaets-Checks (CJ=CityJSON valide, Imp=mit citydpc importierbar, "
            f"Geo=Geometrie konsistent, Disp=Gebaeude darstellbar) gleichzeitig erfuellt sind. "
            f"Beim Code Interpreter wurden in 2 Runs zwar Objekte erzeugt und im Datensatz sichtbar, "
            f"diese erfuellten jedoch nicht die vollstaendigen Integritaetskriterien."
        )
    caption = _escape_latex(caption_base + caption_extra)
    label = _escape_latex(f"tab:combined_{test_case_id}_metrics_{date_value}")

    is_state_change = test_case_id in ["raise_building_by_id", "add_building"]
    is_add_building = test_case_id == "add_building"

    def _is_usable_state_change_run(run: dict) -> bool:
        """Strict usability check for state-changing runs."""
        sv_checks = (run.get("state_validation") or {}).get("checks", {}) or {}
        if test_case_id == "raise_building_by_id":
            return bool(
                sv_checks.get("building_found")
                and sv_checks.get("height_correct")
                and sv_checks.get("cityjson_valid")
            )
        if test_case_id == "add_building":
            return bool(
                sv_checks.get("building_found")
                and sv_checks.get("id_correct")
                and sv_checks.get("height_correct")
                and sv_checks.get("cityjson_valid")
                and sv_checks.get("geometry_valid")
            )
        return False

    if is_state_change:
        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\scriptsize",
            r"\caption{" + caption + "}",
            r"\label{" + label + "}",
            r"\begin{tabular}{l r r r r r r r r l}",
            r"\toprule",
            r"Setup & Runs & Avg Dur. (ms) & Average Input Tokens & Average Output Tokens & Total Tok. & Mod-Dataset OK & Loadable (Imp) & Usable Mods & Assessment \\",
            r"\midrule",
        ]
    else:
        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\scriptsize",
            r"\caption{" + caption + "}",
            r"\label{" + label + "}",
            r"\begin{tabular}{l r r r r r r}",
            r"\toprule",
            r"Setup & Runs & Avg Dur. (ms) & Average Input Tokens & Average Output Tokens & Total Tok. & Success Rate \\",
            r"\midrule",
        ]

    for setup_name, log_data in setup_logs:
        runs = log_data.get("runs", [])
        durations = []
        costs = []
        total_tokens = []
        input_tokens = []
        output_tokens = []
        accuracies = []
        tool_calls = []
        # Integrity counters
        cj_ok_count = 0
        imp_ok_count = 0
        geo_ok_count = 0
        disp_ok_count = 0
        integrity_runs = 0
        usable_ok_count = 0

        for run in runs:
            durations.append(run.get("duration_ms"))
            tokens = run.get("tokens") or {}
            in_t = tokens.get("input_tokens")
            out_t = tokens.get("output_tokens")
            input_tokens.append(in_t)
            output_tokens.append(out_t)
            cached_t = tokens.get("cached_input_tokens")
            costs.append(_calculate_cost(in_t, out_t, cached_t))
            total_tokens.append(tokens.get("total_tokens"))
            tool_calls.append(run.get("tool_calls_count"))
            
            # File-based correctness check
            is_correct = _is_run_correct(test_case_id, run, ground_truth)
            accuracies.append(1.0 if is_correct else 0.0)
            
            # Integrity stats
            if is_state_change:
                sv_checks = (run.get("state_validation") or {}).get("checks", {}) or {}
                if sv_checks.get("cityjson_valid") is not None or sv_checks.get("citydpc_importable") is not None:
                    integrity_runs += 1
                    if sv_checks.get("cityjson_valid"):
                        cj_ok_count += 1
                    if sv_checks.get("citydpc_importable"):
                        imp_ok_count += 1
                    if sv_checks.get("geometry_valid"):
                        geo_ok_count += 1
                    if sv_checks.get("building_displayable"):
                        disp_ok_count += 1
                if _is_usable_state_change_run(run):
                    usable_ok_count += 1

        avg_dur = _central(durations)
        avg_in_tok = _central(input_tokens)
        avg_out_tok = _central(output_tokens)
        total_tok = sum([v for v in total_tokens if v is not None]) if total_tokens else 0
        avg_cost = _central_cost(costs)
        
        numeric_accuracies = [a for a in accuracies if a is not None]
        avg_acc = _format_percentage(mean(numeric_accuracies) if numeric_accuracies else None)

        if is_state_change:
            if integrity_runs > 0:
                integrity_str = (
                    f"CJ:{cj_ok_count}/{integrity_runs} "
                    f"Imp:{imp_ok_count}/{integrity_runs} "
                    f"Geo:{geo_ok_count}/{integrity_runs} "
                    f"Disp:{disp_ok_count}/{integrity_runs}"
                )
                loadable_str = f"{imp_ok_count}/{integrity_runs}"
                usable_str = f"{usable_ok_count}/{len(runs)}"
                if usable_ok_count == len(runs):
                    assessment = "Alle Aenderungen nutzbar"
                elif usable_ok_count == 0:
                    assessment = "Aenderungen nicht nutzbar"
                else:
                    assessment = "Teilweise nutzbar"
                if imp_ok_count == 0 and usable_ok_count > 0:
                    assessment += " (Import-Check fehlgeschlagen)"
                if is_add_building and usable_ok_count < len(runs):
                    assessment += " (oft nicht ladbar/gueltig)"
            else:
                integrity_str = "--"
                loadable_str = "--"
                usable_str = "--"
                assessment = "Keine Integritaetsdaten"
            acc_display = avg_acc
            if test_case_id == "add_building" and setup_name in {"code_interpreter", "no_tools"}:
                acc_display = f"{avg_acc}**"
            lines.append(
                f"{_escape_latex(setup_name)} & {len(runs)} & {avg_dur} & {avg_in_tok} & {avg_out_tok} & "
                f"{_format_number(total_tok)} & {acc_display} & "
                f"{_escape_latex(loadable_str)} & {_escape_latex(usable_str)} & {_escape_latex(assessment)} \\\\"
            )
        else:
            lines.append(
                f"{_escape_latex(setup_name)} & {len(runs)} & {avg_dur} & {avg_in_tok} & {avg_out_tok} & "
                f"{_format_number(total_tok)} & {avg_acc} \\\\"
            )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def build_overall_average_table(combined_logs: dict, test_case_order: list[str], ground_truth: dict, caption_info: str = "") -> str:
    """
    Build one overview table:
    one row per (setup, test_case) with average metrics over all runs.
    """
    # Gather all setup names first, then iterate setup-first for desired ordering
    setup_names = sorted(
        {setup_name for tc in test_case_order for setup_name, _ in (combined_logs.get(tc, []) or [])}
    )
    rows = []
    for setup_name in setup_names:
        for test_case_id in test_case_order:
            setup_logs = combined_logs.get(test_case_id, [])
            log_data = None
            for s_name, s_log in setup_logs:
                if s_name == setup_name:
                    log_data = s_log
                    break
            if log_data is None:
                continue
            runs = log_data.get("runs", []) or []
            if not runs:
                continue

            durations = []
            total_tokens = []
            input_tokens = []
            output_tokens = []
            tool_calls = []
            correctness = []

            for run in runs:
                durations.append(run.get("duration_ms"))
                tok = run.get("tokens") or {}
                total_tokens.append(tok.get("total_tokens"))
                input_tokens.append(tok.get("input_tokens"))
                output_tokens.append(tok.get("output_tokens"))
                tool_calls.append(run.get("tool_calls_count"))
                is_correct = _is_run_correct(test_case_id, run, ground_truth)
                if is_correct is not None:
                    correctness.append(1.0 if is_correct else 0.0)

            success = mean(correctness) if correctness else None

            rows.append(
                [
                    _escape_latex(setup_name),
                    _escape_latex(test_case_id),
                    str(len(runs)),
                    _central(durations),
                    _central(input_tokens),
                    _central(output_tokens),
                    _central(tool_calls),
                    (
                        _format_percentage(success) + "**"
                        if test_case_id == "add_building" and setup_name in {"code_interpreter", "no_tools"}
                        else _format_percentage(success)
                    ),
                ]
            )

    if not rows:
        return "% No runs found"

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\caption{" + _escape_latex(
            f"Overall {_metric_label()} Metrics per Setup and Test Case"
            + (f" ({caption_info})" if caption_info else "")
            + " **: Fuer add_building gilt eine strenge Erfolgsdefinition: nur Runs mit gleichzeitig erfuellten Checks CJ, Imp, Geo und Disp werden als korrekt gewertet."
        ) + r"}",
        r"\begin{tabular}{l l r r r r r r}",
        r"\toprule",
        r"Setup & Test Case & Runs & Avg Dur. (ms) & Average Input Tokens & Average Output Tokens & Avg Calls & Success Rate \\",
        r"\midrule",
    ]
    current_setup = None
    for row in rows:
        setup_cell = row[0]
        if current_setup is not None and setup_cell != current_setup:
            lines.append(r"\midrule")
        lines.append(" & ".join(row) + r" \\")
        current_setup = setup_cell
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def build_outputs_table(log_data: dict, max_len: int = 500) -> str:
    """Build outputs table showing AI responses."""
    meta = log_data.get("meta", {})
    test_case = log_data.get("test_case", {})
    setup_key = meta.get("setup_key", "setup")
    setup_label = meta.get("setup", setup_key)
    date_value = meta.get("log_date", "")
    test_case_id = test_case.get("id", "test")

    runs = log_data.get("runs", [])
    rows = []

    for run in runs:
        output_text = run.get("output_text", "")
        if len(output_text) > max_len:
            output_text = output_text[:max_len] + "..."

        rows.append([
            str(run.get("run", "")),
            _escape_latex(output_text)
        ])

    caption = _escape_latex(f"{setup_label} - {test_case_id} - Outputs ({date_value})")
    label = _escape_latex(f"tab:{setup_key}_{test_case_id}_outputs_{date_value}")

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\scriptsize",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{r p{12cm}}",
        r"\hline",
        "Run & Output \\\\",
        r"\hline",
    ]

    for row in rows:
        lines.append(" & ".join(row) + r" \\")

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# ============================================================================
# DOCUMENT BUILDER
# ============================================================================
def load_ground_truth() -> dict:
    """Load ground truth from JSON file."""
    gt_path = Path("ground_truth.json")
    if not gt_path.exists():
        return {}
    with gt_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _load_revalidation_overrides(date_folder: Path) -> dict[tuple[str, str, int], dict]:
    """
    Load revalidation results and index by (setup, test_case_id, run_index).
    """
    report_path = date_folder / "revalidation_report.json"
    if not report_path.exists():
        return {}

    try:
        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)
    except Exception:
        return {}

    overrides: dict[tuple[str, str, int], dict] = {}
    for item in report.get("files", []) or []:
        setup = item.get("setup")
        test_case_id = item.get("test_case_id")
        for run in item.get("runs", []) or []:
            run_index = run.get("run_index")
            if setup and test_case_id and isinstance(run_index, int):
                overrides[(setup, test_case_id, run_index)] = run
    return overrides


def _apply_revalidation_to_log(
    log_data: dict,
    setup_name: str,
    test_case_id: str,
    overrides: dict[tuple[str, str, int], dict]
) -> dict:
    """
    Merge revalidated validation/state_validation into log_data runs.
    """
    if not overrides:
        return log_data

    runs = log_data.get("runs", []) or []
    for idx, run in enumerate(runs, start=1):
        run_index = run.get("run_index")
        if not isinstance(run_index, int):
            run_index = idx
        ov = overrides.get((setup_name, test_case_id, run_index))
        if not ov:
            continue
        if ov.get("new_validation") is not None:
            run["validation"] = ov.get("new_validation") or {}
        if ov.get("new_state_validation") is not None:
            run["state_validation"] = ov.get("new_state_validation") or {}
    return log_data


def build_combined_document(date_folder: Path) -> str:
    """Build complete LaTeX document with all tables."""
    date_str = date_folder.name
    ground_truth = load_ground_truth()

    # Extract evaluation info (model, reasoning, date/time)
    eval_info = _extract_eval_info(date_folder)
    model_name = eval_info["model"] or "Unknown"
    reasoning = eval_info["reasoning_effort"] or "--"
    eval_date = eval_info["date"] or date_str
    eval_time = eval_info["time"] or ""
    revalidation_overrides = _load_revalidation_overrides(date_folder)

    # Preamble
    lines = [
        r"\documentclass[a4paper,11pt]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[ngerman,english]{babel}",
        r"\usepackage{geometry}",
        r"\geometry{a4paper, margin=2cm}",
        r"\usepackage{booktabs}",
        r"\usepackage{amssymb}",
        r"\usepackage{tabularx}",
        r"\usepackage{longtable}",
        r"\usepackage{float}",
        r"\usepackage{hyperref}",
        r"\usepackage{xcolor}",
        r"\usepackage{graphicx}",
        r"",
        r"\hypersetup{",
        r"    colorlinks=true,",
        r"    linkcolor=blue,",
        r"    urlcolor=blue,",
        r"    citecolor=blue",
        r"}",
        r"",
        r"\title{Evaluation Results}",
        f"\\date{{{_escape_latex(eval_date)}, {_escape_latex(eval_time)} Uhr}}",
        r"",
        r"\begin{document}",
        r"\maketitle",
        r"",
        r"\begin{table}[H]",
        r"\centering",
        r"\begin{tabular}{l l}",
        r"\toprule",
        r"\textbf{Parameter} & \textbf{Value} \\",
        r"\midrule",
        f"Model & {_escape_latex(model_name)} \\\\",
        f"Reasoning Effort & {_escape_latex(reasoning)} \\\\",
        f"Date & {_escape_latex(eval_date)}, {_escape_latex(eval_time)} Uhr \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"",
        r"\tableofcontents",
        r"\clearpage",
        r"",
    ]

    # Get test case order from evals/cases.py
    test_case_order = [tc["id"] for tc in TEST_CASES]

    # Collect all setup directories
    skip_dirs = {"datasets", "workspaces"}
    setup_dirs = [d for d in sorted(date_folder.iterdir()) if d.is_dir() and d.name not in skip_dirs]

    # Check if we have multiple setups - if so, create combined average tables
    if len(setup_dirs) > 1:
        lines.append("\\section{Combined Results - Multiple Setups}")
        lines.append("")
        lines.append("When testing multiple setups simultaneously, only average results across all setups are shown.")
        lines.append("")

        # Organize logs by test case and setup
        combined_logs = {}
        setup_names = []

        for setup_dir in setup_dirs:
            setup_name = setup_dir.name
            setup_names.append(setup_name)

            for log_file in setup_dir.glob("*.json"):
                try:
                    with log_file.open("r", encoding="utf-8") as f:
                        log_data = json.load(f)
                    test_case_id = log_data.get("meta", {}).get("test_case_id")
                    if test_case_id:
                        log_data = _apply_revalidation_to_log(
                            log_data=log_data,
                            setup_name=setup_name,
                            test_case_id=test_case_id,
                            overrides=revalidation_overrides,
                        )
                        if test_case_id not in combined_logs:
                            combined_logs[test_case_id] = []
                        combined_logs[test_case_id].append((setup_name, log_data))
                except:
                    continue

        # Generate combined tables for each test case
        caption_info = f"{model_name}, {reasoning}, {eval_date} {eval_time}"
        lines.append(build_overall_average_table(combined_logs, test_case_order, ground_truth, caption_info))
        lines.append("")
        lines.append(r"\clearpage")
        lines.append("")

        for test_case_id in test_case_order:
            if test_case_id not in combined_logs:
                continue

            setup_logs = combined_logs[test_case_id]
            lines.append(f"\\subsection{{{_escape_latex(test_case_id)}}}")
            lines.append("")

            # Combined metrics table with averages
            lines.append(build_combined_metrics_table(setup_logs, test_case_id, ground_truth, caption_info))
            lines.append("")

            # NEW: All Runs Detail Table
            lines.append(build_all_runs_table(combined_logs, test_case_id, ground_truth, caption_info))
            lines.append("")

            # Ground Truth and Outputs table
            lines.append(build_ground_truth_and_outputs_table(setup_logs, test_case_id, ground_truth, caption_info))
            lines.append("")
            lines.append(r"\clearpage")
            lines.append("")

    else:
        # Single setup - use original logic
        for setup_dir in setup_dirs:
            setup_name = setup_dir.name
            lines.append(f"\\section{{{_escape_latex(setup_name)}}}")
            lines.append("")

            # Organize logs by test case
            test_case_logs = {}
            for log_file in setup_dir.glob("*.json"):
                try:
                    with log_file.open("r", encoding="utf-8") as f:
                        log_data = json.load(f)
                    test_case_id = log_data.get("meta", {}).get("test_case_id")
                    if test_case_id:
                        log_data = _apply_revalidation_to_log(
                            log_data=log_data,
                            setup_name=setup_name,
                            test_case_id=test_case_id,
                            overrides=revalidation_overrides,
                        )
                        test_case_logs[test_case_id] = log_data
                except:
                    continue

            # Generate tables in test case order
            for test_case_id in test_case_order:
                if test_case_id not in test_case_logs:
                    continue

                log_data = test_case_logs[test_case_id]
                lines.append(f"\\subsection{{{_escape_latex(test_case_id)}}}")
                lines.append("")

                # Metrics table
                lines.append(build_metrics_table(log_data, ground_truth))
                lines.append("")
                lines.append(r"\clearpage")
                lines.append("")

                # Outputs table
                lines.append(build_outputs_table(log_data))
                lines.append("")
                lines.append(r"\clearpage")
            lines.append("")

    lines.append(r"\end{document}")

    return "\n".join(lines)


def build_all_runs_table(combined_logs: dict, test_case_id: str, ground_truth: dict, caption_info: str = "") -> str:
    """Build a giant table with ALL runs for a specific test case."""
    setup_logs = combined_logs.get(test_case_id, [])
    if not setup_logs:
        return ""
    
    # Ground Truth Row - extract from output
    gt_solution = ground_truth.get("solutions", {}).get(test_case_id, {})
    gt_output = gt_solution.get("output", {})
    
    if test_case_id == "list_buildings":
        total = gt_output.get("total", len(gt_output.get("building_ids", [])))
        gt_expected = f"Total: {total}"
    elif test_case_id == "highest_measured_height":
        bid = gt_output.get("building_id", "N/A")
        height = gt_output.get("measuredHeight", "N/A")
        gt_expected = f"{bid}, {height}m"
    elif test_case_id == "roof_volume_sum":
        vol = gt_output.get("roof_volume_sum_m3", "N/A")
        gt_expected = f"{vol} m³"
    elif test_case_id == "raise_building_by_id":
        bid = gt_output.get("building_id", "N/A")
        new_h = gt_output.get("new_measuredHeight", "N/A")
        gt_expected = f"{bid}, {new_h}m"
    elif test_case_id == "add_building":
        bid = gt_output.get("building_id", "N/A")
        city_obj = gt_output.get("cityjson_object", {})
        mh = city_obj.get("attributes", {}).get("measuredHeight", "N/A")
        gt_expected = f"{bid}, {mh}m"
    else:
        gt_expected = str(gt_output)[:50] if gt_output else "N/A"
    
    is_state_change = test_case_id in ["raise_building_by_id", "add_building"]
    
    if is_state_change:
        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\scriptsize",
            r"\caption{" + _escape_latex(f"{test_case_id} - Detailed Runs Comparison" + (f" ({caption_info})" if caption_info else "")) + r"}",
            r"\resizebox{\textwidth}{!}{",
            r"\begin{tabular}{l r r r r r r r l c}",
            r"\toprule",
            r"Setup & Run & Dur. & Tok. & In & Out & Cost & Calls & Result & Mod-Dataset OK \\",
            r"\midrule",
            # Ground Truth row
            r"\textbf{Ground Truth} & -- & -- & -- & -- & -- & -- & -- & " + _escape_latex(str(gt_expected)) + r" & -- \\",
            r"\midrule",
        ]
    else:
        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\scriptsize",
            r"\caption{" + _escape_latex(f"{test_case_id} - Detailed Runs Comparison" + (f" ({caption_info})" if caption_info else "")) + r"}",
            r"\resizebox{\textwidth}{!}{",
            r"\begin{tabular}{l r r r r r r r l r}",
            r"\toprule",
            r"Setup & Run & Dur. & Tok. & In & Out & Cost & Calls & Result & OK \\",
            r"\midrule",
            # Ground Truth row
            r"\textbf{Ground Truth} & -- & -- & -- & -- & -- & -- & -- & " + _escape_latex(str(gt_expected)) + r" & -- \\",
            r"\midrule",
        ]
    
    for setup_name, log_data in setup_logs:
        runs = log_data.get("runs", [])
        for run in runs:
            dur = run.get("duration_ms")
            tokens = run.get("tokens") or {}
            tok_total = tokens.get("total_tokens", 0)
            tok_in = tokens.get("input_tokens", 0)
            tok_out = tokens.get("output_tokens", 0)
            cost = _calculate_cost(tok_in, tok_out)
            calls = run.get("tool_calls_count")
            
            # Extracted Value
            validation = run.get("validation", {}) or {}
            state_validation = run.get("state_validation", {}) or {}
            
            if test_case_id == "list_buildings":
                output_text = run.get("output_text", "")
                try:
                    parsed = json.loads(output_text)
                    total = parsed.get("total", len(parsed.get("building_ids", [])))
                    result_str = f"Total: {total}"
                except:
                    result_str = "Error"
            elif test_case_id == "highest_measured_height":
                output_text = run.get("output_text", "")
                try:
                    parsed = json.loads(output_text)
                    bid = parsed.get("building_id", "N/A")
                    height = parsed.get("measuredHeight", "N/A")
                    # Shorten ID for display
                    bid_short = bid[-4:] if len(bid) > 4 else bid
                    result_str = f"...{bid_short}, {height}m"
                except:
                    result_str = "Error"
            elif test_case_id == "roof_volume_sum":
                output_text = run.get("output_text", "")
                try:
                    parsed = json.loads(output_text)
                    vol = parsed.get("roof_volume_sum_m3", "N/A")
                    result_str = f"{vol} m³"
                except:
                    result_str = "Error"
            elif test_case_id in ["raise_building_by_id", "add_building"]:
                expected = state_validation.get("expected", {})
                checks = state_validation.get("checks", {})
                if test_case_id == "raise_building_by_id":
                    bid = expected.get("building_id", "N/A")
                    new_h = checks.get("output_height") or checks.get("actual_height") or "N/A"
                    # Shorten ID for display
                    bid_short = bid[-4:] if len(str(bid)) > 4 else bid
                    result_str = f"...{bid_short}, {new_h}m"
                else:  # add_building
                    bid = checks.get("actual_id") or checks.get("output_building_id") or expected.get("building_id", "N/A")
                    height = checks.get("actual_height", "N/A")
                    # Shorten ID for display
                    bid_short = bid[-4:] if len(str(bid)) > 4 else bid
                    result_str = f"...{bid_short}, {height}m"
            else:
                result_str = "N/A"
            
            # Success (True/False) — File-basierte Prüfung
            if run.get("error") or run.get("retry_exhausted"):
                success_str = "Error"
            else:
                is_correct = _is_run_correct(test_case_id, run, ground_truth)
                success_str = "True" if is_correct is True else ("False" if is_correct is False else "N/A")
            
            # Integrity checks (nur für state-change Tests)
            if is_state_change:
                mod_dataset_ok = _is_run_correct(test_case_id, run, ground_truth)
                mod_dataset_mark = _checkmark(mod_dataset_ok)
                
                lines.append(
                    f"{_escape_latex(setup_name)} & {run.get('run_index')} & {_format_number(dur)} & "
                    f"{_format_number(tok_total)} & {_format_number(tok_in)} & {_format_number(tok_out)} & "
                    f"{_format_cost(cost)} & {calls} & {_escape_latex(result_str)} & "
                    f"{mod_dataset_mark} \\\\"
                )
            else:
                lines.append(
                    f"{_escape_latex(setup_name)} & {run.get('run_index')} & {_format_number(dur)} & "
                    f"{_format_number(tok_total)} & {_format_number(tok_in)} & {_format_number(tok_out)} & "
                    f"{_format_cost(cost)} & {calls} & {_escape_latex(result_str)} & {success_str} \\\\"
                )
        lines.append(r"\midrule") # Separator between setups
        
    lines.pop() # Remove last midrule
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_extracted_values_table(combined_logs: dict, test_case_id: str, ground_truth: dict) -> str:
    """Build a table with extracted values and success for all runs."""
    setup_logs = combined_logs.get(test_case_id, [])
    if not setup_logs:
        return ""
    
    # Ground Truth Row - extract from output
    gt_solution = ground_truth.get("solutions", {}).get(test_case_id, {})
    gt_output = gt_solution.get("output", {})
    
    if test_case_id == "list_buildings":
        total = gt_output.get("total", len(gt_output.get("building_ids", [])))
        gt_expected = f"Total: {total} buildings"
    elif test_case_id == "highest_measured_height":
        bid = gt_output.get("building_id", "N/A")
        height = gt_output.get("measuredHeight", "N/A")
        gt_expected = f"ID: {bid}, Height: {height}m"
    elif test_case_id == "roof_volume_sum":
        vol = gt_output.get("roof_volume_sum_m3", "N/A")
        gt_expected = f"Sum: {vol} m³"
    elif test_case_id == "raise_building_by_id":
        bid = gt_output.get("building_id", "N/A")
        new_h = gt_output.get("new_measuredHeight", "N/A")
        gt_expected = f"ID: {bid}, New Height: {new_h}m"
    elif test_case_id == "add_building":
        bid = gt_output.get("building_id", "N/A")
        city_obj = gt_output.get("cityjson_object", {})
        mh = city_obj.get("attributes", {}).get("measuredHeight", "N/A")
        gt_expected = f"ID: {bid}, Height: {mh}m"
    else:
        gt_expected = str(gt_output)[:100] if gt_output else "N/A"
    
    lines = [
        r"\begin{longtable}{l r l r}",
        r"\caption{" + _escape_latex(f"{test_case_id} - Extracted Values Comparison") + r"} \\",
        r"\toprule",
        r"Setup & Run & Extracted Value & OK \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Setup & Run & Extracted Value & OK \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{4}{r}{Continued...} \\",
        r"\bottomrule",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
        # Ground Truth row
        r"\textbf{Ground Truth} & -- & " + _escape_latex(str(gt_expected)) + r" & -- \\",
        r"\midrule",
    ]
    
    for setup_name, log_data in setup_logs:
        runs = log_data.get("runs", [])
        for run in runs:
            # Extracted Value - same logic as before
            validation = run.get("validation", {}) or {}
            state_validation = run.get("state_validation", {}) or {}
            
            if test_case_id == "list_buildings":
                output_text = run.get("output_text", "")
                try:
                    parsed = json.loads(output_text)
                    total = parsed.get("total", len(parsed.get("building_ids", [])))
                    result_str = f"Total: {total} buildings"
                except:
                    result_str = "Parse Error"
            elif test_case_id == "highest_measured_height":
                output_text = run.get("output_text", "")
                try:
                    parsed = json.loads(output_text)
                    bid = parsed.get("building_id", "N/A")
                    height = parsed.get("measuredHeight", "N/A")
                    result_str = f"ID: {bid}, Height: {height}m"
                except:
                    result_str = "Parse Error"
            elif test_case_id == "roof_volume_sum":
                output_text = run.get("output_text", "")
                try:
                    parsed = json.loads(output_text)
                    vol = parsed.get("roof_volume_sum_m3", "N/A")
                    result_str = f"Sum: {vol} m³"
                except:
                    result_str = "Parse Error"
            elif test_case_id in ["raise_building_by_id", "add_building"]:
                expected = state_validation.get("expected", {})
                checks = state_validation.get("checks", {})
                if test_case_id == "raise_building_by_id":
                    bid = expected.get("building_id", "N/A")
                    new_h = checks.get("output_height") or checks.get("actual_height") or "N/A"
                    result_str = f"ID: {bid}, New Height: {new_h}m"
                else:  # add_building
                    bid = checks.get("actual_id") or checks.get("output_building_id") or expected.get("building_id", "N/A")
                    height = checks.get("actual_height", "N/A")
                    result_str = f"ID: {bid}, Height: {height}m"
            else:
                result_str = "N/A"
            
            # Success (True/False) — File-basierte Prüfung
            if run.get("error") or run.get("retry_exhausted"):
                success_str = "Error"
            else:
                is_correct = _is_run_correct(test_case_id, run, ground_truth)
                success_str = "True" if is_correct is True else ("False" if is_correct is False else "N/A")
            
            lines.append(
                f"{_escape_latex(setup_name)} & {run.get('run_index')} & {_escape_latex(result_str)} & {success_str} \\\\"
            )
        lines.append(r"\midrule")
    
    lines.pop() # Remove last midrule
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================
def main() -> int:
    global CENTRAL_TENDENCY
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from evaluation logs")
    parser.add_argument("--date", help="Date folder (YYYYMMDD_HHMMSS), defaults to latest")
    parser.add_argument("--output", help="Output file name", default=None)
    parser.add_argument("--central", choices=["mean", "median"], default="mean", help="Central tendency for aggregate rows/metrics")

    args = parser.parse_args()
    CENTRAL_TENDENCY = args.central

    # Find date folder
    if args.date:
        date_folder = LOG_DIR / args.date
        if not date_folder.exists():
            print(f"Error: Date folder not found: {date_folder}")
            return 1
    else:
        date_folder = _find_latest_date_folder()
        if not date_folder:
            print("Error: No evaluation logs found")
            return 1

    print(f"Using logs from: {date_folder}")
    print(f"Central tendency: {CENTRAL_TENDENCY}")

    # Build combined document
    document = build_combined_document(date_folder)

    # Write output
    TABLE_DIR.mkdir(exist_ok=True)
    output_name = args.output or f"evaluation_tables_{date_folder.name}.tex"
    output_path = TABLE_DIR / output_name

    output_path.write_text(document, encoding="utf-8")
    print(f"Generated: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
