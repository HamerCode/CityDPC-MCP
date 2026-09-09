"""
Generates concise tool call logs from evaluation results.
For each test case and run, extracts only tool call names + arguments (no output),
plus any errors from the run.
"""

import argparse
import json
from pathlib import Path

LOG_DIR = Path("evaluation_logs")
TOOL_CALLS_DIR = Path("evaluation_tool_calls")


def _find_latest_date_folder() -> Path | None:
    if not LOG_DIR.exists():
        return None
    date_folders = sorted([d for d in LOG_DIR.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
    return date_folders[0] if date_folders else None


def extract_tool_calls_for_run(run: dict) -> dict:
    """Extract concise tool call info from a single run."""
    tool_calls = run.get("tool_calls", [])
    concise_calls = []

    for tc in tool_calls:
        call_type = tc.get("type", "unknown")
        entry = {
            "name": tc.get("name") or tc.get("function", {}).get("name", "unknown"),
            "arguments": tc.get("arguments") or tc.get("function", {}).get("arguments", ""),
        }
        # Include error if present
        if tc.get("error"):
            entry["error"] = tc["error"]
        # Include status if not completed
        status = tc.get("status")
        if status and status != "completed":
            entry["status"] = status

        concise_calls.append(entry)

    # Parse arguments from JSON strings
    for call in concise_calls:
        args = call.get("arguments", "")
        if isinstance(args, str) and args:
            try:
                call["arguments"] = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                pass

    result = {
        "run": run.get("run"),
        "run_index": run.get("run_index"),
        "tool_calls_count": run.get("tool_calls_count", len(concise_calls)),
        "tool_calls": concise_calls,
    }

    # Add run-level error info
    if run.get("error"):
        result["run_error"] = run["error"]
    if run.get("retry_exhausted"):
        result["retry_exhausted"] = True
    if run.get("retryable_error"):
        result["retryable_error"] = run["retryable_error"]

    # Add validation result summary
    validation = run.get("validation", {}) or {}
    result["validated"] = validation.get("validated", False)
    result["accuracy"] = validation.get("accuracy")
    if validation.get("reason"):
        result["validation_reason"] = validation["reason"]

    # Output text empty? (likely max_tool_calls reached)
    output_text = run.get("output_text", "")
    result["output_text_empty"] = (output_text == "" or output_text is None)

    return result


def generate_tool_call_logs(date_folder: Path):
    """Generate tool call log files for all test cases in a date folder."""
    output_base = TOOL_CALLS_DIR / date_folder.name

    # Find all setup directories
    setup_dirs = [d for d in sorted(date_folder.iterdir()) if d.is_dir() and d.name != "datasets" and d.name != "workspaces"]

    for setup_dir in setup_dirs:
        setup_name = setup_dir.name

        for log_file in sorted(setup_dir.glob("*.json")):
            try:
                with log_file.open("r", encoding="utf-8") as f:
                    log_data = json.load(f)
            except Exception as e:
                print(f"  [SKIP] {log_file.name}: {e}")
                continue

            meta = log_data.get("meta", {})
            test_case_id = meta.get("test_case_id", "unknown")
            runs = log_data.get("runs", [])

            if not runs:
                continue

            # Output directory: evaluation_tool_calls/<date>/<setup>/<test_case>/
            out_dir = output_base / setup_name / test_case_id
            out_dir.mkdir(parents=True, exist_ok=True)

            # Generate per-run files
            for run in runs:
                run_num = run.get("run", run.get("run_index", 0))
                run_data = extract_tool_calls_for_run(run)

                out_file = out_dir / f"run_{run_num:03d}_tool_calls.json"
                with out_file.open("w", encoding="utf-8") as f:
                    json.dump(run_data, f, ensure_ascii=False, indent=2)

            # Generate summary file with all runs
            summary = {
                "meta": {
                    "test_case_id": test_case_id,
                    "setup": setup_name,
                    "model": meta.get("model"),
                    "run_count": len(runs),
                },
                "runs_summary": [],
            }

            for run in runs:
                run_num = run.get("run", run.get("run_index", 0))
                tool_calls = run.get("tool_calls", [])
                call_names = [
                    tc.get("name") or tc.get("function", {}).get("name", "?")
                    for tc in tool_calls
                ]
                validation = run.get("validation", {}) or {}

                output_text = run.get("output_text", "")
                has_error = bool(run.get("error")) or run.get("retry_exhausted", False)
                max_tools_hit = (len(call_names) == 20 and (output_text == "" or output_text is None))

                summary["runs_summary"].append({
                    "run": run_num,
                    "tool_calls_count": len(call_names),
                    "call_sequence": call_names,
                    "validated": validation.get("validated", False),
                    "accuracy": validation.get("accuracy"),
                    "has_error": has_error,
                    "max_tool_calls_hit": max_tools_hit,
                    "error": run.get("error"),
                })

            summary_file = out_dir / "summary.json"
            with summary_file.open("w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            print(f"  [{setup_name}] {test_case_id}: {len(runs)} runs -> {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate tool call logs from evaluation results")
    parser.add_argument("--date", help="Date folder (YYYYMMDD_HHMMSS), defaults to latest")
    args = parser.parse_args()

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

    print(f"Extracting tool calls from: {date_folder}")
    generate_tool_call_logs(date_folder)
    print(f"\nOutput in: {TOOL_CALLS_DIR / date_folder.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
