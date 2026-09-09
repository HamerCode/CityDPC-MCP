"""
Revalidate existing evaluation logs without rerunning model calls.

Usage:
  python3 -m evals.tools.revalidate_logs --date 07.02_15-30_gpt-5-mini_medium
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from evals.validation import validate_dataset_modification, validate_test_result

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "evaluation_logs"
GROUND_TRUTH_PATH = ROOT / "ground_truth.json"


def _load_ground_truth() -> dict[str, Any]:
    if not GROUND_TRUTH_PATH.exists():
        return {"solutions": {}}
    with GROUND_TRUTH_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "solutions" not in data:
        return {"solutions": data}
    return data


def _extract_raise_spec_from_prompt(prompt: str) -> dict[str, Any] | None:
    if not isinstance(prompt, str):
        return None
    id_match = re.search(r"ID\s+'([^']+)'", prompt)
    old_match = re.search(r"measuredHeight\s+([0-9]+(?:\.[0-9]+)?)m", prompt)
    new_match = re.search(r"auf\s+([0-9]+(?:\.[0-9]+)?)m", prompt)
    inc_match = re.search(r"um\s+([0-9]+(?:\.[0-9]+)?)m", prompt)
    if not (id_match and old_match and new_match):
        return None
    old_h = float(old_match.group(1))
    new_h = float(new_match.group(1))
    inc = float(inc_match.group(1)) if inc_match else round(new_h - old_h, 3)
    return {
        "target_building_id": id_match.group(1),
        "old_height": old_h,
        "increase_by": inc,
        "new_height": new_h,
    }


def _iter_log_files(date_dir: Path) -> list[Path]:
    files: list[Path] = []
    for setup_dir in sorted([d for d in date_dir.iterdir() if d.is_dir() and d.name not in {"datasets", "workspaces"}]):
        files.extend(sorted(setup_dir.glob("*.json")))
    return files


def revalidate_date(date_dir: Path) -> dict[str, Any]:
    gt = _load_ground_truth()
    gt_solutions = gt.get("solutions", {}) or {}
    expected_building_spec = gt.get("test_building_spec")
    gt_raise_spec = gt.get("raise_building_spec")

    report: dict[str, Any] = {
        "date_folder": date_dir.name,
        "ground_truth_file": str(GROUND_TRUTH_PATH),
        "files": [],
        "summary": {},
    }

    summary: dict[str, dict[str, dict[str, int]]] = {}

    for log_file in _iter_log_files(date_dir):
        with log_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("meta", {}) or {}
        test_case = data.get("test_case", {}) or {}
        setup_key = meta.get("setup_key", "")
        test_id = meta.get("test_case_id") or test_case.get("id") or ""
        runs = data.get("runs", []) or []

        file_item = {
            "file": str(log_file),
            "setup": setup_key,
            "test_case_id": test_id,
            "run_count": len(runs),
            "runs": [],
        }

        summary.setdefault(setup_key, {}).setdefault(test_id, {"runs": 0, "correct": 0})

        for run in runs:
            output_text = run.get("output_text", "") or ""
            val = validate_test_result(test_id, output_text, gt_solutions)

            state_val = None
            if test_case.get("category") == "state_change":
                dataset_state = run.get("dataset_state", {}) or {}
                snapshot_path = dataset_state.get("snapshot_path")
                dataset_path = Path(snapshot_path) if snapshot_path and Path(snapshot_path).exists() else Path(dataset_state.get("path", ""))

                state_eval_setup = setup_key
                if setup_key == "no_tools" and dataset_state.get("patch_applied") and dataset_state.get("changed"):
                    state_eval_setup = "patched_no_tools"
                elif setup_key == "code_interpreter" and (dataset_state.get("ci_fallback", {}) or {}).get("applied"):
                    state_eval_setup = "ci_fallback"

                expected_raise_spec = gt_raise_spec
                if test_id == "raise_building_by_id":
                    prompt_raise = _extract_raise_spec_from_prompt(test_case.get("prompt", ""))
                    if prompt_raise:
                        expected_raise_spec = prompt_raise

                state_val = validate_dataset_modification(
                    test_id=test_id,
                    setup_key=state_eval_setup,
                    output_text=output_text,
                    dataset_path=dataset_path,
                    expected_building_spec=expected_building_spec,
                    expected_raise_spec=expected_raise_spec,
                )

                if not val.get("json_parsed", False):
                    val = {
                        "validated": bool(state_val.get("validated")),
                        "json_parsed": False,
                        "accuracy": state_val.get("accuracy"),
                        "reason": "State-change evaluated via dataset validation fallback",
                    }

            is_correct = False
            if test_id == "raise_building_by_id":
                checks = (state_val or {}).get("checks", {}) or {}
                is_correct = bool(checks.get("building_found") and checks.get("height_correct"))
            elif test_id == "add_building":
                checks = (state_val or {}).get("checks", {}) or {}
                is_correct = bool(checks.get("building_found") and checks.get("id_correct") and checks.get("height_correct"))
            else:
                is_correct = (val.get("accuracy") or 0) >= 1.0

            summary[setup_key][test_id]["runs"] += 1
            summary[setup_key][test_id]["correct"] += 1 if is_correct else 0

            file_item["runs"].append(
                {
                    "run_index": run.get("run_index"),
                    "old_validation": run.get("validation"),
                    "new_validation": val,
                    "new_state_validation": state_val,
                    "correct": is_correct,
                }
            )

        report["files"].append(file_item)

    report["summary"] = summary
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Revalidate logs without rerunning model calls.")
    parser.add_argument("--date", required=True, help="Date folder in evaluation_logs, e.g. 07.02_15-30_gpt-5-mini_medium")
    args = parser.parse_args()

    date_dir = LOG_DIR / args.date
    if not date_dir.exists():
        print(f"Error: date folder not found: {date_dir}")
        return 1

    report = revalidate_date(date_dir)
    out_path = date_dir / "revalidation_report.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Revalidation report written: {out_path}")
    for setup, tests in sorted(report["summary"].items()):
        for test_id, s in sorted(tests.items()):
            runs = s["runs"]
            ok = s["correct"]
            rate = (ok / runs * 100.0) if runs else 0.0
            print(f"{setup:18} {test_id:28} {ok}/{runs} ({rate:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
