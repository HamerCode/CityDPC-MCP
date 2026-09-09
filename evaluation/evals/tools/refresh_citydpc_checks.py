"""
Refresh citydpc-based checks (`citydpc_importable`, `building_displayable`) in
add_building evaluation logs that were validated while the `citydpc` library was
NOT installed.

Symptom: such logs store
    state_validation.integrity.citydpc_import.error == "citydpc_not_installed"
and therefore `citydpc_importable` / `building_displayable` are False for *every*
run. Because the success criterion for add_building requires both, the comparison
tables show 0% success even though the building was generated correctly. Other
models (gpt-oss, mistral) were validated *with* citydpc installed, so this makes
the comparison unfair.

This tool re-runs the full dataset validation against the already-stored
post-modification snapshots
    <run_folder>/datasets/add_building/<setup>/run_<NNN>_post.city.json
now that citydpc is available, and patches the affected logs in place.

Only logs whose stored citydpc error is exactly "citydpc_not_installed" are
touched. A safety guard ensures that *only* citydpc-related check keys (and the
derived accuracy) change; if any other check would change, the run is reported
and skipped (unless --force).

Usage (dry-run, prints what would change, writes nothing):
    python3 -m evals.tools.refresh_citydpc_checks 07.06_20-45_gpt-5.4-mini_low_codex-oauth

Apply the fix (writes a .bak next to each modified log):
    python3 -m evals.tools.refresh_citydpc_checks 07.06_20-45_gpt-5.4-mini_low_codex-oauth --apply

Scan every run folder under evaluation_logs:
    python3 -m evals.tools.refresh_citydpc_checks --all [--apply]
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

# Silence citydpc's very chatty import logging / planarity warnings.
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

from evals.validation import validate_dataset_modification  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "evaluation_logs"
GROUND_TRUTH_PATH = ROOT / "ground_truth.json"

NOT_INSTALLED = "citydpc_not_installed"

# Check keys that are allowed to change as a result of installing citydpc.
CITYDPC_KEYS = {
    "citydpc_importable",
    "building_displayable",
    "building_has_surfaces",
    "building_surface_count",
    "building_has_roof",
    "building_has_wall",
    "building_has_ground",
}


def _load_ground_truth() -> dict[str, Any]:
    if not GROUND_TRUTH_PATH.exists():
        return {}
    with GROUND_TRUTH_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_affected_run(run: dict[str, Any]) -> bool:
    integrity = (run.get("state_validation") or {}).get("integrity") or {}
    err = (integrity.get("citydpc_import") or {}).get("error")
    return err == NOT_INSTALLED


def _snapshot_path(run_folder: Path, setup_dir_name: str, run_no: int) -> Path:
    return (
        run_folder
        / "datasets"
        / "add_building"
        / setup_dir_name
        / f"run_{run_no:03d}_post.city.json"
    )


def _diff_checks(old: dict, new: dict) -> dict[str, tuple[Any, Any]]:
    keys = set(old) | set(new)
    return {k: (old.get(k), new.get(k)) for k in keys if old.get(k) != new.get(k)}


def process_log(
    log_path: Path,
    run_folder: Path,
    gt: dict[str, Any],
    apply: bool,
    force: bool,
) -> dict[str, Any]:
    with log_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    setup_dir_name = log_path.parent.name
    expected_spec = gt.get("test_building_spec")
    runs = data.get("runs") or []

    report = {
        "log": str(log_path.relative_to(LOG_DIR)),
        "runs": len(runs),
        "affected": 0,
        "snapshot_missing": 0,
        "patched": 0,
        "skipped_unexpected_change": 0,
        "success_before": 0,
        "success_after": 0,
        "unexpected_examples": [],
    }

    def _add_building_ok(checks: dict) -> bool:
        return bool(
            checks.get("building_found")
            and checks.get("id_correct")
            and checks.get("height_correct")
            and checks.get("cityjson_valid")
            and checks.get("citydpc_importable")
            and checks.get("geometry_valid")
            and checks.get("building_displayable")
        )

    changed = False
    for run in runs:
        sv = run.get("state_validation") or {}
        old_checks = sv.get("checks") or {}
        if _add_building_ok(old_checks):
            report["success_before"] += 1

        if not _is_affected_run(run):
            # Not a citydpc-artifact run; leave its success contribution as-is.
            if _add_building_ok(old_checks):
                report["success_after"] += 1
            continue

        report["affected"] += 1

        run_no = run.get("run")
        snap = _snapshot_path(run_folder, setup_dir_name, run_no) if run_no else None
        if not snap or not snap.exists():
            report["snapshot_missing"] += 1
            if _add_building_ok(old_checks):
                report["success_after"] += 1
            continue

        state_setup = sv.get("setup_key") or run.get("setup_key") or setup_dir_name
        new_sv = validate_dataset_modification(
            test_id="add_building",
            setup_key=state_setup,
            output_text=run.get("output_text", "") or "",
            dataset_path=snap,
            expected_building_spec=expected_spec,
            expected_raise_spec=None,
        )
        new_checks = new_sv.get("checks") or {}

        diff = _diff_checks(old_checks, new_checks)
        unexpected = {k: v for k, v in diff.items() if k not in CITYDPC_KEYS}
        if unexpected and not force:
            report["skipped_unexpected_change"] += 1
            if len(report["unexpected_examples"]) < 5:
                report["unexpected_examples"].append(
                    {"run": run_no, "unexpected": unexpected}
                )
            if _add_building_ok(old_checks):
                report["success_after"] += 1
            continue

        if apply:
            run["state_validation"] = new_sv
            # Keep the top-level validation.accuracy in sync (used by non-add
            # tables); only update the numeric accuracy, preserve other fields.
            val = run.get("validation")
            if isinstance(val, dict) and "accuracy" in val:
                val["accuracy"] = new_sv.get("accuracy")
            changed = True

        report["patched"] += 1
        if _add_building_ok(new_checks):
            report["success_after"] += 1

    if apply and changed:
        backup = log_path.with_suffix(log_path.suffix + ".bak")
        if not backup.exists():
            backup.write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8")
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return report


def _iter_add_building_logs(run_folder: Path) -> list[Path]:
    logs: list[Path] = []
    for setup_dir in sorted(run_folder.iterdir()):
        if not setup_dir.is_dir() or setup_dir.name in {"datasets", "workspaces"}:
            continue
        for log_path in sorted(setup_dir.glob("*add_building*.json")):
            logs.append(log_path)
    return logs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-validate citydpc checks in add_building logs validated without citydpc."
    )
    parser.add_argument("folders", nargs="*", help="Run folders under evaluation_logs (names or paths).")
    parser.add_argument("--all", action="store_true", help="Scan every run folder under evaluation_logs.")
    parser.add_argument("--apply", action="store_true", help="Write changes (default is dry-run).")
    parser.add_argument("--force", action="store_true", help="Patch even if non-citydpc checks would change.")
    args = parser.parse_args()

    gt = _load_ground_truth()

    if args.all:
        run_folders = [p for p in sorted(LOG_DIR.iterdir()) if p.is_dir()]
    else:
        run_folders = []
        for name in args.folders:
            p = Path(name)
            if not p.is_absolute():
                p = LOG_DIR / name
            if p.exists():
                run_folders.append(p)
            else:
                print(f"WARN: folder not found: {name}")
    if not run_folders:
        print("No run folders to process. Pass folder names or --all.")
        return 1

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== refresh_citydpc_checks [{mode}] ===\n")

    grand = {"affected": 0, "patched": 0, "snapshot_missing": 0, "skipped": 0,
             "success_before": 0, "success_after": 0, "runs": 0}

    for run_folder in run_folders:
        logs = _iter_add_building_logs(run_folder)
        folder_has_affected = False
        folder_reports = []
        for log_path in logs:
            rep = process_log(log_path, run_folder, gt, apply=args.apply, force=args.force)
            if rep["affected"]:
                folder_has_affected = True
            folder_reports.append(rep)

        if not folder_has_affected:
            continue

        print(f"## {run_folder.name}")
        for rep in folder_reports:
            if not rep["affected"]:
                continue
            print(
                f"  {rep['log']}\n"
                f"     runs={rep['runs']} affected={rep['affected']} patched={rep['patched']} "
                f"snap_missing={rep['snapshot_missing']} skipped={rep['skipped_unexpected_change']}\n"
                f"     add_building success: {rep['success_before']}/{rep['runs']} "
                f"-> {rep['success_after']}/{rep['runs']}"
            )
            for ex in rep["unexpected_examples"]:
                print(f"     ! unexpected change run {ex['run']}: {ex['unexpected']}")
            grand["runs"] += rep["runs"]
            grand["affected"] += rep["affected"]
            grand["patched"] += rep["patched"]
            grand["snapshot_missing"] += rep["snapshot_missing"]
            grand["skipped"] += rep["skipped_unexpected_change"]
            grand["success_before"] += rep["success_before"]
            grand["success_after"] += rep["success_after"]
        print()

    print("=== TOTAL ===")
    print(
        f"runs={grand['runs']} affected={grand['affected']} patched={grand['patched']} "
        f"snap_missing={grand['snapshot_missing']} skipped_unexpected={grand['skipped']}"
    )
    print(
        f"add_building success across affected logs: "
        f"{grand['success_before']}/{grand['runs']} -> {grand['success_after']}/{grand['runs']}"
    )
    if not args.apply:
        print("\n(DRY-RUN: no files written. Re-run with --apply to persist.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
