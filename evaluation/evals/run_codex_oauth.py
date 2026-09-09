"""
Separate evaluation runner using Codex OAuth instead of KI:Connect.

The existing evals/run.py runner remains unchanged. This script reuses the same
test cases, setup handling, MCP transports, logging and validation, but swaps
the model client for the ChatGPT Codex OAuth backend.

Examples:
    python3 -m evals.run_codex_oauth --setup all --tests all --runs 100
    python3 -m evals.run_codex_oauth --setup mcp --tests list_buildings --runs 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.cases import TEST_CASES, get_test_cases
from evals.codex_client import DEFAULT_CODEX_RESPONSES_URL, CodexOAuthResponsesClient
from evals.codex_oauth import ensure_fresh_tokens, oauth_status, save_project_tokens
from evals.run import (
    INPUT_DIR,
    LOG_DIR,
    ROOT,
    SETUPS,
    call_model,
    expand,
    load_ground_truth,
    run_once,
    write_log,
    write_summary,
)


DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_REASONING_EFFORT = "xhigh"


def default_project_token_path() -> Path:
    return ROOT / "evals" / "data" / "codex_oauth.json"


def make_codex_date_prefix(model: str, reasoning: str | None) -> str:
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-") or "model"
    effort = f"_{reasoning}" if reasoning else ""
    return datetime.now().strftime("%d.%m_%H-%M") + f"_{model_slug}{effort}_codex-oauth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation Runner mit Codex OAuth")
    parser.add_argument(
        "--setup",
        default=os.getenv("CODEX_OAUTH_EVAL_SETUP", "all"),
        help="Komma-getrennt: mcp,no_tools,code_interpreter oder all",
    )
    parser.add_argument(
        "--tests",
        default=os.getenv("CODEX_OAUTH_EVAL_TESTS", "all"),
        help="Komma-getrennte Test-IDs oder all",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=int(os.getenv("CODEX_OAUTH_RUNS_PER_TEST", "100")),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("CODEX_OAUTH_EVAL_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument(
        "--reasoning-effort",
        default=os.getenv("CODEX_OAUTH_REASONING_EFFORT", DEFAULT_REASONING_EFFORT),
        help="Codex reasoning effort: minimal, low, medium, high oder xhigh.",
    )
    parser.add_argument(
        "--dataset",
        default=os.getenv("EVAL_DATASET_PATH", str(INPUT_DIR / "evaluation.city.json")),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("CODEX_OAUTH_EVAL_WORKERS", os.getenv("EVAL_WORKERS", "1"))),
        help="Anzahl paralleler Lanes (1 = sequentiell)",
    )
    parser.add_argument(
        "--output-prefix",
        default=os.getenv("CODEX_OAUTH_EVAL_OUTPUT_PREFIX", ""),
        help="Bestehenden evaluation_logs-Unterordner fortsetzen, z.B. 04.06_00-51_gpt-5.4-mini_xhigh_codex-oauth.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Vorhandene Runs im Output-Ordner ignorieren und Lane wieder bei 1 starten.",
    )
    parser.add_argument(
        "--rerun-errors",
        action="store_true",
        help="Vorhandene Runs mit technischem error-Feld als fehlend behandeln und neu ausfuehren.",
    )
    parser.add_argument(
        "--rerun-below-accuracy",
        type=float,
        default=None,
        help=(
            "Vorhandene Runs mit Accuracy unter diesem Wert neu ausfuehren "
            "(z.B. 1.0 fuer alle nicht-perfekten Runs)."
        ),
    )
    parser.add_argument(
        "--parallel-runs",
        action="store_true",
        help="Fehlende einzelne Run-Nummern parallel ausfuehren statt je Lane sequenziell.",
    )
    parser.add_argument(
        "--codex-auth-file",
        type=Path,
        default=Path(os.getenv("CODEX_OAUTH_TOKEN_FILE", str(default_project_token_path()))),
        help="Projektlokale Token-Datei aus evals/00_codex_oauth.py.",
    )
    parser.add_argument(
        "--codex-responses-url",
        default=os.getenv("CODEX_OAUTH_RESPONSES_URL", DEFAULT_CODEX_RESPONSES_URL),
    )
    parser.add_argument(
        "--no-global-auth",
        action="store_true",
        help="Nicht auf ~/.codex/auth.json zurueckfallen.",
    )
    parser.add_argument("--list-tests", action="store_true")
    return parser.parse_args()


def load_existing_runs(date_prefix: str, setup_key: str, test_case: dict, target_runs: int) -> list[dict]:
    log_path = LOG_DIR / date_prefix / setup_key / f"{test_case['id']}.json"
    if not log_path.exists():
        return []
    try:
        data = json.loads(log_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [{setup_key}/{test_case['id']}] Resume ignoriert: {exc}", flush=True)
        return []
    runs = data.get("runs")
    if not isinstance(runs, list):
        return []
    by_run: dict[int, dict] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        run_no = run.get("run")
        if not isinstance(run_no, int) or run_no < 1 or run_no > target_runs:
            continue
        by_run.setdefault(run_no, run)
    return [by_run[run_no] for run_no in sorted(by_run)]


def run_accuracy(run: dict) -> float | None:
    validation = run.get("validation")
    if isinstance(validation, dict):
        accuracy = validation.get("accuracy")
        if isinstance(accuracy, (int, float)):
            return float(accuracy)
    state_validation = run.get("state_validation")
    if isinstance(state_validation, dict):
        accuracy = state_validation.get("accuracy")
        if isinstance(accuracy, (int, float)):
            return float(accuracy)
    return None


def should_rerun_existing_run(run: dict, args: argparse.Namespace) -> bool:
    if args.rerun_errors and run.get("error") is not None:
        return True
    if args.rerun_below_accuracy is not None:
        accuracy = run_accuracy(run)
        if accuracy is None or accuracy < args.rerun_below_accuracy:
            return True
    return False


def filter_existing_runs_for_resume(
    runs: list[dict],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    if args.no_resume:
        return [], runs
    kept = []
    rerun = []
    for run in runs:
        if should_rerun_existing_run(run, args):
            rerun.append(run)
        else:
            kept.append(run)
    return kept, rerun


def ensure_project_auth(auth_file: Path, *, include_global: bool) -> None:
    tokens = ensure_fresh_tokens(auth_file, include_global=include_global)
    save_project_tokens(auth_file, tokens)
    expires = datetime.fromtimestamp(tokens.expires, tz=timezone.utc).isoformat()
    print(f"[Auth] Codex OAuth: accountId={tokens.account_id or 'unknown'}")
    print(f"[Auth] Expires:     {expires}")
    print(f"[Auth] Token file:  {auth_file}")


def main() -> None:
    load_dotenv(ROOT / ".env")
    args = parse_args()

    if args.list_tests:
        for tc in TEST_CASES:
            print(f"{tc['id']}: {tc.get('category', '?')}")
        return

    setup_ids = expand(args.setup, list(SETUPS))
    valid_test_ids = [tc["id"] for tc in TEST_CASES]
    selected = expand(args.tests, valid_test_ids)

    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if not dataset_path.exists():
        print(f"Dataset nicht gefunden: {dataset_path}")
        sys.exit(1)

    include_global = not args.no_global_auth
    auth_file = args.codex_auth_file.expanduser()
    if not auth_file.is_absolute():
        auth_file = ROOT / auth_file
    ensure_project_auth(auth_file, include_global=include_global)
    print(f"[Auth] Status:      {oauth_status(auth_file, include_global=False)}")

    format_type = "gml" if dataset_path.suffix.lower() in {".gml", ".xml"} else "city.json"
    test_cases = get_test_cases(format_type=format_type, test_ids=selected)
    if not test_cases:
        print("Keine passenden Test-Cases.")
        return

    mcp_dir = Path(os.getenv("MCP_SERVER_DIR", str(ROOT.parent / "mcp-server"))).expanduser()
    mcp_script = Path(os.getenv("MCP_SERVER_SCRIPT", str(mcp_dir / "cityDPC.py")))
    mcp_python = os.getenv("MCP_SERVER_PYTHON", sys.executable)
    mcp_dataset_dir = Path(os.getenv("MCP_DATASET_DIR", str(mcp_dir / "data" / "datasets")))
    mcp_log = LOG_DIR / "mcp_server_stdio.log"

    client = CodexOAuthResponsesClient(
        auth_file=auth_file,
        include_global_auth=include_global,
        api_url=args.codex_responses_url,
        model=args.model.strip(),
        timeout_seconds=int(os.getenv("TEST_TIMEOUT_SECONDS", "300")),
        max_tool_rounds=int(os.getenv("CODEX_OAUTH_MAX_TOOL_ROUNDS", os.getenv("KI_CONNECT_MAX_TOOL_ROUNDS", "80"))),
        reasoning_effort=args.reasoning_effort.strip() or None,
    )

    date_prefix = args.output_prefix.strip() or make_codex_date_prefix(client.model, client.reasoning_effort)
    ground_truth = load_ground_truth()

    print(f"[Eval] Modell:        {client.model}")
    print(f"[Eval] Provider:      {client.provider}")
    print(f"[Eval] Reasoning:     {client.reasoning_effort or '(default)'}")
    print(f"[Eval] Dataset:       {dataset_path}")
    print(f"[Eval] Setups:        {setup_ids}")
    print(f"[Eval] Test-Cases:    {[tc['id'] for tc in test_cases]}")
    print(f"[Eval] Runs je Lane:  {args.runs}")
    print(f"[Eval] Workers:       {args.workers}")
    print(f"[Eval] Resume:        {'nein' if args.no_resume else 'ja'}")
    print(f"[Eval] Rerun Errors:  {'ja' if args.rerun_errors else 'nein'}")
    print(
        "[Eval] Rerun < Acc:  "
        + (str(args.rerun_below_accuracy) if args.rerun_below_accuracy is not None else "nein")
    )
    print(f"[Eval] Parallel Runs: {'ja' if args.parallel_runs else 'nein'}")
    print(f"[Eval] Output-Ordner: {LOG_DIR / date_prefix}")
    print()

    lanes = [(setup, case) for setup in setup_ids for case in test_cases]
    all_runs: dict[tuple[str, str], list[dict]] = {}
    write_lock = threading.Lock()
    t_start = time.perf_counter()

    def run_lane(setup_key: str, test_case: dict) -> tuple[tuple[str, str], list[dict]]:
        loaded_runs = [] if args.no_resume else load_existing_runs(date_prefix, setup_key, test_case, args.runs)
        runs, rerun_runs = filter_existing_runs_for_resume(loaded_runs, args)
        existing_run_nos = {run["run"] for run in runs}
        missing_run_nos = [run_no for run_no in range(1, args.runs + 1) if run_no not in existing_run_nos]

        if not missing_run_nos:
            with write_lock:
                print(f"  [{setup_key}/{test_case['id']}] Resume: {len(runs)}/{args.runs} bereits vorhanden", flush=True)
                write_log(date_prefix, setup_key, test_case, runs, client, dataset_path)
            return (setup_key, test_case["id"]), runs

        if runs or rerun_runs:
            with write_lock:
                msg = f"  [{setup_key}/{test_case['id']}] Resume: {len(runs)}/{args.runs} behalten"
                if rerun_runs:
                    msg += f", {len(rerun_runs)} werden ersetzt"
                print(msg, flush=True)

        for run_no in missing_run_nos:
            res = run_once(
                client,
                setup_key,
                test_case,
                dataset_path,
                run_no,
                date_prefix,
                ground_truth,
                mcp_dir,
                mcp_script,
                mcp_python,
                mcp_log,
                mcp_dataset_dir,
            )
            runs.append(res)
            acc = res.get("validation", {}).get("accuracy")
            if res.get("error"):
                status, acc_text = "ERR", "-"
            elif isinstance(acc, (int, float)):
                status = "PASS" if acc >= 1 else "FAIL"
                acc_text = f"{acc:.0%}"
            else:
                status, acc_text = "DONE", "N/A"
            with write_lock:
                runs.sort(key=lambda run: run["run"])
                print(
                    f"  [{setup_key}/{test_case['id']}] Run {run_no}/{args.runs} "
                    f"... {status} {acc_text} ({res['duration_ms']/1000:.1f}s)",
                    flush=True,
                )
                write_log(date_prefix, setup_key, test_case, runs, client, dataset_path)
        return (setup_key, test_case["id"]), runs

    def run_missing_once(setup_key: str, test_case: dict, run_no: int) -> dict:
        return run_once(
            client,
            setup_key,
            test_case,
            dataset_path,
            run_no,
            date_prefix,
            ground_truth,
            mcp_dir,
            mcp_script,
            mcp_python,
            mcp_log,
            mcp_dataset_dir,
        )

    def print_result(setup_key: str, test_case: dict, res: dict) -> None:
        acc = res.get("validation", {}).get("accuracy")
        if res.get("error"):
            status, acc_text = "ERR", "-"
        elif isinstance(acc, (int, float)):
            status = "PASS" if acc >= 1 else "FAIL"
            acc_text = f"{acc:.0%}"
        else:
            status, acc_text = "DONE", "N/A"
        print(
            f"  [{setup_key}/{test_case['id']}] Run {res['run']}/{args.runs} "
            f"... {status} {acc_text} ({res['duration_ms']/1000:.1f}s)",
            flush=True,
        )

    if args.parallel_runs:
        pending_runs: list[tuple[str, dict, int]] = []
        missing_by_lane: list[tuple[str, dict, list[int]]] = []
        for setup_key, test_case in lanes:
            key = (setup_key, test_case["id"])
            loaded_runs = [] if args.no_resume else load_existing_runs(date_prefix, setup_key, test_case, args.runs)
            runs, rerun_runs = filter_existing_runs_for_resume(loaded_runs, args)
            all_runs[key] = runs
            existing_run_nos = {run["run"] for run in runs}
            missing_run_nos = [run_no for run_no in range(1, args.runs + 1) if run_no not in existing_run_nos]
            if missing_run_nos:
                print(
                    f"  [{setup_key}/{test_case['id']}] Queue: {len(missing_run_nos)} offen, "
                    f"{len(runs)}/{args.runs} behalten"
                    + (f", {len(rerun_runs)} werden ersetzt" if rerun_runs else ""),
                    flush=True,
                )
                missing_by_lane.append((setup_key, test_case, missing_run_nos))
            else:
                print(f"  [{setup_key}/{test_case['id']}] Resume: {len(runs)}/{args.runs} bereits vorhanden", flush=True)
                write_log(date_prefix, setup_key, test_case, runs, client, dataset_path)

        max_missing = max((len(missing) for _, _, missing in missing_by_lane), default=0)
        for idx in range(max_missing):
            for setup_key, test_case, missing_run_nos in missing_by_lane:
                if idx < len(missing_run_nos):
                    pending_runs.append((setup_key, test_case, missing_run_nos[idx]))

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_missing_once, setup_key, test_case, run_no): (setup_key, test_case)
                for setup_key, test_case, run_no in pending_runs
            }
            for future in as_completed(futures):
                setup_key, test_case = futures[future]
                key = (setup_key, test_case["id"])
                res = future.result()
                with write_lock:
                    runs = all_runs[key]
                    runs[:] = [run for run in runs if run.get("run") != res.get("run")]
                    runs.append(res)
                    runs.sort(key=lambda run: run["run"])
                    print_result(setup_key, test_case, res)
                    write_log(date_prefix, setup_key, test_case, runs, client, dataset_path)
    elif args.workers <= 1:
        for setup_key, test_case in lanes:
            key, runs = run_lane(setup_key, test_case)
            all_runs[key] = runs
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_lane, setup, case) for setup, case in lanes]
            for future in as_completed(futures):
                key, runs = future.result()
                all_runs[key] = runs

    elapsed = time.perf_counter() - t_start
    write_summary(date_prefix, all_runs, client, dataset_path)
    print(f"\n[Eval] Fertig in {elapsed/60:.1f} min. Logs unter: {LOG_DIR / date_prefix}")


if __name__ == "__main__":
    main()
