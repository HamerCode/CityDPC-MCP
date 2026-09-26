"""
Benchmark runner: five tasks x three setups (MCP, No Tools, Code Interpreter).

Writes one JSON log per setup/task to evaluation/logs/<run>/ (same format as the
paper runs); `python analyze.py logs/<run>` turns it into the result tables.

Examples:
    python evaluation/run.py --list-tests
    python evaluation/run.py --setup mcp --tests list_buildings --runs 1
    python evaluation/run.py --model gpt-oss-120b --reasoning-effort high --api completions --runs 100
"""

import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv(Path(__file__).resolve().parent / ".env")  # before importing client (reads throttle settings)

from cases import (
    BASE_INSTRUCTIONS,
    RAISE_BUILDING_SPEC,
    TEST_BUILDING_SPEC,
    TEST_CASES,
    get_setup_specific_prompt,
    get_test_cases,
)
from client import LLMClient
from schemas import RESPONSE_SCHEMAS
from validation import validate_dataset_modification, validate_test_result


# ---------------------------------------------------------------------------
# Konstanten & Pfade
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
GROUND_TRUTH_FILE = DATA_DIR / "ground_truth.json"

DEFAULT_MODEL = "gpt-oss-120b"
DEFAULT_BASE_URL = "https://chat.kiconnect.nrw/api/v1"

# Setup-Key -> Label (taucht im Logfile als meta.setup auf)
SETUPS = {
    "mcp": "MCP",
    "no_tools": "No Tools",
    "code_interpreter": "Code Interpreter",
}


# ---------------------------------------------------------------------------
# Kleine Helfer
# ---------------------------------------------------------------------------
def load_ground_truth() -> dict:
    if not GROUND_TRUTH_FILE.exists():
        return {"solutions": {}}
    data = json.loads(GROUND_TRUTH_FILE.read_text(encoding="utf-8"))
    return data if "solutions" in data else {"solutions": data}


def expand(selection: str, valid: list) -> list:
    items = [s.strip() for s in selection.split(",") if s.strip()]
    if not items or items == ["all"]:
        return valid
    filtered = [i for i in items if i in valid]
    return filtered or valid


def dataset_ext(name: str) -> str:
    return ".city.json" if name.endswith(".city.json") else Path(name).suffix


def is_dataset_modification_case(test_case: dict) -> bool:
    return test_case.get("category") == "state_change"


def make_date_prefix(model: str, reasoning, completions: bool) -> str:
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-") or "model"
    effort = f"_{reasoning}" if reasoning else ""
    provider = "completions" if completions else "responses"
    return datetime.now().strftime("%d.%m_%H-%M") + f"_{model_slug}{effort}_{provider}"


# ---------------------------------------------------------------------------
# JSON-Patch (RFC 6902, nur add / replace / remove)
# Fuer no_tools und code_interpreter State-Change Cases: das Modell liefert
# einen Patch, den wir lokal auf das Dataset anwenden, damit die Validation
# wie bei MCP gegen die Datei pruefen kann.
# ---------------------------------------------------------------------------
def extract_patch(text: str):
    cleaned = re.sub(r"\[THINK\].*?\[/THINK\]", "", text, flags=re.DOTALL).strip()
    candidates = [cleaned]
    candidates += [m.group(1).strip()
                   for m in re.finditer(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)]
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            if isinstance(obj.get("patch"), list):
                return obj["patch"]
            if "op" in obj and "path" in obj:
                return [obj]
    return None


def apply_patch(doc, ops):
    for op in ops:
        path = op.get("path", "")
        if not path.startswith("/"):
            continue
        tokens = [p.replace("~1", "/").replace("~0", "~") for p in path[1:].split("/")]
        parent = doc
        for t in tokens[:-1]:
            if isinstance(parent, list):
                t = int(t)
            parent = parent[t]
        key = tokens[-1]
        if isinstance(parent, list) and key != "-":
            key = int(key)

        action = op.get("op")
        if action == "add":
            if isinstance(parent, list):
                if key == "-":
                    parent.append(op.get("value"))
                else:
                    parent.insert(key, op.get("value"))
            else:
                parent[key] = op.get("value")
        elif action == "replace":
            parent[key] = op.get("value")
        elif action == "remove":
            del parent[key]


# ---------------------------------------------------------------------------
# MCP-Transports
# ---------------------------------------------------------------------------
def make_citydpc_transport(log_file, mcp_dataset_dir):
    """Start the CityDPC MCP server of this repository (fresh process per run)."""
    from fastmcp.client.transports import StdioTransport
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MCP_DATASET_DIR"] = str(mcp_dataset_dir)
    command = os.getenv("MCP_SERVER_COMMAND")
    if command:
        cmd, *args = command.split()
    else:
        cmd, args = sys.executable, ["-m", "citydpc_mcp"]
    return StdioTransport(
        command=cmd,
        args=args,
        env=env,
        cwd=str(mcp_dataset_dir),
        keep_alive=False,
        log_file=log_file,
    )


def make_code_sandbox_transport():
    from fastmcp.client.transports import StdioTransport
    # External Docker sandbox MCP server, e.g. github.com/Automata-Labs-team/code-sandbox-mcp
    binary = os.getenv("CODE_SANDBOX_MCP_BINARY", "code-sandbox-mcp")
    return StdioTransport(command=binary, args=[], keep_alive=False)


# ---------------------------------------------------------------------------
# Prompt zusammenbauen
# ---------------------------------------------------------------------------
def build_prompt(setup_key, test_case, dataset_path, dataset_filename):
    base_prompt = test_case["prompt"]
    base_prompt = base_prompt.replace("{dataset_filename}", dataset_filename)

    # Bei MCP-Stdio bekommt das Dataset pro Run einen eindeutigen Filename.
    # Der Prompt referenziert sonst nur "evaluation.city.json" bzw. "evaluation.gml".
    if setup_key == "mcp":
        for original in ("evaluation.city.json", "evaluation.gml", "emptyCampus.city.json"):
            base_prompt = base_prompt.replace(original, dataset_filename)

    setup_addon = get_setup_specific_prompt(setup_key, test_case["id"])
    parts = []
    if setup_addon:
        parts.append(setup_addon)
    parts.append(f"Aufgabe:\n{base_prompt}")

    # Bei no_tools / code_interpreter Dataset inline anhaengen
    if setup_key in {"no_tools", "code_interpreter"}:
        lang = "xml" if dataset_path.suffix.lower() in {".gml", ".xml"} else "json"
        content = dataset_path.read_text(encoding="utf-8")
        header = f"Dataset-Dateiname: {dataset_filename}"
        # Code-Interpreter darf die Datei direkt vom Host in die Sandbox kopieren.
        if setup_key == "code_interpreter":
            header += f"\nDataset-Pfad: {dataset_path}"
        parts.append(f"{header}\n```{lang}\n{content}\n```")

    return "\n\n".join(parts)


def call_model(client, setup_key, prompt, schema, mcp_log, mcp_dataset_dir):
    if setup_key == "mcp":
        return client.responses.create(
            instructions=BASE_INSTRUCTIONS,
            input=prompt,
            schema=schema,
            tools=[{
                "type": "mcp",
                "server_label": "citydpc",
                "server_description": "CityDPC MCP Server",
                "server_transport": make_citydpc_transport(mcp_log, mcp_dataset_dir),
                "require_approval": "never",
            }],
        )
    if setup_key == "no_tools":
        return client.responses.create(
            instructions=BASE_INSTRUCTIONS,
            input=prompt,
            schema=schema,
        )
    if setup_key == "code_interpreter":
        return client.responses.create(
            instructions=BASE_INSTRUCTIONS,
            input=prompt,
            schema=schema,
            tools=[{
                "type": "mcp",
                "server_label": "code-sandbox-mcp",
                "server_description": "Docker Code Sandbox MCP Server",
                "server_transport": make_code_sandbox_transport(),
                "require_approval": "never",
            }],
        )
    raise ValueError(f"Unbekanntes Setup: {setup_key}")


# ---------------------------------------------------------------------------
# Dataset pro Run vorbereiten
# ---------------------------------------------------------------------------
def prepare_dataset(dataset_path, setup_key, test_id, run_no, workspace, mcp_dataset_dir):
    """Legt eine Arbeitskopie an. Liefert (pfad, filename, cleanup)."""
    ext = dataset_ext(dataset_path.name)
    base = dataset_path.name[: -len(ext)] if ext else dataset_path.stem

    if setup_key == "mcp":
        # MCP-Server arbeitet auf einem festen Daten-Ordner. Damit ein vorheriger
        # Run nichts cached, gibt es pro Run einen eindeutigen Filename.
        run_filename = f"{base}__{setup_key}__{test_id}__run_{run_no:03d}{ext}"
        target = mcp_dataset_dir / run_filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dataset_path, target)
        return target, run_filename, True

    target = workspace / dataset_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset_path, target)
    return target, dataset_path.name, False


# ---------------------------------------------------------------------------
# Validation Wrapper
# ---------------------------------------------------------------------------
def validate(test_case, setup_key, output_text, dataset_path, ground_truth, patch_applied):
    try:
        validation = validate_test_result(
            test_case["id"], output_text, ground_truth.get("solutions", {})
        )
    except Exception as exc:
        validation = {"validated": False, "json_parsed": False,
                      "accuracy": None, "reason": str(exc)}

    state_validation = None
    if is_dataset_modification_case(test_case):
        # Bei no_tools entscheidet patch_applied, ob die Datei oder nur der
        # Output geprueft wird (siehe validation.py).
        if setup_key == "no_tools":
            eval_setup = "patched_no_tools" if patch_applied else "no_tools"
        else:
            eval_setup = setup_key

        state_validation = validate_dataset_modification(
            test_id=test_case["id"],
            setup_key=eval_setup,
            output_text=output_text,
            dataset_path=dataset_path,
            expected_building_spec=ground_truth.get("test_building_spec") or TEST_BUILDING_SPEC,
            expected_raise_spec=ground_truth.get("raise_building_spec") or RAISE_BUILDING_SPEC,
        )
        # Bei State-Change-Tests ist die Dataset-Pruefung aussagekraeftiger.
        validation = {
            "validated": bool(state_validation.get("validated")) or validation.get("validated", False),
            "json_parsed": validation.get("json_parsed", False),
            "accuracy": state_validation.get("accuracy"),
            "reason": "state_validation",
            "extracted_data": validation.get("extracted_data"),
        }

    return validation, state_validation


# ---------------------------------------------------------------------------
# Tool-Calls aus Response extrahieren
# ---------------------------------------------------------------------------
def extract_tool_calls(response):
    if response is None:
        return [], [], {}
    data = response.model_dump(mode="json") if hasattr(response, "model_dump") else {}
    output_items = data.get("output") or []
    usage = data.get("usage") or {}
    calls = []
    for step, item in enumerate(output_items, 1):
        if not isinstance(item, dict):
            continue
        if item.get("type") not in {"function_call", "tool_call", "mcp_call"}:
            continue
        calls.append({
            "step": step,
            "name": item.get("name"),
            "status": item.get("status"),
            "arguments": item.get("arguments"),
            "output": item.get("output"),
            "error": item.get("error"),
        })
    return calls, output_items, usage


# ---------------------------------------------------------------------------
# Einzelner Run
# ---------------------------------------------------------------------------
def run_once(client, setup_key, test_case, dataset_path, run_no, date_prefix,
             ground_truth, mcp_log, mcp_dataset_dir):
    started_at = datetime.now()
    t0 = time.perf_counter()

    output_text = ""
    response = None
    error = None
    patch_applied = False
    work_path = None
    cleanup = False
    state_validation = None
    validation = {"validated": False, "json_parsed": False,
                  "accuracy": None, "reason": "not_run"}

    workspace = (
        LOG_DIR / date_prefix / "workspaces" / setup_key
        / test_case["id"] / f"run_{run_no:03d}"
    )

    try:
        work_path, dataset_filename, cleanup = prepare_dataset(
            dataset_path, setup_key, test_case["id"], run_no, workspace, mcp_dataset_dir
        )

        prompt = build_prompt(setup_key, test_case, work_path, dataset_filename)
        schema = RESPONSE_SCHEMAS.get(test_case["id"])

        response = call_model(client, setup_key, prompt, schema, mcp_log, mcp_dataset_dir)
        output_text = response.output_text

        # Patch lokal anwenden (no_tools, code_interpreter, state_change)
        if (is_dataset_modification_case(test_case)
                and setup_key in {"no_tools", "code_interpreter"}):
            ops = extract_patch(output_text)
            if ops:
                try:
                    data = json.loads(work_path.read_text(encoding="utf-8"))
                    apply_patch(data, ops)
                    work_path.write_text(
                        json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    patch_applied = True
                except Exception as exc:
                    print(f"    [Patch-Fehler] {exc}")

        validation, state_validation = validate(
            test_case, setup_key, output_text, work_path, ground_truth, patch_applied
        )

        # Snapshot fuer State-Change Cases
        if (is_dataset_modification_case(test_case)
                and work_path is not None and work_path.exists()):
            snap_dir = LOG_DIR / date_prefix / "datasets" / test_case["id"] / setup_key
            snap_dir.mkdir(parents=True, exist_ok=True)
            snap_path = snap_dir / f"run_{run_no:03d}_post{dataset_ext(work_path.name)}"
            shutil.copy2(work_path, snap_path)

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        validation = {"validated": False, "json_parsed": False,
                      "accuracy": None, "reason": error}
    finally:
        if cleanup and work_path:
            work_path.unlink(missing_ok=True)

    tool_calls, output_items, usage = extract_tool_calls(response)

    return {
        "run": run_no,
        "started_at": started_at.isoformat(),
        "duration_ms": int((time.perf_counter() - t0) * 1000),
        "setup_key": setup_key,
        "setup": SETUPS[setup_key],
        "output_text": output_text,
        "tokens": {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
        "tool_calls_count": len(tool_calls),
        "tool_calls": tool_calls,
        "output_items": output_items,
        "validation": validation,
        "state_validation": state_validation,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Logfiles schreiben
# ---------------------------------------------------------------------------
def write_log(date_prefix, setup_key, test_case, runs, client, dataset_path):
    log_dir = LOG_DIR / date_prefix / setup_key
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / f"{test_case['id']}.json"

    log = {
        "meta": {
            "updated_at": datetime.now().isoformat(),
            "model": client.model,
            "provider": client.provider,
            "api_url": client.api_url,
            "reasoning_effort": client.reasoning_effort,
            "dataset": str(dataset_path),
            "setup_key": setup_key,
            "setup": SETUPS[setup_key],
            "test_case_id": test_case["id"],
            "run_count": len(runs),
        },
        "test_case": {
            "id": test_case["id"],
            "name": test_case.get("name", test_case["id"]),
            "category": test_case.get("category"),
            "prompt": test_case["prompt"],
        },
        "runs": runs,
    }
    out.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary(date_prefix, all_runs, client, dataset_path):
    flat = [r for runs in all_runs.values() for r in runs]
    accs = [r["validation"]["accuracy"] for r in flat
            if isinstance(r.get("validation", {}).get("accuracy"), (int, float))]
    summary = {
        "created_at": datetime.now().isoformat(),
        "model": client.model,
        "provider": client.provider,
        "reasoning_effort": client.reasoning_effort,
        "dataset": str(dataset_path),
        "runs_total": len(flat),
        "mean_accuracy": sum(accs) / len(accs) if accs else None,
        "logs": [
            {"setup": setup, "test_case": test_id, "runs": len(runs)}
            for (setup, test_id), runs in sorted(all_runs.items())
        ],
    }
    (LOG_DIR / date_prefix / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="CityDPC-MCP benchmark runner")
    parser.add_argument("--setup", default=os.getenv("EVAL_SETUP", "all"),
                        help="Comma-separated: mcp,no_tools,code_interpreter or all")
    parser.add_argument("--tests", default=os.getenv("EVAL_TESTS", "all"),
                        help="Comma-separated task IDs or all")
    parser.add_argument("--runs", type=int, default=int(os.getenv("EVAL_RUNS", "10")),
                        help="Repetitions per setup/task (paper: 100)")
    parser.add_argument("--model", default=os.getenv("EVAL_MODEL", DEFAULT_MODEL))
    parser.add_argument("--reasoning-effort", default=os.getenv("EVAL_REASONING_EFFORT", ""),
                        help="e.g. high (paper)")
    parser.add_argument("--api", choices=["responses", "completions"],
                        default=os.getenv("EVAL_API", "responses"),
                        help="OpenAI Responses or Chat Completions API")
    parser.add_argument("--base-url", default=os.getenv("EVAL_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--dataset",
                        default=os.getenv("EVAL_DATASET", str(DATA_DIR / "evaluation.city.json")))
    parser.add_argument("--workers", type=int, default=int(os.getenv("EVAL_WORKERS", "1")),
                        help="Parallel setup/task lanes (1 = sequential)")
    parser.add_argument("--list-tests", action="store_true")
    return parser.parse_args()


def main():
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

    format_type = "gml" if dataset_path.suffix.lower() in {".gml", ".xml"} else "city.json"
    test_cases = get_test_cases(format_type=format_type, test_ids=selected)
    if not test_cases:
        print("Keine passenden Test-Cases.")
        return

    completions = args.api == "completions"
    client = LLMClient(
        base_url=args.base_url,
        completions_mode=completions,
        api_key=os.getenv("EVAL_API_KEY", ""),
        model=args.model.strip(),
        timeout_seconds=int(os.getenv("EVAL_TIMEOUT_SECONDS", "300")),
        max_tool_rounds=int(os.getenv("EVAL_MAX_TOOL_ROUNDS", "80")),
        reasoning_effort=args.reasoning_effort.strip() or None,
    )

    date_prefix = make_date_prefix(client.model, client.reasoning_effort, completions)
    mcp_log = LOG_DIR / date_prefix / "mcp_server_stdio.log"
    mcp_dataset_dir_override = os.getenv("MCP_DATASET_DIR")
    mcp_dataset_dir = (
        Path(mcp_dataset_dir_override).expanduser()
        if mcp_dataset_dir_override
        else LOG_DIR / date_prefix / "mcp_datasets"
    )
    ground_truth = load_ground_truth()

    print(f"[Eval] Modell:        {client.model}")
    print(f"[Eval] Provider:      {client.provider}")
    print(f"[Eval] Reasoning:     {client.reasoning_effort or '(default)'}")
    print(f"[Eval] Dataset:       {dataset_path}")
    print(f"[Eval] Setups:        {setup_ids}")
    print(f"[Eval] Test-Cases:    {[tc['id'] for tc in test_cases]}")
    print(f"[Eval] Runs je Lane:  {args.runs}")
    print(f"[Eval] Workers:       {args.workers}")
    print(f"[Eval] Output-Ordner: {LOG_DIR / date_prefix}")
    print(f"[Eval] MCP-Datasets:  {mcp_dataset_dir}")
    print()

    lanes = [(s, c) for s in setup_ids for c in test_cases]
    all_runs = {}
    write_lock = threading.Lock()
    t_start = time.perf_counter()

    def run_lane(setup_key, test_case):
        runs = []
        for run_no in range(1, args.runs + 1):
            res = run_once(
                client, setup_key, test_case, dataset_path, run_no, date_prefix,
                ground_truth, mcp_log, mcp_dataset_dir,
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
                print(f"  [{setup_key}/{test_case['id']}] Run {run_no}/{args.runs} "
                      f"... {status} {acc_text} ({res['duration_ms']/1000:.1f}s)", flush=True)
                write_log(date_prefix, setup_key, test_case, runs, client, dataset_path)
        return (setup_key, test_case["id"]), runs

    if args.workers <= 1:
        for setup_key, test_case in lanes:
            key, runs = run_lane(setup_key, test_case)
            all_runs[key] = runs
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(run_lane, s, c) for s, c in lanes]
            for fut in as_completed(futures):
                key, runs = fut.result()
                all_runs[key] = runs

    elapsed = time.perf_counter() - t_start
    write_summary(date_prefix, all_runs, client, dataset_path)
    print(f"\n[Eval] Fertig in {elapsed/60:.1f} min. Logs unter: {LOG_DIR / date_prefix}")


if __name__ == "__main__":
    main()
