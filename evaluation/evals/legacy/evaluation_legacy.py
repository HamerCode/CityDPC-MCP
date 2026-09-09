"""
AI Model Evaluation System für CityJSON/CityGML Datasets.
"""

from gettext import ngettext
import json
import logging
import os
import re
import signal
import subprocess
import time
import warnings
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from datetime import datetime
from pathlib import Path
from copy import deepcopy

from dotenv import load_dotenv
try:
    from openai import OpenAI, NOT_GIVEN, APIStatusError
except Exception:  # pragma: no cover - Fallback for SDK variants
    OpenAI = None
    NOT_GIVEN = None
    APIStatusError = None
from model_client import OpenAICompatibleChatClient, env_flag, run_complete_with_mcp

# Logging/Warnings beruhigen
warnings.filterwarnings("ignore", message="Pydantic serializer warnings:*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*not planar.*")
logging.disable(logging.INFO)

from test_cases import TEST_CASES, get_test_cases_for_format
from schemas import RESPONSE_SCHEMAS, TASK_INSTRUCTIONS, BASE_INSTRUCTIONS, SETUP_SPECIFIC_INSTRUCTIONS
from validation import (
    validate_test_result,
    validate_dataset_modification,
    extract_json_from_response,
    extract_city_object_from_text,
    diff_json,
)
from dataset_manager import backup_datasets, restore_datasets, compute_file_hash
from ground_truth_generator import generate_ground_truth

# Log-Komprimierung: kompakte Tool-Traces statt voller MCP-Objekte/Schemas
MAX_LOG_TEXT_LEN = 500

load_dotenv(Path(__file__).resolve().parent / ".env")

# ============================================================================
# KONFIGURATION - Hier anpassen!
# ============================================================================
def _csv_env(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


# Welche Setups sollen getestet werden?
# Verfügbar: "mcp", "code_interpreter", "no_tools", "code_interpreter_cityjson"
SELECTED_SETUPS = _csv_env("SELECTED_SETUPS", ["mcp"])

# Timeout pro Test in Sekunden (bei Überschreitung gilt Test als nicht bestanden)
TEST_TIMEOUT_SECONDS = int(os.getenv("TEST_TIMEOUT_SECONDS", "300"))

# Wie viele Runs pro Test?
RUNS_PER_TEST = int(os.getenv("RUNS_PER_TEST", "10"))

# 1 = Runs eines Testcases sequentiell abarbeiten.
# 0 = alle Runs eines Testcases parallel starten (nur fuer kleine Smoke-Tests sinnvoll).
MAX_PARALLEL_RUNS = int(os.getenv("MAX_PARALLEL_RUNS", "1"))

# 0 = alle Testcases eines Setups parallel starten.
MAX_PARALLEL_TESTS = int(os.getenv("MAX_PARALLEL_TESTS", "0"))

# Raeumt lokale MCP-Serverprozesse auf, die nach Timeouts/Abbruechen verwaist sind.
CLEAN_STALE_MCP_SERVERS = env_flag("CLEAN_STALE_MCP_SERVERS", default=True)

# Retry-Logik für Tool-Fehler (z.B. 424: Error retrieving tool)
MAX_RETRIES = 10
RETRY_BACKOFF_SECONDS = 2

# Welche Test-Cases? (None = alle)
# Verfügbar: "list_buildings", "highest_measured_height", "roof_volume_sum",
#            "raise_building_by_id", "add_building"
SELECTED_TEST_CASES = _csv_env("SELECTED_TEST_CASES", ["list_buildings", "highest_measured_height", "roof_volume_sum", "raise_building_by_id", "add_building"])

# Model Provider:
# - "kiconnect": KI:connect Chat Completions API (default)
# - "openai_responses": legacy OpenAI Responses API with hosted tools
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "kiconnect").strip().lower()

KI_CONNECT_MODEL_ALIASES = {
    "mistral small 4 119b": "mistralai-mistral-small-4-119b",
    "mistral-small-4-119b": "mistralai-mistral-small-4-119b",
    "mistralai mistral small 4 119b": "mistralai-mistral-small-4-119b",
    "gpt oss 120b": "gpt-oss-120b",
    "gpt-oss-120b": "gpt-oss-120b",
    "e5 mistral 7b instruct": "e5-mistral-7b-instruct",
    "e5-mistral-7b-instruct": "e5-mistral-7b-instruct",
    "qwen3 embedding 8b": "qwen3-embedding-8b",
    "qwen3-embedding-8b": "qwen3-embedding-8b",
}

MCP_ALLOWED_TOOLS = {
    "list_buildings": {"load_dataset", "get_buiding_Id_list", "number_of_buildings"},
    "highest_measured_height": {"load_dataset", "get_all_buildings", "get_building_by_id"},
    "roof_volume_sum": {"load_dataset", "get_buiding_Id_list", "calculate_roof_volume_by_id"},
    "raise_building_by_id": {"load_dataset", "get_building_by_id", "enrich_building", "save_dataset"},
    "add_building": {"load_dataset", "create_building", "enrich_building", "save_dataset"},
}


def normalize_model_id(model_name: str) -> str:
    if MODEL_PROVIDER != "kiconnect":
        return model_name
    alias_key = re.sub(r"[^a-z0-9]+", " ", model_name.lower()).strip()
    dash_key = re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")
    return KI_CONNECT_MODEL_ALIASES.get(alias_key) or KI_CONNECT_MODEL_ALIASES.get(dash_key) or model_name


# Model/deployment ID. KI:connect requires API IDs, not UI display names.
MODEL = normalize_model_id(os.getenv(
    "EVAL_MODEL",
    "mistralai-mistral-small-4-119b" if MODEL_PROVIDER == "kiconnect" else "gpt-5.2",
).strip())
MODEL_SLUG = re.sub(r"[^A-Za-z0-9._-]+", "-", MODEL).strip("-") or "model"

# Reasoning Effort fuer OpenAI Responses. KI:connect nutzt optional
# KI_CONNECT_REASONING_EFFORT, weil nicht jedes Modell dieselben Werte erlaubt.
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "medium").strip()
KI_CONNECT_REASONING_EFFORT = os.getenv("KI_CONNECT_REASONING_EFFORT", "").strip().lower() or None
LOG_REASONING_EFFORT = KI_CONNECT_REASONING_EFFORT or REASONING_EFFORT


# ============================================================================
# PFADE
# ============================================================================
ROOT_DIR = Path(__file__).resolve().parent
INPUT_DIR = ROOT_DIR / "input"
LOG_DIR = ROOT_DIR / "evaluation_logs"
# Zentraler Schalter: aktives Dataset
# Beispiele: "evaluation.city.json", "evaluationBig.city.json"
DATASET_FILENAME = "evaluation.city.json"
DATASET_PATH = INPUT_DIR / DATASET_FILENAME
GROUND_TRUTH_PATH = ROOT_DIR / "ground_truth.json"
BACKUP_DIR = ROOT_DIR / "dataset_backups"

# MCP Server Dataset Path
MCP_DATASET_DIR = Path(os.getenv("MCP_DATASET_DIR", str(LOG_DIR / "mcp_datasets"))).expanduser()
MCP_DATASET_PATH = MCP_DATASET_DIR / DATASET_FILENAME

# MCP Server
MCP_SERVER_TRANSPORT = os.getenv("MCP_SERVER_TRANSPORT", "stdio").strip().lower()
MCP_SERVER_DIR = Path(os.getenv("MCP_SERVER_DIR", "/Users/tammo/Documents/UNI/BA/mcp-server"))
MCP_SERVER_SCRIPT = Path(os.getenv("MCP_SERVER_SCRIPT", str(MCP_SERVER_DIR / "cityDPC.py")))
MCP_SERVER_PYTHON = os.getenv("MCP_SERVER_PYTHON", str(MCP_SERVER_DIR / ".venv/bin/python"))
MCP_SERVER_LOG_FILE = Path(os.getenv("MCP_SERVER_LOG_FILE", str(LOG_DIR / "mcp_server_stdio.log")))
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "https://unblossoming-harley-attendantly.ngrok-free.dev/mcp")
CITYJSON_URL = "https://cj-mcp-264879243442.europe-west4.run.app/mcp"

# KI:connect API
KI_CONNECT_API_URL = os.getenv("KI_CONNECT_API_URL", "https://chat.kiconnect.nrw/api/v1/chat/completions")
KI_CONNECT_API_KEY = os.getenv("KI_CONNECT_API_KEY") or os.getenv("KICONNECT_API_KEY")
KI_CONNECT_ENABLE_JSON_SCHEMA = env_flag("KI_CONNECT_ENABLE_JSON_SCHEMA", default=False)
KI_CONNECT_VALIDATE_MODEL = env_flag("KI_CONNECT_VALIDATE_MODEL", default=True)

client = OpenAI() if MODEL_PROVIDER == "openai_responses" and OpenAI else None
chat_client: OpenAICompatibleChatClient | None = None
_DATASET_FILE_ID_CACHE: dict[str, str] = {}
_DATASET_FILE_ID_LOCK = threading.Lock()


def is_kiconnect_provider() -> bool:
    return MODEL_PROVIDER in {"kiconnect", "ki_connect", "chat_completions", "openai_compatible"}


def get_chat_client() -> OpenAICompatibleChatClient:
    global chat_client
    if chat_client is None:
        chat_client = OpenAICompatibleChatClient(
            api_url=KI_CONNECT_API_URL,
            api_key=KI_CONNECT_API_KEY or "",
            model=MODEL,
            timeout_seconds=TEST_TIMEOUT_SECONDS,
            supports_json_schema=KI_CONNECT_ENABLE_JSON_SCHEMA,
            reasoning_effort=KI_CONNECT_REASONING_EFFORT,
        )
    return chat_client


def get_mcp_transport():
    if MCP_SERVER_TRANSPORT in {"http", "https", "streamable_http", "sse"}:
        return MCP_SERVER_URL
    if MCP_SERVER_TRANSPORT == "stdio":
        from fastmcp.client.transports import PythonStdioTransport

        if not MCP_SERVER_SCRIPT.exists():
            raise FileNotFoundError(f"MCP_SERVER_SCRIPT nicht gefunden: {MCP_SERVER_SCRIPT}")
        if not Path(MCP_SERVER_PYTHON).exists():
            raise FileNotFoundError(f"MCP_SERVER_PYTHON nicht gefunden: {MCP_SERVER_PYTHON}")
        MCP_SERVER_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["MCP_DATASET_DIR"] = str(MCP_DATASET_DIR)
        return PythonStdioTransport(
            script_path=MCP_SERVER_SCRIPT,
            env=env,
            cwd=str(MCP_SERVER_DIR),
            python_cmd=MCP_SERVER_PYTHON,
            keep_alive=False,
            log_file=MCP_SERVER_LOG_FILE,
        )
    raise ValueError(
        f"Unbekannter MCP_SERVER_TRANSPORT: {MCP_SERVER_TRANSPORT}. "
        "Erlaubt: 'stdio' oder 'http'."
    )


def cleanup_stale_mcp_servers(reason: str) -> int:
    """Kill orphaned local stdio MCP server processes from earlier interrupted runs."""
    if not CLEAN_STALE_MCP_SERVERS or MCP_SERVER_TRANSPORT != "stdio":
        return 0

    script_path = str(MCP_SERVER_SCRIPT)
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,command="],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return 0

    stale_pids: list[int] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        pid_raw, ppid_raw, command = parts
        try:
            pid = int(pid_raw)
            ppid = int(ppid_raw)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        if ppid == 1 and script_path in command:
            stale_pids.append(pid)

    for pid in stale_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass

    if stale_pids:
        time.sleep(0.5)
        for pid in stale_pids:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            except PermissionError:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                pass
        print(f"[Evaluation] Bereinigt: {len(stale_pids)} verwaiste MCP-Serverprozesse ({reason})")

    return len(stale_pids)


def require_openai_responses(setup_key: str):
    if MODEL_PROVIDER != "openai_responses" or client is None:
        raise RuntimeError(
            f"Setup '{setup_key}' benoetigt OpenAI Responses hosted tools. "
            "Mit MODEL_PROVIDER=kiconnect sind aktuell 'mcp' und 'no_tools' nutzbar."
        )


# ============================================================================
# SETUP FUNKTIONEN
# ============================================================================
def setup_mcp(prompt: str, test_case_id: str) -> tuple[str, object]:
    """MCP Server Setup"""
    schema = RESPONSE_SCHEMAS.get(test_case_id)
    instructions = build_instructions("mcp", test_case_id)

    if is_kiconnect_provider():
        resp = run_complete_with_mcp(
            get_chat_client(),
            server_transport=get_mcp_transport(),
            instructions=instructions,
            prompt=prompt,
            schema=schema,
            allowed_tool_names=MCP_ALLOWED_TOOLS.get(test_case_id),
        )
        return resp.output_text, resp

    require_openai_responses("mcp")
    resp = client.responses.create(
        model=MODEL,
        tools=[{"type": "mcp", "server_label": "cityDPC", "server_url": MCP_SERVER_URL, "require_approval": "never"}],
        instructions=instructions,
        input=prompt,
        text={"format": schema} if schema else NOT_GIVEN,
        reasoning={"effort": REASONING_EFFORT} if REASONING_EFFORT else NOT_GIVEN,
    )
    return resp.output_text, resp


def setup_code_interpreter(prompt: str, test_case_id: str) -> tuple[str, object]:
    """Code Interpreter Setup - Dataset wird im Prompt mitgegeben"""
    require_openai_responses("code_interpreter")
    schema = RESPONSE_SCHEMAS.get(test_case_id)
    instructions = build_instructions("code_interpreter", test_case_id)

    # Bei JSON/GML/XML als Inline-Text arbeiten: verhindert File-Type-400 beim Upload.
    suffix = DATASET_PATH.suffix.lower()
    inline_only = suffix in {".json", ".gml", ".xml"}

    if inline_only:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            dataset_content = f.read()
        block_lang = "json" if suffix == ".json" else "xml"
        full_prompt = (
            f"{prompt}\n\n"
            f"Hier ist das Dataset '{DATASET_FILENAME}':\n"
            f"```{block_lang}\n{dataset_content}\n```"
        )
        input_payload = full_prompt
    else:
        # Fallback-Strategie für andere Dateitypen: Upload versuchen
        try:
            dataset_file_id = ensure_dataset_file_uploaded(DATASET_PATH)
            input_payload = [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_text", "text": f"Nutze die angehängte Datei '{DATASET_FILENAME}' als einziges Dataset."},
                        {"type": "input_file", "file_id": dataset_file_id},
                    ],
                }
            ]
        except Exception:
            # Wenn Upload fehlschlägt, robust auf Inline zurückfallen
            with open(DATASET_PATH, "r", encoding="utf-8") as f:
                dataset_content = f.read()
            full_prompt = (
                f"{prompt}\n\n"
                f"Hier ist das Dataset '{DATASET_FILENAME}':\n"
                f"```text\n{dataset_content}\n```"
            )
            input_payload = full_prompt

    resp = client.responses.create(
        model=MODEL,
        tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
        instructions=instructions,
        input=input_payload,
        text={"format": schema} if schema else NOT_GIVEN,
        reasoning={"effort": REASONING_EFFORT} if REASONING_EFFORT else NOT_GIVEN,
    )
    return resp.output_text, resp


def setup_no_tools(prompt: str, test_case_id: str) -> tuple[str, object]:
    """Ohne Tools Setup - Dataset wird als Inline-JSON im Prompt mitgegeben."""
    schema = RESPONSE_SCHEMAS.get(test_case_id)
    instructions = build_instructions("no_tools", test_case_id)
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset_content = f.read()
    is_gml = DATASET_PATH.suffix.lower() == ".gml"
    block_lang = "xml" if is_gml else "json"
    dataset_label = "CityGML XML" if is_gml else "CityJSON"
    full_prompt = (
        f"{prompt}\n\n"
        f"Hier ist das Dataset im {dataset_label} Format:\n"
        f"```{block_lang}\n{dataset_content}\n```"
    )

    if is_kiconnect_provider():
        resp = get_chat_client().complete(
            instructions=instructions,
            prompt=full_prompt,
            schema=schema,
        )
        return resp.output_text, resp

    require_openai_responses("no_tools")
    resp = client.responses.create(
        model=MODEL,
        instructions=instructions,
        input=full_prompt,
        text={"format": schema} if schema else NOT_GIVEN,
        reasoning={"effort": REASONING_EFFORT} if REASONING_EFFORT else NOT_GIVEN,
    )
    return resp.output_text, resp


def setup_code_interpreter_cityjson(prompt: str, test_case_id: str) -> tuple[str, object]:
    """Code Interpreter + CityJSON MCP Setup - Dataset wird NICHT im Prompt mitgegeben"""
    require_openai_responses("code_interpreter_cityjson")
    schema = RESPONSE_SCHEMAS.get(test_case_id)
    instructions = build_instructions("code_interpreter_cityjson", test_case_id)
    full_prompt = f"{prompt}\n\nDataset-Pfad (nur Pfad, kein Inhalt): {DATASET_PATH}"

    resp = client.responses.create(
        model=MODEL,
        tools=[
            {"type": "code_interpreter", "container": {"type": "auto"}},
            {"type": "mcp", "server_label": "cityjson-spec", "server_url": CITYJSON_URL, "require_approval": "never"},
        ],
        instructions=instructions,
        input=full_prompt,
        text={"format": schema} if schema else NOT_GIVEN,
        reasoning={"effort": REASONING_EFFORT} if REASONING_EFFORT else NOT_GIVEN,
    )
    return resp.output_text, resp


SETUPS = {
    "mcp": ("MCP Server", setup_mcp),
    "code_interpreter": ("Code Interpreter", setup_code_interpreter),
    "no_tools": ("No Tools", setup_no_tools),
    "code_interpreter_cityjson": ("Code Interpreter + CityJSON", setup_code_interpreter_cityjson),
}


def ensure_dataset_file_uploaded(dataset_path: Path) -> str:
    """Lädt Dataset als OpenAI File hoch und cached file_id per Dateihash."""
    with _DATASET_FILE_ID_LOCK:
        file_hash = compute_file_hash(dataset_path)
        if not file_hash:
            raise FileNotFoundError(f"Dataset nicht gefunden oder nicht lesbar: {dataset_path}")
        cached = _DATASET_FILE_ID_CACHE.get(file_hash)
        if cached:
            return cached

        with dataset_path.open("rb") as f:
            uploaded = client.files.create(file=f, purpose="user_data")
        file_id = getattr(uploaded, "id", None)
        if not file_id:
            raise RuntimeError("Datei-Upload fehlgeschlagen: keine file_id")
        _DATASET_FILE_ID_CACHE[file_hash] = file_id
        return file_id


# ============================================================================
# HELPER
# ============================================================================
def load_ground_truth() -> dict:
    """Lädt Ground Truth aus JSON (inkl. Specs)."""
    if not GROUND_TRUTH_PATH.exists():
        return {"solutions": {}}
    with GROUND_TRUTH_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "solutions" not in data and isinstance(data, dict):
        return {"solutions": data}
    return data


def build_instructions(setup_key: str, test_case_id: str) -> str:
    """Baut Instructions inkl. setup-spezifischer Hinweise."""
    instructions = BASE_INSTRUCTIONS + "\n\n" + TASK_INSTRUCTIONS.get(test_case_id, "")
    setup_specific = SETUP_SPECIFIC_INSTRUCTIONS.get(setup_key, {}).get(test_case_id)
    if setup_specific:
        instructions += "\n\n" + setup_specific
    return instructions


def apply_dataset_filename_to_test_cases(test_cases: list[dict], dataset_filename: str) -> list[dict]:
    """
    Ersetzt den Dataset-Dateinamen zentral in allen Test-Case-Prompts.
    So kann DATASET_FILENAME umgestellt werden, ohne test_cases.py anzufassen.
    """
    patched = deepcopy(test_cases)
    for tc in patched:
        prompt = tc.get("prompt")
        if isinstance(prompt, str):
            tc["prompt"] = prompt.replace("evaluation.city.json", dataset_filename)
    return patched


def extract_raise_spec_from_prompt(prompt: str) -> dict | None:
    """
    Extrahiert target_building_id sowie alte/neue Höhe aus dem Prompt.
    Fallback für Fälle, in denen Ground-Truth-Spec und Prompt auseinanderlaufen.
    """
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
    increase = float(inc_match.group(1)) if inc_match else round(new_h - old_h, 3)

    return {
        "target_building_id": id_match.group(1),
        "old_height": old_h,
        "increase_by": increase,
        "new_height": new_h,
    }


def _parse_json_pointer(path: str) -> list[str] | None:
    if path == "":
        return []
    if not path.startswith("/"):
        return None
    parts = path.lstrip("/").split("/")
    tokens = []
    for part in parts:
        tokens.append(part.replace("~1", "/").replace("~0", "~"))
    return tokens


def _get_parent_and_key(doc: object, tokens: list[str]) -> tuple[object, str | int] | tuple[None, None]:
    current = doc
    for token in tokens[:-1]:
        if isinstance(current, dict):
            if token not in current:
                return None, None
            current = current[token]
        elif isinstance(current, list):
            try:
                index = int(token)
            except ValueError:
                return None, None
            if index < 0 or index >= len(current):
                return None, None
            current = current[index]
        else:
            return None, None

    last = tokens[-1] if tokens else None
    if last is None:
        return None, None
    if isinstance(current, dict):
        return current, last
    if isinstance(current, list):
        if last == "-":
            return current, last
        try:
            index = int(last)
        except ValueError:
            return None, None
        return current, index
    return None, None


def apply_json_patch(doc: dict, patch_ops: list[dict]) -> dict:
    """
    Wendet eine JSON-Patch Liste auf das Dokument an.
    Unterstützt: add, replace, remove.
    """
    result = {"applied": False, "errors": []}
    if not isinstance(patch_ops, list):
        result["errors"].append("patch_ops_not_list")
        return result

    for op in patch_ops:
        if not isinstance(op, dict):
            result["errors"].append("patch_op_not_object")
            continue
        op_type = op.get("op")
        path = op.get("path")
        if not op_type or path is None:
            result["errors"].append("missing_op_or_path")
            continue

        tokens = _parse_json_pointer(path)
        if tokens is None:
            result["errors"].append(f"invalid_path:{path}")
            continue

        parent, key = _get_parent_and_key(doc, tokens)
        if parent is None:
            result["errors"].append(f"path_not_found:{path}")
            continue

        if op_type == "add":
            value = op.get("value")
            if isinstance(parent, dict):
                parent[key] = value
            elif isinstance(parent, list):
                if key == "-":
                    parent.append(value)
                elif isinstance(key, int):
                    if key < 0 or key > len(parent):
                        result["errors"].append(f"index_out_of_range:{path}")
                        continue
                    parent.insert(key, value)
            else:
                result["errors"].append(f"invalid_parent:{path}")
        elif op_type == "replace":
            value = op.get("value")
            if isinstance(parent, dict):
                if key not in parent:
                    result["errors"].append(f"replace_missing_key:{path}")
                    continue
                parent[key] = value
            elif isinstance(parent, list) and isinstance(key, int):
                if key < 0 or key >= len(parent):
                    result["errors"].append(f"index_out_of_range:{path}")
                    continue
                parent[key] = value
            else:
                result["errors"].append(f"invalid_parent:{path}")
        elif op_type == "remove":
            if isinstance(parent, dict):
                if key not in parent:
                    result["errors"].append(f"remove_missing_key:{path}")
                    continue
                del parent[key]
            elif isinstance(parent, list) and isinstance(key, int):
                if key < 0 or key >= len(parent):
                    result["errors"].append(f"index_out_of_range:{path}")
                    continue
                del parent[key]
            else:
                result["errors"].append(f"invalid_parent:{path}")
        else:
            result["errors"].append(f"unsupported_op:{op_type}")

    result["applied"] = len(result["errors"]) == 0
    return result


def extract_patch_from_output(output_text: str) -> list[dict] | None:
    parsed = extract_json_from_response(output_text)
    if parsed is None:
        return None
    if isinstance(parsed, dict) and isinstance(parsed.get("patch"), list):
        return parsed.get("patch")
    if isinstance(parsed, list):
        return parsed
    return None


def extract_ci_file_id(output_items: list[dict], filename: str) -> str | None:
    """Extrahiert file_id aus Code-Interpreter Output Items."""
    for item in output_items:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or []
        if isinstance(content, list):
            for c in content:
                annotations = c.get("annotations") if isinstance(c, dict) else None
                if not annotations:
                    continue
                for ann in annotations:
                    if ann.get("filename") == filename and ann.get("file_id"):
                        return ann.get("file_id")
    return None


def sync_ci_file_to_local(file_id: str, target_path: Path) -> bool:
    """Lädt eine CI-Datei über file_id und schreibt sie lokal."""
    if not file_id:
        return False
    try:
        content = client.files.content(file_id)
        data = content.read() if hasattr(content, "read") else content
        if hasattr(data, "decode"):
            data = data.decode("utf-8")
        target_path.write_text(data, encoding="utf-8")
        return True
    except Exception:
        return False


def apply_ci_fallback_state_change(
    test_id: str,
    output_text: str,
    dataset_path: Path,
    expected_raise_spec: dict | None,
    expected_building_spec: dict | None,
) -> dict:
    """Fallback: wende CI-Output minimal auf lokales Dataset an."""
    result = {"applied": False, "reason": None}
    if not dataset_path.exists():
        result["reason"] = "dataset_missing"
        return result

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception as e:
        result["reason"] = f"dataset_read_error:{type(e).__name__}"
        return result

    city_objects = dataset.get("CityObjects", {})

    if test_id == "raise_building_by_id":
        target_id = expected_raise_spec.get("target_building_id") if expected_raise_spec else None
        if not target_id or target_id not in city_objects:
            result["reason"] = "target_missing"
            return result

        extracted = extract_city_object_from_text(output_text) or {}
        attrs = extracted.get("extracted_object", {}).get("attributes", {})
        new_height = attrs.get("measuredHeight")
        if new_height is None and expected_raise_spec:
            new_height = expected_raise_spec.get("new_height")
        if new_height is None:
            result["reason"] = "new_height_missing"
            return result

        city_objects[target_id].setdefault("attributes", {})["measuredHeight"] = new_height
        dataset["CityObjects"] = city_objects

        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        result.update({"applied": True, "target_id": target_id, "new_height": new_height})
        return result

    if test_id == "add_building":
        extracted = extract_city_object_from_text(output_text) or {}
        city_obj = extracted.get("extracted_object")
        building_id = extracted.get("building_id")
        if not building_id and expected_building_spec:
            building_id = expected_building_spec.get("id")
        if not building_id or not city_obj:
            result["reason"] = "city_object_missing"
            return result

        city_objects[building_id] = city_obj
        dataset["CityObjects"] = city_objects
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        result.update({"applied": True, "building_id": building_id})
        return result

    result["reason"] = "unsupported_test_id"
    return result


def extract_from_response(response) -> dict:
    """Extrahiert relevante Daten aus API Response"""
    if response is None:
        return {"usage": None, "tool_calls": [], "output_items": [], "raw_output_items": []}

    def _truncate(value, limit: int = MAX_LOG_TEXT_LEN):
        text = str(value)
        return text if len(text) <= limit else text[:limit] + "..."

    def _compact(value, depth: int = 0):
        if depth >= 2:
            return _truncate(value, 120)
        if isinstance(value, dict):
            out = {}
            for i, (k, v) in enumerate(value.items()):
                if i >= 20:
                    out["..."] = f"{len(value) - 20} more keys"
                    break
                out[str(k)] = _compact(v, depth + 1)
            return out
        if isinstance(value, list):
            items = [_compact(v, depth + 1) for v in value[:20]]
            if len(value) > 20:
                items.append(f"... {len(value) - 20} more items")
            return items
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value if not isinstance(value, str) else _truncate(value, 200)
        return _truncate(value, 120)

    def _safe_json(val):
        if isinstance(val, str):
            try:
                return _compact(json.loads(val))
            except Exception:
                return _truncate(val)
        return _compact(val)

    def _summarize_tool_call(item: dict, idx: int) -> dict:
        error_obj = item.get("error")
        error_msg = None
        if isinstance(error_obj, dict):
            error_msg = error_obj.get("message") or error_obj.get("code") or _truncate(error_obj)
        elif error_obj:
            error_msg = _truncate(error_obj)

        output_val = item.get("output")
        if isinstance(output_val, str):
            output_preview = _truncate(output_val)
        elif output_val is not None:
            output_preview = _truncate(json.dumps(output_val, ensure_ascii=False))
        else:
            output_preview = None

        return {
            "step": idx,
            "type": item.get("type"),
            "id": item.get("id"),
            "name": item.get("name"),
            "server_label": item.get("server_label"),
            "status": item.get("status"),
            "arguments": _safe_json(item.get("arguments")),
            "error": error_msg,
            "output_preview": output_preview,
        }

    resp_dict = response.model_dump(mode="json") if hasattr(response, "model_dump") else {}
    output_items = resp_dict.get("output", []) or []

    # Tool Calls extrahieren und kompakt loggen
    raw_tool_calls = []
    for item in output_items:
        if isinstance(item, dict):
            if item.get("type", "").endswith("_call"):
                raw_tool_calls.append(item)
            for call in item.get("tool_calls", []):
                if isinstance(call, dict):
                    raw_tool_calls.append(call)

    tool_calls = [_summarize_tool_call(call, i + 1) for i, call in enumerate(raw_tool_calls)]

    output_items_summary = []
    for i, item in enumerate(output_items, start=1):
        if not isinstance(item, dict):
            continue
        output_items_summary.append({
            "step": i,
            "type": item.get("type"),
            "id": item.get("id"),
            "status": item.get("status"),
            "role": item.get("role"),
            "content_preview": _truncate(item.get("content")),
        })

    return {
        "usage": resp_dict.get("usage"),
        "tool_calls": tool_calls,
        "output_items": output_items_summary,
        "raw_output_items": output_items,
        "response_dict": resp_dict,
    }


def write_log(log_path: Path, data: dict):
    """Schreibt Log-Datei"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_log(log_path: Path) -> dict | None:
    """Lädt existierende Log-Datei"""
    if not log_path.exists():
        return None
    with log_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset_snapshot(dataset_path: Path, snapshot_path: Path) -> bool:
    """Speichert eine Dataset-Datei als Snapshot."""
    if not dataset_path.exists():
        return False
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset_path, snapshot_path)
    return True


def ensure_working_dataset(base_path: Path, working_path: Path, force_reset: bool = False) -> bool:
    """Stellt sicher, dass ein Setup-separates Dataset existiert (kopiert bei Bedarf)."""
    if force_reset or not working_path.exists():
        working_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base_path, working_path)
        return True
    return False


# ============================================================================
# MAIN EVALUATION
# ============================================================================
def run_evaluation(
    test_case_ids: list[str] | None = None,
    setup_ids: list[str] | None = None,
    runs_per_test: int | None = None,
    append_to_logs: bool = False,
):
    """Führt die Evaluation durch"""
    started_at = datetime.now()
    effort_suffix = f"_{LOG_REASONING_EFFORT}" if LOG_REASONING_EFFORT else ""

    # Parameter priorisieren: Argumente > Globale Konstanten
    setups_to_run = setup_ids if setup_ids is not None else SELECTED_SETUPS
    runs_count = runs_per_test if runs_per_test is not None else RUNS_PER_TEST
    test_cases_to_run_ids = test_case_ids if test_case_ids is not None else SELECTED_TEST_CASES

    if is_kiconnect_provider():
        unsupported_setups = [sk for sk in setups_to_run if sk not in {"mcp", "no_tools"}]
        if unsupported_setups:
            raise ValueError(
                "KI:Connect unterstuetzt in diesem Runner nur die Setups "
                f"'mcp' und 'no_tools'. Nicht verfuegbar: {unsupported_setups}. "
                "OpenAI Code Interpreter ist ein OpenAI-Responses-hosted Tool und "
                "hat bei KI:Connect kein direktes Gegenstueck."
            )
        if KI_CONNECT_VALIDATE_MODEL:
            get_chat_client().validate_model()

    if append_to_logs and LOG_DIR.exists():
        # Suche nach dem neuesten Log-Verzeichnis für dieses Modell
        pattern = f"*_{MODEL_SLUG}{effort_suffix}"
        dirs = sorted([d for d in LOG_DIR.glob(pattern) if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
        if dirs:
            date_prefix = dirs[0].name
        else:
            date_prefix = started_at.strftime("%d.%m_%H-%M") + f"_{MODEL_SLUG}{effort_suffix}"
    else:
        date_prefix = started_at.strftime("%d.%m_%H-%M") + f"_{MODEL_SLUG}{effort_suffix}"

    print(f"[Evaluation] Start: {started_at.isoformat()}")
    print(f"[Evaluation] Provider: {MODEL_PROVIDER}")
    print(f"[Evaluation] Model: {MODEL}")
    if is_kiconnect_provider():
        print(f"[Evaluation] KI:connect API: {KI_CONNECT_API_URL}")
        print(f"[Evaluation] MCP Transport: {MCP_SERVER_TRANSPORT}")
        if MCP_SERVER_TRANSPORT == "stdio":
            print(f"[Evaluation] MCP Script: {MCP_SERVER_SCRIPT}")
        else:
            print(f"[Evaluation] MCP URL: {MCP_SERVER_URL}")
    print(f"[Evaluation] Dataset: {DATASET_FILENAME}")
    print(f"[Evaluation] Setups: {setups_to_run}")
    print(f"[Evaluation] Runs pro Test: {runs_count}")
    print(f"[Evaluation] Parallel Test-Cases: {'alle' if MAX_PARALLEL_TESTS <= 0 else MAX_PARALLEL_TESTS}")
    print(f"[Evaluation] Parallel Runs je Testcase: {'alle' if MAX_PARALLEL_RUNS <= 0 else MAX_PARALLEL_RUNS}")

    cleanup_stale_mcp_servers("start")

    # Backups erstellen
    print("[Evaluation] Erstelle Backups...")
    if not MCP_DATASET_PATH.exists():
        print(f"[WARN] MCP_DATASET_PATH nicht gefunden: {MCP_DATASET_PATH}")
    backups = backup_datasets({
        "evaluation": DATASET_PATH,
        "mcp": MCP_DATASET_PATH
    }, BACKUP_DIR)

    # Ground Truth laden
    ground_truth_data = load_ground_truth()
    ground_truth = ground_truth_data.get("solutions", {})
    print(f"[Evaluation] Ground Truth: {len(ground_truth)} Test-Cases")

    # Test-Cases laden (CityJSON vs CityGML Prompt-Set)
    format_type = "citygml" if DATASET_FILENAME.lower().endswith(".gml") else "cityjson"
    test_cases = get_test_cases_for_format(format_type=format_type, only_first_three=False)
    if test_cases_to_run_ids:
        test_cases = [tc for tc in test_cases if tc["id"] in test_cases_to_run_ids]

    def run_setup_for_test(test_case: dict, setup_key: str):
        if setup_key not in SETUPS:
            print(f"  [SKIP] Unbekanntes Setup: {setup_key}")
            return

        test_id = test_case["id"]
        prompt = test_case["prompt"]
        setup_label, setup_func = SETUPS[setup_key]
        log_path = LOG_DIR / date_prefix / setup_key / f"{test_id}.json"

        # Existierende Runs laden
        existing_log = load_log(log_path)
        existing_runs = existing_log.get("runs", []) if existing_log else []

        # Working-Dataset pro Setup+Test (verhindert Parallel-Konflikte)
        working_dataset_path = DATASET_PATH
        if setup_key != "mcp":
            working_dataset_path = LOG_DIR / date_prefix / "workspaces" / setup_key / test_id / "evaluation.city.json"
            ensure_working_dataset(DATASET_PATH, working_dataset_path, force_reset=True)

        for run_num in range(1, runs_count + 1):
            global_run = len(existing_runs) + 1
            print(f"  [{setup_label}] {test_id} Run {run_num}/{runs_count}...", end=" ", flush=True)

            # Setup ausführen mit Timeout
            start_time = time.perf_counter()
            run_started_at = datetime.now().isoformat()
            error = None
            retry_exhausted = False
            retryable_error = None
            output_text = ""
            response = None
            state_validation = None
            dataset_state = {}

            is_state_change = test_case.get("category") == "state_change"
            dataset_path = MCP_DATASET_PATH if setup_key == "mcp" else working_dataset_path

            # Für state_change Tests: pre-run restore, um sauberen Zustand zu garantieren
            if is_state_change:
                if setup_key == "mcp":
                    restore_datasets(backups)
                else:
                    ensure_working_dataset(DATASET_PATH, working_dataset_path, force_reset=True)
            else:
                if setup_key != "mcp":
                    ensure_working_dataset(DATASET_PATH, working_dataset_path, force_reset=False)

            pre_dataset = None
            if is_state_change and dataset_path.exists():
                try:
                    with open(dataset_path, "r", encoding="utf-8") as f:
                        pre_dataset = json.load(f)
                except Exception:
                    pre_dataset = None

            pre_hash = compute_file_hash(dataset_path) if dataset_path.exists() else None
            dataset_state["path"] = str(dataset_path)
            dataset_state["pre_hash"] = pre_hash

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    timeout_executor = ThreadPoolExecutor(max_workers=1)
                    future = timeout_executor.submit(setup_func, prompt, test_id)
                    try:
                        output_text, response = future.result(timeout=TEST_TIMEOUT_SECONDS)
                        error = None
                        retryable_error = None
                        timeout_executor.shutdown(wait=False)
                        break
                    except FuturesTimeoutError:
                        error = f"TIMEOUT: Test überschritt {TEST_TIMEOUT_SECONDS}s Limit"
                        print(f"TIMEOUT ({TEST_TIMEOUT_SECONDS}s)", end=" ")
                        timeout_executor.shutdown(wait=False, cancel_futures=True)
                        break
                except KeyboardInterrupt:
                    print("ABGEBROCHEN")
                    restore_datasets(backups)
                    return
                except Exception as e:
                    is_tool_retrieve_error = False
                    if APIStatusError and isinstance(e, APIStatusError):
                        try:
                            is_tool_retrieve_error = e.status_code == 424
                        except Exception:
                            is_tool_retrieve_error = False
                    if "Error retrieving tool" in str(e):
                        is_tool_retrieve_error = True

                    if is_tool_retrieve_error and attempt < MAX_RETRIES:
                        retryable_error = str(e)
                        time.sleep(RETRY_BACKOFF_SECONDS)
                        continue

                    if is_tool_retrieve_error:
                        retry_exhausted = True
                        retryable_error = str(e)
                        error = None
                    else:
                        error = f"{type(e).__name__}: {e}"
                    break

            duration_ms = int((time.perf_counter() - start_time) * 1000)

            # Response verarbeiten (vor evtl. File-Sync)
            extracted = extract_from_response(response)
            usage = extracted["usage"] or {}

            expected_building_spec = ground_truth_data.get("test_building_spec")
            expected_raise_spec = ground_truth_data.get("raise_building_spec")
            if test_id == "raise_building_by_id":
                prompt_raise_spec = extract_raise_spec_from_prompt(prompt)
                if prompt_raise_spec:
                    expected_raise_spec = prompt_raise_spec
                    dataset_state["expected_raise_source"] = "prompt"
                else:
                    dataset_state["expected_raise_source"] = "ground_truth"

            # Code-Interpreter: geänderte Datei aus dem Container zurückholen
            if is_state_change and setup_key == "code_interpreter":
                file_id = extract_ci_file_id(extracted["raw_output_items"], "evaluation.city.json")
                dataset_state["ci_file_id"] = file_id
                if file_id:
                    dataset_state["ci_file_synced"] = sync_ci_file_to_local(file_id, dataset_path)
                if not dataset_state.get("ci_file_synced"):
                    dataset_state["ci_fallback"] = apply_ci_fallback_state_change(
                        test_id=test_id,
                        output_text=output_text,
                        dataset_path=dataset_path,
                        expected_raise_spec=expected_raise_spec,
                        expected_building_spec=expected_building_spec,
                    )

            # No-Tools: Patch anwenden (fairer Write-Flow)
            if is_state_change and setup_key == "no_tools":
                patch_ops = extract_patch_from_output(output_text)
                dataset_state["patch_ops_count"] = len(patch_ops) if patch_ops else 0
                if patch_ops:
                    try:
                        with open(dataset_path, "r", encoding="utf-8") as f:
                            dataset_json = json.load(f)
                        patch_result = apply_json_patch(dataset_json, patch_ops)
                        dataset_state["patch_result"] = patch_result
                        if patch_result.get("applied"):
                            with open(dataset_path, "w", encoding="utf-8") as f:
                                json.dump(dataset_json, f, ensure_ascii=False, indent=2)
                            dataset_state["patch_applied"] = True
                        else:
                            dataset_state["patch_applied"] = False
                    except Exception as e:
                        dataset_state["patch_error"] = f"{type(e).__name__}: {e}"
                        dataset_state["patch_applied"] = False
                else:
                    dataset_state["patch_applied"] = False
                    dataset_state["patch_error"] = "No patch found in output"

            post_dataset = None
            if is_state_change and dataset_path.exists():
                try:
                    with open(dataset_path, "r", encoding="utf-8") as f:
                        post_dataset = json.load(f)
                except Exception:
                    post_dataset = None

            post_hash = compute_file_hash(dataset_path) if dataset_path.exists() else None
            dataset_state["post_hash"] = post_hash
            dataset_state["changed"] = (pre_hash is not None and post_hash is not None and pre_hash != post_hash)

            # Validierung
            if retry_exhausted:
                validation = {
                    "validated": False,
                    "json_parsed": False,
                    "accuracy": None,
                    "reason": "retry_exhausted",
                }
                accuracy = None
            else:
                validation = validate_test_result(test_id, output_text, ground_truth)
                accuracy = validation.get("accuracy")

            # State-Change Validierung (nur für state_change Tests)
            if is_state_change:
                state_eval_setup = setup_key
                if setup_key == "no_tools" and dataset_state.get("patch_applied") and dataset_state.get("changed"):
                    state_eval_setup = "patched_no_tools"
                elif setup_key == "code_interpreter" and dataset_state.get("ci_fallback", {}).get("applied"):
                    state_eval_setup = "ci_fallback"
                state_validation = validate_dataset_modification(
                    test_id=test_id,
                    setup_key=state_eval_setup,
                    output_text=output_text,
                    dataset_path=dataset_path,
                    expected_building_spec=expected_building_spec,
                    expected_raise_spec=expected_raise_spec
                )
                state_validation["evaluation_mode"] = state_eval_setup
                state_validation["dataset_changed"] = dataset_state.get("changed", False)

                # Für State-Change zählt primär die nachgewiesene Dataset-Änderung,
                # auch wenn das Modell kein parsebares JSON ausgegeben hat.
                if not retry_exhausted and (
                    (validation.get("json_parsed") is False) or (validation.get("accuracy") is None)
                ):
                    state_acc = state_validation.get("accuracy")
                    validation = {
                        "validated": bool(state_validation.get("validated")),
                        "json_parsed": False,
                        "accuracy": state_acc,
                        "reason": "State-change evaluated via dataset validation fallback",
                    }
                    accuracy = state_acc

                # Diff-basierte Kontrolle (erlaubte Pfade)
                if pre_dataset is not None and post_dataset is not None:
                    changes = diff_json(pre_dataset, post_dataset)
                    allowed_prefixes = []
                    if test_id == "raise_building_by_id" and expected_raise_spec:
                        target_id = expected_raise_spec.get("target_building_id")
                        if target_id:
                            allowed_prefixes = [f"/CityObjects/{target_id}/attributes/measuredHeight"]
                    elif test_id == "add_building" and expected_building_spec:
                        new_id = expected_building_spec.get("id")
                        if new_id:
                            allowed_prefixes = [
                                f"/CityObjects/{new_id}",
                                "/vertices",
                                "/metadata/geographicalExtent"
                            ]

                    def _is_allowed(path: str) -> bool:
                        return any(path.startswith(prefix) for prefix in allowed_prefixes)

                    unexpected = [c for c in changes if not _is_allowed(c.get("path", ""))]
                    dataset_state["diff"] = {
                        "change_count": len(changes),
                        "unexpected_count": len(unexpected),
                        "allowed_prefixes": allowed_prefixes,
                        "changes": changes[:200],
                        "unexpected_changes": unexpected[:50],
                        "diff_ok": len(unexpected) == 0
                    }

            # Dataset Snapshot speichern (für alle Setups/Testcases)
            snapshot_rel = Path("datasets") / test_id / setup_key / f"run_{global_run:03d}_post.json"
            snapshot_path = LOG_DIR / date_prefix / snapshot_rel
            dataset_state["snapshot_path"] = str(snapshot_path)
            dataset_state["snapshot_saved"] = save_dataset_snapshot(dataset_path, snapshot_path)

            # Status ausgeben
            display_accuracy = accuracy
            if is_state_change and state_validation and state_validation.get("accuracy") is not None:
                display_accuracy = state_validation.get("accuracy")
            acc_str = f"{display_accuracy:.0%}" if display_accuracy is not None else "N/A"
            status = "RETRY_EXHAUSTED" if retry_exhausted else ("ERROR" if error else "OK")
            error_msg = f" - {error[:80]}" if error else ""
            print(f"{acc_str} | {duration_ms/1000:.1f}s | {status}{error_msg}")

            # Bei modifizierenden Tests: Dataset nach jedem Run zurücksetzen
            if is_state_change:
                if setup_key == "mcp":
                    restore_datasets(backups)
                else:
                    ensure_working_dataset(DATASET_PATH, working_dataset_path, force_reset=True)

            # Run-Daten
            run_data = {
                "run": global_run,
                "run_index": run_num,
                "global_run": global_run,
                "started_at": run_started_at,
                "duration_ms": duration_ms,
                "setup_key": setup_key,
                "setup": setup_label,
                "output_text": output_text,
                "tokens": {
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                },
                "tool_calls_count": len(extracted["tool_calls"]),
                "tool_calls": extracted["tool_calls"],
                "output_items": extracted["output_items"],
                "validation": validation,
                "state_validation": state_validation,
                "dataset_state": dataset_state,
                "error": error,
                "retry_exhausted": retry_exhausted,
                "retryable_error": retryable_error,
            }
            existing_runs.append(run_data)

            # Log schreiben
            log_data = {
                "meta": {
                    "created_at": existing_log.get("meta", {}).get("created_at") if existing_log else datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "log_date": started_at.date().isoformat(),
                    "model": MODEL,
                    "reasoning_effort": LOG_REASONING_EFFORT,
                    "test_case_id": test_id,
                    "setup_key": setup_key,
                    "setup": setup_label,
                    "runs_per_test": runs_count,
                    "run_count": len(existing_runs),
                },
                "test_case": {
                    "id": test_id,
                    "name": test_case.get("name", test_id),
                    "category": test_case.get("category"),
                    "prompt": prompt,
                },
                "runs": existing_runs,
            }
            write_log(log_path, log_data)

    def _dataset_name_parts(filename: str) -> tuple[str, str]:
        if filename.endswith(".city.json"):
            return filename[:-len(".city.json")], ".city.json"
        path = Path(filename)
        return path.stem, path.suffix

    def _safe_filename_part(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "run"

    def _run_dataset_filename(setup_key: str, test_id: str, global_run: int) -> str:
        base, ext = _dataset_name_parts(DATASET_FILENAME)
        parts = [
            base,
            _safe_filename_part(date_prefix),
            _safe_filename_part(setup_key),
            _safe_filename_part(test_id),
            f"run_{global_run:03d}",
        ]
        return "__".join(parts) + ext

    def run_setup_for_test(test_case: dict, setup_key: str):
        """Run all runs for one setup/testcase with configured concurrency and isolated datasets."""
        if setup_key not in SETUPS:
            print(f"  [SKIP] Unbekanntes Setup: {setup_key}")
            return

        test_id = test_case["id"]
        base_prompt = test_case["prompt"]
        setup_label, setup_func = SETUPS[setup_key]
        log_path = LOG_DIR / date_prefix / setup_key / f"{test_id}.json"

        existing_log = load_log(log_path)
        existing_runs = existing_log.get("runs", []) if existing_log else []
        existing_count = len(existing_runs)

        def prepare_dataset_for_run(global_run: int) -> tuple[Path, str, list[Path]]:
            cleanup_paths: list[Path] = []
            if setup_key == "mcp":
                run_filename = _run_dataset_filename(setup_key, test_id, global_run)
                run_path = MCP_DATASET_PATH.parent / run_filename
                ensure_working_dataset(DATASET_PATH, run_path, force_reset=True)
                cleanup_paths.append(run_path)
                return run_path, run_filename, cleanup_paths

            run_path = LOG_DIR / date_prefix / "workspaces" / setup_key / test_id / f"run_{global_run:03d}" / DATASET_FILENAME
            ensure_working_dataset(DATASET_PATH, run_path, force_reset=True)
            return run_path, DATASET_FILENAME, cleanup_paths

        def run_single(run_num: int, global_run: int) -> dict:
            print(f"  [{setup_label}] {test_id} Run {run_num}/{runs_count}...", flush=True)
            start_time = time.perf_counter()
            run_started_at = datetime.now().isoformat()
            error = None
            retry_exhausted = False
            retryable_error = None
            output_text = ""
            response = None
            state_validation = None
            dataset_state = {}
            validation = {
                "validated": False,
                "json_parsed": False,
                "accuracy": None,
                "reason": "not_run",
            }
            accuracy = None
            extracted = {"usage": None, "tool_calls": [], "output_items": [], "raw_output_items": []}
            cleanup_paths: list[Path] = []

            try:
                is_state_change = test_case.get("category") == "state_change"
                dataset_path, run_dataset_filename, cleanup_paths = prepare_dataset_for_run(global_run)
                run_prompt = base_prompt
                if setup_key == "mcp":
                    run_prompt = base_prompt.replace(DATASET_FILENAME, run_dataset_filename)

                pre_dataset = None
                if is_state_change and dataset_path.exists():
                    try:
                        with dataset_path.open("r", encoding="utf-8") as f:
                            pre_dataset = json.load(f)
                    except Exception:
                        pre_dataset = None

                pre_hash = compute_file_hash(dataset_path) if dataset_path.exists() else None
                dataset_state["path"] = str(dataset_path)
                dataset_state["filename"] = run_dataset_filename
                dataset_state["pre_hash"] = pre_hash

                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        timeout_executor = ThreadPoolExecutor(max_workers=1)
                        future = timeout_executor.submit(setup_func, run_prompt, test_id)
                        try:
                            output_text, response = future.result(timeout=TEST_TIMEOUT_SECONDS)
                            error = None
                            retryable_error = None
                            timeout_executor.shutdown(wait=False)
                            break
                        except FuturesTimeoutError:
                            error = f"TIMEOUT: Test überschritt {TEST_TIMEOUT_SECONDS}s Limit"
                            timeout_executor.shutdown(wait=False, cancel_futures=True)
                            break
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        is_tool_retrieve_error = False
                        if APIStatusError and isinstance(e, APIStatusError):
                            try:
                                is_tool_retrieve_error = e.status_code == 424
                            except Exception:
                                is_tool_retrieve_error = False
                        if "Error retrieving tool" in str(e):
                            is_tool_retrieve_error = True

                        if is_tool_retrieve_error and attempt < MAX_RETRIES:
                            retryable_error = str(e)
                            time.sleep(RETRY_BACKOFF_SECONDS)
                            continue

                        if is_tool_retrieve_error:
                            retry_exhausted = True
                            retryable_error = str(e)
                            error = None
                        else:
                            error = f"{type(e).__name__}: {e}"
                        break

                extracted = extract_from_response(response)
                usage = extracted["usage"] or {}

                expected_building_spec = ground_truth_data.get("test_building_spec")
                expected_raise_spec = ground_truth_data.get("raise_building_spec")
                if test_id == "raise_building_by_id":
                    prompt_raise_spec = extract_raise_spec_from_prompt(run_prompt)
                    if prompt_raise_spec:
                        expected_raise_spec = prompt_raise_spec
                        dataset_state["expected_raise_source"] = "prompt"
                    else:
                        dataset_state["expected_raise_source"] = "ground_truth"

                if is_state_change and setup_key == "code_interpreter":
                    file_id = extract_ci_file_id(extracted["raw_output_items"], DATASET_FILENAME)
                    dataset_state["ci_file_id"] = file_id
                    if file_id:
                        dataset_state["ci_file_synced"] = sync_ci_file_to_local(file_id, dataset_path)
                    if not dataset_state.get("ci_file_synced"):
                        dataset_state["ci_fallback"] = apply_ci_fallback_state_change(
                            test_id=test_id,
                            output_text=output_text,
                            dataset_path=dataset_path,
                            expected_raise_spec=expected_raise_spec,
                            expected_building_spec=expected_building_spec,
                        )

                if is_state_change and setup_key == "no_tools":
                    patch_ops = extract_patch_from_output(output_text)
                    dataset_state["patch_ops_count"] = len(patch_ops) if patch_ops else 0
                    if patch_ops:
                        try:
                            with dataset_path.open("r", encoding="utf-8") as f:
                                dataset_json = json.load(f)
                            patch_result = apply_json_patch(dataset_json, patch_ops)
                            dataset_state["patch_result"] = patch_result
                            if patch_result.get("applied"):
                                with dataset_path.open("w", encoding="utf-8") as f:
                                    json.dump(dataset_json, f, ensure_ascii=False, indent=2)
                                dataset_state["patch_applied"] = True
                            else:
                                dataset_state["patch_applied"] = False
                        except Exception as e:
                            dataset_state["patch_error"] = f"{type(e).__name__}: {e}"
                            dataset_state["patch_applied"] = False
                    else:
                        dataset_state["patch_applied"] = False
                        dataset_state["patch_error"] = "No patch found in output"

                post_dataset = None
                if is_state_change and dataset_path.exists():
                    try:
                        with dataset_path.open("r", encoding="utf-8") as f:
                            post_dataset = json.load(f)
                    except Exception:
                        post_dataset = None

                post_hash = compute_file_hash(dataset_path) if dataset_path.exists() else None
                dataset_state["post_hash"] = post_hash
                dataset_state["changed"] = (pre_hash is not None and post_hash is not None and pre_hash != post_hash)

                if retry_exhausted:
                    validation = {
                        "validated": False,
                        "json_parsed": False,
                        "accuracy": None,
                        "reason": "retry_exhausted",
                    }
                    accuracy = None
                else:
                    try:
                        validation = validate_test_result(test_id, output_text, ground_truth)
                        accuracy = validation.get("accuracy")
                    except Exception as e:
                        validation = {
                            "validated": False,
                            "json_parsed": False,
                            "accuracy": None,
                            "reason": f"validation_error:{type(e).__name__}: {e}",
                        }
                        accuracy = None
                        if error is None:
                            error = f"VALIDATION_{type(e).__name__}: {e}"

                if is_state_change:
                    state_eval_setup = setup_key
                    if setup_key == "no_tools" and dataset_state.get("patch_applied") and dataset_state.get("changed"):
                        state_eval_setup = "patched_no_tools"
                    elif setup_key == "code_interpreter" and dataset_state.get("ci_fallback", {}).get("applied"):
                        state_eval_setup = "ci_fallback"
                    state_validation = validate_dataset_modification(
                        test_id=test_id,
                        setup_key=state_eval_setup,
                        output_text=output_text,
                        dataset_path=dataset_path,
                        expected_building_spec=expected_building_spec,
                        expected_raise_spec=expected_raise_spec,
                    )
                    state_validation["evaluation_mode"] = state_eval_setup
                    state_validation["dataset_changed"] = dataset_state.get("changed", False)

                    if not retry_exhausted and (
                        (validation.get("json_parsed") is False) or (validation.get("accuracy") is None)
                    ):
                        state_acc = state_validation.get("accuracy")
                        validation = {
                            "validated": bool(state_validation.get("validated")),
                            "json_parsed": False,
                            "accuracy": state_acc,
                            "reason": "State-change evaluated via dataset validation fallback",
                        }
                        accuracy = state_acc

                    if pre_dataset is not None and post_dataset is not None:
                        changes = diff_json(pre_dataset, post_dataset)
                        allowed_prefixes = []
                        if test_id == "raise_building_by_id" and expected_raise_spec:
                            target_id = expected_raise_spec.get("target_building_id")
                            if target_id:
                                allowed_prefixes = [f"/CityObjects/{target_id}/attributes/measuredHeight"]
                        elif test_id == "add_building" and expected_building_spec:
                            new_id = expected_building_spec.get("id")
                            if new_id:
                                allowed_prefixes = [
                                    f"/CityObjects/{new_id}",
                                    "/vertices",
                                    "/metadata/geographicalExtent",
                                ]

                        def _is_allowed(path: str) -> bool:
                            return any(path.startswith(prefix) for prefix in allowed_prefixes)

                        unexpected = [c for c in changes if not _is_allowed(c.get("path", ""))]
                        dataset_state["diff"] = {
                            "change_count": len(changes),
                            "unexpected_count": len(unexpected),
                            "allowed_prefixes": allowed_prefixes,
                            "changes": changes[:200],
                            "unexpected_changes": unexpected[:50],
                            "diff_ok": len(unexpected) == 0,
                        }

                snapshot_rel = Path("datasets") / test_id / setup_key / f"run_{global_run:03d}_post.json"
                snapshot_path = LOG_DIR / date_prefix / snapshot_rel
                dataset_state["snapshot_path"] = str(snapshot_path)
                dataset_state["snapshot_saved"] = save_dataset_snapshot(dataset_path, snapshot_path)
                usage = usage

            except KeyboardInterrupt:
                raise
            except Exception as e:
                usage = (extracted.get("usage") if isinstance(extracted, dict) else None) or {}
                error = f"{type(e).__name__}: {e}"
                validation = {
                    "validated": False,
                    "json_parsed": False,
                    "accuracy": None,
                    "reason": error,
                }
                accuracy = None
            finally:
                for cleanup_path in cleanup_paths:
                    try:
                        cleanup_path.unlink(missing_ok=True)
                    except Exception:
                        pass

            duration_ms = int((time.perf_counter() - start_time) * 1000)
            display_accuracy = accuracy
            if is_state_change and state_validation and state_validation.get("accuracy") is not None:
                display_accuracy = state_validation.get("accuracy")
            acc_str = f"{display_accuracy:.0%}" if display_accuracy is not None else "N/A"
            status = "RETRY_EXHAUSTED" if retry_exhausted else ("ERROR" if error else "OK")
            error_msg = f" - {error[:80]}" if error else ""
            print(f"  [{setup_label}] {test_id} Run {run_num}/{runs_count}: {acc_str} | {duration_ms/1000:.1f}s | {status}{error_msg}")

            return {
                "run": global_run,
                "run_index": run_num,
                "global_run": global_run,
                "started_at": run_started_at,
                "duration_ms": duration_ms,
                "setup_key": setup_key,
                "setup": setup_label,
                "output_text": output_text,
                "tokens": {
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                },
                "tool_calls_count": len(extracted["tool_calls"]),
                "tool_calls": extracted["tool_calls"],
                "output_items": extracted["output_items"],
                "validation": validation,
                "state_validation": state_validation,
                "dataset_state": dataset_state,
                "error": error,
                "retry_exhausted": retry_exhausted,
                "retryable_error": retryable_error,
            }

        new_runs = []

        def write_current_log() -> None:
            combined_runs = existing_runs + sorted(new_runs, key=lambda item: item.get("global_run", 0))
            log_data = {
                "meta": {
                    "created_at": existing_log.get("meta", {}).get("created_at") if existing_log else datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "log_date": started_at.date().isoformat(),
                    "model": MODEL,
                    "reasoning_effort": LOG_REASONING_EFFORT,
                    "test_case_id": test_id,
                    "setup_key": setup_key,
                    "setup": setup_label,
                    "runs_per_test": runs_count,
                    "run_count": len(combined_runs),
                },
                "test_case": {
                    "id": test_id,
                    "name": test_case.get("name", test_id),
                    "category": test_case.get("category"),
                    "prompt": base_prompt,
                },
                "runs": combined_runs,
            }
            write_log(log_path, log_data)

        worker_count = runs_count if MAX_PARALLEL_RUNS <= 0 else min(runs_count, MAX_PARALLEL_RUNS)
        with ThreadPoolExecutor(max_workers=max(1, worker_count)) as run_executor:
            futures = {
                run_executor.submit(run_single, run_num, existing_count + run_num): run_num
                for run_num in range(1, runs_count + 1)
            }
            for future in as_completed(futures):
                try:
                    new_runs.append(future.result())
                    write_current_log()
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    run_num = futures[future]
                    print(f"  [FEHLER] {setup_key}/{test_id} Run {run_num}: {type(e).__name__}: {e}")
                    write_current_log()

        existing_runs.extend(sorted(new_runs, key=lambda item: item.get("global_run", 0)))
        log_data = {
            "meta": {
                "created_at": existing_log.get("meta", {}).get("created_at") if existing_log else datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "log_date": started_at.date().isoformat(),
                "model": MODEL,
                "reasoning_effort": LOG_REASONING_EFFORT,
                "test_case_id": test_id,
                "setup_key": setup_key,
                "setup": setup_label,
                "runs_per_test": runs_count,
                "run_count": len(existing_runs),
            },
            "test_case": {
                "id": test_id,
                "name": test_case.get("name", test_id),
                "category": test_case.get("category"),
                "prompt": base_prompt,
            },
            "runs": existing_runs,
        }
        write_log(log_path, log_data)

    # Setups, die sequentiell laufen müssen (shared MCP server state)
    SEQUENTIAL_SETUPS = set()

    def run_tests_for_setup(setup_key: str):
        if setup_key in SEQUENTIAL_SETUPS:
            for tc in test_cases:
                run_setup_for_test(tc, setup_key)
        else:
            worker_count = len(test_cases) if MAX_PARALLEL_TESTS <= 0 else min(len(test_cases), MAX_PARALLEL_TESTS)
            with ThreadPoolExecutor(max_workers=max(1, worker_count)) as tc_executor:
                tc_futures = [tc_executor.submit(run_setup_for_test, tc, setup_key) for tc in test_cases]
                for f in as_completed(tc_futures):
                    try:
                        f.result()
                    except Exception as e:
                        print(f"  [FEHLER] {setup_key}: {type(e).__name__}: {e}")

    # Evaluation durchführen: Setups parallel, Test-Cases parallel (außer MCP)
    with ThreadPoolExecutor(max_workers=len(setups_to_run)) as executor:
        futures = {executor.submit(run_tests_for_setup, sk): sk for sk in setups_to_run}
        for future in as_completed(futures):
            setup_key = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  [FEHLER] Setup {setup_key}: {type(e).__name__}: {e}")

    # Backups wiederherstellen
    print("\n[Evaluation] Stelle Backups wieder her...")
    results = restore_datasets(backups)
    for name, modified in results.items():
        status = "WIEDERHERGESTELLT" if modified else "unverändert"
        print(f"  {name}: {status}")

    print(f"\n[Evaluation] Fertig! Logs in: {LOG_DIR / date_prefix}")
    cleanup_stale_mcp_servers("ende")


# ============================================================================
# GROUND TRUTH GENERIEREN
# ============================================================================
def generate_ground_truth_file():
    """Generiert Ground Truth aus Dataset"""
    print("Generiere Ground Truth...")
    data = generate_ground_truth(DATASET_PATH)
    GROUND_TRUTH_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Gespeichert in: {GROUND_TRUTH_PATH}")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--ground-truth":
        generate_ground_truth_file()
    else:
        run_evaluation()
