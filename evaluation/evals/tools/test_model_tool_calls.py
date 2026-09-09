"""
Schnelltest: Vergleicht wie gut verschiedene Modelle via MCP-Server
die Dachvolumina aller Gebäude berechnen können.

Zeigt Tool-Call-Sequenzen, Errors und ob max_tool_calls erreicht wurde.
"""

import json
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent / ".env")
client = OpenAI()

# ============================================================================
# KONFIGURATION
# ============================================================================
MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini"]
RUNS_PER_MODEL = 3
MAX_TOOL_CALLS = 50
TIMEOUT_SECONDS = 300

NGROK_URL = "https://unblossoming-harley-attendantly.ngrok-free.dev/mcp"

PROMPT = (
    "Berechne die Dachvolumina aller Gebaeude im Dataset 'evaluation.city.json' "
    "und gib nur die summierte Gesamtmenge in m3 aus."
)

INSTRUCTIONS = """
You are a helpful CityJSON expert that can answer questions and help with tasks.

WICHTIG - Befolge diese Regeln:
1. KEINE RÜCKFRAGEN: Beantworte alle Fragen direkt und vollständig. Stelle KEINE Rückfragen zur Klärung.
2. Gib immer konkrete Zahlen und Ergebnisse aus, keine theoretischen Erklärungen.
3. Gib die summierte Gesamtmenge als Zahl in m3 aus.
"""

# Ground Truth
EXPECTED_VOLUME = 407.725  # m³ (Toleranz ±50)

LOG_DIR = Path("evaluation_logs")


# ============================================================================
# HELPER
# ============================================================================
def extract_tool_calls(response) -> list[dict]:
    """Extrahiert Tool-Call-Namen und Argumente (ohne Output)."""
    resp_dict = response.model_dump(mode="json") if hasattr(response, "model_dump") else {}
    output_items = resp_dict.get("output", []) or []
    calls = []
    for item in output_items:
        if isinstance(item, dict) and item.get("type", "").endswith("_call"):
            entry = {
                "name": item.get("name", "?"),
                "arguments": item.get("arguments", ""),
            }
            if item.get("error"):
                entry["error"] = item["error"]
            status = item.get("status")
            if status and status != "completed":
                entry["status"] = status
            # Parse JSON arguments
            if isinstance(entry["arguments"], str) and entry["arguments"]:
                try:
                    entry["arguments"] = json.loads(entry["arguments"])
                except (json.JSONDecodeError, ValueError):
                    pass
            calls.append(entry)
    return calls


def run_single_test(model: str, run_num: int) -> dict:
    """Führt einen einzelnen Test-Run durch."""
    print(f"  Run {run_num}...", end=" ", flush=True)
    start = time.perf_counter()

    try:
        resp = client.responses.create(
            model=model,
            tools=[{
                "type": "mcp",
                "server_label": "cityDPC",
                "server_url": NGROK_URL,
                "require_approval": "never",
            }],
            instructions=INSTRUCTIONS,
            input=PROMPT,
            max_tool_calls=MAX_TOOL_CALLS,
        )
        duration_ms = int((time.perf_counter() - start) * 1000)
        output_text = resp.output_text or ""
        tool_calls = extract_tool_calls(resp)
        resp_dump = resp.model_dump(mode="json")
        usage = resp_dump.get("usage", {})

        # API-Status: "completed" vs "incomplete" (+ reason)
        api_status = resp_dump.get("status", "?")
        incomplete_details = resp_dump.get("incomplete_details") or {}
        stop_reason = incomplete_details.get("reason", "")

        # Parse result - versuche Zahl aus Freitext zu extrahieren
        volume = None
        # Erst JSON probieren
        try:
            parsed = json.loads(output_text)
            volume = parsed.get("roof_volume_sum_m3")
        except (json.JSONDecodeError, ValueError):
            pass
        # Dann Regex auf Zahlen im Text
        if volume is None and output_text:
            import re
            numbers = re.findall(r"(\d+[\.,]\d+)", output_text)
            for n in numbers:
                val = float(n.replace(",", "."))
                if 50 < val < 5000:  # plausible Dachvolumen-Range
                    volume = val
                    break

        is_empty = output_text == "" or output_text is None
        max_hit = len(tool_calls) >= MAX_TOOL_CALLS or (len(tool_calls) >= 20 and is_empty)
        correct = volume is not None and abs(volume - EXPECTED_VOLUME) < 50

        # Kurzübersicht
        call_names = [c["name"] for c in tool_calls]
        status = "OK" if correct else ("MAX_CALLS" if max_hit else ("EMPTY" if is_empty else "WRONG"))
        vol_str = f"{volume:.1f}" if volume is not None else "N/A"
        reason_str = f" [{stop_reason}]" if stop_reason else ""
        print(f"{status} | {len(tool_calls)} calls | {vol_str} m³ | {duration_ms/1000:.1f}s | api={api_status}{reason_str}")

        # Error-Calls anzeigen
        error_calls = [c for c in tool_calls if c.get("error")]
        if error_calls:
            for ec in error_calls:
                print(f"    ERROR in {ec['name']}: {ec['error'][:100]}")

        return {
            "run": run_num,
            "model": model,
            "duration_ms": duration_ms,
            "tool_calls_count": len(tool_calls),
            "tool_calls": tool_calls,
            "call_sequence": call_names,
            "output_text": output_text[:500],
            "volume": volume,
            "correct": correct,
            "max_tool_calls_hit": max_hit,
            "output_empty": is_empty,
            "status": status,
            "api_status": api_status,
            "stop_reason": stop_reason,
            "tokens": {
                "input": usage.get("input_tokens"),
                "output": usage.get("output_tokens"),
                "total": usage.get("total_tokens"),
            },
            "error": None,
        }

    except Exception as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        print(f"ERROR | {type(e).__name__}: {str(e)[:80]}")
        return {
            "run": run_num,
            "model": model,
            "duration_ms": duration_ms,
            "tool_calls_count": 0,
            "tool_calls": [],
            "call_sequence": [],
            "output_text": "",
            "volume": None,
            "correct": False,
            "max_tool_calls_hit": False,
            "output_empty": True,
            "status": "EXCEPTION",
            "tokens": {},
            "error": f"{type(e).__name__}: {e}",
        }


# ============================================================================
# MAIN
# ============================================================================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"{'='*70}")
    print(f"Model-Vergleich: Dachvolumen-Berechnung via MCP")
    print(f"Models: {MODELS}")
    print(f"Runs pro Model: {RUNS_PER_MODEL}")
    print(f"Max Tool Calls: {MAX_TOOL_CALLS}")
    print(f"Expected Volume: ~{EXPECTED_VOLUME} m³ (±50)")
    print(f"{'='*70}\n")

    all_results = {}

    for model in MODELS:
        print(f"\n{'─'*50}")
        print(f"Model: {model}")
        print(f"{'─'*50}")

        runs = []
        for i in range(1, RUNS_PER_MODEL + 1):
            result = run_single_test(model, i)
            runs.append(result)

        all_results[model] = runs

        # Model-Zusammenfassung
        ok_count = sum(1 for r in runs if r["correct"])
        max_hit = sum(1 for r in runs if r["max_tool_calls_hit"])
        avg_calls = sum(r["tool_calls_count"] for r in runs) / len(runs)
        avg_dur = sum(r["duration_ms"] for r in runs) / len(runs)
        print(f"\n  Summary: {ok_count}/{len(runs)} correct, "
              f"{max_hit} max_calls_hit, "
              f"avg {avg_calls:.0f} calls, "
              f"avg {avg_dur/1000:.1f}s")

    # ============================================================
    # Gesamtvergleich
    # ============================================================
    print(f"\n\n{'='*70}")
    print("GESAMTVERGLEICH")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'OK':>4} {'MaxHit':>7} {'AvgCalls':>9} {'AvgTime':>8} {'AvgVol':>10}")
    print(f"{'─'*60}")

    for model, runs in all_results.items():
        ok = sum(1 for r in runs if r["correct"])
        mh = sum(1 for r in runs if r["max_tool_calls_hit"])
        ac = sum(r["tool_calls_count"] for r in runs) / len(runs)
        at = sum(r["duration_ms"] for r in runs) / len(runs) / 1000
        vols = [r["volume"] for r in runs if r["volume"] is not None]
        av = f"{sum(vols)/len(vols):.1f}" if vols else "N/A"
        print(f"{model:<20} {ok:>2}/{len(runs):<2} {mh:>5}x   {ac:>7.0f}   {at:>6.1f}s   {av:>8}")

    # Tool-Call-Sequenzen pro Model (nur fehlgeschlagene)
    print(f"\n\n{'='*70}")
    print("FEHLGESCHLAGENE RUNS - Tool-Call-Sequenzen")
    print(f"{'='*70}")

    for model, runs in all_results.items():
        failed = [r for r in runs if not r["correct"]]
        if not failed:
            print(f"\n{model}: Alle Runs erfolgreich!")
            continue

        print(f"\n{model}: {len(failed)} fehlgeschlagene Runs")
        for r in failed:
            print(f"  Run {r['run']} [{r['status']}] ({r['tool_calls_count']} calls):")
            for i, name in enumerate(r["call_sequence"], 1):
                print(f"    {i:2d}. {name}")
            if r["error"]:
                print(f"    >>> ERROR: {r['error'][:120]}")
            if r["output_text"]:
                print(f"    >>> Output: {r['output_text'][:120]}")

    # Log speichern
    log_path = LOG_DIR / f"model_comparison_{timestamp}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "timestamp": timestamp,
                "models": MODELS,
                "runs_per_model": RUNS_PER_MODEL,
                "max_tool_calls": MAX_TOOL_CALLS,
                "expected_volume": EXPECTED_VOLUME,
                "response_schema": "none (plain text)",
            },
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n\nLog gespeichert: {log_path}")


if __name__ == "__main__":
    main()
