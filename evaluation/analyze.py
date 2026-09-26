"""
Turn evaluation logs into a per-run CSV and compute the paper's result tables.

    # Tables 2 and 3 of the paper from the published per-run results
    python evaluation/analyze.py tables

    # Your own runs: logs -> CSV -> tables
    python evaluation/analyze.py extract evaluation/logs/<run> -o my_runs.csv
    python evaluation/analyze.py tables my_runs.csv

Metric definitions (identical to the scripts used for the paper):
- tool_calls: tool calls per run; adjacent request/result duplicates are merged and,
  in the MCP setup, the bookkeeping tools list_datasets/take_snapshot are ignored.
- successful_tool_calls (STC): completed calls of a task's expected tools whose
  prerequisites (TOOL_DEPENDENCIES) completed earlier in the same run.
- TUE (tool usage efficiency, "WA" in the paper) = sum(STC) / sum(tool_calls).
- success: query/analytic tasks need accuracy 1.0; raise_building_by_id and
  add_building are judged on the saved dataset (see run_success).
- Setup averages are means of task means; "Total" is the mean over all cells.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
PAPER_CSV = ROOT / "results" / "paper_runs.csv"

SETUPS = {"mcp": "MCP", "no_tools": "No Tools", "code_interpreter": "Sandbox"}
TASKS = {
    "list_buildings": "List buildings",
    "highest_measured_height": "Highest height",
    "raise_building_by_id": "Raise building",
    "add_building": "Add building",
    "count_party_walls": "Party walls",
}
MODEL_LABELS = {
    "gpt-oss-120b": "GPT-OSS",
    "mistralai-mistral-small-4-119b": "Mistral",
    "gpt-5.4-mini": "GPT-5.4 Mini",
}

IGNORED_MCP_TOOLS = {"list_datasets", "take_snapshot"}
TOOL_DEPENDENCIES = {
    "list_buildings": {
        "load_dataset": [],
        "get_buiding_Id_list": ["load_dataset"],
    },
    "highest_measured_height": {
        "load_dataset": [],
        "get_all_buildings": ["load_dataset"],
        "number_of_buildings": ["get_all_buildings"],
        "get_buiding_Id_list": ["get_all_buildings"],
        "get_building_by_id": ["get_all_buildings"],
    },
    "raise_building_by_id": {
        "load_dataset": [],
        "get_building_by_id": ["load_dataset"],
        "enrich_building": ["get_building_by_id"],
        "save_dataset": ["enrich_building"],
    },
    "add_building": {
        "load_dataset": [],
        "create_building": ["load_dataset"],
        "enrich_building": ["create_building"],
        "save_dataset": ["enrich_building"],
    },
    "count_party_walls": {
        "load_dataset": [],
        "get_party_walls": ["load_dataset"],
    },
}

CSV_FIELDS = [
    "model", "run_folder", "setup", "test_case", "repetition",
    "input_tokens", "output_tokens", "total_tokens", "tool_calls",
    "successful_tool_calls", "duration_ms", "success", "tool_sequence",
]


# ---------------------------------------------------------------------------
# Per-run metrics from a log entry
# ---------------------------------------------------------------------------
def _name(call):
    return call.get("name") or (call.get("function") or {}).get("name")


def _args(call):
    return call.get("arguments") or (call.get("function") or {}).get("arguments")


def _norm_args(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    try:
        return json.dumps(json.loads(str(value)), sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(value).replace(" ", "")


def _has_output(call):
    return bool(call.get("output") or call.get("result") or call.get("output_preview"))


def dedupe_tool_calls(calls):
    """Merge request/result duplicate entries (as logged by streaming clients)."""
    out, i = [], 0
    while i < len(calls):
        cur = calls[i]
        if i + 1 < len(calls):
            nxt = calls[i + 1]
            if (_name(cur) == _name(nxt) and _norm_args(_args(cur)) == _norm_args(_args(nxt))
                    and not _has_output(cur) and _has_output(nxt)):
                out.append(nxt)
                i += 2
                continue
        out.append(cur)
        i += 1
    return out


def count_tool_calls(calls, setup):
    if setup == "mcp":
        calls = [c for c in calls if _name(c) not in IGNORED_MCP_TOOLS]
    return len(calls)


def count_successful_tool_calls(test_case, calls):
    deps = TOOL_DEPENDENCIES.get(test_case, {})
    completed, successful = set(), 0
    for call in calls:
        name = _name(call)
        if not name or name in IGNORED_MCP_TOOLS or name not in deps:
            continue
        if call.get("status") != "completed":
            continue
        if all(req in completed for req in deps[name]):
            successful += 1
            completed.add(name)
    return successful


def run_success(test_case, run):
    if run.get("error") or run.get("retry_exhausted"):
        return False
    checks = (run.get("state_validation") or {}).get("checks") or {}
    if test_case == "raise_building_by_id" and checks:
        return bool(checks.get("building_found") and checks.get("height_correct"))
    if test_case == "add_building" and checks:
        return all(checks.get(k) for k in (
            "building_found", "id_correct", "height_correct", "cityjson_valid",
            "citydpc_importable", "geometry_valid", "building_displayable",
        ))
    accuracy = (run.get("validation") or {}).get("accuracy")
    try:
        return float(accuracy) >= 1.0
    except (TypeError, ValueError):
        return False


def _total_tokens(tokens):
    if tokens.get("total_tokens") is not None:
        return int(tokens["total_tokens"])
    parts = [tokens.get("input_tokens"), tokens.get("output_tokens")]
    return sum(int(p) for p in parts if p is not None) if any(p is not None for p in parts) else None


def extract(run_dirs):
    rows = []
    for run_dir in map(Path, run_dirs):
        for setup in SETUPS:
            for task in TASKS:
                log_path = run_dir / setup / f"{task}.json"
                if not log_path.exists():
                    continue
                log = json.loads(log_path.read_text(encoding="utf-8"))
                model = (log.get("meta") or {}).get("model", "")
                for run in log.get("runs") or []:
                    tokens = run.get("tokens") or {}
                    calls = dedupe_tool_calls([c for c in run.get("tool_calls") or [] if isinstance(c, dict)])
                    rows.append({
                        "model": model,
                        "run_folder": run_dir.name,
                        "setup": setup,
                        "test_case": task,
                        "repetition": run.get("run"),
                        "input_tokens": tokens.get("input_tokens"),
                        "output_tokens": tokens.get("output_tokens"),
                        "total_tokens": _total_tokens(tokens),
                        "tool_calls": count_tool_calls(calls, setup),
                        "successful_tool_calls": count_successful_tool_calls(task, calls),
                        "duration_ms": run.get("duration_ms"),
                        "success": int(run_success(task, run)),
                        "tool_sequence": ";".join(f"{_name(c)}:{c.get('status')}" for c in calls),
                    })
    return rows


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(value):
    return float(value) if value not in (None, "") else None


def _avg(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def cell_metrics(rows):
    """(model, setup, task) -> dict of averaged metrics."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["setup"], r["test_case"])].append(r)
    out = {}
    for key, runs in groups.items():
        stc = sum(_num(r["successful_tool_calls"]) or 0 for r in runs)
        atc = sum(_num(r["tool_calls"]) or 0 for r in runs)
        out[key] = {
            "runs": len(runs),
            "tokens": _avg(_num(r["total_tokens"]) for r in runs),
            "tool_calls": _avg(_num(r["tool_calls"]) for r in runs),
            "time_s": _avg(_num(r["duration_ms"]) for r in runs) / 1000,
            "success": mean(int(r["success"]) for r in runs),
            "stc": stc / len(runs),
            "stc_sum": stc,
            "atc_sum": atc,
        }
    return out


def _fmt(value, kind):
    if value is None:
        return "–"
    return {"tokens": f"{value:,.0f}", "tool_calls": f"{value:.2f}", "time_s": f"{value:.1f}",
            "success": f"{value * 100:.0f}%", "pct1": f"{value * 100:.1f}"}[kind]


def table_overview(rows):
    """Paper Table 2: tokens, tool calls, time and success per model/setup/task."""
    cells = cell_metrics(rows)
    models = list(dict.fromkeys(r["model"] for r in rows))
    setups = [s for s in SETUPS if any(k[1] == s for k in cells)]
    tasks = [t for t in TASKS if any(k[2] == t for k in cells)]
    metrics = ["tokens", "tool_calls", "time_s", "success"]

    head = "| Setup | Task | " + " | ".join(
        f"{MODEL_LABELS.get(m, m)} {h}" for m in models for h in ("Tokens", "Calls", "Time (s)", "SR")) + " |"
    lines = [head, "|" + "---|" * (2 + 4 * len(models))]

    def row(setup_label, task_label, values_per_model, bold=False):
        cols = [_fmt(v.get(k) if v else None, k) for v in values_per_model for k in metrics]
        if bold:
            cols = [f"**{c}**" for c in cols]
            task_label = f"**{task_label}**"
        lines.append(f"| {setup_label} | {task_label} | " + " | ".join(cols) + " |")

    def averaged(keys):
        found = [cells[k] for k in keys if k in cells]
        return {k: _avg(c[k] for c in found) for k in metrics} if found else None

    for setup in setups:
        for i, task in enumerate(tasks):
            row(SETUPS[setup] if i == 0 else "", TASKS[task], [cells.get((m, setup, task)) for m in models])
        row("", "Setup avg.", [averaged([(m, setup, t) for t in tasks]) for m in models], bold=True)
    row("**Total**", "All tasks", [averaged([(m, s, t) for s in setups for t in tasks]) for m in models], bold=True)
    return "\n".join(lines)


def table_tool_usage(rows, setup="mcp"):
    """Paper Table 3: successful vs. actual tool calls in the MCP setup."""
    cells = cell_metrics([r for r in rows if r["setup"] == setup])
    models = list(dict.fromkeys(r["model"] for r in rows if r["setup"] == setup))
    tasks = [t for t in TASKS if any(k[2] == t for k in cells)]
    head = "| Task | " + " | ".join(
        f"{MODEL_LABELS.get(m, m)} {h}" for m in models for h in ("STC", "ATC", "TUE (%)", "SR (%)")) + " |"
    lines = [head, "|" + "---|" * (1 + 4 * len(models))]

    def cols(runs, stc_sum, atc_sum, succ):
        return [_fmt(stc_sum / runs, "tool_calls"), _fmt(atc_sum / runs, "tool_calls"),
                _fmt(stc_sum / atc_sum if atc_sum else 0.0, "pct1"), _fmt(succ, "pct1")]

    for task in tasks:
        out = []
        for m in models:
            c = cells.get((m, setup, task))
            out += cols(c["runs"], c["stc_sum"], c["atc_sum"], c["success"]) if c else ["–"] * 4
        lines.append(f"| {TASKS[task]} | " + " | ".join(out) + " |")
    total = []
    for m in models:
        cs = [cells[(m, setup, t)] for t in tasks if (m, setup, t) in cells]
        runs = sum(c["runs"] for c in cs)
        total += [f"**{v}**" for v in cols(runs, sum(c["stc_sum"] for c in cs), sum(c["atc_sum"] for c in cs),
                                            sum(c["success"] * c["runs"] for c in cs) / runs)]
    lines.append("| **Total** | " + " | ".join(total) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_extract = sub.add_parser("extract", help="evaluation logs -> per-run CSV")
    p_extract.add_argument("run_dirs", nargs="+", help="run folders under evaluation/logs/")
    p_extract.add_argument("-o", "--output", required=True)
    p_tables = sub.add_parser("tables", help="per-run CSV -> result tables (Markdown)")
    p_tables.add_argument("csv", nargs="?", default=str(PAPER_CSV))
    p_tables.add_argument("-o", "--output", help="also write the tables to this Markdown file")
    args = parser.parse_args()

    if args.command == "extract":
        rows = extract(args.run_dirs)
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"{len(rows)} runs -> {args.output}", file=sys.stderr)
        return

    rows = load_rows(args.csv)
    text = (
        "## Overview (paper Table 2)\n\n"
        "Averages per run; SR = success rate. Setup avg. and Total are means of task means.\n\n"
        + table_overview(rows)
        + "\n\n## Tool usage in the MCP setup (paper Table 3)\n\n"
        "STC = successful tool calls, ATC = actual tool calls (per run), "
        "TUE = tool usage efficiency (STC/ATC; \"WA\" in the paper), SR = success rate.\n\n"
        + table_tool_usage(rows)
        + "\n"
    )
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
