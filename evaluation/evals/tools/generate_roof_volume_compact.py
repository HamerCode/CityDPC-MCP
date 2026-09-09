"""
Generate compact roof_volume_sum comparison artifacts for two runs:
- a short LaTeX summary table
- a compact boxplot PNG

Usage:
  python generate_roof_volume_compact.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "evaluation_tables"

RUN_MEDIUM = "07.02_15-30_gpt-5-mini_medium"
RUN_MINIMAL = "07.02_19-32_gpt-5-mini_minimal"

MEDIUM_LOG = ROOT / "evaluation_logs" / RUN_MEDIUM / "mcp" / "evaluation_07.02_11-46_gpt-5-mini_medium_roof_volume_sum_mcp.json"
MINIMAL_LOG = ROOT / "evaluation_logs" / RUN_MINIMAL / "mcp" / "roof_volume_sum.json"

MEDIUM_REVAL = ROOT / "evaluation_logs" / RUN_MEDIUM / "revalidation_report.json"
MINIMAL_REVAL = ROOT / "evaluation_logs" / RUN_MINIMAL / "revalidation_report.json"

GT_PATH = ROOT / "ground_truth.json"


def esc(s: str) -> str:
    out = str(s)
    repl = [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")]
    for a, b in repl:
        out = out.replace(a, b)
    return out


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_value(run: dict) -> float | None:
    extracted = (run.get("validation") or {}).get("extracted_data") or {}
    if isinstance(extracted, dict) and extracted.get("roof_volume_sum_m3") is not None:
        try:
            return float(extracted["roof_volume_sum_m3"])
        except Exception:
            pass
    text = run.get("output_text") or ""
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*m[³3]", text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def basic_stats(values: list[float], gt: float) -> dict:
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "mae": None,
            "rmse": None,
            "exact_hits": 0,
            "tol_hits": 0,
        }
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var)
    abs_err = [abs(v - gt) for v in values]
    mae = sum(abs_err) / n
    rmse = math.sqrt(sum((v - gt) ** 2 for v in values) / n)
    exact_hits = sum(1 for v in values if abs(v - gt) < 1e-3)
    tol_hits = sum(1 for v in values if abs(v - gt) <= 0.1)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "mae": mae,
        "rmse": rmse,
        "exact_hits": exact_hits,
        "tol_hits": tol_hits,
    }


def fmt(v: float | None, d: int = 3) -> str:
    if v is None:
        return "--"
    return f"{v:.{d}f}"


def load_success_rate(reval_path: Path) -> tuple[int, int, float]:
    j = load_json(reval_path)
    tc = ((j.get("summary") or {}).get("mcp") or {}).get("roof_volume_sum", {})
    ok = int(tc.get("correct", 0) or 0)
    runs = int(tc.get("runs", 0) or 0)
    rate = (100.0 * ok / runs) if runs else 0.0
    return ok, runs, rate


def build_compact_table(
    gt: float,
    med_values: list[float],
    min_values: list[float],
    med_calls: list[int],
    min_calls: list[int],
    med_sr: tuple[int, int, float],
    min_sr: tuple[int, int, float],
) -> str:
    med = basic_stats(med_values, gt)
    mn = basic_stats(min_values, gt)
    med_calls_mean = (sum(med_calls) / len(med_calls)) if med_calls else 0.0
    min_calls_mean = (sum(min_calls) / len(min_calls)) if min_calls else 0.0

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\caption{roof\_volume\_sum: Kompakter Vergleich (gpt-5 medium vs. gpt-5 minimal, MCP)}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Kennzahl & gpt-5 medium & gpt-5 minimal \\",
        r"\midrule",
        rf"Ground Truth (m$^3$) & \multicolumn{{2}}{{c}}{{{fmt(gt)}}} \\",
        rf"Runs & {med['n']} & {mn['n']} \\",
        rf"\textbf{{Success Rate}} & \textbf{{{med_sr[0]}/{med_sr[1]} ({med_sr[2]:.1f}\%)}} & \textbf{{{min_sr[0]}/{min_sr[1]} ({min_sr[2]:.1f}\%)}} \\",
        rf"Mittelwert Ergebnis (m$^3$) & {fmt(med['mean'])} & {fmt(mn['mean'])} \\",
        rf"StdAbw Ergebnis (m$^3$) & {fmt(med['std'])} & {fmt(mn['std'])} \\",
        rf"Min / Max (m$^3$) & {fmt(med['min'])} / {fmt(med['max'])} & {fmt(mn['min'])} / {fmt(mn['max'])} \\",
        rf"MAE zu Ground Truth (m$^3$) & {fmt(med['mae'])} & {fmt(mn['mae'])} \\",
        rf"RMSE zu Ground Truth (m$^3$) & {fmt(med['rmse'])} & {fmt(mn['rmse'])} \\",
        rf"Exakte Treffer (±0.001) & {med['exact_hits']}/{med['n']} & {mn['exact_hits']}/{mn['n']} \\",
        rf"Treffer (±0.1 m$^3$) & {med['tol_hits']}/{med['n']} & {mn['tol_hits']}/{mn['n']} \\",
        rf"Ø Tool Calls & {med_calls_mean:.2f} & {min_calls_mean:.2f} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def build_plot(gt: float, med_values: list[float], min_values: list[float], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.boxplot([med_values, min_values], labels=["gpt-5 medium", "gpt-5 minimal"], showmeans=True)
    ax.axhline(gt, color="red", linestyle="--", linewidth=1.2, label=f"Ground Truth: {gt:.3f}")
    ax.set_ylabel("roof_volume_sum (m³)")
    ax.set_title("roof_volume_sum Verteilung pro Modell")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gt = ((load_json(GT_PATH).get("solutions") or {}).get("roof_volume_sum") or {}).get("output", {}).get("roof_volume_sum_m3")
    gt = float(gt)

    med_runs = (load_json(MEDIUM_LOG).get("runs") or [])[:20]
    min_runs = (load_json(MINIMAL_LOG).get("runs") or [])[:20]

    med_values = [v for v in (extract_value(r) for r in med_runs) if v is not None]
    min_values = [v for v in (extract_value(r) for r in min_runs) if v is not None]

    med_calls = [int(r.get("tool_calls_count", 0) or 0) for r in med_runs]
    min_calls = [int(r.get("tool_calls_count", 0) or 0) for r in min_runs]

    med_sr = load_success_rate(MEDIUM_REVAL)
    min_sr = load_success_rate(MINIMAL_REVAL)

    out_tex = OUT_DIR / "roof_volume_sum_compare_compact.tex"
    out_png = OUT_DIR / "roof_volume_sum_compare_compact.png"

    out_tex.write_text(
        build_compact_table(
            gt=gt,
            med_values=med_values,
            min_values=min_values,
            med_calls=med_calls,
            min_calls=min_calls,
            med_sr=med_sr,
            min_sr=min_sr,
        ),
        encoding="utf-8",
    )

    build_plot(gt=gt, med_values=med_values, min_values=min_values, out_png=out_png)

    print(f"Generated: {out_tex}")
    print(f"Generated: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
