"""
Interactive GUI for selecting specific evaluation runs and generating LaTeX tables.

Run with:
    streamlit run evals/tools/generate_tables_gui.py
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from . import generate_tables as gt


LOG_DIR = Path("evaluation_logs")
TABLE_DIR = Path("evaluation_tables")
SKIP_DIRS = {"datasets", "workspaces"}


def _integrity_summary(checks: dict[str, Any]) -> str:
    if not checks:
        return "--"
    labels = [
        ("cityjson_valid", "CJ"),
        ("citydpc_importable", "Imp"),
        ("geometry_valid", "Geo"),
        ("building_displayable", "Disp"),
    ]
    parts: list[str] = []
    for key, short in labels:
        value = checks.get(key)
        if value is True:
            parts.append(f"{short}:ok")
        elif value is False:
            parts.append(f"{short}:fail")
    return " | ".join(parts) if parts else "--"


def _success_label(run: dict[str, Any], test_case_id: str, ground_truth: dict[str, Any]) -> str:
    if run.get("error") or run.get("retry_exhausted"):
        return "Error"
    is_correct = gt._is_run_correct(test_case_id, run, ground_truth)
    if is_correct is True:
        return "True"
    if is_correct is False:
        return "False"
    return "N/A"


@st.cache_data(show_spinner=False)
def discover_runs(log_dir: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(log_dir)
    ground_truth = gt.load_ground_truth()
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}

    if not root.exists():
        return pd.DataFrame(), {}

    date_folders = sorted(
        [p for p in root.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for date_folder in date_folders:
        for setup_dir in sorted(date_folder.iterdir()):
            if not setup_dir.is_dir() or setup_dir.name in SKIP_DIRS:
                continue

            for log_file in sorted(setup_dir.glob("*.json")):
                try:
                    with log_file.open("r", encoding="utf-8") as f:
                        log_data = json.load(f)
                except Exception:
                    continue

                meta = log_data.get("meta", {}) or {}
                test_case = log_data.get("test_case", {}) or {}
                test_case_id = meta.get("test_case_id") or test_case.get("id") or "unknown_test_case"
                setup_key = meta.get("setup_key") or setup_dir.name
                setup_label = meta.get("setup") or setup_key
                model = meta.get("model") or "unknown"
                reasoning = meta.get("reasoning_effort") or "--"

                for run in log_data.get("runs", []):
                    run_index = run.get("run_index")
                    run_index = run_index if run_index is not None else run.get("run")
                    if run_index is None:
                        continue

                    tokens = run.get("tokens") or {}
                    validation = run.get("validation", {}) or {}
                    state_validation = run.get("state_validation", {}) or {}
                    checks = state_validation.get("checks", {}) or {}

                    input_tokens = tokens.get("input_tokens")
                    output_tokens = tokens.get("output_tokens")
                    cached_input_tokens = tokens.get("cached_input_tokens")
                    total_tokens = tokens.get("total_tokens")
                    cost = gt._calculate_cost(input_tokens, output_tokens, cached_input_tokens)

                    extracted_data = validation.get("extracted_data", {})
                    result_preview = gt._format_extracted_value(test_case_id, extracted_data, state_validation)
                    if len(result_preview) > 95:
                        result_preview = result_preview[:95] + "..."

                    run_key = (
                        f"{date_folder.name}::{setup_key}::{test_case_id}::"
                        f"{log_file.name}::run_{run_index}"
                    )
                    success = _success_label(run, test_case_id, ground_truth)
                    accuracy = state_validation.get("accuracy")
                    if accuracy is None:
                        accuracy = validation.get("accuracy")

                    rows.append(
                        {
                            "run_key": run_key,
                            "date_folder": date_folder.name,
                            "model": model,
                            "reasoning": reasoning,
                            "setup": setup_label,
                            "setup_key": setup_key,
                            "test_case_id": test_case_id,
                            "test_case_name": test_case.get("name", test_case_id),
                            "run_index": run_index,
                            "duration_ms": run.get("duration_ms"),
                            "total_tokens": total_tokens,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "cost_usd": cost,
                            "tool_calls": run.get("tool_calls_count"),
                            "success": success,
                            "accuracy": accuracy,
                            "result_preview": result_preview,
                            "integrity": _integrity_summary(checks),
                            "has_error": bool(run.get("error")),
                            "error": run.get("error") or "",
                            "started_at": run.get("started_at") or "",
                            "log_path": str(log_file),
                            "output_preview": (run.get("output_text") or "")[:140].replace("\n", " "),
                        }
                    )

                    details[run_key] = {
                        "log_path": str(log_file),
                        "meta": meta,
                        "test_case": test_case,
                        "run": run,
                        "validation": validation,
                        "state_validation": state_validation,
                        "tool_calls": run.get("tool_calls", []),
                        "output_items": run.get("output_items", []),
                    }

    if not rows:
        return pd.DataFrame(), details

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["date_folder", "setup", "test_case_id", "run_index"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    return df, details


def _filtered_df(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.markdown("### Filter")

        date_options = sorted(df["date_folder"].dropna().unique().tolist(), reverse=True)
        model_options = sorted(df["model"].dropna().unique().tolist())
        setup_options = sorted(df["setup"].dropna().unique().tolist())
        test_case_options = sorted(df["test_case_id"].dropna().unique().tolist())
        success_options = ["True", "False", "Error", "N/A"]

        selected_dates = st.multiselect("Date folders", date_options, default=date_options[:6])
        selected_models = st.multiselect("Models", model_options, default=model_options)
        selected_setups = st.multiselect("Setups", setup_options, default=setup_options)
        selected_cases = st.multiselect("Test cases", test_case_options, default=test_case_options)
        selected_success = st.multiselect("Success state", success_options, default=success_options)

        search_text = st.text_input("Quick search", placeholder="model, setup, test_case, output...")
        token_range = st.slider(
            "Total tokens",
            min_value=int(df["total_tokens"].fillna(0).min()),
            max_value=int(df["total_tokens"].fillna(0).max()),
            value=(
                int(df["total_tokens"].fillna(0).min()),
                int(df["total_tokens"].fillna(0).max()),
            ),
            step=100,
        )

    filtered = df.copy()
    if selected_dates:
        filtered = filtered[filtered["date_folder"].isin(selected_dates)]
    if selected_models:
        filtered = filtered[filtered["model"].isin(selected_models)]
    if selected_setups:
        filtered = filtered[filtered["setup"].isin(selected_setups)]
    if selected_cases:
        filtered = filtered[filtered["test_case_id"].isin(selected_cases)]
    if selected_success:
        filtered = filtered[filtered["success"].isin(selected_success)]

    filtered = filtered[
        filtered["total_tokens"].fillna(0).between(token_range[0], token_range[1], inclusive="both")
    ]

    if search_text:
        needle = search_text.lower().strip()
        haystack = (
            filtered["model"].astype(str)
            + " "
            + filtered["setup"].astype(str)
            + " "
            + filtered["test_case_id"].astype(str)
            + " "
            + filtered["output_preview"].astype(str)
            + " "
            + filtered["error"].astype(str)
        ).str.lower()
        filtered = filtered[haystack.str.contains(needle, regex=False)]

    return filtered.reset_index(drop=True)


def _render_metrics(total_df: pd.DataFrame, selected_df: pd.DataFrame) -> None:
    success_numeric = selected_df["success"].map({"True": 1.0, "False": 0.0, "Error": 0.0})
    measurable = success_numeric.dropna()
    success_rate = f"{measurable.mean() * 100:.1f}%" if not measurable.empty else "--"

    total_selected = len(selected_df)
    total_cost = selected_df["cost_usd"].sum() if not selected_df.empty else 0.0
    total_tokens = int(selected_df["total_tokens"].fillna(0).sum()) if not selected_df.empty else 0
    unique_cases = int(selected_df["test_case_id"].nunique()) if not selected_df.empty else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Visible runs", f"{len(total_df)}")
    col2.metric("Selected runs", f"{total_selected}")
    col3.metric("Selected test cases", f"{unique_cases}")
    col4.metric("Success rate", success_rate)
    col5.metric("Estimated cost", f"${total_cost:.4f}")
    st.caption(f"Total selected tokens: {total_tokens:,}".replace(",", " "))


def _generate_tex_from_selection(df: pd.DataFrame, selected_keys: set[str], output_filename: str) -> tuple[Path, int]:
    selected = df[df["run_key"].isin(selected_keys)].copy()
    if selected.empty:
        raise ValueError("No runs selected.")

    grouped: dict[str, set[str]] = {}
    for row in selected.itertuples(index=False):
        key = str(row.log_path)
        grouped.setdefault(key, set()).add(str(row.run_index))

    with tempfile.TemporaryDirectory(prefix="table_selection_") as tmp_root:
        tmp_root_path = Path(tmp_root)
        pseudo_folder = tmp_root_path / f"{datetime.now().strftime('%d.%m_%H-%M')}_selection_custom"
        pseudo_folder.mkdir(parents=True, exist_ok=True)

        written_files = 0
        for log_path_str, run_ids in grouped.items():
            log_path = Path(log_path_str)
            if not log_path.exists():
                continue

            try:
                with log_path.open("r", encoding="utf-8") as f:
                    original = json.load(f)
            except Exception:
                continue

            filtered_runs = []
            for run in original.get("runs", []):
                rid = run.get("run_index")
                if rid is None:
                    rid = run.get("run")
                if str(rid) in run_ids:
                    filtered_runs.append(run)
            if not filtered_runs:
                continue

            filtered_log = dict(original)
            filtered_log["runs"] = filtered_runs
            meta = dict(filtered_log.get("meta", {}) or {})
            meta["runs_per_test"] = len(filtered_runs)
            meta["run_count"] = len(filtered_runs)
            filtered_log["meta"] = meta

            setup_name = log_path.parent.name
            date_name = log_path.parent.parent.name
            target_setup_dir = pseudo_folder / setup_name
            target_setup_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_setup_dir / f"{date_name}__{log_path.name}"
            target_file.write_text(json.dumps(filtered_log, indent=2, ensure_ascii=False), encoding="utf-8")
            written_files += 1

        if written_files == 0:
            raise ValueError("Selected runs could not be mapped to valid log files.")

        document = gt.build_combined_document(pseudo_folder)

    TABLE_DIR.mkdir(exist_ok=True)
    output_path = TABLE_DIR / output_filename
    output_path.write_text(document, encoding="utf-8")
    return output_path, len(selected)


def _render_detail_panel(selected_df: pd.DataFrame, details: dict[str, Any]) -> None:
    st.markdown("### Run details")
    if selected_df.empty:
        st.info("Select at least one run to inspect all details.")
        return

    labels: list[str] = []
    run_key_by_label: dict[str, str] = {}
    for row in selected_df.itertuples(index=False):
        label = (
            f"{row.date_folder} | {row.setup} | {row.test_case_id} | "
            f"run {row.run_index} | {row.success}"
        )
        labels.append(label)
        run_key_by_label[label] = row.run_key

    selected_label = st.selectbox("Inspect selected run", labels, index=0)
    run_key = run_key_by_label[selected_label]
    data = details.get(run_key, {})
    run = data.get("run", {})

    tab_summary, tab_output, tab_validation, tab_state, tab_raw = st.tabs(
        ["Summary", "Output", "Validation", "State checks", "Raw run JSON"]
    )

    with tab_summary:
        c1, c2, c3 = st.columns(3)
        c1.metric("Duration (ms)", run.get("duration_ms"))
        c2.metric("Tool calls", run.get("tool_calls_count"))
        tokens = run.get("tokens") or {}
        c3.metric("Total tokens", tokens.get("total_tokens"))
        st.json(
            {
                "meta": data.get("meta", {}),
                "test_case": data.get("test_case", {}),
                "error": run.get("error"),
                "retry_exhausted": run.get("retry_exhausted"),
            },
            expanded=False,
        )

    with tab_output:
        output_text = run.get("output_text", "")
        language = "json" if output_text.strip().startswith("{") else "text"
        st.code(output_text, language=language)

    with tab_validation:
        st.json(data.get("validation", {}), expanded=True)

    with tab_state:
        st.json(data.get("state_validation", {}), expanded=True)

    with tab_raw:
        st.json(run, expanded=False)


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');
        :root {
          --bg: #f5f3eb;
          --panel: #fffdf7;
          --ink: #1b1f24;
          --muted: #5c6874;
          --accent: #d24f31;
          --accent-soft: #f6d7c9;
          --line: #dad4c8;
        }
        .stApp {
          font-family: "Outfit", sans-serif;
          background:
            radial-gradient(circle at 12% 0%, #f0e8d8 0%, transparent 37%),
            radial-gradient(circle at 82% 15%, #f4d9b2 0%, transparent 31%),
            linear-gradient(140deg, #f4f1e9 0%, #f8f6f1 45%, #efeae0 100%);
          color: var(--ink);
        }
        .main .block-container { padding-top: 1.4rem; max-width: 88rem; }
        .hero {
          border: 1px solid var(--line);
          background: linear-gradient(120deg, rgba(255,255,255,.88), rgba(255,250,238,.9));
          border-radius: 18px;
          padding: 1.1rem 1.3rem;
          margin-bottom: 0.7rem;
          box-shadow: 0 8px 24px rgba(83,64,39,.08);
        }
        .hero h1 {
          font-size: 2rem;
          line-height: 1.05;
          margin: 0;
          letter-spacing: -0.02em;
        }
        .hero p {
          margin: .45rem 0 0;
          color: var(--muted);
          font-size: .95rem;
        }
        div[data-testid="stMetric"] {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 12px;
          padding: 10px 12px;
        }
        div[data-testid="stMetricValue"] { font-size: 1.5rem; }
        .stCode pre, code, .monospace {
          font-family: "IBM Plex Mono", monospace !important;
        }
        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, #f3efe6 0%, #ece6d7 100%);
          border-right: 1px solid var(--line);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Evaluation Table Studio",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_theme()

    st.markdown(
        """
        <div class="hero">
          <h1>Evaluation Table Studio</h1>
          <p>Select exact runs, inspect all run details, and generate LaTeX tables from only those selected runs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Data source")
        st.code(str(LOG_DIR))
        if st.button("Reload logs", width="stretch"):
            discover_runs.clear()
            st.rerun()

    df, details = discover_runs(str(LOG_DIR))
    if df.empty:
        st.error("No evaluation log runs found in evaluation_logs/.")
        return

    if "selected_run_keys" not in st.session_state:
        st.session_state.selected_run_keys = set()

    filtered = _filtered_df(df)

    selected_set: set[str] = set(st.session_state.selected_run_keys)
    selected_filtered = filtered[filtered["run_key"].isin(selected_set)]
    _render_metrics(filtered, selected_filtered)

    btn1, btn2, btn3, btn4 = st.columns(4)
    if btn1.button("Select visible", width="stretch"):
        selected_set.update(filtered["run_key"].tolist())
        st.session_state.selected_run_keys = selected_set
        st.rerun()
    if btn2.button("Select failed visible", width="stretch"):
        failed = filtered[filtered["success"].isin(["False", "Error"])]
        selected_set.update(failed["run_key"].tolist())
        st.session_state.selected_run_keys = selected_set
        st.rerun()
    if btn3.button("Clear visible", width="stretch"):
        selected_set.difference_update(filtered["run_key"].tolist())
        st.session_state.selected_run_keys = selected_set
        st.rerun()
    if btn4.button("Clear all", width="stretch"):
        st.session_state.selected_run_keys = set()
        st.rerun()

    table_df = filtered[
        [
            "run_key",
            "date_folder",
            "model",
            "reasoning",
            "setup",
            "test_case_id",
            "run_index",
            "success",
            "duration_ms",
            "total_tokens",
            "input_tokens",
            "output_tokens",
            "cost_usd",
            "tool_calls",
            "integrity",
            "result_preview",
            "error",
        ]
    ].copy()
    table_df["select"] = table_df["run_key"].isin(selected_set)
    table_df = table_df.set_index("run_key")

    edited = st.data_editor(
        table_df,
        width="stretch",
        height=420,
        hide_index=True,
        column_config={
            "select": st.column_config.CheckboxColumn("Pick"),
            "date_folder": st.column_config.TextColumn("Date"),
            "model": st.column_config.TextColumn("Model"),
            "reasoning": st.column_config.TextColumn("Reasoning"),
            "setup": st.column_config.TextColumn("Setup"),
            "test_case_id": st.column_config.TextColumn("Test case"),
            "run_index": st.column_config.NumberColumn("Run", format="%d"),
            "success": st.column_config.TextColumn("Success"),
            "duration_ms": st.column_config.NumberColumn("Duration (ms)", format="%d"),
            "total_tokens": st.column_config.NumberColumn("Tok total", format="%d"),
            "input_tokens": st.column_config.NumberColumn("Tok in", format="%d"),
            "output_tokens": st.column_config.NumberColumn("Tok out", format="%d"),
            "cost_usd": st.column_config.NumberColumn("Cost (USD)", format="$%.5f"),
            "tool_calls": st.column_config.NumberColumn("Calls", format="%d"),
            "integrity": st.column_config.TextColumn("Integrity"),
            "result_preview": st.column_config.TextColumn("Result"),
            "error": st.column_config.TextColumn("Error"),
        },
        disabled=[
            "date_folder",
            "model",
            "reasoning",
            "setup",
            "test_case_id",
            "run_index",
            "success",
            "duration_ms",
            "total_tokens",
            "input_tokens",
            "output_tokens",
            "cost_usd",
            "tool_calls",
            "integrity",
            "result_preview",
            "error",
        ],
    )

    for run_key, row in edited.iterrows():
        if bool(row["select"]):
            selected_set.add(run_key)
        else:
            selected_set.discard(run_key)
    st.session_state.selected_run_keys = selected_set

    selected_df = df[df["run_key"].isin(selected_set)].copy()
    _render_detail_panel(selected_df, details)

    st.markdown("### Generate tables from selected runs")
    default_name = f"evaluation_tables_selection_{datetime.now().strftime('%d.%m_%H-%M')}.tex"
    output_name = st.text_input("Output filename", value=default_name)

    if st.button("Generate .tex from selected runs", type="primary", width="stretch"):
        clean_name = output_name.strip()
        if not clean_name:
            st.error("Please provide an output filename.")
        else:
            if not clean_name.endswith(".tex"):
                clean_name = f"{clean_name}.tex"
            try:
                with st.spinner("Generating LaTeX document..."):
                    safe_name = Path(clean_name).name
                    output_path, selected_count = _generate_tex_from_selection(df, selected_set, safe_name)
                st.success(f"Generated {output_path} from {selected_count} selected runs.")
                st.code(str(output_path))
            except Exception as exc:
                st.error(f"Generation failed: {exc}")


if __name__ == "__main__":
    main()
