# Paper evidence

Paper: *CityDPC-MCP: Enabling 3D city model processing via Model Context Protocol* (supplied revised blind draft).

## Identified runs

| Model | Run folder |
|---|---|
| GPT-OSS | `04.06_13-41_gpt-oss-120b_high_completions` |
| Mistral Small 4 | `04.06_00-43_mistralai-mistral-small-4-119b_high_completions` |
| GPT-5.4 Mini | `11.06_14-51_gpt-5.4-mini_high_codex-oauth` |

Each has 100 repetitions for five tasks in each of `mcp`, `no_tools`, `code_interpreter`: 4500 runs total. `run_metrics.csv` contains scalar per-run token, tool-call, duration and success values extracted with `evals.tools.generate_model_comparison_table`. Success values inherit the recorded validation and that script's success logic. They are not newly adjudicated labels.

- Table 2 source: `../evaluation_tables/compare_gptoss_vs_mistral_100runs_all_calls_time.tex`.
- Table 3 source: `../evaluation_tables/compare_gptoss_vs_mistral_mcp_tool_usage_efficiency.tex` (the draft relabels TUE as WA).
- Both sources were reproduced byte-for-byte from the original local logs; all 300 numeric cells match the paper.
- `source_manifest.json` records the 45 original JSON source paths and hashes. Raw transcripts are not included. Table 3 requires those original tool-call traces to independently recalculate workflow adherence.

The active party-wall task and 36-building sample match the paper. Older roof-volume result files are historical experiments. The optional GPT-5.4 runner uses Codex OAuth; the manuscript's API description should distinguish this from a normal API-key run if relevant.

The server and evaluation source are the current local versions. Their identity with the exact source code used during every historical run is not independently established. The PDF is a blind draft and is not uploaded.
