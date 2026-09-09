# Evaluation

Run from this directory with the repository virtual environment activated. Copy `.env.example` to `.env` and enter your own KI:Connect API key.

```bash
python -m evals.run --list-tests
python -m evals.run --setup mcp,no_tools --tests list_buildings,highest_measured_height,count_party_walls,raise_building_by_id,add_building --runs 10 --workers 2 --completions
```

The main runner supports KI:Connect Responses and Chat Completions; `--completions` selects the latter. Server paths default to the sibling `mcp-server/` and the active Python interpreter. Environment variables `MCP_SERVER_DIR`, `MCP_SERVER_SCRIPT` and `MCP_SERVER_PYTHON` override these defaults.

`evals/run_codex_oauth.py` is an optional provider-specific research runner. It requires separately configured credentials; none are distributed. The `code_interpreter` setup requires a separate code-sandbox MCP executable configured via `CODE_SANDBOX_MCP_BINARY` (see runner). It is not needed for MCP/no-tools comparisons.

Task definitions include additional experimental reconstruction tasks requiring external reference images. Those images are not distributed; use the explicit five-task command above for the core benchmark.

## Data and analysis

`input/` contains the evaluation datasets; `ground_truth.json` retains the existing ground truth. The root-level working dataset and historical per-run mutated copies are not substitutes for `input/`.

```bash
python -m evals.tools.generate_tables --help
```

The publication fixes `evals.ground_truth` to regenerate the current party-wall task. Regenerated building IDs, highest-height values and party-wall metrics match the retained ground truth. Run `python -m evals.ground_truth` to regenerate; it overwrites `ground_truth.json`. Table/plot scripts may require their original raw evaluation logs and some historical helpers retain workstation-specific defaults. Existing tables and plots are included, but raw model transcripts, credentials, temporary workspaces, and large dataset backups are excluded from this source publication. The paper-specific runs and numerical table comparison are documented in `paper/README.md`.

## Benchmark tasks

| Task ID | Check |
|---|---|
| `list_buildings` | Exact building-ID list and total. |
| `highest_measured_height` | Highest measured-height value and corresponding ID. |
| `count_party_walls` | Number of CityDPC party-wall records. |
| `raise_building_by_id` | Persisted semantic height increase for the specified building. |
| `add_building` | New building and required properties in the resulting dataset. |

Task prompts and specifications live in `evals/cases.py`; response schemas in `evals/schemas.py`; validation in `evals/validation.py`. State-changing runs are checked against the resulting file. The core runner's workspaces keep per-run copies separate. Additional generative cases are experimental and are not part of the paper's five-task quantitative benchmark.

## Configuration reference

From the repository root:

```bash
source .venv/bin/activate
cd evaluation
cp .env.example .env
```

Edit `.env` before starting model runs. Listing tasks and running the server smoke test need no model credentials; real evaluations send requests to the configured provider and may incur charges.

| Setting / option | Purpose |
|---|---|
| `KI_CONNECT_API_KEY` | Your provider credential; never commit `.env`. |
| `KI_CONNECT_RESPONSES_URL` | Responses endpoint used by the main client. |
| `KI_CONNECT_API_URL` | Chat Completions endpoint selected with `--completions`. |
| `--model` / `EVAL_MODEL` | Provider model identifier. |
| `--reasoning-effort` | Reasoning setting, e.g. `high` in the paper runs. |
| `--runs` | Repetitions per task/setup; paper: 100. |
| `--workers` | Concurrent task/setup lanes; repetitions within a lane run sequentially. |
| `--dataset` | Input path; defaults to `input/evaluation.city.json`. |
| `MCP_SERVER_DIR`, `MCP_SERVER_SCRIPT`, `MCP_SERVER_PYTHON` | Optional server overrides; use absolute paths. |
| `CODE_SANDBOX_MCP_BINARY` | External sandbox MCP executable for `code_interpreter`. |

Example for the two setups that work with the bundled server:

```bash
python -m evals.run \
  --setup mcp,no_tools \
  --tests list_buildings,highest_measured_height,count_party_walls,raise_building_by_id,add_building \
  --model gpt-oss-120b --reasoning-effort high \
  --runs 100 --workers 2 --completions
```

Add `code_interpreter` to `--setup` only after configuring the external sandbox executable. Availability of a historical model or endpoint is not guaranteed by this repository. Avoid `--tests all` for a paper-only run, because it also selects the later generative experiments.

## Outputs and reproduction boundaries

A new run writes `evaluation_logs/<timestamp-model>/`, per-setup JSON logs and a summary. The logs can contain model text, tool arguments and paths, so they are ignored by Git. `evals/tools/` provides table generation, state validation refresh and tool-call analysis. Run each tool with `--help` to inspect its supported arguments.

The published `paper/run_metrics.csv` enables analysis of the recorded scalar measurements without access to a provider. It contains `run_folder`, `setup`, `test_case`, `repetition`, `total_tokens`, `tool_calls`, `duration_ms` and binary `success`. Duration is in milliseconds; the paper reports seconds. Setup-level averages in the paper are arithmetic means of the five task-level means. Raw traces and per-run geometry are not included, so the CSV does not enable independent revalidation of model answers or workflow adherence.

The exact three historical run folders and source-file hashes are listed in [paper evidence](paper/README.md). Re-running today's models should not be expected to reproduce their outputs bit-for-bit.
