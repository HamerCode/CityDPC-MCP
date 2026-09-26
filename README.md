# CityDPC-MCP

A [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server that lets LLM agents such as Claude Code or Codex inspect, analyse and edit **CityJSON** and **CityGML** building models. It is built on [CityDPC](https://github.com/RWTH-E3D/CityDPC), the Python library for 3D city models maintained by RWTH-E3D.

Companion repository to the paper *CityDPC-MCP: Enabling 3D city model processing via Model Context Protocol*. It contains the server, the benchmark used in the paper and the per-run results.

## Installation

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). It fetches Python 3.12+ and all dependencies, CityDPC included.

```bash
uv tool install git+https://github.com/HamerCode/CityDPC-MCP
```

This puts a `citydpc-mcp` command on your `PATH`. The server communicates over stdio and works with the `*.json` / `*.gml` files in one **dataset directory**, which you set with `--dataset-dir` (default: the current directory).

> **Note:** `save_dataset` overwrites files in that directory, so work on copies of important data.

### Claude Code

```bash
claude mcp add citydpc -- citydpc-mcp --dataset-dir /absolute/path/to/your/data
claude mcp list          # should show: citydpc: citydpc-mcp ... - ✔ Connected
```

Add `--scope user` to make the server available in all projects. If you leave out `--dataset-dir`, the server uses the project directory Claude Code was started in.

### Codex

```bash
codex mcp add citydpc -- citydpc-mcp --dataset-dir /absolute/path/to/your/data
```

Or add it to `~/.codex/config.toml` yourself:

```toml
[mcp_servers.citydpc]
command = "citydpc-mcp"
args = ["--dataset-dir", "/absolute/path/to/your/data"]
```

### Other MCP clients (Claude Desktop, Cursor, …)

```json
{
  "mcpServers": {
    "citydpc": {
      "command": "citydpc-mcp",
      "args": ["--dataset-dir", "/absolute/path/to/your/data"]
    }
  }
}
```

Desktop apps often don't search `~/.local/bin`. If yours can't find the command, use the absolute path that `which citydpc-mcp` prints. To run the server without installing it, use `uvx --from git+https://github.com/HamerCode/CityDPC-MCP citydpc-mcp` as the command. The first start takes a while because uv installs the dependencies then.

### Try it

Point the server at the sample dataset ([`evaluation/data/evaluation.city.json`](evaluation/data/evaluation.city.json), 36 LoD2 buildings) and ask something like:

> Load evaluation.city.json. Which building has the highest measuredHeight, and how many party walls does the dataset have?

## Tools

| Group | Tools |
|---|---|
| Datasets | `list_datasets`, `load_dataset`, `create_dataset` (new empty file), `filter_dataset` (by address or bounding polygon), `save_dataset` (CityJSON or CityGML) |
| Analysis | `analyse_dataset`, `number_of_buildings`, `get_buiding_Id_list`, `get_all_buildings`, `get_building_by_id`, `get_party_walls`, `calculate_roof_volume_by_id` |
| Editing | `create_building` (LoD 0/1/2 with roof types), `remove_building_from_dataset`, `enrich_building`, `remove_building_attributes` |
| Versioning | `take_snapshot`, `rollback_to_snapshot`, `get_dataset_history` |

All edits happen in memory until `save_dataset` is called. The paper evaluated the other 18 tools; `create_dataset` was added afterwards for quick testing. Their names and (German) descriptions are unchanged, including the historical spelling `get_buiding_Id_list`. See [`src/citydpc_mcp/server.py`](src/citydpc_mcp/server.py) for signatures.

## Evaluation

The benchmark asks an LLM to solve five tasks on the 36-building sample, in three setups:

- **MCP**: the model uses this server.
- **No Tools**: the dataset is pasted into the prompt.
- **Sandbox**: the dataset is in the prompt and the model also has a Python code sandbox.

| Task | Category | Checked against |
|---|---|---|
| `list_buildings` | query | all 36 building IDs and the total |
| `highest_measured_height` | query | the ID and height of the tallest building |
| `count_party_walls` | analytic | the number of party walls (CityDPC: 10) |
| `raise_building_by_id` | state change | the saved file: measuredHeight +2 m |
| `add_building` | state change | the saved file: new building, attributes and valid geometry, importable by CityDPC |

The paper tested GPT-OSS-120B, Mistral Small 4 and GPT-5.4 Mini, all with high reasoning effort. Each combination of model, setup and task ran 100 times, for 4,500 runs in total.

### Results

Averages across the five tasks ([full tables](evaluation/results/tables.md)):

| Setup | GPT-OSS success | Mistral success | GPT-5.4 Mini success | Avg. tokens (GPT-OSS / Mistral / GPT-5.4 Mini) |
|---|---|---|---|---|
| **MCP** | **93%** | **80%** | **97%** | **14,865 / 25,755 / 15,429** |
| No Tools | 40% | 34% | 61% | 62,188 / 80,633 / 71,879 |
| Sandbox | 34% | 30% | 64% | 236,803 / 83,517 / 117,703 |

[`evaluation/results/paper_runs.csv`](evaluation/results/paper_runs.csv) has one row per run: tokens, tool calls, the sequence of tool calls, duration and success. The paper's Tables 2 and 3 are computed from this file, and all 300 numbers match the paper:

```bash
git clone https://github.com/HamerCode/CityDPC-MCP && cd CityDPC-MCP
uv sync --extra eval
uv run python evaluation/analyze.py tables
```

### Running the benchmark yourself

```bash
cp evaluation/.env.example evaluation/.env    # add your endpoint and API key
uv run python evaluation/run.py --setup mcp,no_tools --model gpt-oss-120b \
    --api completions --reasoning-effort high --runs 100
uv run python evaluation/analyze.py extract evaluation/logs/<run> -o my_runs.csv
uv run python evaluation/analyze.py tables my_runs.csv
```

The runner works with any OpenAI-compatible endpoint, through either the Responses or the Chat Completions API. The paper ran GPT-OSS and Mistral through [KI:Connect NRW](https://kiconnect.nrw) Chat Completions, and GPT-5.4 Mini through the Responses API of ChatGPT's Codex backend. Prompts, task specifications, validation and success criteria are the ones used for the paper. Current models will not reproduce the historical runs exactly.

The `code_interpreter` (Sandbox) setup also needs a Docker code-sandbox MCP server (`CODE_SANDBOX_MCP_BINARY`, e.g. [code-sandbox-mcp](https://github.com/Automata-Labs-team/code-sandbox-mcp)). Leave that setup out if you don't have one.

The following check needs no API key. It exercises the server over stdio, solves all five tasks with the reference tool calls and confirms that the evaluation's validation accepts each solution:

```bash
uv run python tests/smoke_test.py
```

## Repository layout

```
src/citydpc_mcp/server.py   MCP server (19 tools)
evaluation/
  cases.py, schemas.py      task prompts and answer schemas
  run.py, client.py         benchmark runner and OpenAI-compatible tool-calling client
  validation.py             checks answers and saved datasets
  analyze.py                logs -> per-run CSV -> result tables
  ground_truth.py           regenerates data/ground_truth.json with CityDPC
  data/                     sample dataset and ground truth
  results/                  per-run results and tables of the paper
tests/smoke_test.py         offline end-to-end check
```

## Credits and license

CityDPC is developed by Maxim Shamovich and Simon Raming at [RWTH-E3D](https://github.com/RWTH-E3D/CityDPC) under the Apache 2.0 license. This server installs CityDPC v0.1.22 pinned to commit `4ede6d7`. This repository is licensed under Apache 2.0 as well ([LICENSE](LICENSE)).

If you use this work, please cite the paper *CityDPC-MCP: Enabling 3D city model processing via Model Context Protocol* and link a specific commit of this repository.
