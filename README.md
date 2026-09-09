# CityDPC MCP Server

A Model Context Protocol (MCP) server for inspecting, analysing and editing CityJSON and CityGML building datasets through CityDPC. The accompanying evaluation compares language-model workflows with MCP, without tools, and with a separately configured code interpreter.

**Built on [CityDPC](https://github.com/RWTH-E3D/CityDPC)**, the separate open-source Python library for processing CityGML and CityJSON. This repository provides the MCP interface and accompanying evaluation; the original CityDPC project is maintained in its own upstream repository.

## Quick start

Python 3.12 or newer is required; the pinned dependency set was verified with Python 3.12 on macOS. `requirements-lock.txt` records that environment, while `requirements.txt` contains the direct dependency ranges. Run these commands from the repository root:

```bash
git clone https://github.com/HamerCode/CityDPC-MCP.git
cd CityDPC-MCP
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python mcp-server/cityDPC.py
```

The server uses **stdio**. Configure an MCP client with the absolute path to `.venv/bin/python` as its command and the absolute path to `mcp-server/cityDPC.py` as its single argument. Set `MCP_DATASET_DIR` to a dataset directory if needed; the default is `mcp-server/data/datasets/`, which contains a sample dataset. Use a copy of valuable data: editing and save tools can change files. This is a local research server, not a hosted multi-user service.

## Documentation

- [Server guide](docs/mcp-server.md): client configuration, dataset lifecycle, editing, persistence and troubleshooting.
- [Tool reference](docs/tools.md): all 18 names, signatures and original descriptions.
- [Read-only Python example](examples/query_dataset.py): use the server without an LLM.
- [Evaluation guide](evaluation/README.md): benchmark tasks, configuration and analysis.
- [Paper evidence](evaluation/paper/README.md): identified runs, 4,500 per-run metrics and table verification.
- [Publication review](REVIEW.md): checks, provenance and limitations.

## Tools

The server exposes 18 tools: dataset discovery/loading, filtering and analysis, building counts and IDs, building details, party-wall analysis, roof volume per building, creation/removal and attribute editing, snapshots, rollback, saving and history. Tool names and schemas are defined in [`mcp-server/cityDPC.py`](mcp-server/cityDPC.py). The historical tool spelling `get_buiding_Id_list` is retained for compatibility with recorded evaluations.

## Repository

- **`mcp-server/`**: the primary MCP server.
- **`CityDPC/`**: the exact local CityDPC source used with the server, including its original Apache-2.0 license and attribution.
- **`evaluation/`**: runners, task definitions, validation, ground truth, input datasets, analysis scripts, and existing result tables/plots. See [evaluation instructions](evaluation/README.md).
- **`tests/`**: offline MCP integration smoke test.

## Verification

```bash
python tests/smoke_mcp.py
cd evaluation
python -m evals.run --list-tests
```

These checks do not run paid model evaluations or re-evaluate model outputs. The paper's Tables 2 and 3 were reproduced from existing logs and all 300 numeric cells matched the supplied PDF. Historical tables and plots are retained as existing research outputs. Credentials, session tokens, temporary dataset copies and large raw execution traces are excluded.

## Attribution and provenance

CityDPC is developed by Maxim Shamovich and Simon Raming: https://github.com/RWTH-E3D/CityDPC. Its license applies to the bundled `CityDPC/` directory. No additional license for the research server/evaluation is asserted here.

This repository brings together the locally updated MCP server and evaluation as of 2026-09-09. Earlier separate copies existed in the private `HamerJava/BA_Evaluation` GitHub repository and the `shamovich/ma_hamer` institutional GitLab repository. See [publication review](REVIEW.md) for scope and validation.

## Referencing this repository

Repository URL: **https://github.com/HamerCode/CityDPC-MCP**. For a fixed source snapshot in a paper, link to a specific Git commit as well. The accompanying paper is titled *CityDPC-MCP: Enabling 3D city model processing via Model Context Protocol*. Bibliographic author/DOI metadata is not inferred from its anonymized draft.
