# Publication review — 2026-09-09

## Existing remotes

- GitHub `HamerJava/BA_Evaluation`: private, main `c64f09473df40c0fc93e8a515b51c44d03ecd8e6`, last commit 2026-02-08.
- Institutional GitLab `shamovich/ma_hamer`: HEAD `b726452784043536ab72a3dac65ad85e94ee097f`, dated 2026-02-19; server is under `BA/mcp-server/`.
- The current local server differs from the GitLab server (SHA-256 comparison). The original working directories and their staged changes were preserved.

## Publication changes

Combined the current server, bundled CityDPC source, evaluation Python modules, four input datasets, existing ground truth, tables and plots in a new repository. Added portable server/interpreter defaults for both current evaluation runners, installation requirements, documentation and an offline protocol integration test. The server implementation itself was copied unchanged.

Excluded `.env`, OAuth credential files, virtual environments, caches, unrelated presentation/export artifacts, external reconstruction reference images and large raw logs/workspaces/backups. This is a source and derived-results publication, not a complete archive of every model transcript.

## Validation

- Fresh Python 3.12 environment: installation from root requirements succeeded.
- Actual stdio MCP client/server integration: 18 tools discovered; sample with 36 buildings loaded; IDs/count and roof-volume query passed; removal, rollback, save and reload passed on a temporary copy.
- Python syntax compilation passed for server, library, evaluation and smoke test.
- Main evaluation task listing and optional runner help passed.
- Publication text scanned for common API-key, GitHub-token, JWT and private-key patterns; no matches. Credential files are excluded independently.

## Research limitations

The active analytic task is `count_party_walls`; historical tables also contain `roof_volume_sum`. The older ground-truth generator was corrected to compute the active party-wall answer and avoid a stale roof-volume output lookup. Regenerated query and party-wall values match the existing ground truth, which was preserved. Some legacy/analysis scripts retain historical assumptions. Extra reconstruction cases require excluded reference images.

CityDPC reports non-planar surface warnings for the sample, especially after export. The successful transport/state checks do not establish geometric correctness. No paid LLM runs were started, and the existing logs were used for the subsequent paper reconciliation described below.

## Supplied paper reconciliation

The supplied revised blind paper defines the current five tasks, three models, three setups and 100 repetitions. Its Tables 2 and 3 were regenerated from the identified original logs with the existing analysis scripts; both generated LaTeX files are byte-identical to the existing table sources. All 228 numeric cells in Table 2 and 72 numeric cells in Table 3 match the PDF extraction; page 5 was also rendered and inspected. All 45 task/model/setup combinations contain 100 runs (4500 total).

`evaluation/paper/run_metrics.csv` contains the per-run scalar metrics used for Table 2, without transcripts or credentials; `source_manifest.json` records source-file SHA-256 hashes. This confirms aggregation and manuscript transcription, not an independent reassessment of every model response or geometric result. The paper describes the GPT-5.4 Mini interface as OpenAI Responses; the matching local run was produced by `run_codex_oauth.py`, which should be described precisely if provider/authentication details matter. The blind manuscript itself is not included in this repository.
