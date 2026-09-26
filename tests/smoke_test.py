"""Offline check of the MCP server and the evaluation validation (no LLM calls).

    uv run python tests/smoke_test.py

Talks to the real server over stdio, performs the reference solution of every
benchmark task on a temporary copy of the dataset and checks that the
evaluation's validation and success criteria accept it.
"""
import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

from analyze import run_success  # noqa: E402
from cases import RAISE_BUILDING_SPEC as RAISE, TEST_BUILDING_SPEC as NEW  # noqa: E402
from validation import validate_dataset_modification, validate_test_result  # noqa: E402

SAMPLE = ROOT / "evaluation" / "data" / "evaluation.city.json"
GROUND_TRUTH = json.loads((ROOT / "evaluation" / "data" / "ground_truth.json").read_text())


def check_answer(task, answer):
    result = validate_test_result(task, json.dumps(answer), GROUND_TRUTH["solutions"])
    assert result["accuracy"] == 1.0, (task, result)


def check_state(task, dataset_path):
    state = validate_dataset_modification(task, "mcp", "", dataset_path, NEW, RAISE)
    assert run_success(task, {"state_validation": state}), (task, state["checks"])


async def main():
    original = SAMPLE.read_bytes()
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / SAMPLE.name
        shutil.copy2(SAMPLE, data)
        transport = StdioTransport(
            command=sys.executable, args=["-m", "citydpc_mcp"],
            env={**os.environ, "MCP_DATASET_DIR": tmp}, cwd=tmp,
        )
        async with Client(transport) as client:
            async def call(name, **args):
                result = await client.call_tool(name, args)
                assert not result.is_error, (name, result)
                content = result.structured_content
                if isinstance(content, dict):  # plain JSON instead of generated dataclasses
                    return content.get("result", content) if len(content) == 1 else content
                return result.data

            tools = await client.list_tools()
            assert len(tools) == 19, len(tools)
            assert SAMPLE.name in await call("list_datasets")
            assert "Erfolg" in await call("load_dataset", filename=SAMPLE.name)

            ids = await call("get_buiding_Id_list")
            check_answer("list_buildings", {"building_ids": ids, "total": await call("number_of_buildings")})

            buildings = await call("get_all_buildings")
            with_height = [b for b in buildings if b["measured_height"] is not None]
            top = max(with_height, key=lambda b: b["measured_height"])
            check_answer("highest_measured_height", {
                "building_id": top["gml_id"], "measuredHeight": top["measured_height"],
                "missing_count": len(buildings) - len(with_height)})

            check_answer("count_party_walls", {"party_wall_count": len(await call("get_party_walls"))})

            # snapshot / rollback
            await call("take_snapshot", description="smoke test")
            await call("remove_building_from_dataset", building_id=ids[0])
            assert await call("number_of_buildings") == len(ids) - 1
            await call("rollback_to_snapshot", snapshot_index=1)
            assert await call("number_of_buildings") == len(ids)

            # raise_building_by_id
            old = (await call("get_building_by_id", building_id=RAISE["target_building_id"]))["measured_height"]
            await call("enrich_building", building_id=RAISE["target_building_id"],
                       measured_height=old + RAISE["increase_by"])
            await call("save_dataset")
            check_state("raise_building_by_id", data)

            # add_building
            shutil.copy2(SAMPLE, data)
            await call("load_dataset", filename=SAMPLE.name)
            await call("create_building", id=NEW["id"], groundsCoordinates=NEW["groundCoordinates"],
                       groundSurfaceHeight=NEW["groundSurfaceHeight"], geometryHeight=NEW["measuredHeight"],
                       roofType=NEW["roofType"], roofHeight=NEW["roofHeight"], roofOrientation=0)
            await call("enrich_building", building_id=NEW["id"], measured_height=NEW["measuredHeight"],
                       storeys_above_ground=NEW["storeysAboveGround"], function=NEW["function"])
            await call("save_dataset")
            check_state("add_building", data)

            # create_dataset: new empty file, add a building, save, reload
            for name in ("new.city.json", "new.gml"):
                assert "Erfolg" in await call("create_dataset", filename=name, title="smoke test")
                assert await call("number_of_buildings") == 0
                await call("create_building", id="B1", groundsCoordinates=NEW["groundCoordinates"],
                           groundSurfaceHeight=0.0, geometryHeight=10.0, lod=1)
                await call("save_dataset")
                assert "Erfolg" in await call("load_dataset", filename=name)
                assert await call("get_buiding_Id_list") == ["B1"]
                assert "existiert bereits" in await call("create_dataset", filename=name)

    assert SAMPLE.read_bytes() == original, "sample dataset was modified"
    print(f"PASS: 19 tools, {len(ids)} buildings, all 5 benchmark tasks validated, snapshot/rollback and create_dataset OK")


if __name__ == "__main__":
    asyncio.run(main())
