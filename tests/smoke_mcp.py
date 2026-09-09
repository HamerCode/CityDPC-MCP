"""Exercise the actual stdio MCP protocol without model calls or source-data edits."""
import asyncio
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport

ROOT = Path(__file__).resolve().parents[1]

async def main():
    with tempfile.TemporaryDirectory() as tmp:
        sample = ROOT / 'evaluation/input/evaluation.city.json'
        shutil.copy2(sample, Path(tmp) / sample.name)
        source = json.loads(sample.read_text())
        expected = sorted(k for k, v in source['CityObjects'].items() if v['type'] == 'Building')
        transport = PythonStdioTransport(
            script_path=ROOT / 'mcp-server/cityDPC.py',
            python_cmd=sys.executable,
            cwd=str(ROOT),
            env={**os.environ, 'MCP_DATASET_DIR': tmp},
        )
        async with Client(transport) as client:
            tools = await client.list_tools()
            assert len(tools) == 18, len(tools)
            async def call(name, args=None):
                result = await client.call_tool(name, args or {})
                assert not result.is_error, (name, result)
                return result.data
            assert sample.name in await call('list_datasets')
            loaded = await call('load_dataset', {'filename': sample.name})
            assert 'Erfolg' in loaded, loaded
            assert await call('number_of_buildings') == len(expected)
            assert sorted(await call('get_buiding_Id_list')) == expected
            building = await call('get_building_by_id', {'building_id': expected[0]})
            assert building
            volume = await call('calculate_roof_volume_by_id', {'building_id': expected[0]})
            assert isinstance(volume, (float, int))
            await call('take_snapshot', {'description': 'smoke test'})
            await call('remove_building_from_dataset', {'building_id': expected[0]})
            assert await call('number_of_buildings') == len(expected) - 1
            await call('rollback_to_snapshot', {'snapshot_index': 1})
            assert await call('number_of_buildings') == len(expected)
            await call('save_dataset', {'description': 'smoke test'})
            await call('load_dataset', {'filename': sample.name})
            assert sorted(await call('get_buiding_Id_list')) == expected
        assert sample.read_bytes() == (ROOT / 'mcp-server/data/datasets/evaluation.city.json').read_bytes()
        print(f'PASS: 18 tools; {len(expected)} buildings; query, roof volume, remove, rollback, save and reload')

if __name__ == '__main__':
    asyncio.run(main())
