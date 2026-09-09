"""Read-only example: connect to the local stdio server, load a sample, query it."""
import asyncio
import os
from pathlib import Path
import sys

from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport

ROOT = Path(__file__).resolve().parents[1]

async def main():
    transport = PythonStdioTransport(
        script_path=ROOT / "mcp-server/cityDPC.py",
        python_cmd=sys.executable,
        cwd=str(ROOT),
        env=os.environ.copy(),
    )
    async with Client(transport) as client:
        for name, arguments in [
            ("list_datasets", {}),
            ("load_dataset", {"filename": "evaluation.city.json"}),
            ("number_of_buildings", {}),
            ("get_buiding_Id_list", {}),
        ]:
            result = await client.call_tool(name, arguments)
            print(f"{name}: {result.data}")

if __name__ == "__main__":
    asyncio.run(main())
