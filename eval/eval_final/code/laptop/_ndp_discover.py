"""Helper: discover NDP datasets in an isolated subprocess."""
import asyncio
import pickle
import sys

sys.path.insert(0, "src")

from clio_agentic_search.connectors.ndp.mcp_client import NDPMCPClient


async def run():
    terms = sys.argv[1:]
    binary = ".venv/bin/ndp-mcp"
    c = NDPMCPClient(ndp_mcp_binary=binary)
    async with c.connect() as cc:
        return await cc.search_datasets(terms, limit=25)


ds = asyncio.run(run())
sys.stdout.buffer.write(pickle.dumps(ds))
