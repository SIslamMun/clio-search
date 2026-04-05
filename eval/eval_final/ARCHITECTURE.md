# CLIO Search vs MCP — What They Are and How They Differ

## MCP (Model Context Protocol)

**What it is:** A protocol specification. A JSON-RPC-over-stdio (or HTTP/SSE)
contract that lets an LLM agent call **tools** exposed by a **server**.

**Shape:** Server ↔ Client.
- The **server** is a subprocess that listens for requests. It exposes a list
  of tools via `list_tools` and executes them via `call_tool`.
- The **client** is usually an LLM agent. It sends requests through the protocol
  and receives results.

**Example: NDP-MCP** (`clio-kit/clio-kit-mcp-servers/ndp`)
- It's a Python process that runs via `ndp-mcp --transport stdio`.
- It exposes 3 tools:
  - `search_datasets(terms, limit, …)` → list of CKAN datasets
  - `list_organizations(server)` → list of NDP organizations
  - `get_dataset_details(dataset_identifier, …)` → single dataset metadata
- When the agent calls `search_datasets`, the MCP server internally makes an
  HTTP call to `http://155.101.6.191:8003/search`, parses the JSON, and
  returns it to the agent.

**MCP's role:** A standardized wrapper around external services so any
LLM agent can call them uniformly. That's it. It doesn't do any processing,
indexing, or reasoning — it just translates "tool call" into "API call" and
back.

## CLIO Search

**What it is:** A Python framework (library + data pipeline + optional MCP wrapper).
Not a protocol. It's the code that runs **behind** or **alongside** an MCP.

**Shape:** Pipeline with four stages plus a storage layer.

```
Discovery ──→ Reasoning ──→ Transform & Search ──→ Execute
    │             │                │                    │
    ▼             ▼                ▼                    ▼
 NDP-MCP    Corpus profile   Unit conversion        Connectors
 HDF5-MCP   Branch select    Formula norm           (filesystem, NDP,
 filesystem Strategy pick    Ranked retrieval        HDF5, NetCDF, S3)
 ...        Metadata quality
```

**What CLIO actually contains:**

1. **SI unit registry** (`indexing/scientific.py`)
   - 46+ units across 12 physical domains (pressure, temperature, velocity, …)
   - Dimensional analysis: each unit is a 7-tuple of SI base dimensions + scale
   - `canonicalize_measurement(value, unit)` returns a canonical value in SI
     base units. `30 degC → 303.15 K`. `5 km → 5000 m`.
   - This is arithmetic, not string matching.

2. **Connectors** (`connectors/`) — each one knows how to read a specific
   backend format:
   - `FilesystemConnector` (text, CSV)
   - `NDPConnector` (CKAN catalog via HTTP or via NDP-MCP client)
   - `HDF5Connector` (h5py, reads `units` attributes)
   - `NetCDFConnector` (xarray, CF conventions)
   - `ObjectStoreConnector` (S3)
   - `VectorStoreConnector` (Qdrant stub)

3. **DuckDB index** (`storage/duckdb_store.py`)
   - When a connector indexes a resource, CLIO extracts:
     - Text chunks (for BM25 + vector search)
     - Scientific measurements `(chunk_id, canonical_unit, canonical_value, raw_unit, raw_value)`
     - Formulas (normalized form)
     - Metadata (document-level + chunk-level key-value pairs)
   - All stored in a single DuckDB file with B-tree indices on
     `canonical_unit, canonical_value`.

4. **Retrieval coordinator** (`retrieval/coordinator.py`)
   - Given a query, it dispatches to 4 branches in parallel:
     - Lexical (BM25)
     - Vector (384-d MiniLM embeddings)
     - Graph (Neo4j stub)
     - Scientific (canonical measurement range queries on DuckDB)
   - Merges by chunk ID, keeps max per-branch score, reranks.

5. **Branch selection** (`retrieval/strategy.py`)
   - Before dispatching, CLIO profiles the namespace:
     `build_corpus_profile(storage, namespace)` returns
     `{documents, chunks, measurements, formulas, metadata_density, …}`.
   - Rule-based decisions:
     - If `profile.has_measurements == False`, skip the scientific branch.
     - If `profile.has_embeddings == False`, skip the vector branch.
     - If `operators.is_active()` and the corpus has matching data, enable scientific.
   - Result: fewer branches activated = fewer SQL queries = less work.

6. **Agentic loop** (`retrieval/agentic.py`)
   - Multi-hop wrapper around the coordinator.
   - Can call a query rewriter (LLM or deterministic SI-unit-expansion fallback)
     between hops.
   - Tracks convergence, merges citations across hops.

## Where MCP and CLIO Meet

```
                    User prompt
                         │
                         ▼
               ┌──────────────────┐
               │  Claude Agent    │
               └────────┬─────────┘
                        │  calls tools
           ┌────────────┴────────────┐
           │                         │
           ▼                         ▼
     ┌──────────┐            ┌─────────────┐
     │ NDP-MCP  │            │ CLIO tools  │
     │ (stdio)  │            │ (in-process)│
     └────┬─────┘            └──────┬──────┘
          │                         │
   HTTP   │                         │  function calls
          ▼                         ▼
   ┌─────────────┐         ┌──────────────────┐
   │ NDP CKAN    │         │ CLIO framework   │
   │ server      │         │  • SI registry   │
   └─────────────┘         │  • DuckDB index  │
                           │  • Connectors    │
                           │  • Coordinator   │
                           └──────┬───────────┘
                                  │
                         (may itself call NDP-MCP
                          or download URLs directly)
                                  ▼
                           Data sources (NDP, HDF5, …)
```

**Key points:**
- **CLIO is not an alternative to MCP.** It's a framework that can *consume*
  MCPs (CLIO's NDPConnector can use NDP-MCP internally) AND can be exposed as
  its own MCP tool for agents to call.
- **MCP handles the protocol layer.** CLIO handles the data layer (parsing,
  unit arithmetic, indexing, ranking).
- **They can coexist.** An agent with both NDP-MCP and CLIO available can use
  NDP-MCP for discovery and CLIO for precise unit-aware filtering.

## Why this matters for evaluation

A fair test must:

1. **Use the same agent** (Claude, via `claude_agent_sdk.query`).
2. **Use the same system prompt** — no "hints" telling one mode what to do.
3. **Use the same user question.**
4. **Differ only in the available tools:**
   - Mode A: `NDP-MCP` server + `Bash`
   - Mode B: `NDP-MCP` server + `Bash` + `CLIO helper tools`
5. **Not pre-index or pre-compute anything** — CLIO runs on demand inside Mode B.
6. **Let the agent freely choose** which tools to call and in what order.

That way the only variable is: *does giving the agent access to CLIO's
science-aware helpers change what it does?*

## What CLIO tools will be exposed to the agent in Mode B

These are the primitives CLIO provides. The agent can pick them, combine them
with NDP-MCP calls, or ignore them and use Bash — its choice.

| Tool | What it does |
|------|--------------|
| `clio_canonicalize_unit(value, unit)` | Convert a measurement to canonical SI base (e.g. `30 degC → 303.15 K`). Handles 46+ units. |
| `clio_parse_csv(url, column_hint, min_value, unit)` | Download a CSV, find the requested column, infer its unit from the header, canonicalize, filter by threshold. Returns count + samples. |
| `clio_search_ndp(query, min_value, max_value, unit)` | Full pipeline: discover via NDP-MCP internally, index candidate resources, apply unit-aware filter, return ranked results. |

**Note**: `clio_search_ndp` is the "high-level" one that does everything.
`clio_canonicalize_unit` and `clio_parse_csv` are "low-level" building blocks
the agent can use as helpers alongside raw NDP-MCP calls.

## What we'll measure

For each mode, we capture:
1. **Total tool calls** (raw count)
2. **Distinct tools used** (which ones the agent reached for)
3. **Input / output / cache tokens** (from `ResultMessage.usage`)
4. **Turns** (distinct assistant messages)
5. **Wall time**
6. **Full trace** (every step: thinking → tool call → tool result → next step)
7. **Final answer** (for correctness check)
8. **Workflow description** — human-readable summary of what the agent did

No forced prompts. No pre-indexing. The question is simple and the same for
both modes. The only difference is which tools are on the table.
