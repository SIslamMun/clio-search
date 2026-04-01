# System Design: clio-agentic-search

## v0.5 — SC2026: Science-Aware Hybrid Retrieval with Dimensional Conversion for HPC Data Discovery

---

## 1. System Overview

clio-agentic-search is a hybrid retrieval engine for scientific computing corpora. It combines lexical search (BM25), dense vector similarity, graph traversal, and novel science-aware operators into a unified retrieval pipeline with federated multi-backend support.

```
┌──────────────────────────────────────────────────────────────┐
│                     Query Interface                          │
│              (FastAPI REST  /  CLI)                           │
└────────────────────────┬─────────────────────────────────────┘
                         │  QueryRequest
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                  Namespace Registry                           │
│         (connector lifecycle, capability discovery)           │
│                                                              │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│   │local_fs  │ │object_s3 │ │vector_   │ │graph_    │      │
│   │Filesystem│ │S3 Object │ │qdrant    │ │neo4j     │      │
│   │+DuckDB   │ │Store     │ │Vector DB │ │Graph DB  │      │
│   └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘      │
│        │            │            │            │              │
│        └────────────┴────────────┴────────────┘              │
│                         │                                    │
│              Capability Negotiation                           │
│     (detect: lexical? vector? graph? scientific?)            │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                 Retrieval Coordinator                         │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           Parallel Branch Execution                  │    │
│  │                                                      │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │    │
│  │  │ Lexical  │ │ Vector   │ │ Graph    │ │Science │ │    │
│  │  │ (BM25)   │ │ (Embed)  │ │(Traverse)│ │(Units) │ │    │
│  │  │ top_k×4  │ │ top_k×4  │ │ top_k×2  │ │top_k×4 │ │    │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘ │    │
│  │       └─────────────┴────────────┴───────────┘      │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Candidate Merging (deduplicate by chunk_id)         │   │
│  │  → Scientific Filtering (if operators active)        │   │
│  │  → Metadata Filtering (key=value constraints)        │   │
│  │  → Heuristic Reranking (lexical + vector + metadata) │   │
│  │  → Top-K Selection                                   │   │
│  └──────────────────────────────────────────────────────┘   │
│                         │                                    │
│                    Citations + Trace Events                   │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Unit-Aware Scientific Operators (Core Novelty)

### 2.1 The Problem

Standard retrieval treats measurements as strings:
- Query: "pressure between 190 and 360 kPa"
- Document contains: "The measured pressure was 200000 Pa"
- BM25: no token overlap on the value → **miss**
- Vector: weak semantic similarity → **low rank or miss**
- Result: **false negative** on a scientifically correct match

### 2.2 SI Dimensional Conversion (Key Differentiator from Prior Art)

Prior quantity-aware retrieval [Almasian et al., 2022; 2024] normalizes unit *strings* — mapping "km/h" to "kilometer per hour." This handles synonyms but **cannot resolve cross-prefix equivalence**: "200 kPa" and "200000 Pa" remain unmatched because "kilopascal" ≠ "pascal."

Our approach performs **arithmetic conversion** using SI multiplication factors:

```
Prior art (string normalization):
  "200 kPa"    → unit="kilopascal", value=200
  "200000 Pa"  → unit="pascal",     value=200000
  ❌ Different unit strings → NO MATCH

Our system (SI dimensional conversion):
  "200 kPa"    → value × factor = 200 × 1000 = 200000, canonical_unit = "pa"
  "200000 Pa"  → value × factor = 200000 × 1  = 200000, canonical_unit = "pa"
  ✓ Same canonical value + unit → MATCH
```

At **index time**, we extract measurements from text and canonicalize to base SI units:

```
Input text: "The pressure reached 250 kPa at 15 min"

Extracted measurements:
  (250, "kPa") → canonical: (250 × 1000 = 250000.0, "pa")
  (15, "min")  → canonical: (15 × 60 = 900.0, "s")
```

**SI Conversion Table (implemented):**

| Domain | Raw Units | Base SI Unit | Conversion Factor |
|--------|-----------|-------------|-------------------|
| Distance | mm, cm, m, km | m (meter) | ×10⁻³, ×10⁻², ×1, ×10³ |
| Mass | mg, g, kg | kg (kilogram) | ×10⁻⁶, ×10⁻³, ×1 |
| Time | s, min, h | s (second) | ×1, ×60, ×3600 |
| Pressure | Pa, kPa, MPa | Pa (pascal) | ×1, ×10³, ×10⁶ |
| Velocity | m/s, km/h | m/s | ×1, ×(1000/3600) |

At **query time**, numeric range operators are canonicalized identically:

```
Query: --numeric-range "190:360:kPa"
Canonical: min = 190 × 1000 = 190000 Pa, max = 360 × 1000 = 360000 Pa

Document A: "250 kPa"   → 250 × 1000 = 250000 Pa → 190000 ≤ 250000 ≤ 360000 → MATCH ✓
Document B: "200000 Pa"  → 200000 × 1 = 200000 Pa → 190000 ≤ 200000 ≤ 360000 → MATCH ✓
Document C: "0.3 MPa"   → 0.3 × 1e6  = 300000 Pa → 190000 ≤ 300000 ≤ 360000 → MATCH ✓

String normalization would only find Document A. We find all three.
```

### 2.3 Measurement Extraction

Pattern-based extraction from text using regex:

```
Pattern: [+-]?\d+(\.\d+)?\s*(km/h|m/s|km|cm|mm|m|kg|mg|g|h|min|s|mpa|kpa|pa)
```

Each match produces: `(raw_value, raw_unit, canonical_value, canonical_unit)`

Stored in DuckDB `scientific_measurements` table per chunk.

### 2.4 Formula Normalization

Mathematical expressions are normalized for matching:

```
Input:  "E = m * c^{2}"
Step 1: strip whitespace     → "E=m*c^{2}"
Step 2: lowercase            → "e=m*c^{2}"
Step 3: normalize superscripts → "e=m*c^2"  (^{2}→^2, **2→^2)
Step 4: split on =           → ["e", "m*c^2"]
Step 5: sort factors          → ["e", "c^2*m"]
Step 6: canonical form        → "e=c^2*m"
```

Stored in DuckDB `scientific_formulas` table per chunk.

### 2.5 Scientific Scoring

When scientific operators are active, each candidate chunk receives a science score:

| Operator | Match Score |
|----------|------------|
| Numeric range match | +1.2 |
| Unit match | +1.0 |
| Formula match | +1.4 |
| No match on active operator | 0.0 (hard filter — chunk dropped) |

---

## 3. Hybrid Retrieval Pipeline

### 3.1 Branch Execution

For each query, the Retrieval Coordinator runs available branches in parallel:

1. **Lexical Branch (Okapi BM25, k1=1.2, b=0.75)**
   - Tokenize query → look up posting lists in DuckDB `lexical_postings`
   - Score: IDF(t) × (tf × (k1+1)) / (tf + k1 × (1 - b + b × dl/avgdl))
   - IDF, document frequency, average chunk length computed in-query via DuckDB CTEs
   - Return `top_k × 4` candidates (over-fetch for merge)

2. **Vector Branch (Embedding Similarity)**
   - Embed query using sentence-transformers (all-MiniLM-L6-v2, 384-dim)
   - Cosine similarity against stored chunk embeddings
   - Uses ExactANNAdapter (sharded hash-based) or HnswANNAdapter
   - Return `top_k × 4` candidates

3. **Graph Branch (Traversal)**
   - Graph-based neighbor expansion from initial matches
   - Return `top_k × 2` candidates

4. **Scientific Branch (Unit/Formula)**
   - Only active when scientific operators are specified
   - Query DuckDB `scientific_measurements` and `scientific_formulas`
   - Filter by numeric range, unit match, or formula match
   - Return `top_k × 4` candidates

### 3.2 Merge and Rerank

```
All candidates (deduplicated by chunk_id)
    │
    ▼
Scientific Filtering (drop chunks failing active operators)
    │
    ▼
Metadata Filtering (key=value constraints)
    │
    ▼
Heuristic Reranking: score = lexical_score + vector_score + metadata_score
    │
    ▼
Top-K selection → Citations with snippets
```

### 3.3 Trace Events

Every query produces a trace with 8-10 events:
`query_started → lexical_completed → vector_completed → graph_completed → merge_completed → scientific_filter_completed → metadata_completed → rerank_completed → query_completed`

---

## 4. Federated Multi-Namespace Search

### 4.1 Connector Protocol

Each backend implements the `NamespaceConnector` protocol:

```python
class NamespaceConnector(Protocol):
    def connect(self) -> None
    def teardown(self) -> None
    def index(self, ...) -> IndexReport
    def descriptor(self) -> NamespaceDescriptor
    def build_citation(self, chunk) -> Citation
```

### 4.2 Capability Negotiation

The coordinator dynamically detects what each connector supports:

```python
class RetrievalCapabilities(Protocol):
    def search_lexical(query, top_k) -> list[ScoredChunk]    # optional
    def search_vector(query, top_k) -> list[ScoredChunk]     # optional
    def search_graph(query, top_k) -> list[ScoredChunk]      # optional
    def search_scientific(query, top_k, operators) -> list    # optional
```

Only branches with available capabilities are executed. This means:
- FilesystemConnector: lexical ✓, vector ✓, scientific ✓
- QdrantConnector: vector ✓
- Neo4jConnector: graph ✓
- RedisConnector: (log retrieval)

### 4.3 Multi-Namespace Query

```
Query → for each namespace:
          │
          ├→ single-namespace pipeline (parallel branches → merge → rerank)
          │
        collect all citations across namespaces
          │
        global sort by score → deduplicate → top-k
```

---

## 5. Indexing Pipeline

### 5.1 Document Processing

```
Filesystem scan
    │
    ▼
Change detection (SHA256 + mtime)
    │
    ▼ (only changed/new files)
    │
Chunking (fixed-size text chunks)
    │
    ├→ BM25 postings (LexicalPostingsIngestor)
    │    - tokenize, stopword filter, DF pruning (threshold 0.98)
    │    - batch insert to `lexical_postings` table
    │
    ├→ Embeddings (SentenceTransformerEmbedder or HashEmbedder)
    │    - 384-dim vectors stored in `embeddings` table
    │    - indexed in ExactANNAdapter (sharded, 16 shards)
    │
    ├→ Scientific extraction
    │    - measurement extraction → `scientific_measurements` table
    │    - formula extraction → `scientific_formulas` table
    │
    └→ Metadata extraction → `metadata` table
```

### 5.2 Incremental Indexing

```
file_index table: (namespace, path, document_id, mtime_ns, content_hash)

On index:
  1. Scan filesystem, compute SHA256 for each file
  2. Compare against stored state
  3. If mtime + hash match → skip
  4. If changed → re-index chunks
  5. If deleted → remove from index
  6. Report: (scanned, indexed, skipped, removed, elapsed_seconds)
```

---

## 6. Storage Layer

**DuckDB** as the single embedded storage backend (no external dependencies for basic use).

| Table | Purpose |
|-------|---------|
| `documents` | Document metadata (URI, namespace, checksums) |
| `chunks` | Text chunks with document references |
| `embeddings` | Dense vectors (384-dim) per chunk |
| `metadata` | Key-value metadata per document |
| `file_index` | Incremental indexing state |
| `lexical_postings` | BM25 token → chunk mappings with term frequency |
| `scientific_measurements` | Extracted measurements (raw + canonical) |
| `scientific_formulas` | Extracted formulas (raw + normalized) |

Write safety: file-based locking (fcntl on Unix) for concurrent access.

---

## 7. API and CLI

### REST API (FastAPI)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness probe |
| GET | `/version` | Package version |
| GET | `/documents` | List indexed documents |
| POST | `/query` | Hybrid retrieval with scientific operators |
| POST | `/jobs/index` | Async background indexing |
| GET | `/jobs/{id}` | Job status |
| DELETE | `/jobs/{id}` | Cancel job |
| GET | `/metrics` | Prometheus metrics |

### CLI

```bash
clio query --q "pressure > 200 kPa" --namespace local_fs --top-k 5
clio query --numeric-range "190:360:kPa"
clio query --unit-match "200:kg"
clio query --formula "F=ma"
clio index --namespace local_fs
clio list --namespace local_fs
clio seed
clio serve --port 8000
```

---

## 8. Observability

- **OpenTelemetry** tracing (opt-in via `CLIO_OTEL_ENABLED`)
- **Prometheus** metrics at `/metrics`
- **Trace events** in every query response (8-10 stages with timestamps)
- **Retry with backoff** via tenacity for transient failures
