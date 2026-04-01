# Argument Map

## Central Thesis

General-purpose retrieval fails on scientific data. We introduce science-aware retrieval operators — dimensional conversion, formula matching, and federated multi-backend search — as a new class of retrieval primitives for AI-driven HPC data discovery.

## Supporting Claims

### Claim 1: Retrieval systems are dimensionally blind
- Evidence: "200 kPa" ≠ "200000 Pa" in every existing system
- Evidence: Numeracy Gap (EACL 2026) — 0.54 embedding accuracy on numbers
- Evidence: NC-Retriever NumConQ — 16.3% dense retrieval accuracy on numeric constraints
- Evidence: GSM-Symbolic (Apple, ICLR 2025) — LLMs drop 65% on numeric changes
- Differentiation: Numbers Matter! does string normalization ("kilopascal"→"pascal" still fails); CONE learns embeddings (probabilistic, not guaranteed); we do arithmetic (200×1000=200000, guaranteed)

### Claim 2: Formula matching is absent from retrieval
- Evidence: No system matches "F=ma" with "F = m · a" with "ma=F"
- Evidence: Math-aware search (Approach0, SSEmb) handles LaTeX structure but not physical measurement integration
- Differentiation: We normalize formulas AND integrate with dimensional operators in one pipeline

### Claim 3: HPC scientific data is unfindable
- Evidence: OpenScholar searches 45M papers, not data; HiPerRAG: 3.6M papers, not data
- Evidence: ScienceAgentBench (ICLR 2025): best agent = 32.4%, failures at data handling
- Evidence: HDF Clinic: KG over HDF5 envisioned, not implemented
- Evidence: PROV-IO+: captures provenance but no search interface
- Differentiation: We search the data itself across heterogeneous HPC storage

### Claim 4: These problems compound
- The motivating example: 3 documents across 3 backends with 3 unit representations
- Only our system finds all three
- No prior system addresses even two of the three failures

## Logical Flow

```
Numbers are broken in retrieval (0.54 accuracy)
  ↓
Scientific data has numbers with units (dimensional quantities)
  ↓
String normalization can't cross SI prefixes (Numbers Matter! fails)
  ↓
Learned embeddings can't guarantee equivalence (CONE: probabilistic)
  ↓
We do arithmetic conversion (guaranteed correct by construction)
  ↓
Scientific data also has formulas → we normalize and match
  ↓
Scientific data lives across HPC storage tiers → we federate
  ↓
Combined: science-aware operators as a new class of retrieval primitives
  ↓
Evaluation: dimensional conversion recovers false negatives that all baselines miss
```

## Gaps / Risks

- Evaluation data doesn't exist yet — must build benchmark
- ~~HDF5/NetCDF connector~~ — IMPLEMENTED (2026-04-01)
- ~~Multi-hop / LLM rewriting~~ — IMPLEMENTED (2026-04-01)
- Real Qdrant/Neo4j integration needed (currently in-memory mocks)
- Only 13 units supported vs CQE's 531 — must acknowledge
- No HPC-scale testing — cannot claim "HPC-scale" without evidence
- PANGAEA-GPT (2026) searches geoscientific data — must differentiate clearly
