# Figures Plan — v0.5 SC2026: Science-Aware Hybrid Retrieval with Dimensional Conversion for HPC Data Discovery

## Figure 1: System Architecture Overview
- Full system diagram showing: Query Interface → Namespace Registry → Retrieval Coordinator → Branch Execution → Merge/Rerank → Citations
- Show all 5 connector types (filesystem, S3, Qdrant, Neo4j, Redis)
- Highlight the scientific branch as the novel component
- **Type**: Architecture block diagram

## Figure 2: Unit Canonicalization Pipeline
- Show the flow: Raw text → Measurement extraction → Canonicalization → Storage
- Example: "250 kPa" → (250, kPa) → (250000, Pa) → DuckDB
- Query side: "190:360:kPa" → (190000, 360000, Pa) → range check → match
- **Type**: Pipeline flow diagram

## Figure 3: Hybrid Retrieval Pipeline
- Four parallel branches: BM25, Vector, Graph, Scientific
- Show over-fetching (top_k × 4)
- Merge with deduplication → Scientific filter → Metadata filter → Rerank → Top-K
- Trace events alongside each stage
- **Type**: Pipeline flow diagram

## Figure 4: Evaluation — Precision/Recall with/without Scientific Operators
- Bar chart: BM25-only vs Vector-only vs Hybrid vs Hybrid+Scientific
- On scientific queries with unit variations
- Show the recovery of false negatives when scientific operators are ON
- **Type**: Bar chart (TO BE GENERATED from evaluation)

## Figure 5: Evaluation — Ablation Study
- Table or chart showing contribution of each component
- Dimensions: lexical, vector, scientific, combined
- **Type**: Table or grouped bar chart (TO BE GENERATED)

## Figure 6: Incremental Indexing Efficiency
- Line chart: corpus size vs indexing time
- Full reindex vs incremental (with 5%, 10%, 25% changes)
- **Type**: Line chart (TO BE GENERATED)

## Table 1: Unit Canonicalization Table
- Complete mapping: raw units → canonical unit → conversion factor
- 5 domains: distance, mass, time, pressure, velocity

## Table 2: Comparison with Related Systems
- Columns: System, Dimensional Conversion, Formula Matching, Cross-Unit Range, Federated, Searches Data (not papers), BM25+Vector Hybrid
- Rows: Numbers Matter! [2024], CONE [2026], Context-1 [2026], HiPerRAG [2025], OpenScholar [2026], Datum [INL], **clio-agentic-search**
- Show that the first 5 rows (our novel capabilities) are empty for ALL other systems
