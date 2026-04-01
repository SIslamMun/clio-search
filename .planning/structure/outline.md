# Paper Outline — IEEE CS (SC2026)

Total budget: ~7,500 words (10 IEEE double-column pages)

| § | Section | Word Budget | Word Ratio | Status |
|---|---------|-------------|------------|--------|
| 1 | Introduction | 1,125 | 15% | Not started |
| 2 | Background and Related Work | 1,125 | 15% | Not started |
| 3 | System Design | 1,650 | 22% | Not started |
| 4 | Implementation | 750 | 10% | Not started |
| 5 | Evaluation | 1,875 | 25% | Not started |
| 6 | Conclusion | 600 | 8% | Not started |
| — | Abstract | 250 | — | Not started |
| — | References | — | — | Not started |

---

## Abstract (250 words)
- Problem: 3 failures (dimensional, formula, storage)
- Solution: science-aware retrieval operators
- Results: dimensional conversion recovers X% false negatives
- Significance: new class of retrieval primitives for HPC data

## §1 Introduction (1,125 words)

### 1.1 Motivation — AI agents entering HPC (~300 words)
- Agents doing science: Coscientist, AI Scientist, ALS, ORNL SC'25
- MCP as interface: MCP for HPC [arXiv:2508.18489]
- But: ScienceAgentBench 32%, agents fail at data not reasoning

### 1.2 Three Failures (~400 words)
- Dimensional blindness: 0.54 accuracy, kPa≠Pa in all systems
- Formula opacity: no system normalizes across notations
- Storage fragmentation: data across 5+ tiers, no unified search

### 1.3 Contributions (~300 words)
1. Dimensional-conversion measurement retrieval (arithmetic SI)
2. Formula normalization unified with dimensional operators
3. Federated multi-namespace search with capability negotiation
4. HDF5/NetCDF metadata indexing
5. Multi-hop agentic retrieval with LLM query rewriting

### 1.4 Paper Organization (~125 words)

## §2 Background and Related Work (1,125 words)

### 2.1 Hybrid Retrieval (~200 words)
- BM25 + dense, fusion methods, SPLADE

### 2.2 The Numerical Crisis (~200 words)
- Numeracy Gap, NumConQ, Numbers Not Continuous, NumericBench

### 2.3 Quantity-Aware Retrieval (~300 words)
- QFinder, CQE, Numbers Matter! (string normalization)
- CONE, NC-Retriever (learned embeddings)
- SciHarvester, MeasEval
- Gap: no arithmetic conversion

### 2.4 Scientific Data Discovery in HPC (~200 words)
- OpenScholar, HiPerRAG: papers not data
- PROV-IO+, HDF Clinic: provenance not search
- BRINDEXER: POSIX metadata only

### 2.5 Agentic Retrieval (~225 words)
- Context-1, A-RAG, MA-RAG: better orchestration, same retrieval
- Academy, ORNL agents: federated agent infrastructure
- Gap: no science-aware operators

## §3 System Design (1,650 words)

### 3.1 Architecture Overview (~300 words)
- System diagram (Fig 1)
- Query → Namespace Registry → Coordinator → Branches → Merge → Citations

### 3.2 Dimensional Conversion Operators (~400 words)
- SI conversion table, arithmetic canonicalization
- Measurement extraction pipeline (Fig 2)
- Numeric range, unit match, value match operators
- Correctness guarantee argument

### 3.3 Formula Normalization (~200 words)
- Normalization steps: whitespace, superscripts, factor ordering, side sorting
- Integration with dimensional operators

### 3.4 Hybrid Retrieval Pipeline (~300 words)
- 4 parallel branches: BM25, vector, graph, scientific (Fig 3)
- Merge, deduplication, scientific filtering, heuristic reranking

### 3.5 Federated Multi-Namespace Search (~250 words)
- Connector protocol, capability negotiation
- FS, S3, Qdrant, Neo4j, HDF5/NetCDF connectors
- Per-backend branch execution

### 3.6 Multi-hop Agentic Retrieval (~200 words)
- Iterative search loop with LLM query rewriting
- Refinement and re-query strategy

## §4 Implementation (750 words)

### 4.1 System Implementation (~400 words)
- Python, FastAPI, DuckDB, sentence-transformers
- DuckDB schema: 8 tables
- Incremental indexing with SHA256 change detection

### 4.2 HDF5/NetCDF Connector (~200 words)
- h5py/xarray integration
- Attribute extraction, metadata indexing

### 4.3 Experimental Platform (~150 words)
- Hardware, software versions, model specs

## §5 Evaluation (1,875 words)

### 5.1 Experimental Setup (~300 words)
- Benchmark: scientific retrieval tasks with unit-variation queries
- Baselines: BM25, dense vector, hybrid (no science), Numbers Matter!-style string normalization
- Metrics: Precision@K, Recall@K, F1, MRR

### 5.2 Dimensional Conversion Results (~400 words)
- Cross-prefix matching: recall improvement over baselines
- Range query accuracy across SI domains
- Bar chart (Fig 4): Hybrid vs Hybrid+Scientific

### 5.3 Formula Matching Results (~250 words)
- Formula retrieval accuracy across notation variants

### 5.4 Federated Search Results (~300 words)
- Multi-backend discovery vs single-backend
- Coverage analysis

### 5.5 Ablation Study (~325 words)
- Contribution of each branch: lexical, vector, scientific, graph
- Table/chart (Fig 5)

### 5.6 Indexing Performance (~300 words)
- Full vs incremental indexing
- Corpus size scaling (Fig 6)

## §6 Conclusion (600 words)

### 6.1 Summary (~300 words)
- Three contributions, key results

### 6.2 Limitations (~150 words)
- 14 units (small coverage), no HPC-scale test, heuristic reranking

### 6.3 Future Work (~150 words)
- Learned reranking, larger unit ontology, HPC deployment
