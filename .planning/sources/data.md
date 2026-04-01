# Data and Evidence Index

## Figures (Planned)

| ID | Description | Status | Target Section |
|----|-------------|--------|----------------|
| Fig 1 | System architecture: Query → Registry → Coordinator → Branches → Citations | Not created | §3 Design |
| Fig 2 | Unit canonicalization pipeline: raw text → extraction → SI conversion → DuckDB → range check | Not created | §3.2 |
| Fig 3 | Hybrid retrieval pipeline: 4 parallel branches → merge → filter → rerank → top-K | Not created | §3.3 |
| Fig 4 | Evaluation bar chart: BM25 vs Vector vs Hybrid vs Hybrid+Scientific on unit-variation queries | **Not run** | §4 Evaluation |
| Fig 5 | Ablation study: contribution of each retrieval branch | **Not run** | §4 Evaluation |
| Fig 6 | Incremental indexing efficiency: corpus size vs time, full vs incremental | **Not run** | §4 Evaluation |

## Tables (Planned)

| ID | Description | Status | Target Section |
|----|-------------|--------|----------------|
| Table 1 | Comparison vs prior systems (7 capabilities × 7 systems) | Draft exists in novelty.md | §2 Background |
| Table 2 | SI unit canonicalization table (5 domains × conversions) | Defined in architecture.md | §3 Design |
| Table 3 | Evaluation results (TBD) | **Not run** | §4 Evaluation |

## Key Statistics Available

| Stat | Source | Use |
|------|--------|-----|
| 0.54 accuracy on numbers | Deng et al., EACL 2026 (Numeracy Gap) | Introduction + Background |
| 16.3% accuracy on numeric constraint queries | NC-Retriever, NumConQ 2025 | Background §2.2 |
| 32% agent success on science tasks | ScienceAgentBench, NeurIPS 2024 | Introduction |
| 88% answer recall (Context-1) vs 58% single-pass | Chroma 2026 | Background §2.4 |
| 136/138 tests passing | clio-agentic-search test suite | §3 Implementation |
| 25% Recall@10 improvement | CONE, arXiv:2603.04741 | Background §2.3 |
| 77.5% query reduction | RAGRoute, EuroMLSys 2025 | Background §2.5 |
| <3.5% overhead | PROV-IO+, IEEE TPDS 2024 | Background §2.5 |
| 45M papers | OpenScholar, Nature 2026 | Introduction |
| 3.6M articles | HiPerRAG, PASC 2025 | Introduction |
| 100× beamline prep speedup | ALS Agents, arXiv:2509.17255 | Introduction |

## Critical Gap: Evaluation Data

**The paper currently has NO experimental evaluation.** The evaluation section (§4) requires:
- Benchmark dataset with unit-variation queries (needs to be created or adapted)
- Comparison baselines: BM25, dense vector, Numbers Matter! string normalization, CONE-style learned
- Metrics: Precision@K, Recall@K, F1, MRR
- Ablation across retrieval branches
- Indexing performance benchmarks

This is the largest missing piece for SC2026.
