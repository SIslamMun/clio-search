# Roadmap: Pluggable Science-Aware Operators for Agentic Retrieval over Federated HPC Data

## Overview

SC2026 conference paper (IEEE double-column, 10 pages). Core argument: general-purpose retrieval fails on scientific data because it lacks dimensional reasoning, formula understanding, and unified search across HPC storage. We introduce science-aware retrieval operators — arithmetic SI conversion, formula normalization — as first-class retrieval branches in a federated agentic pipeline.

## Document Structure

**Type:** Conference Paper (SC2026 Data Analytics, Visualization & Storage)
**Target Length:** ~7,500 words + references
**Format:** IEEE CS double-column, 10 pages

## Domain Expertise

- .planning/structure/argument-map.md
- .planning/structure/outline.md
- .planning/structure/narrative-arc.md
- .planning/sources/literature.md
- .planning/GAP-VALIDATION.md

## Sections

- [ ] **Section 1: Introduction** — Establish the three compounding failures; state contributions
- [ ] **Section 2: Background and Related Work** — Position against hybrid retrieval, numeracy crisis, quantity-aware retrieval, HPC data discovery
- [ ] **Section 3: System Design** — Architecture, dimensional conversion, formula normalization, hybrid pipeline, federated search, agentic loop
- [ ] **Section 4: Implementation** — Code details, DuckDB schema, HDF5/NetCDF, experimental platform
- [ ] **Section 5: Evaluation** — Benchmark, baselines, dimensional results, formula results, federated results, ablation
- [ ] **Section 6: Conclusion** — Summary, limitations, future work

## Section Details

### Section 1: Introduction
**Goal**: Hook with AI agents in science, reveal three failures (dimensional blindness, formula opacity, storage fragmentation), present contributions
**Depends on**: Nothing
**Research**: Unlikely (already surveyed 55 papers)
**Wave**: 1
**Plans**: 2 plans
**Word budget**: 1,125

Plans:
- [ ] 01-01: AI agents + three failures (hook through gap)
- [ ] 01-02: Contributions list + paper organization

### Section 2: Background and Related Work
**Goal**: Position against 4 areas: hybrid retrieval, numeracy crisis, quantity-aware retrieval, scientific data discovery. End with gap table.
**Depends on**: Section 1 (builds on gap framing)
**Research**: Unlikely (literature verified and indexed)
**Wave**: 1
**Plans**: 2 plans
**Word budget**: 1,125

Plans:
- [ ] 02-01: Hybrid retrieval foundations + numeracy crisis + quantity-aware retrieval
- [ ] 02-02: Scientific data discovery + agentic retrieval + gap table

### Section 3: System Design
**Goal**: Full architecture — dimensional conversion pipeline, formula normalization, hybrid 4-branch retrieval, federated multi-namespace, agentic loop
**Depends on**: Sections 1-2 (uses gap framing and positioning)
**Research**: Unlikely (describing own system)
**Wave**: 2
**Plans**: 3 plans
**Word budget**: 1,650

Plans:
- [ ] 03-01: Architecture overview + dimensional conversion operators (core novelty)
- [ ] 03-02: Formula normalization + hybrid retrieval pipeline
- [ ] 03-03: Federated multi-namespace search + multi-hop agentic retrieval

### Section 4: Implementation
**Goal**: DuckDB schema, connector details, HDF5/NetCDF, experimental platform
**Depends on**: Section 3 (implements the design)
**Research**: Unlikely (describing own code)
**Wave**: 2
**Plans**: 1 plan
**Word budget**: 750

Plans:
- [ ] 04-01: Full implementation details (schema, connectors, HDF5/NetCDF, platform)

### Section 5: Evaluation
**Goal**: Benchmark construction, baselines, dimensional conversion results, formula matching, federated search, ablation, indexing performance
**Depends on**: Sections 3-4 (evaluates the system)
**Research**: Unlikely (presenting own results)
**Wave**: 3
**Plans**: 2 plans
**Word budget**: 1,875

Plans:
- [ ] 05-01: Setup + dimensional conversion results + formula matching
- [ ] 05-02: Federated search + ablation + agentic retrieval + indexing performance

### Section 6: Conclusion
**Goal**: Summary with evidence, limitations (honest), future work
**Depends on**: All sections
**Research**: Unlikely
**Wave**: 3
**Plans**: 1 plan
**Word budget**: 600

Plans:
- [ ] 06-01: Summary, limitations, future work

## Word Budget

| Section | Target | Current | Status |
|---------|--------|---------|--------|
| 1. Introduction | 1,125 | 0 | Not started |
| 2. Background | 1,125 | 0 | Not started |
| 3. Design | 1,650 | 0 | Not started |
| 4. Implementation | 750 | 0 | Not started |
| 5. Evaluation | 1,875 | 0 | Not started |
| 6. Conclusion | 600 | 0 | Not started |
| **Total** | **7,125** | **0** | **0%** |

## Progress

**Execution Order (waves):**
- Wave 1: Sections 1, 2 (parallel — independent)
- Wave 2: Sections 3, 4 (parallel — depend on wave 1)
- Wave 3: Sections 5, 6 (parallel — depend on wave 2)

| Section | Plans Complete | Status | Completed |
|---------|---------------|--------|-----------|
| 1. Introduction | 0/2 | Not started | - |
| 2. Background | 0/2 | Not started | - |
| 3. Design | 0/3 | Not started | - |
| 4. Implementation | 0/1 | Not started | - |
| 5. Evaluation | 0/2 | Not started | - |
| 6. Conclusion | 0/1 | Not started | - |
