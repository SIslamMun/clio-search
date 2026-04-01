# Literature Index

Source: `background/papers/01_prior_art_and_competitors.md` + `background/background.md`

## Core References

| Key | Citation | Relevance | Status |
|-----|----------|-----------|--------|
| Numbers Matter! | Almasian et al., EMNLP 2024, arXiv:2407.10283 | Closest prior art — quantity-aware retrieval, string normalization | Must cite + differentiate |
| QFinder | Almasian & Gertz, SIGIR 2022 | Elasticsearch quantity ranking, string normalization | Must cite |
| CQE | Almasian et al., EMNLP 2023, arXiv:2305.08853 | 531-unit extraction framework | Must cite |
| CONE | Shrestha et al., arXiv:2603.04741, March 2026 | Learned unit embeddings, 25% Recall@10 gain | Must cite + differentiate |
| NC-Retriever | Tongji-KGLLM, SciOpen 2025 | NumConQ: 16.3% accuracy on numeric constraints | Cite as evidence of problem |
| Numeracy Gap | Deng et al., EACL 2026, arXiv:2509.05691 | 0.54 embedding accuracy on numbers | Key evidence stat |
| Context-1 | Chroma, March 2026 | 20B agentic search, 88% recall, BM25+dense | Background motivation |
| HiPerRAG | Gokdemir et al., PASC 2025, arXiv:2505.04846 | HPC RAG over 3.6M papers, Argonne/ORNL | HPC domain context |
| OpenScholar | Asai et al., Nature 2026 | 45M papers, citation accuracy | Contrast: searches papers not data |
| ScienceAgentBench | Chen et al., NeurIPS 2024 | 32% agent success on science tasks | Agent failure evidence |
| MLAgentBench | Huang et al., ICML 2024 | Agents fail at data prep | Confirms data retrieval gap |
| PROV-IO+ | IEEE TPDS 2024 | HPC I/O provenance, <3.5% overhead | No search interface |
| HDF Clinic | HDF Group, 2025 | KG over HDF5 — vision, no implementation | Gap confirmation |
| RAGRoute | Guerraoui et al., EuroMLSys 2025, arXiv:2502.19280 | Federated RAG routing, 77.5% query reduction | Federated RAG context |
| MCP for HPC | arXiv:2508.18489 | Wrap HPC resources under MCP | Architectural predecessor |
| Souza et al. SC'25 | arXiv:2509.13978 | MCP-based LLM agents at ORNL | SC precedent |
| ORNL Agents SC'25 | dl.acm.org/10.1145/3731599.3767592 | Autonomous cross-facility experiments | SC precedent |
| Agentic AI Survey | ICLR 2025, arXiv:2503.08979 | Data mgmt underexplored | Gap confirmation |
| ALS Beamline | arXiv:2509.17255 | 100× beamline prep speedup | Agent in HPC context |
| Coscientist | Nature 2023 | Autonomous wet-lab experiments | Agents doing science |
| AI Scientist v2 | Yamada et al., arXiv:2504.08066 | End-to-end paper writing | Agents doing science |
| GSM-Symbolic | Apple, ICLR 2025, arXiv:2410.05229 | LLMs drop 65% on numeric changes | Numbers crisis |
| Numbers Not Continuous | arXiv:2510.08009 | Numbers encoded per-digit, not magnitude | Numbers crisis |
| BGE-M3 | Chen et al., ACL 2024, arXiv:2402.03216 | Hybrid retrieval foundation | Background |
| Bruch et al. | ACM TOIS 2023, arXiv:2210.11934 | Convex fusion > RRF | Background |
| BRINDEXER | Paul et al., CCGrid 2020 | POSIX metadata indexing on Lustre | HPC storage context |
| A-RAG | Du et al., arXiv:2602.03442 | Hierarchical retrieval interfaces | Related retrieval |
| Academy | Pauloski et al., arXiv:2505.05428 | Federated agent middleware | HPC agent context |
| SciHarvester | Rybinski et al., SIGIR 2023 | Numerical search for agronomy | Quantity retrieval |

## To Find / Verify

- [ ] Datum (INL, 2024) — exact citation needed
- [ ] NC-Retriever full citation — SciOpen journal details
- [ ] SSEmb CIKM 2025 — formula retrieval
- [ ] MIRB 2025 — math-aware retrieval
- [ ] Approach0 — equation search engine
- [ ] NumericBench ACL 2025
- [ ] Number Cookbook ICLR 2025
- [ ] Chakraborty et al. 2025 — federated RAG survey, arXiv:2505.18906
- [ ] HiPRAG — Wu et al. arXiv:2510.07794
- [ ] DF-RAG 2026 — need citation
- [ ] PROV-AGENT eScience 2025, arXiv:2508.02866
