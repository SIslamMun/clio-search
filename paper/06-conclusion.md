# 6. Conclusion

## 6.1 Summary

This paper introduced science-aware retrieval operators---a new class of deterministic, pluggable retrieval primitives that execute as parallel branches alongside standard BM25 and dense vector search. We demonstrated that three compounding failures in scientific data retrieval---dimensional blindness, formula opacity, and storage fragmentation---can be addressed through explicit arithmetic and symbolic transformations rather than learned approximations. Five contributions substantiate this claim.

First, dimensional-conversion measurement retrieval canonicalizes physical quantities to base SI units via explicit arithmetic multiplication. By converting all indexed measurements to canonical form at ingest time and applying the same conversion to query-time range bounds, the operator recovers [TBD]\% of false negatives that lexical, dense, and string-normalization systems miss entirely. The correctness guarantee is structural: if the SI conversion factor is correct, the match is correct. No learned embedding or string normalization can provide this guarantee.

Second, unified formula normalization reduces inconsistent mathematical surface forms to a canonical representation through a deterministic six-step pipeline. Integrating formula matching with dimensional operators as parallel branches in a single hybrid retrieval pipeline yields a [TBD] F1 improvement over retrieval without formula awareness. No prior system combines formula normalization with quantity-aware search.

Third, federated multi-namespace search with capability negotiation unifies retrieval across filesystem, S3, Qdrant, Neo4j, HDF5, and NetCDF backends without requiring data migration. The connector architecture discovers [TBD]\% more relevant documents than any single-backend configuration, because scientific data is inherently distributed across heterogeneous storage tiers.

Fourth, dedicated HDF5 and NetCDF connectors extract datasets, attributes, variables, dimensions, and units from binary scientific formats and pass them through the same SI canonicalization pipeline as text-based connectors. This enables cross-format dimensional search: a pressure range query matches documents on a filesystem and variables inside an HDF5 file through a single, unified retrieval call.

Fifth, multi-hop agentic retrieval employs an LLM-driven query rewriting loop that iteratively refines searches using expand, narrow, and pivot strategies. This mechanism improves Recall by [TBD] over single-shot retrieval, converging in 2--3 hops as result sets stabilize.

Taken together, these contributions establish science-aware operators as a distinct class of retrieval primitives that complement rather than replace standard hybrid retrieval. As AI agents increasingly orchestrate scientific workflows \cite{coscientist2023, aiscientistv2_2025, ornl_agents_sc25}, the retrieval layer they depend on must understand the data it searches---not merely the text that describes it.

## 6.2 Limitations

We identify four limitations that bound the scope of our claims.

**Unit coverage.** The current conversion table spans 13 units across 5 SI domains. CQE \cite{cqe2023} supports 531 units via dependency parsing, and the QUDT ontology covers thousands. Our system does not handle derived units (e.g., N = kg$\cdot$m/s$^2$), compound units with multiple denominators, or non-SI unit systems such as imperial or CGS. Extending coverage requires populating the conversion table, not changing the architecture, but the gap is real for corpora using specialized units.

**Heuristic reranking.** The composite fusion score is a hand-tuned weighted combination of branch scores. We do not employ learned reranking via cross-encoders \cite{bruch2023fusion} or late-interaction models such as ColBERT. A learned fusion strategy would likely improve precision at the cost of additional inference latency.

**Regex-based extraction.** Measurement and formula extraction relies on regular expressions rather than NLP-based parsing. CQE \cite{cqe2023} uses dependency parsing to handle complex quantity expressions (e.g., "between 200 and 300 kPa at 25 degrees Celsius"). Our regex patterns may miss such expressions or produce incomplete extractions on syntactically complex scientific prose.

**No HPC-scale deployment.** All experiments run on a benchmark corpus, not a petabyte-scale HPC filesystem. While the federated architecture is designed for scale---connectors are stateless, indexing is incremental, and DuckDB handles analytical queries efficiently---we have not validated performance on production HPC systems at facilities such as ALCF or OLCF.

## 6.3 Future Work

Five directions follow directly from these limitations. First, we plan to expand the unit ontology by integrating the QUDT knowledge graph or the CQE 531-unit dictionary \cite{cqe2023}, adding support for derived and compound units. Second, we intend to replace heuristic fusion with a learned reranking stage, evaluating cross-encoder and ColBERT-based approaches on scientific retrieval benchmarks. Third, wrapping the retrieval API as a Model Context Protocol (MCP) server \cite{mcp_hpc2025} would expose science-aware operators to any MCP-compatible agent, enabling plug-and-play integration with frameworks such as LangChain, AutoGen, and Claude. Fourth, deployment on DOE leadership-class systems---Polaris at ALCF and Frontier at OLCF---will validate federated search at HPC scale, where corpora span millions of files across parallel filesystems and object stores. Fifth, integrating PROV-IO+ \cite{provio2024} provenance graphs would enable retrieval queries that incorporate data lineage, connecting search results to the experimental workflows that produced them.
