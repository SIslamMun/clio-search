"""Multi-hop agentic retrieval with iterative query refinement."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from clio_agentic_search.core.connectors import NamespaceConnector
from clio_agentic_search.models.contracts import CitationRecord, TraceEvent
from clio_agentic_search.retrieval.coordinator import RetrievalCoordinator
from clio_agentic_search.retrieval.query_rewriter import (
    FallbackQueryRewriter,
    QueryRewriter,
    RewriteResult,
)
from clio_agentic_search.retrieval.scientific import ScientificQueryOperators


@dataclass(frozen=True, slots=True)
class HopRecord:
    hop_number: int
    query: str
    strategy: str
    reasoning: str
    citations_found: int
    new_citations: int


@dataclass(frozen=True, slots=True)
class AgenticQueryResult:
    namespace: str
    original_query: str
    final_query: str
    citations: list[CitationRecord]
    hops: list[HopRecord]
    trace: list[TraceEvent]
    total_hops: int


@dataclass(slots=True)
class AgenticRetriever:
    coordinator: RetrievalCoordinator = field(default_factory=RetrievalCoordinator)
    rewriter: QueryRewriter | FallbackQueryRewriter = field(
        default_factory=FallbackQueryRewriter
    )
    max_hops: int = 3
    min_score_threshold: float = 0.5
    convergence_threshold: int = 0  # stop if no new citations found

    def query(
        self,
        *,
        connector: NamespaceConnector,
        query: str,
        top_k: int = 5,
        metadata_filters: dict[str, str] | None = None,
        scientific_operators: ScientificQueryOperators | None = None,
    ) -> AgenticQueryResult:
        """Run multi-hop agentic retrieval against a single namespace."""
        namespace = connector.descriptor().name
        trace: list[TraceEvent] = []
        hops: list[HopRecord] = []
        all_citations: dict[str, CitationRecord] = {}  # chunk_id -> best citation
        current_query = query

        self._append_trace(
            trace=trace,
            stage="agentic_started",
            message="agentic retrieval loop started",
            attributes={
                "query": query,
                "namespace": namespace,
                "max_hops": str(self.max_hops),
            },
        )

        for hop_number in range(1, self.max_hops + 1):
            self._append_trace(
                trace=trace,
                stage="hop_started",
                message=f"hop {hop_number} started",
                attributes={
                    "hop": str(hop_number),
                    "query": current_query,
                },
            )

            result = self.coordinator.query(
                connector=connector,
                query=current_query,
                top_k=top_k,
                metadata_filters=metadata_filters,
                scientific_operators=scientific_operators,
            )
            trace.extend(result.trace)

            # Merge citations — keep highest score per chunk_id.
            new_count = 0
            for citation in result.citations:
                existing = all_citations.get(citation.chunk_id)
                if existing is None:
                    all_citations[citation.chunk_id] = citation
                    new_count += 1
                elif citation.score > existing.score:
                    all_citations[citation.chunk_id] = citation

            hop_record = HopRecord(
                hop_number=hop_number,
                query=current_query,
                strategy="initial" if hop_number == 1 else "rewrite",
                reasoning="",
                citations_found=len(result.citations),
                new_citations=new_count,
            )

            self._append_trace(
                trace=trace,
                stage="hop_completed",
                message=f"hop {hop_number} completed",
                attributes={
                    "citations_found": str(len(result.citations)),
                    "new_citations": str(new_count),
                    "total_accumulated": str(len(all_citations)),
                },
            )

            # Check if best results meet score threshold.
            max_score = max(
                (c.score for c in result.citations), default=0.0
            )
            results_sufficient = (
                max_score >= self.min_score_threshold and len(result.citations) > 0
            )

            # Decide whether to continue.
            if hop_number >= self.max_hops:
                hops.append(hop_record)
                break

            # Convergence: no new citations.
            if new_count <= self.convergence_threshold and hop_number > 1:
                hops.append(
                    HopRecord(
                        hop_number=hop_record.hop_number,
                        query=hop_record.query,
                        strategy="converged",
                        reasoning="No new citations found; stopping.",
                        citations_found=hop_record.citations_found,
                        new_citations=hop_record.new_citations,
                    )
                )
                self._append_trace(
                    trace=trace,
                    stage="convergence_reached",
                    message="no new citations; stopping",
                    attributes={"hop": str(hop_number)},
                )
                break

            # Ask the rewriter whether/how to refine.
            snippets = [c.snippet for c in result.citations if c.snippet]
            rewrite_result: RewriteResult = self.rewriter.rewrite(
                query=current_query,
                retrieved_snippets=snippets,
                hop_number=hop_number,
                max_hops=self.max_hops,
            )

            hop_record = HopRecord(
                hop_number=hop_record.hop_number,
                query=hop_record.query,
                strategy=rewrite_result.strategy,
                reasoning=rewrite_result.reasoning,
                citations_found=hop_record.citations_found,
                new_citations=hop_record.new_citations,
            )
            hops.append(hop_record)

            self._append_trace(
                trace=trace,
                stage="rewrite_completed",
                message=f"rewriter chose strategy={rewrite_result.strategy}",
                attributes={
                    "hop": str(hop_number),
                    "strategy": rewrite_result.strategy,
                    "rewritten_query": rewrite_result.rewritten_query,
                    "reasoning": rewrite_result.reasoning,
                },
            )

            if rewrite_result.strategy == "done":
                break
            if results_sufficient and rewrite_result.strategy == "done":
                break

            current_query = rewrite_result.rewritten_query
            continue

        # Final citation list sorted by score descending, capped at top_k.
        final_citations = sorted(
            all_citations.values(),
            key=lambda c: (-c.score, c.namespace, c.document_id, c.chunk_id),
        )[:top_k]

        self._append_trace(
            trace=trace,
            stage="agentic_completed",
            message="agentic retrieval loop completed",
            attributes={
                "total_hops": str(len(hops)),
                "final_citations": str(len(final_citations)),
                "final_query": current_query,
            },
        )

        return AgenticQueryResult(
            namespace=namespace,
            original_query=query,
            final_query=current_query,
            citations=final_citations,
            hops=hops,
            trace=trace,
            total_hops=len(hops),
        )

    def query_namespaces(
        self,
        *,
        connectors: list[NamespaceConnector],
        query: str,
        top_k: int = 5,
        metadata_filters: dict[str, str] | None = None,
        scientific_operators: ScientificQueryOperators | None = None,
    ) -> AgenticQueryResult:
        """Run multi-hop agentic retrieval across multiple namespaces."""
        trace: list[TraceEvent] = []
        hops: list[HopRecord] = []
        all_citations: dict[str, CitationRecord] = {}
        current_query = query
        namespaces = [c.descriptor().name for c in connectors]

        self._append_trace(
            trace=trace,
            stage="agentic_multi_started",
            message="agentic multi-namespace retrieval started",
            attributes={
                "query": query,
                "namespaces": ",".join(namespaces),
                "max_hops": str(self.max_hops),
            },
        )

        for hop_number in range(1, self.max_hops + 1):
            self._append_trace(
                trace=trace,
                stage="hop_started",
                message=f"hop {hop_number} started (multi-namespace)",
                attributes={
                    "hop": str(hop_number),
                    "query": current_query,
                },
            )

            result = self.coordinator.query_namespaces(
                connectors=connectors,
                query=current_query,
                top_k=top_k,
                metadata_filters=metadata_filters,
                scientific_operators=scientific_operators,
            )
            trace.extend(result.trace)

            new_count = 0
            for citation in result.citations:
                existing = all_citations.get(citation.chunk_id)
                if existing is None:
                    all_citations[citation.chunk_id] = citation
                    new_count += 1
                elif citation.score > existing.score:
                    all_citations[citation.chunk_id] = citation

            hop_record = HopRecord(
                hop_number=hop_number,
                query=current_query,
                strategy="initial" if hop_number == 1 else "rewrite",
                reasoning="",
                citations_found=len(result.citations),
                new_citations=new_count,
            )

            self._append_trace(
                trace=trace,
                stage="hop_completed",
                message=f"hop {hop_number} completed (multi-namespace)",
                attributes={
                    "citations_found": str(len(result.citations)),
                    "new_citations": str(new_count),
                    "total_accumulated": str(len(all_citations)),
                },
            )

            max_score = max(
                (c.score for c in result.citations), default=0.0
            )
            results_sufficient = (
                max_score >= self.min_score_threshold and len(result.citations) > 0
            )

            if hop_number >= self.max_hops:
                hops.append(hop_record)
                break

            if new_count <= self.convergence_threshold and hop_number > 1:
                hops.append(
                    HopRecord(
                        hop_number=hop_record.hop_number,
                        query=hop_record.query,
                        strategy="converged",
                        reasoning="No new citations found; stopping.",
                        citations_found=hop_record.citations_found,
                        new_citations=hop_record.new_citations,
                    )
                )
                self._append_trace(
                    trace=trace,
                    stage="convergence_reached",
                    message="no new citations; stopping (multi-namespace)",
                    attributes={"hop": str(hop_number)},
                )
                break

            snippets = [c.snippet for c in result.citations if c.snippet]
            rewrite_result: RewriteResult = self.rewriter.rewrite(
                query=current_query,
                retrieved_snippets=snippets,
                hop_number=hop_number,
                max_hops=self.max_hops,
            )

            hop_record = HopRecord(
                hop_number=hop_record.hop_number,
                query=hop_record.query,
                strategy=rewrite_result.strategy,
                reasoning=rewrite_result.reasoning,
                citations_found=hop_record.citations_found,
                new_citations=hop_record.new_citations,
            )
            hops.append(hop_record)

            self._append_trace(
                trace=trace,
                stage="rewrite_completed",
                message=f"rewriter chose strategy={rewrite_result.strategy}",
                attributes={
                    "hop": str(hop_number),
                    "strategy": rewrite_result.strategy,
                    "rewritten_query": rewrite_result.rewritten_query,
                    "reasoning": rewrite_result.reasoning,
                },
            )

            if rewrite_result.strategy == "done":
                break

            current_query = rewrite_result.rewritten_query
            continue

        final_citations = sorted(
            all_citations.values(),
            key=lambda c: (-c.score, c.namespace, c.document_id, c.chunk_id),
        )[:top_k]

        self._append_trace(
            trace=trace,
            stage="agentic_multi_completed",
            message="agentic multi-namespace retrieval completed",
            attributes={
                "total_hops": str(len(hops)),
                "final_citations": str(len(final_citations)),
                "final_query": current_query,
            },
        )

        return AgenticQueryResult(
            namespace=",".join(namespaces),
            original_query=query,
            final_query=current_query,
            citations=final_citations,
            hops=hops,
            trace=trace,
            total_hops=len(hops),
        )

    @staticmethod
    def _append_trace(
        *,
        trace: list[TraceEvent],
        stage: str,
        message: str,
        attributes: dict[str, str],
    ) -> None:
        trace.append(
            TraceEvent(
                stage=stage,
                message=message,
                timestamp_ns=time.time_ns(),
                attributes=attributes,
            )
        )
