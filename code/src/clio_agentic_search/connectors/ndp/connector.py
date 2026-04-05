"""NDP (National Data Platform) connector for dataset discovery via CKAN API.

This connector implements the Discover stage of the CLIO Search agentic pipeline:
it queries the NDP CKAN API to discover scientific datasets, extracts metadata
(titles, descriptions, resources, organizations), and indexes them into the
CLIO Search pipeline for science-aware retrieval.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from clio_agentic_search.indexing.scientific import (
    build_structure_aware_chunk_plan,
    canonicalize_measurement,
    normalize_formula,
)
from clio_agentic_search.indexing.text_features import Embedder, HashEmbedder, tokenize
from clio_agentic_search.models.contracts import (
    ChunkRecord,
    CitationRecord,
    DocumentRecord,
    EmbeddingRecord,
    MetadataRecord,
    NamespaceDescriptor,
)
from clio_agentic_search.retrieval.ann import ANNAdapter, build_ann_adapter
from clio_agentic_search.retrieval.capabilities import ScoredChunk
from clio_agentic_search.retrieval.corpus_profile import CorpusProfile, build_corpus_profile
from clio_agentic_search.retrieval.scientific import (
    ScientificQueryOperators,
    score_scientific_metadata,
)
from clio_agentic_search.storage.contracts import DocumentBundle, FileIndexState, StorageAdapter


NDP_BASE_URL = "http://155.101.6.191:8003"


@dataclass(slots=True)
class NDPConnector:
    """Connector that discovers and indexes datasets from the NDP CKAN API."""

    namespace: str
    storage: StorageAdapter
    base_url: str = NDP_BASE_URL
    server: str = "global"
    embedder: Embedder = field(default_factory=HashEmbedder)
    embedding_model: str = "hash16-v1"
    chunk_size: int = 400
    _connected: bool = False
    _ann_index: ANNAdapter | None = field(default=None, init=False, repr=False)

    def descriptor(self) -> NamespaceDescriptor:
        return NamespaceDescriptor(
            name=self.namespace,
            connector_type="ndp",
            root_uri=self.base_url,
        )

    def connect(self) -> None:
        self.storage.connect()
        self._connected = True

    def teardown(self) -> None:
        self.storage.teardown()
        self._connected = False

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("NDPConnector not connected. Call connect() first.")

    def discover_datasets(
        self,
        search_terms: list[str] | None = None,
        search_term: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Query NDP API to discover datasets. Returns raw dataset dicts."""
        self._ensure_connected()
        with httpx.Client(timeout=30.0) as client:
            if search_terms:
                params: dict[str, Any] = {
                    "terms": search_terms[0],
                    "server": self.server,
                }
                resp = client.get(f"{self.base_url}/search", params=params)
            elif search_term:
                data = {"search_term": search_term, "server": self.server}
                resp = client.post(f"{self.base_url}/search", json=data)
            else:
                params = {"terms": "", "server": self.server}
                resp = client.get(f"{self.base_url}/search", params=params)

            resp.raise_for_status()
            datasets = resp.json()

        if isinstance(datasets, list):
            return datasets[:limit]
        return []

    def index_datasets(self, datasets: list[dict[str, Any]]) -> int:
        """Index discovered NDP datasets into the CLIO Search storage."""
        self._ensure_connected()
        indexed = 0

        for ds in datasets:
            ds_id = ds.get("id", "")
            ds_name = ds.get("name", ds_id)
            ds_title = ds.get("title", ds_name)
            ds_notes = ds.get("notes", "") or ""
            ds_org = ds.get("owner_org", "") or ""
            resources = ds.get("resources", [])

            # Build document text as a SINGLE flat paragraph (no sub-headings)
            # to prevent the chunker from splitting resources into fake datasets.
            text_parts = [f"# {ds_title}", ""]
            if ds_org:
                text_parts.append(f"Organization: {ds_org}")
            if ds_notes:
                text_parts.append(ds_notes)

            # Flatten resources into a single "Resources:" block — one line each.
            if resources:
                text_parts.append("")
                text_parts.append("Resources:")
                for i, res in enumerate(resources):
                    res_name = res.get("name", f"Resource {i + 1}")
                    res_fmt = res.get("format", "") or ""
                    res_url = res.get("url", "") or ""
                    res_desc = res.get("description", "") or ""
                    line = f"- {res_name}"
                    if res_fmt:
                        line += f" [{res_fmt}]"
                    if res_desc:
                        line += f": {res_desc[:100]}"
                    if res_url:
                        line += f" ({res_url})"
                    text_parts.append(line)

            text = "\n".join(text_parts)

            # Build structured metadata for URLs, formats, DOIs, organization
            resource_urls = "; ".join(
                r.get("url", "") for r in resources if r.get("url")
            )
            resource_formats = "; ".join(
                sorted({r.get("format", "") for r in resources if r.get("format")})
            )
            # Try to find DOI in notes or resource URLs
            doi = ""
            if "doi.org" in (ds_notes or ""):
                import re as _re
                m = _re.search(r"(10\.\d{4,}[-._;()/:A-Za-z0-9]+)", ds_notes or "")
                if m:
                    doi = m.group(1)
            if not text.strip():
                continue

            document_id = f"ndp_{ds_id[:12]}" if ds_id else f"ndp_{ds_name}"
            uri = f"ndp://{self.server}/{ds_name}"
            checksum = hashlib.sha256(text.encode()).hexdigest()

            doc = DocumentRecord(
                namespace=self.namespace,
                document_id=document_id,
                uri=uri,
                checksum=checksum,
                modified_at_ns=time.time_ns(),
            )

            # Build chunks using science-aware chunking
            plan = build_structure_aware_chunk_plan(
                namespace=self.namespace,
                document_id=document_id,
                text=text,
                chunk_size=self.chunk_size,
            )

            # Build embeddings
            embeddings = [
                EmbeddingRecord(
                    namespace=self.namespace,
                    chunk_id=chunk.chunk_id,
                    model=self.embedding_model,
                    vector=self.embedder.embed(chunk.text),
                )
                for chunk in plan.chunks
            ]

            # Build metadata (preserve URLs, formats, DOI for result display)
            metadata_records: list[MetadataRecord] = [
                MetadataRecord(namespace=self.namespace, record_id=document_id,
                               scope="document", key="source", value="ndp"),
                MetadataRecord(namespace=self.namespace, record_id=document_id,
                               scope="document", key="organization", value=ds_org),
                MetadataRecord(namespace=self.namespace, record_id=document_id,
                               scope="document", key="ndp_id", value=ds_id),
                MetadataRecord(namespace=self.namespace, record_id=document_id,
                               scope="document", key="title", value=ds_title),
                MetadataRecord(namespace=self.namespace, record_id=document_id,
                               scope="document", key="resource_urls", value=resource_urls),
                MetadataRecord(namespace=self.namespace, record_id=document_id,
                               scope="document", key="resource_formats", value=resource_formats),
                MetadataRecord(namespace=self.namespace, record_id=document_id,
                               scope="document", key="doi", value=doi),
            ]

            # Add chunk-level scientific metadata
            for chunk in plan.chunks:
                chunk_meta = plan.metadata_by_chunk_id.get(chunk.chunk_id, {})
                for key, value in sorted(chunk_meta.items()):
                    metadata_records.append(
                        MetadataRecord(
                            namespace=self.namespace,
                            record_id=chunk.chunk_id,
                            scope="chunk",
                            key=key,
                            value=value,
                        )
                    )

            file_state = FileIndexState(
                namespace=self.namespace,
                path=uri,
                document_id=document_id,
                mtime_ns=time.time_ns(),
                content_hash=checksum,
            )

            self.storage.upsert_document_bundle(
                document=doc,
                chunks=plan.chunks,
                embeddings=embeddings,
                metadata=metadata_records,
                file_state=file_state,
            )
            indexed += 1

        # Build ANN index from stored embeddings
        embeddings_map = self.storage.list_embeddings(
            self.namespace, self.embedding_model
        )
        if embeddings_map:
            sample_vec = next(iter(embeddings_map.values()))
            dims = len(sample_vec)
            ann = build_ann_adapter(backend="exact", dimensions=dims, shard_count=4)
            ann.build(embeddings_map)
            self._ann_index = ann

        return indexed

    def search_lexical(self, query: str, top_k: int) -> list[ScoredChunk]:
        self._ensure_connected()
        query_tokens = tuple(sorted(set(tokenize(query))))
        if not query_tokens:
            return []
        matches = self.storage.query_chunks_lexical(
            namespace=self.namespace, query_tokens=query_tokens, limit=top_k
        )
        return [
            ScoredChunk(
                chunk_id=m.chunk.chunk_id,
                document_id=m.chunk.document_id,
                text=m.chunk.text,
                lexical_score=m.bm25_score,
            )
            for m in matches
        ]

    def search_vector(self, query: str, top_k: int) -> list[ScoredChunk]:
        self._ensure_connected()
        if self._ann_index is None:
            return []
        query_vec = self.embedder.embed(query)
        results = self._ann_index.query(query_vector=query_vec, top_k=top_k)
        chunks = []
        for r in results:
            stored = self.storage.get_chunk(self.namespace, r.chunk_id)
            if stored:
                chunks.append(
                    ScoredChunk(
                        chunk_id=r.chunk_id,
                        document_id=stored.document_id,
                        text=stored.text,
                        vector_score=r.score,
                    )
                )
        return chunks

    def search_scientific(
        self, query: str, top_k: int, operators: ScientificQueryOperators
    ) -> list[ScoredChunk]:
        self._ensure_connected()
        if not operators.is_active():
            return []

        candidate_ids: set[str] | None = None

        if operators.numeric_range is not None:
            try:
                canonical_unit = canonicalize_measurement(
                    0.0, operators.numeric_range.unit
                )[1]
                canonical_min = None
                canonical_max = None
                if operators.numeric_range.minimum is not None:
                    canonical_min = canonicalize_measurement(
                        operators.numeric_range.minimum, operators.numeric_range.unit
                    )[0]
                if operators.numeric_range.maximum is not None:
                    canonical_max = canonicalize_measurement(
                        operators.numeric_range.maximum, operators.numeric_range.unit
                    )[0]

                range_chunks = self.storage.query_chunks_by_measurement_range(
                    self.namespace,
                    canonical_unit=canonical_unit,
                    minimum=canonical_min,
                    maximum=canonical_max,
                )
                candidate_ids = {c.chunk_id for c in range_chunks}
            except (KeyError, ValueError):
                candidate_ids = set()

        if operators.formula is not None:
            norm = normalize_formula(operators.formula)
            formula_chunks = self.storage.query_chunks_by_formula(
                self.namespace, formula_signature=norm,
            )
            formula_ids = {c.chunk_id for c in formula_chunks}
            if candidate_ids is None:
                candidate_ids = formula_ids
            else:
                candidate_ids &= formula_ids

        if candidate_ids is None:
            return []

        results: list[ScoredChunk] = []
        for chunk_id in list(candidate_ids)[:top_k]:
            stored = self.storage.get_chunk(self.namespace, chunk_id)
            if stored:
                meta = self.storage.get_chunk_metadata(self.namespace, chunk_id)
                sci_score = score_scientific_metadata(meta, operators)
                results.append(
                    ScoredChunk(
                        chunk_id=chunk_id,
                        document_id=stored.document_id,
                        text=stored.text,
                        metadata_score=sci_score,
                    )
                )
        return sorted(results, key=lambda c: -c.metadata_score)[:top_k]

    def filter_metadata(
        self, candidates: list[ScoredChunk], required: dict[str, str]
    ) -> list[ScoredChunk]:
        if not required:
            return candidates
        filtered = []
        for c in candidates:
            meta = self.storage.get_chunk_metadata(self.namespace, c.chunk_id)
            if all(meta.get(k) == v for k, v in required.items()):
                filtered.append(c)
        return filtered

    def corpus_profile(self) -> CorpusProfile:
        self._ensure_connected()
        return build_corpus_profile(self.storage, self.namespace)

    def build_citation(self, chunk: ScoredChunk) -> CitationRecord:
        self._ensure_connected()
        meta = self.storage.get_chunk_metadata(self.namespace, chunk.chunk_id)
        uri = meta.get("path", f"ndp://{self.namespace}/{chunk.document_id}")
        return CitationRecord(
            namespace=self.namespace,
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            uri=uri,
            snippet=chunk.text.strip()[:160],
            score=round(chunk.combined_score, 6),
        )
