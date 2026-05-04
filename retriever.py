"""
Hybrid Retriever
-----------------
Combines dense (semantic) and sparse (keyword/BM25) retrieval,
then applies a cross-encoder reranker for precision.

Architecture decision:
  Dense alone: misses exact keyword matches (model numbers, IDs, names)
  Sparse alone: misses semantic similarity ("failure" vs "error")
  Hybrid + rerank: best of both, production-grade precision

Tenant isolation: each tenant gets its own FAISS index partition.
"""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document

logger = logging.getLogger(__name__)

RetrievalStrategy = Literal["dense", "sparse", "hybrid"]


class HybridRetriever:
    """
    Retrieval strategies:
      dense   — pure semantic similarity (embedding cosine)
      sparse  — BM25 keyword matching
      hybrid  — RRF (Reciprocal Rank Fusion) of both, then reranked

    Cross-encoder reranking is applied in hybrid mode to re-score
    the merged candidate set against the original query.
    """

    def __init__(self, config: dict):
        self.embed_model = config.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.store_dir = Path(config.get("store_dir", "vectorstore"))
        self.rerank_model = config.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._embeddings = HuggingFaceEmbeddings(model_name=self.embed_model)
        self._faiss_indices: dict[str, FAISS] = {}
        self._bm25_indices: dict[str, BM25Retriever] = {}
        self._reranker = None  # lazy-loaded

    # ── Public Interface ───────────────────────────────────────────────────────

    def ingest(self, docs: list[Document], tenant_id: str, index: str = "default"):
        """Ingest documents into both dense and sparse indices for a tenant."""
        key = f"{tenant_id}::{index}"
        logger.info("Ingesting %d docs → [%s]", len(docs), key)

        # Dense index
        if key in self._faiss_indices:
            self._faiss_indices[key].add_documents(docs)
        else:
            self._faiss_indices[key] = FAISS.from_documents(docs, self._embeddings)

        # Sparse index (BM25)
        texts = [d.page_content for d in docs]
        self._bm25_indices[key] = BM25Retriever.from_texts(texts)
        self._bm25_indices[key].k = 20  # retrieve wider set before rerank

        # Persist dense index
        partition_dir = self.store_dir / tenant_id / index
        partition_dir.mkdir(parents=True, exist_ok=True)
        self._faiss_indices[key].save_local(str(partition_dir))
        logger.info("Index persisted → %s", partition_dir)

    def retrieve(
        self,
        query: str,
        tenant_id: str,
        index: str = "default",
        strategy: RetrievalStrategy = "hybrid",
        top_k: int = 5,
    ) -> list[dict]:
        key = f"{tenant_id}::{index}"
        self._ensure_loaded(key, tenant_id, index)

        if strategy == "dense":
            return self._dense_retrieve(key, query, top_k)
        elif strategy == "sparse":
            return self._sparse_retrieve(key, query, top_k)
        else:
            return self._hybrid_retrieve(key, query, top_k)

    # ── Retrieval Strategies ───────────────────────────────────────────────────

    def _dense_retrieve(self, key: str, query: str, top_k: int) -> list[dict]:
        results = self._faiss_indices[key].similarity_search_with_score(query, k=top_k)
        return [
            {"text": doc.page_content, "metadata": doc.metadata, "score": float(1 / (1 + score))}
            for doc, score in results
        ]

    def _sparse_retrieve(self, key: str, query: str, top_k: int) -> list[dict]:
        docs = self._bm25_indices[key].get_relevant_documents(query)[:top_k]
        return [
            {"text": doc.page_content, "metadata": doc.metadata, "score": 1.0 / (i + 1)}
            for i, doc in enumerate(docs)
        ]

    def _hybrid_retrieve(self, key: str, query: str, top_k: int) -> list[dict]:
        """
        Reciprocal Rank Fusion (RRF) merges dense and sparse rankings.
        RRF score = Σ 1/(k + rank_i) where k=60 (empirically optimal).
        Then cross-encoder reranks the merged candidate set.
        """
        dense_results = self._dense_retrieve(key, query, top_k * 2)
        sparse_results = self._sparse_retrieve(key, query, top_k * 2)

        # RRF fusion
        rrf_k = 60
        scores: dict[str, float] = {}
        text_map: dict[str, dict] = {}

        for rank, r in enumerate(dense_results):
            tid = r["text"][:100]  # use text prefix as ID
            scores[tid] = scores.get(tid, 0) + 1 / (rrf_k + rank + 1)
            text_map[tid] = r

        for rank, r in enumerate(sparse_results):
            tid = r["text"][:100]
            scores[tid] = scores.get(tid, 0) + 1 / (rrf_k + rank + 1)
            text_map[tid] = r

        # Sort by RRF score, take top_k * 2 candidates for reranking
        candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: top_k * 2]
        candidate_docs = [text_map[tid] for tid, _ in candidates]

        # Cross-encoder rerank
        reranked = self._rerank(query, candidate_docs, top_k)
        return reranked

    # ── Reranker ───────────────────────────────────────────────────────────────

    def _rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """Cross-encoder reranking — more accurate than bi-encoder for final scoring."""
        try:
            from sentence_transformers import CrossEncoder
            if self._reranker is None:
                logger.info("Loading reranker: %s", self.rerank_model)
                self._reranker = CrossEncoder(self.rerank_model)

            pairs = [[query, c["text"]] for c in candidates]
            scores = self._reranker.predict(pairs)

            for i, c in enumerate(candidates):
                c["score"] = float(scores[i])

            candidates.sort(key=lambda x: x["score"], reverse=True)
            return candidates[:top_k]

        except ImportError:
            logger.warning("sentence-transformers not available, skipping rerank")
            return candidates[:top_k]

    # ── Loader ─────────────────────────────────────────────────────────────────

    def _ensure_loaded(self, key: str, tenant_id: str, index: str):
        if key not in self._faiss_indices:
            partition_dir = self.store_dir / tenant_id / index
            if partition_dir.exists():
                logger.info("Loading index from disk: %s", partition_dir)
                self._faiss_indices[key] = FAISS.load_local(
                    str(partition_dir), self._embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                raise ValueError(
                    f"No index found for tenant='{tenant_id}' index='{index}'. "
                    "Run ingest() first."
                )
