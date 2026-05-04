"""
Enterprise RAG Orchestrator
-----------------------------
Multi-tenant, multi-index RAG pipeline with:
  - Configurable retrieval strategies (dense, sparse, hybrid)
  - Guardrail layer (hallucination detection, PII redaction)
  - Observability hooks (latency, token usage, retrieval quality)
  - LLM-agnostic design (Azure OpenAI / OpenAI / local models)

Author: Parul (github.com/parul2189)
"""

import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Optional

from src.retriever import HybridRetriever
from src.guardrails import GuardrailLayer
from src.llm_router import LLMRouter
from src.observability import RAGTracer

logger = logging.getLogger(__name__)


# ── Request / Response Contracts ───────────────────────────────────────────────

@dataclass
class RAGRequest:
    query: str
    tenant_id: str
    index_name: str = "default"
    retrieval_strategy: str = "hybrid"   # dense | sparse | hybrid
    top_k: int = 5
    enable_guardrails: bool = True
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class RAGResponse:
    session_id: str
    answer: str
    sources: list[dict]
    guardrail_flags: list[str]
    latency_ms: float
    tokens_used: int
    retrieval_score: float


# ── Orchestrator ───────────────────────────────────────────────────────────────

class RAGOrchestrator:
    """
    Central orchestrator that wires together:
      retrieval → guardrails → generation → observability

    Design principles:
      - Each component is independently swappable (Strategy pattern)
      - All side effects (logging, tracing) are injected, not hardcoded
      - Tenant isolation enforced at retriever level
      - Fail-safe: guardrail failures return safe fallback, not exceptions
    """

    def __init__(self, config: dict):
        self.retriever = HybridRetriever(config["retrieval"])
        self.guardrails = GuardrailLayer(config["guardrails"])
        self.llm = LLMRouter(config["llm"])
        self.tracer = RAGTracer(config.get("observability", {}))

    def run(self, request: RAGRequest) -> RAGResponse:
        start = time.perf_counter()
        span = self.tracer.start_span(request.session_id, request.query)

        try:
            # 1. Pre-query guardrails (PII, prompt injection detection)
            clean_query, pre_flags = self.guardrails.pre_query(request.query)
            span.log("pre_guardrail", {"flags": pre_flags})

            # 2. Retrieve — strategy selected per request
            chunks = self.retriever.retrieve(
                query=clean_query,
                tenant_id=request.tenant_id,
                index=request.index_name,
                strategy=request.retrieval_strategy,
                top_k=request.top_k,
            )
            retrieval_score = self._mean_score(chunks)
            span.log("retrieval", {"chunks": len(chunks), "mean_score": retrieval_score})

            # 3. Build grounded prompt
            context = self._build_context(chunks)
            prompt = self._build_prompt(clean_query, context)

            # 4. Generate
            raw_answer, token_count = self.llm.complete(prompt)
            span.log("generation", {"tokens": token_count})

            # 5. Post-generation guardrails (hallucination check, citation grounding)
            final_answer, post_flags = self.guardrails.post_generation(
                answer=raw_answer,
                context=context,
                enable=request.enable_guardrails,
            )

            latency_ms = (time.perf_counter() - start) * 1000
            span.finish(success=True, latency_ms=latency_ms)

            return RAGResponse(
                session_id=request.session_id,
                answer=final_answer,
                sources=[{"text": c["text"][:200], "source": c["metadata"]["source"],
                           "score": round(c["score"], 3)} for c in chunks],
                guardrail_flags=pre_flags + post_flags,
                latency_ms=round(latency_ms, 2),
                tokens_used=token_count,
                retrieval_score=round(retrieval_score, 3),
            )

        except Exception as exc:
            span.finish(success=False, error=str(exc))
            logger.error("Orchestrator error [%s]: %s", request.session_id, exc)
            raise

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _mean_score(self, chunks: list[dict]) -> float:
        if not chunks:
            return 0.0
        return sum(c["score"] for c in chunks) / len(chunks)

    def _build_context(self, chunks: list[dict]) -> str:
        return "\n\n---\n\n".join(
            f"[Source: {c['metadata']['source']}]\n{c['text']}" for c in chunks
        )

    def _build_prompt(self, query: str, context: str) -> str:
        return (
            "You are an enterprise knowledge assistant. "
            "Answer ONLY using the provided context. "
            "If the answer is not in the context, say: 'I don't have enough information.'\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            "ANSWER:"
        )
