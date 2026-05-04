# Architecture Decision Records (ADRs)

This document captures the key design decisions made in this pipeline
and the reasoning behind them. ADRs are a standard practice for
communicating architectural intent to reviewers and future maintainers.

---

## ADR-001: Hybrid Retrieval over Dense-Only

**Status:** Accepted

**Context:**
Initial prototype used dense (embedding) retrieval only.
During evaluation, precision dropped sharply on queries containing
exact identifiers (product codes, policy numbers, proper nouns).

**Decision:**
Implement Reciprocal Rank Fusion (RRF) combining dense + BM25 sparse,
followed by cross-encoder reranking.

**Consequences:**
- +12% improvement in context_precision on eval set
- +200ms latency overhead (acceptable for async use cases)
- Requires maintaining two index types per tenant

---

## ADR-002: Tenant Isolation at Index Level

**Status:** Accepted

**Context:**
Multi-tenant SaaS deployment requires strict data segregation.
Row-level filtering in a shared index is error-prone and hard to audit.

**Decision:**
Each tenant gets a dedicated FAISS partition directory.
Retriever enforces tenant_id on every query; cross-tenant access
is architecturally impossible, not just policy-controlled.

**Consequences:**
- Storage overhead proportional to number of tenants
- Simpler audit trail (no filtering logic to mis-configure)
- Index management (ingest, delete) operates per tenant

---

## ADR-003: Guardrails as Non-Blocking Flags

**Status:** Accepted

**Context:**
Early design blocked requests on any guardrail trigger.
This caused high false-positive rates in testing, making the
system appear broken to users for legitimate queries.

**Decision:**
Guardrails FLAG issues and return safe fallback answers rather
than raising exceptions. Operators monitor flag rates via observability
layer. Only prompt injection triggers a hard block.

**Consequences:**
- Better user experience; operators can tune thresholds post-deployment
- Requires active monitoring of flag rates (not set-and-forget)
- Hallucination fallback must be tested for quality regression

---

## ADR-004: LLM Provider Abstraction with Fallback Chain

**Status:** Accepted

**Context:**
Azure OpenAI is the primary provider for GCC deployments (data residency,
SLA). However, quota limits and regional outages have caused production
incidents in prior projects.

**Decision:**
Implement provider-agnostic router with priority-ordered fallback:
Azure OpenAI → OpenAI → local model. Each provider implements
BaseLLMProvider — adding a new provider requires no changes to
orchestrator or calling code.

**Consequences:**
- Resilience against single-provider failures
- Local fallback degrades quality but maintains availability
- Cost increases if fallback to OpenAI is frequent (monitor)

---

## ADR-005: RAGAS for Retrieval Quality Evaluation

**Status:** Accepted

**Context:**
"Eyeballing" answers is not a scalable quality signal.
Need reproducible, automated quality measurement to support
continuous deployment and regression detection.

**Decision:**
Adopt RAGAS as the evaluation framework. Run evaluation on
a curated Q&A ground truth set on every significant config change.

**Metrics tracked:**
- Faithfulness (hallucination risk)
- Answer Relevancy
- Context Precision
- Context Recall

**Consequences:**
- Evaluation requires a maintained ground truth dataset
- RAGAS itself uses an LLM internally — adds cost per eval run
- Enables data-driven tuning of chunk size, top_k, reranker threshold
