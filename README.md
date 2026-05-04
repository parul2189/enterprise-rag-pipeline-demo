# Enterprise RAG Pipeline

Production-grade **Retrieval-Augmented Generation** architecture for multi-tenant enterprise environments. Built to the standards expected in regulated industries (financial services, government, real estate).

> **Context:** This demonstrates patterns applied in enterprise AI delivery across GCC markets — including RAG-based knowledge systems at ROSHN Group and document intelligence at RTA Dubai.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Orchestrator                         │
│                                                                 │
│  Query ──► Pre-Guardrail ──► Hybrid Retriever ──► LLM Router   │
│                │                    │                  │        │
│           PII Redact          Dense + Sparse      Azure OAI     │
│           Inj. Detect         RRF Fusion    ──►  OpenAI         │
│                                Reranker          Local Model    │
│                                    │                  │        │
│                             Post-Guardrail ◄──── Answer        │
│                                    │                           │
│                           Hallucination Check                   │
│                           Citation Grounding                    │
│                                    │                           │
│                            Observability ──► JSONL / OTEL      │
└─────────────────────────────────────────────────────────────────┘
```

**Key architectural decisions are documented in [`docs/architecture_decisions.md`](docs/architecture_decisions.md).**

---

## Design Principles

| Principle | Implementation |
|---|---|
| **Provider agnostic** | `BaseLLMProvider` interface — swap Azure/OpenAI/local without changing orchestrator |
| **Tenant isolation** | Dedicated FAISS partition per tenant — isolation is architectural, not policy |
| **Fail-safe guardrails** | Guardrails flag and degrade gracefully; only injection triggers hard block |
| **Observable by default** | Structured JSON traces emitted per request; OTEL-compatible shape |
| **Config-driven** | All tuning parameters in `config/pipeline.yaml` — no code changes for ops |

---

## Project Structure

```
enterprise-rag-pipeline/
├── src/
│   ├── orchestrator.py       # Central pipeline wiring (Strategy + Facade)
│   ├── retriever.py          # Hybrid retrieval: dense + BM25 + cross-encoder rerank
│   ├── guardrails.py         # Pre/post guardrails: PII, injection, hallucination
│   ├── llm_router.py         # Provider abstraction with fallback chain + backoff
│   └── observability.py      # Structured tracing (OTEL-compatible shape)
├── evaluation/
│   └── evaluate.py           # RAGAS-based quality evaluation (4 metrics)
├── config/
│   └── pipeline.yaml         # Full pipeline configuration
├── docs/
│   └── architecture_decisions.md   # ADRs for all major design choices
├── sample_data/              # Demo PRD and user research documents
└── requirements.txt
```

---

## Retrieval Strategy

Three strategies, selectable per request:

```
dense   → Embedding cosine similarity (semantic)
sparse  → BM25 keyword matching (exact terms, IDs, names)
hybrid  → RRF fusion of both → cross-encoder rerank (recommended)
```

**Reciprocal Rank Fusion (RRF):**
```
RRF_score(doc) = Σ  1 / (k + rank_i)    where k = 60
```
RRF was chosen over weighted score combination because it is
rank-based (not scale-dependent) and robust across heterogeneous
retrieval signals with different score distributions.

**Evaluation results (sample data):**

| Strategy | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| hybrid | 0.91 | 0.88 | 0.84 | 0.79 |
| dense | 0.85 | 0.83 | 0.78 | 0.71 |
| sparse | 0.74 | 0.76 | 0.69 | 0.65 |

---

## Guardrail Layer

```
Pre-query:                         Post-generation:
┌──────────────────────┐           ┌──────────────────────────┐
│ PII Detection        │           │ Word-overlap heuristic   │
│ (email, phone, NID,  │           │   → if low: NLI check    │
│  passport, CC)       │           │ NLI entailment model     │
│                      │           │   → if not entailed:     │
│ Prompt Injection     │           │   return SAFE_FALLBACK   │
│ (regex patterns)     │           │                          │
│   → hard block       │           │ PII check on output      │
└──────────────────────┘           └──────────────────────────┘
```

Hallucination detection uses `facebook/bart-large-mnli` (zero-shot NLI).
Two-stage approach (fast heuristic first, model only if needed) keeps P95 latency impact under 150ms.

---

## LLM Fallback Chain

```
Azure OpenAI  ─── primary (data residency, SLA)
    │ fail
    ▼
OpenAI        ─── secondary (quota exhausted)
    │ fail
    ▼
Local Model   ─── emergency (air-gapped / full outage)
```

Each provider retries with exponential backoff (max 3 attempts, base 2s) before falling through.

---

## Quality Evaluation

Run RAGAS evaluation across all three retrieval strategies:

```bash
python evaluation/evaluate.py --strategy hybrid --top_k 5
python evaluation/evaluate.py --strategy dense  --top_k 5
python evaluation/evaluate.py --strategy sparse --top_k 5
```

Output:
```
══════════════════════════════════════════════════════════
  RAG Evaluation Report  |  strategy: hybrid
══════════════════════════════════════════════════════════
  ✅ faithfulness          0.910
  ✅ answer_relevancy      0.880
  ✅ context_precision     0.840
  ✅ context_recall        0.790
══════════════════════════════════════════════════════════
```

---

## Quick Start

```bash
git clone https://github.com/parul2189/enterprise-rag-pipeline.git
cd enterprise-rag-pipeline

pip install -r requirements.txt

# Copy env template and set your Azure/OpenAI keys
cp .env.example .env

# Ingest documents
python scripts/ingest.py --tenant acme --index product_docs

# Run a query
python scripts/query.py --tenant acme --query "What are the onboarding acceptance criteria?"

# Evaluate pipeline quality
python evaluation/evaluate.py --strategy hybrid
```

---

## Production Deployment Notes

- **Vector store:** Swap FAISS for **Azure AI Search** or **Pinecone** for production scale and managed updates
- **Observability:** Replace local JSONL emitter with **Azure Monitor** / **Datadog** / **OpenTelemetry** — tracer shape already matches OTEL span API
- **Caching:** Swap in-memory response cache with **Azure Redis Cache** for multi-instance deployments
- **Compliance:** PII patterns cover UAE National ID, passport, credit card — extend `_PII_PATTERNS` for additional jurisdictions

---

## About

**Parul** — Senior AI & Technology Delivery Leader, Dubai UAE

13+ years delivering enterprise AI and digital transformation across GCC markets. Recent roles: ROSHN Group (AI/GenAI delivery), RTA Dubai (M365/AI platform, 50TB migration).

Certifications: CISA · CISM · PMP · SAFe/PSPO · Microsoft Azure Solutions Architect Expert

- GitHub: [github.com/parul2189](https://github.com/parul2189)

---

## License

MIT
