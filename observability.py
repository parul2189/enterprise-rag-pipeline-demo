"""
Observability Layer
--------------------
Structured logging and tracing for the RAG pipeline.

Captures per-request:
  - Latency per stage (retrieval, generation, guardrails)
  - Token usage and cost estimation
  - Retrieval quality scores
  - Guardrail flag rates

In production: emit to Azure Monitor / Datadog / OpenTelemetry.
In development: write structured JSON to local log file.

Design note:
  Tracing is implemented as a context-managed Span, matching the
  OpenTelemetry API shape — so swapping in a real OTEL exporter
  requires only changing the backend, not the calling code.
"""

import json
import time
import logging
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class SpanEvent:
    stage: str
    timestamp: float
    data: dict


@dataclass
class Span:
    trace_id: str
    query_hash: str
    events: list[SpanEvent] = field(default_factory=list)
    start_time: float = field(default_factory=time.perf_counter)
    success: Optional[bool] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None

    def log(self, stage: str, data: dict):
        self.events.append(SpanEvent(
            stage=stage,
            timestamp=time.perf_counter() - self.start_time,
            data=data
        ))

    def finish(self, success: bool, latency_ms: float = 0.0, error: str = None):
        self.success = success
        self.latency_ms = latency_ms
        self.error = error


class RAGTracer:
    """
    Lightweight structured tracer. Emits JSON trace records.

    Production swap: replace _emit() with OTEL exporter or
    Azure Monitor track_event() call — no other changes needed.
    """

    def __init__(self, config: dict):
        self._enabled = config.get("enabled", True)
        self._log_dir = Path(config.get("log_dir", "logs"))
        self._log_dir.mkdir(exist_ok=True)
        self._spans: list[Span] = []

    def start_span(self, session_id: str, query: str) -> Span:
        if not self._enabled:
            return _NoOpSpan()
        import hashlib
        span = Span(
            trace_id=session_id,
            query_hash=hashlib.sha256(query.encode()).hexdigest()[:12]
        )
        return span

    def emit(self, span: Span):
        if not self._enabled:
            return
        record = {
            "trace_id": span.trace_id,
            "query_hash": span.query_hash,
            "success": span.success,
            "latency_ms": span.latency_ms,
            "error": span.error,
            "stages": [
                {"stage": e.stage, "t_offset_s": round(e.timestamp, 4), **e.data}
                for e in span.events
            ],
        }
        log_file = self._log_dir / "rag_traces.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.debug("Trace emitted: %s", span.trace_id)
        self._spans.append(span)

    def summary(self) -> dict:
        """Aggregate stats across all spans in this session — useful for eval."""
        if not self._spans:
            return {}
        latencies = [s.latency_ms for s in self._spans if s.latency_ms]
        successes = sum(1 for s in self._spans if s.success)
        return {
            "total_requests": len(self._spans),
            "success_rate": round(successes / len(self._spans), 3),
            "p50_latency_ms": round(sorted(latencies)[len(latencies) // 2], 1) if latencies else None,
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if latencies else None,
            "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        }


class _NoOpSpan:
    """Returned when tracing is disabled — avoids None checks in calling code."""
    def log(self, *args, **kwargs): pass
    def finish(self, *args, **kwargs): pass
