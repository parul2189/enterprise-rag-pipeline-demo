"""
Guardrail Layer
----------------
Enterprise-grade safety controls applied at two points:
  1. Pre-query:  PII redaction, prompt injection detection
  2. Post-gen:   Hallucination detection, citation grounding check

Design intent:
  Guardrails are non-blocking by default — they FLAG issues and
  return a safe response rather than crashing the pipeline.
  This is deliberate: in production, a silent failure (no answer)
  is worse than a flagged, degraded answer that operators can review.

Hallucination detection approach:
  Entailment-based — checks if the generated answer is semantically
  entailed by the retrieved context using a NLI (Natural Language
  Inference) model. This is more robust than simple string matching.
"""

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Compiled regex patterns — compiled once at module load, not per-call
_PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "phone_intl": re.compile(r"\+?\d[\d\s\-().]{8,}\d"),
    "uae_national_id": re.compile(r"\b784-\d{4}-\d{7}-\d\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "passport": re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),
}

_INJECTION_PATTERNS = [
    re.compile(r"ignore (all |previous |above )?instructions", re.IGNORECASE),
    re.compile(r"you are now", re.IGNORECASE),
    re.compile(r"pretend (you are|to be)", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"DAN mode", re.IGNORECASE),
]

SAFE_FALLBACK = (
    "I'm unable to provide a verified answer to this question based on the "
    "available documents. Please contact your support team for assistance."
)


@dataclass
class GuardrailConfig:
    redact_pii: bool = True
    detect_injection: bool = True
    hallucination_threshold: float = 0.4   # NLI entailment score min
    citation_min_overlap: float = 0.3      # Min word overlap for grounding check
    block_on_injection: bool = True


class GuardrailLayer:
    """
    Two-phase guardrail layer with configurable sensitivity.

    Hallucination check uses NLI entailment model:
      score < threshold → answer not grounded → replaced with safe fallback
    """

    def __init__(self, config: dict):
        self.cfg = GuardrailConfig(**config)
        self._nli_model = None  # lazy-loaded

    # ── Pre-Query ──────────────────────────────────────────────────────────────

    def pre_query(self, query: str) -> tuple[str, list[str]]:
        """Clean and validate incoming query. Returns (clean_query, flags)."""
        flags = []

        if self.cfg.detect_injection:
            for pattern in _INJECTION_PATTERNS:
                if pattern.search(query):
                    flags.append("PROMPT_INJECTION_DETECTED")
                    logger.warning("Prompt injection attempt detected: %r", query[:80])
                    if self.cfg.block_on_injection:
                        return "[BLOCKED]", flags

        if self.cfg.redact_pii:
            query, pii_flags = self._redact_pii(query)
            flags.extend(pii_flags)

        return query, flags

    # ── Post-Generation ────────────────────────────────────────────────────────

    def post_generation(
        self, answer: str, context: str, enable: bool = True
    ) -> tuple[str, list[str]]:
        """Validate generated answer against retrieved context."""
        if not enable:
            return answer, []

        flags = []

        # PII check on output
        _, pii_flags = self._redact_pii(answer)
        if pii_flags:
            flags.extend([f"OUTPUT_PII:{f}" for f in pii_flags])

        # Grounding check (fast word-overlap heuristic)
        overlap = self._citation_overlap(answer, context)
        if overlap < self.cfg.citation_min_overlap:
            flags.append(f"LOW_GROUNDING_SCORE:{overlap:.2f}")
            logger.warning("Low grounding score %.2f — checking entailment", overlap)

            # Escalate to NLI model if overlap is low
            if not self._entailment_check(answer, context):
                flags.append("HALLUCINATION_DETECTED")
                logger.error("Hallucination detected. Returning safe fallback.")
                return SAFE_FALLBACK, flags

        return answer, flags

    # ── PII Redaction ──────────────────────────────────────────────────────────

    def _redact_pii(self, text: str) -> tuple[str, list[str]]:
        flags = []
        for label, pattern in _PII_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                text = pattern.sub(f"[REDACTED_{label.upper()}]", text)
                flags.append(f"PII_{label.upper()}_REDACTED")
                logger.info("Redacted %d %s match(es)", len(matches), label)
        return text, flags

    # ── Grounding Checks ───────────────────────────────────────────────────────

    def _citation_overlap(self, answer: str, context: str) -> float:
        """
        Fast word-overlap heuristic (Jaccard similarity).
        Used as a cheap first-pass before invoking the NLI model.
        """
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        if not answer_words:
            return 0.0
        intersection = answer_words & context_words
        return len(intersection) / len(answer_words)

    def _entailment_check(self, answer: str, context: str) -> bool:
        """
        NLI-based entailment check.
        Returns True if the answer is entailed by the context.
        Uses facebook/bart-large-mnli (zero-shot classification).
        """
        try:
            from transformers import pipeline
            if self._nli_model is None:
                logger.info("Loading NLI model for hallucination check...")
                self._nli_model = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"
                )

            # Truncate context to avoid token limit
            truncated_ctx = context[:1500]
            result = self._nli_model(
                answer,
                candidate_labels=["supported by context", "not supported by context"],
                hypothesis_template=f"Based on: {truncated_ctx}. This answer is {{}}.",
            )
            score = result["scores"][result["labels"].index("supported by context")]
            logger.info("Entailment score: %.3f (threshold: %.3f)", score, self.cfg.hallucination_threshold)
            return score >= self.cfg.hallucination_threshold

        except Exception as exc:
            logger.warning("NLI check failed (%s) — defaulting to pass", exc)
            return True  # fail-open: don't block if model unavailable
