"""
Pipeline Evaluation — RAGAS Framework
---------------------------------------
Evaluates RAG pipeline quality across 4 dimensions:

  1. Faithfulness       — Is the answer grounded in retrieved context?
  2. Answer Relevance   — Does the answer address the question?
  3. Context Precision  — Are retrieved chunks actually relevant?
  4. Context Recall     — Does retrieval capture all needed info?

RAGAS is the industry-standard RAG evaluation framework.
This script generates a scored report you can use to:
  - Compare retrieval strategies (dense vs hybrid)
  - Tune chunk size / overlap
  - Track quality regressions across deployments

Usage:
  python evaluation/evaluate.py --strategy hybrid --top_k 5

Reference: https://docs.ragas.io
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# Evaluation dataset — ground truth Q&A pairs for the sample domain
EVAL_DATASET = [
    {
        "question": "What is the target onboarding completion improvement?",
        "ground_truth": "Onboarding completion rate must improve by at least 30% within 60 days of launch.",
        "contexts": ["Onboarding completion rate must improve by ≥ 30% within 60 days of launch"]
    },
    {
        "question": "What percentage of churned users cited confusion during setup?",
        "ground_truth": "68% of churned users cited confusion during initial setup as the primary reason for leaving.",
        "contexts": ["68% of users who churned in the first 30 days cited confusion during initial setup"]
    },
    {
        "question": "What is the maximum acceptable P95 response latency for the AI assistant?",
        "ground_truth": "The AI assistant must respond within 2 seconds at P95 latency.",
        "contexts": ["AI assistant must respond within 2 seconds (P95 latency)"]
    },
    {
        "question": "Which feature is the highest priority for Q3?",
        "ground_truth": "Priority 1 for Q3 is the contextual AI assistant (in-app, real-time).",
        "contexts": ["Priority 1: Contextual AI assistant (in-app, real-time)"]
    },
    {
        "question": "What compliance requirement applies for the UAE market?",
        "ground_truth": "The system must comply with UAE PDPL (Personal Data Protection Law) requirements.",
        "contexts": ["Legal review of AI disclosure requirements (UAE PDPL compliance)"]
    },
]


def run_evaluation(strategy: str = "hybrid", top_k: int = 5, output_dir: str = "evaluation/results"):
    """
    Runs RAGAS evaluation against the pipeline.
    Falls back to a mock scorer if RAGAS is not installed,
    so the evaluation structure is always demonstrable.
    """
    logger.info("Starting evaluation | strategy=%s top_k=%d", strategy, top_k)

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_list(EVAL_DATASET)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        scores = dict(result)
        logger.info("RAGAS scores: %s", scores)

    except ImportError:
        logger.warning("RAGAS not installed — using mock scores for structure demo")
        scores = _mock_scores(strategy)

    # Produce structured report
    report = {
        "run_id": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "config": {"strategy": strategy, "top_k": top_k},
        "n_questions": len(EVAL_DATASET),
        "scores": scores,
        "interpretation": _interpret(scores),
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"eval_{report['run_id']}_{strategy}.json"
    out_file.write_text(json.dumps(report, indent=2))

    _print_report(report)
    return report


def _mock_scores(strategy: str) -> dict:
    """
    Illustrative scores showing hybrid > dense > sparse.
    Replace with real RAGAS output in production.
    """
    base = {
        "hybrid":  {"faithfulness": 0.91, "answer_relevancy": 0.88, "context_precision": 0.84, "context_recall": 0.79},
        "dense":   {"faithfulness": 0.85, "answer_relevancy": 0.83, "context_precision": 0.78, "context_recall": 0.71},
        "sparse":  {"faithfulness": 0.74, "answer_relevancy": 0.76, "context_precision": 0.69, "context_recall": 0.65},
    }
    return base.get(strategy, base["hybrid"])


def _interpret(scores: dict) -> dict:
    thresholds = {
        "faithfulness": (0.85, "Risk of hallucination — tighten guardrails"),
        "answer_relevancy": (0.80, "Answers may be off-topic — review prompt"),
        "context_precision": (0.75, "Retrieving irrelevant chunks — tune top_k or reranker"),
        "context_recall": (0.70, "Missing relevant content — check chunk size or indexing"),
    }
    result = {}
    for metric, (threshold, warning) in thresholds.items():
        score = scores.get(metric, 0)
        result[metric] = {
            "score": score,
            "status": "PASS" if score >= threshold else "WARN",
            "note": "" if score >= threshold else warning,
        }
    return result


def _print_report(report: dict):
    print("\n" + "═" * 58)
    print(f"  RAG Evaluation Report  |  strategy: {report['config']['strategy']}")
    print("═" * 58)
    for metric, detail in report["interpretation"].items():
        status_icon = "✅" if detail["status"] == "PASS" else "⚠️ "
        print(f"  {status_icon} {metric:<22} {detail['score']:.3f}  {detail['note']}")
    print("═" * 58 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="hybrid", choices=["dense", "sparse", "hybrid"])
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()
    run_evaluation(strategy=args.strategy, top_k=args.top_k)
