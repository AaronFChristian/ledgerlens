"""LedgerLens eval harness — Day 3.

Measures three dimensions:
  1. Field-level extraction accuracy (vs CORD ground truth)
  2. Groundedness — answer supported by retrieved graph context
  3. Context relevance — retrieved context relevant to the question

Run:
    python scripts/run_evals.py
    pytest tests/test_evals.py -v          # CI-gated version
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


# ── Field accuracy ────────────────────────────────────────────────────────────

def _normalize(value: str) -> str:
    s = str(value).strip().lower()
    s = re.sub(r"[$,\s₩¥€£]", "", s)
    try:
        f = float(s)
        s = str(int(f)) if f == int(f) else f"{f:.2f}"
    except (ValueError, OverflowError):
        pass
    return s


def field_match(predicted, ground_truth) -> Optional[bool]:
    """Compare predicted vs ground truth. Returns None if no GT."""
    if ground_truth is None or str(ground_truth).strip() == "":
        return None
    if predicted is None:
        return False
    return _normalize(str(predicted)) == _normalize(str(ground_truth))


def evaluate_extraction_accuracy(
    results_dir: str = "data/extraction_results",
    gt_dir: str = "data/ground_truth/cord",
    n_samples: int = 20,
) -> dict:
    """Compute field-level extraction accuracy against CORD ground truth.

    Returns:
        {
          "total_amount":    {"correct": int, "total": int, "accuracy": float},
          "line_item_count": {"correct": int, "total": int, "accuracy": float},
          "overall_accuracy": float,
          "auto_approval_rate": float,
          "cost_per_doc": float,
        }
    """
    from ledgerlens.extraction.schemas import ExtractionResult

    gt_files = sorted(Path(gt_dir).glob("*.json"))[:n_samples]
    result_files = {f.stem: f for f in Path(results_dir).glob("*.json")}

    metrics: dict = {
        "total_amount":    {"correct": 0, "total": 0},
        "line_item_count": {"correct": 0, "total": 0},
    }
    auto_approved = 0
    processed = 0
    total_cost = 0.0

    for gt_file in gt_files:
        stem = gt_file.stem
        if stem not in result_files:
            continue

        gt_data   = json.loads(gt_file.read_text())
        result    = ExtractionResult.model_validate(json.loads(result_files[stem].read_text()))
        gt_parse  = gt_data.get("ground_truth", {}).get("gt_parse", {})

        processed  += 1
        total_cost += result.token_usage.get("cost_usd", 0.0)
        if result.auto_approved:
            auto_approved += 1

        # Total amount
        total_section = gt_parse.get("total", {})
        if "total_price" in total_section:
            metrics["total_amount"]["total"] += 1
            if field_match(result.invoice.total_amount, total_section["total_price"]):
                metrics["total_amount"]["correct"] += 1

        # Line item count
        menus = gt_parse.get("menu", [])
        if menus:
            metrics["line_item_count"]["total"] += 1
            if len(result.invoice.line_items) == len(menus):
                metrics["line_item_count"]["correct"] += 1

    def pct(m: dict) -> float:
        return round(m["correct"] / m["total"] * 100, 1) if m["total"] > 0 else 0.0

    overall = sum(
        m["correct"] for m in metrics.values()
    ) / max(sum(m["total"] for m in metrics.values()), 1) * 100

    return {
        "total_amount":       {**metrics["total_amount"],    "accuracy": pct(metrics["total_amount"])},
        "line_item_count":    {**metrics["line_item_count"], "accuracy": pct(metrics["line_item_count"])},
        "overall_accuracy":   round(overall, 1),
        "auto_approval_rate": round(auto_approved / max(processed, 1) * 100, 1),
        "cost_per_doc":       round(total_cost / max(processed, 1), 5),
        "samples_evaluated":  processed,
    }


# ── Groundedness ──────────────────────────────────────────────────────────────

GROUNDEDNESS_PROMPT = """You are an evaluation system. Given a question, an answer, and the source graph data used to generate the answer, rate how well the answer is grounded in the source data.

Score from 0.0 to 1.0:
  1.0 = every claim in the answer is directly supported by the source data
  0.7 = most claims supported, minor unsupported additions
  0.4 = some claims supported, significant unsupported content
  0.0 = answer not supported by source data at all

Return ONLY a JSON object: {"score": 0.0_to_1.0, "reason": "one sentence"}"""


def evaluate_groundedness(
    question: str,
    answer: str,
    graph_results: list[dict],
    client,
    model: str,
) -> dict:
    """Score whether the answer is grounded in the retrieved graph data."""
    context = json.dumps(graph_results, default=str)[:2000]  # Truncate for cost control

    response = client.messages.create(
        model=model,
        max_tokens=256,
        system=GROUNDEDNESS_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Question: {question}\n\nAnswer: {answer}\n\nSource graph data:\n{context}",
        }],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:-1])

    try:
        parsed = json.loads(raw)
        return {"score": float(parsed["score"]), "reason": parsed.get("reason", "")}
    except Exception:
        return {"score": 0.5, "reason": "Parse error — manual review needed"}


# ── Context relevance ─────────────────────────────────────────────────────────

RELEVANCE_PROMPT = """You are an evaluation system. Given a question and the graph data retrieved to answer it, rate how relevant the retrieved data is to the question.

Score from 0.0 to 1.0:
  1.0 = retrieved data directly contains what's needed to answer the question
  0.7 = mostly relevant with some noise
  0.4 = partially relevant
  0.0 = retrieved data is unrelated to the question

Return ONLY a JSON object: {"score": 0.0_to_1.0, "reason": "one sentence"}"""


def evaluate_context_relevance(
    question: str,
    graph_results: list[dict],
    client,
    model: str,
) -> dict:
    """Score whether the retrieved graph context is relevant to the question."""
    context = json.dumps(graph_results, default=str)[:2000]

    response = client.messages.create(
        model=model,
        max_tokens=256,
        system=RELEVANCE_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Question: {question}\n\nRetrieved graph data:\n{context}",
        }],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:-1])

    try:
        parsed = json.loads(raw)
        return {"score": float(parsed["score"]), "reason": parsed.get("reason", "")}
    except Exception:
        return {"score": 0.5, "reason": "Parse error — manual review needed"}


# ── Full eval suite ───────────────────────────────────────────────────────────

EVAL_QUESTIONS = [
    "Which suppliers appear on the most invoices?",
    "What is the total invoice value across all suppliers?",
    "Which invoices have the highest total amounts?",
    "List all unique suppliers in the graph",
    "What are the most common line item descriptions?",
]


def run_full_eval(
    results_dir: str = "data/extraction_results",
    gt_dir: str = "data/ground_truth/cord",
    n_extraction_samples: int = 20,
    n_agent_questions: int = 3,
) -> dict:
    """Run the complete eval suite and return all metrics."""
    import anthropic
    from ledgerlens.agent.graphrag_agent import GraphRAGAgent
    from ledgerlens.config import settings
    from ledgerlens.graph.neo4j_client import Neo4jClient

    logger.info("Running extraction accuracy eval...")
    extraction_metrics = evaluate_extraction_accuracy(
        results_dir=results_dir,
        gt_dir=gt_dir,
        n_samples=n_extraction_samples,
    )

    logger.info("Running agent groundedness + context relevance evals...")
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    agent_metrics: list[dict] = []

    with Neo4jClient() as neo4j:
        agent = GraphRAGAgent(neo4j)
        for question in EVAL_QUESTIONS[:n_agent_questions]:
            result = agent.ask(question)
            if result.get("error"):
                continue

            groundedness = evaluate_groundedness(
                question, result["answer"], result["graph_results"],
                client, settings.claude_model,
            )
            relevance = evaluate_context_relevance(
                question, result["graph_results"],
                client, settings.claude_model,
            )

            agent_metrics.append({
                "question":         question,
                "answer":           result["answer"][:200],
                "groundedness":     groundedness["score"],
                "context_relevance": relevance["score"],
            })

    avg_groundedness = sum(m["groundedness"] for m in agent_metrics) / max(len(agent_metrics), 1)
    avg_relevance    = sum(m["context_relevance"] for m in agent_metrics) / max(len(agent_metrics), 1)

    return {
        "extraction":        extraction_metrics,
        "agent":             agent_metrics,
        "avg_groundedness":  round(avg_groundedness, 3),
        "avg_context_relevance": round(avg_relevance, 3),
    }
