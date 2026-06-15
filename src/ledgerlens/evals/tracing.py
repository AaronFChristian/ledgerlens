"""Langfuse observability integration.

Wraps extraction pipeline and GraphRAG agent with span-level tracing.
Tracks token cost per document across all stages.

Usage:
    from ledgerlens.evals.tracing import traced_extract, traced_ask

    result = traced_extract(pipeline, image_path)
    answer = traced_ask(agent, question)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from ledgerlens.agent.graphrag_agent import GraphRAGAgent
    from ledgerlens.extraction.pipeline import ExtractionPipeline


def _get_langfuse():
    """Return a Langfuse client if credentials are configured, else None."""
    pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sk = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not pk or not sk:
        logger.debug("Langfuse credentials not set — tracing disabled")
        return None

    try:
        from langfuse import Langfuse
        return Langfuse(public_key=pk, secret_key=sk, host=host)
    except ImportError:
        logger.warning("langfuse not installed — pip install langfuse")
        return None


def traced_extract(pipeline: "ExtractionPipeline", image_path: str | Path) -> dict:
    """Run extraction with Langfuse span tracing.

    Returns ExtractionResult dict + trace_id.
    """
    lf = _get_langfuse()
    image_path = Path(image_path)

    if lf:
        trace = lf.trace(
            name="ledgerlens-extraction",
            metadata={"image": image_path.name},
        )
        span = trace.span(name="claude-vision-extraction", input={"image": image_path.name})

    start = time.perf_counter()
    result = pipeline.extract(image_path)
    elapsed = time.perf_counter() - start

    if lf:
        span.end(
            output={
                "vendor":     result.invoice.vendor_name,
                "total":      result.invoice.total_amount,
                "confidence": result.invoice.confidence_scores.overall if result.invoice.confidence_scores else 0,
                "needs_review": result.needs_human_review,
            },
            usage={
                "input":  result.token_usage.get("input_tokens", 0),
                "output": result.token_usage.get("output_tokens", 0),
                "unit":   "TOKENS",
            },
            metadata={
                "cost_usd":    result.token_usage.get("cost_usd", 0),
                "latency_ms":  round(elapsed * 1000),
                "model":       result.model_used,
            },
        )
        lf.flush()
        logger.info(f"Langfuse trace: {trace.id}")
        return {**result.model_dump(), "trace_id": trace.id}

    return result.model_dump()


def traced_ask(agent: "GraphRAGAgent", question: str) -> dict:
    """Run GraphRAG agent with Langfuse span tracing.

    Returns answer dict + trace_id.
    """
    lf = _get_langfuse()

    if lf:
        trace = lf.trace(
            name="ledgerlens-graphrag",
            input={"question": question},
        )

    start = time.perf_counter()
    result = agent.ask(question)
    elapsed = time.perf_counter() - start

    if lf:
        # Span for Cypher generation
        gen_span = trace.span(name="cypher-generation")
        gen_span.end(output={"queries": result.get("cypher_queries", [])})

        # Span for graph query
        query_span = trace.span(name="graph-query")
        query_span.end(
            output={"records_returned": sum(
                r.get("count", 0) for r in result.get("graph_results", [])
            )},
        )

        # Span for synthesis
        synth_span = trace.span(name="answer-synthesis")
        synth_span.end(
            output={"answer": result.get("answer", "")[:200]},
            metadata={
                "latency_ms": round(elapsed * 1000),
                "traversal_steps": len(result.get("traversal_path", [])),
            },
        )

        trace.update(
            output={"answer": result.get("answer", "")[:200]},
            metadata={"total_latency_ms": round(elapsed * 1000)},
        )
        lf.flush()
        result["trace_id"] = trace.id

    return result
