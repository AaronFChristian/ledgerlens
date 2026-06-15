from __future__ import annotations
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel
from ledgerlens.agent.graphrag_agent import GraphRAGAgent
from ledgerlens.config import settings
from ledgerlens.extraction.pipeline import ExtractionPipeline
from ledgerlens.graph.neo4j_client import Neo4jClient

class AppState:
    pipeline: ExtractionPipeline
    neo4j: Neo4jClient
    agent: GraphRAGAgent
    total_cost_usd: float = 0.0
    total_extractions: int = 0
    total_questions: int = 0

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting LedgerLens API...")
    state.pipeline = ExtractionPipeline()
    state.neo4j    = Neo4jClient()
    state.agent    = GraphRAGAgent(state.neo4j)
    logger.info("Ready")
    yield
    state.neo4j.close()
    logger.info("Shutdown complete")

app = FastAPI(title="LedgerLens", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://ledgerlens.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str

@app.get("/health")
async def health():
    neo4j_ok = state.neo4j.verify_connection()
    return {"status": "ok" if neo4j_ok else "degraded", "neo4j_connected": neo4j_ok,
            "total_extractions": state.total_extractions, "total_questions": state.total_questions,
            "total_cost_usd": round(state.total_cost_usd, 4), "model": settings.claude_model}

@app.post("/extract")
async def extract_invoice(file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(413, "File too large. Max 10MB.")
    suffix = Path(file.filename or "invoice.png").suffix or ".png"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)
        result = state.pipeline.extract(tmp_path)
        state.total_extractions += 1
        state.total_cost_usd += result.cost_usd
        return {
            "vendor_name":          result.invoice.vendor_name,
            "invoice_number":       result.invoice.invoice_number,
            "invoice_date":         result.invoice.invoice_date,
            "total_amount":         float(result.invoice.total_amount) if result.invoice.total_amount is not None else None,
            "line_item_count":      len(result.invoice.line_items),
            "overall_confidence":   result.invoice.confidence_scores.overall if result.invoice.confidence_scores else 0.0,
            "needs_human_review":   result.needs_human_review,
            "low_confidence_fields": result.low_confidence_fields,
            "extraction_time_ms":   result.extraction_time_ms,
            "cost_usd":             result.cost_usd,
            "model_used":           result.model_used,
        }
    except Exception as exc:
        logger.error(f"Extraction error: {exc}")
        raise HTTPException(500, f"Extraction failed: {str(exc)}")
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)

@app.post("/ask")
async def ask_question(body: AskRequest):
    if not body.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    result = state.agent.ask(body.question)
    state.total_questions += 1
    return {"question": body.question, "answer": result.get("answer", ""),
            "traversal_path": result.get("traversal_path", []),
            "cypher_queries": result.get("cypher_queries", []),
            "error": result.get("error")}

@app.get("/graph/stats")
async def graph_stats():
    return state.neo4j.get_stats()

@app.get("/suppliers")
async def list_suppliers():
    rows = state.neo4j.run("MATCH (s:Supplier) RETURN s.canonical_name AS name, s.address AS address ORDER BY s.canonical_name")
    return {"suppliers": rows, "count": len(rows)}

@app.get("/invoices")
async def list_invoices():
    rows = state.neo4j.run("""
        MATCH (s:Supplier)-[:ISSUED]->(i:Invoice)
        RETURN s.canonical_name AS supplier, i.invoice_number AS invoice_number,
               i.date AS date, i.total_amount AS total_amount
        ORDER BY i.total_amount DESC LIMIT 100
    """)
    return {"invoices": rows, "count": len(rows)}
