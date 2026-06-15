"""Load ExtractionResult objects into the Neo4j knowledge graph.

Graph schema:
  (:Supplier)-[:ISSUED]->(:Invoice)-[:CONTAINS]->(:LineItem)

Nodes are MERGE'd on stable IDs so re-running is safe (idempotent).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from loguru import logger

from ..extraction.schemas import ExtractionResult
from .entity_resolution import resolve_batch
from .neo4j_client import Neo4jClient


# ── ID helpers ────────────────────────────────────────────────────────────────

def _id(prefix: str, value: str) -> str:
    """Stable 12-char MD5-based ID."""
    return f"{prefix}_{hashlib.md5(value.encode()).hexdigest()[:12]}"


# ── Single result loader ──────────────────────────────────────────────────────

def load_result(client: Neo4jClient, result: ExtractionResult) -> dict:
    """Load one ExtractionResult into Neo4j.

    Returns stats dict: {supplier, invoice, line_items}.
    """
    inv = result.invoice
    stats = {"supplier": False, "invoice": False, "line_items": 0}

    # 1. Supplier node
    supplier_id = None
    if inv.vendor_name:
        supplier_id = _id("sup", inv.vendor_name.lower())
        client.run_write(
            """
            MERGE (s:Supplier {id: $id})
            ON CREATE SET
                s.name            = $name,
                s.canonical_name  = $canonical_name,
                s.address         = $address,
                s.created_at      = datetime()
            ON MATCH SET
                s.last_seen       = datetime()
            """,
            id=supplier_id,
            name=inv.vendor_name,
            canonical_name=inv.vendor_name,   # Updated after entity resolution
            address=inv.vendor_address or "",
        )
        stats["supplier"] = True

    # 2. Invoice node
    invoice_id = _id("inv", result.image_path)
    client.run_write(
        """
        MERGE (i:Invoice {id: $id})
        ON CREATE SET
            i.invoice_number  = $invoice_number,
            i.date            = $date,
            i.due_date        = $due_date,
            i.total_amount    = $total,
            i.subtotal        = $subtotal,
            i.tax             = $tax,
            i.currency        = $currency,
            i.payment_terms   = $payment_terms,
            i.image_path      = $image_path,
            i.confidence      = $confidence,
            i.created_at      = datetime()
        """,
        id=invoice_id,
        invoice_number=inv.invoice_number or "",
        date=inv.invoice_date or "",
        due_date=inv.due_date or "",
        total=float(inv.total_amount) if inv.total_amount else 0.0,
        subtotal=float(inv.subtotal) if inv.subtotal else 0.0,
        tax=float(inv.tax) if inv.tax else 0.0,
        currency=inv.currency,
        payment_terms=inv.payment_terms or "",
        image_path=result.image_path,
        confidence=inv.confidence_scores.overall if inv.confidence_scores else 0.0,
    )
    stats["invoice"] = True

    # 3. Supplier → Invoice
    if supplier_id:
        client.run_write(
            """
            MATCH (s:Supplier {id: $sid})
            MATCH (i:Invoice  {id: $iid})
            MERGE (s)-[:ISSUED]->(i)
            """,
            sid=supplier_id,
            iid=invoice_id,
        )

    # 4. LineItem nodes + Invoice → LineItem
    for idx, item in enumerate(inv.line_items):
        item_id = _id("item", f"{invoice_id}_{idx}")
        client.run_write(
            """
            MERGE (l:LineItem {id: $id})
            ON CREATE SET
                l.description = $description,
                l.quantity    = $quantity,
                l.unit_price  = $unit_price,
                l.total       = $total,
                l.position    = $position
            """,
            id=item_id,
            description=item.description,
            quantity=float(item.quantity) if item.quantity else 0.0,
            unit_price=float(item.unit_price) if item.unit_price else 0.0,
            total=float(item.total) if item.total else 0.0,
            position=idx,
        )
        client.run_write(
            """
            MATCH (i:Invoice  {id: $iid})
            MATCH (l:LineItem {id: $lid})
            MERGE (i)-[:CONTAINS]->(l)
            """,
            iid=invoice_id,
            lid=item_id,
        )
        stats["line_items"] += 1

    return stats


# ── Batch loader ──────────────────────────────────────────────────────────────

def load_all(
    client: Neo4jClient,
    results_dir: str = "data/extraction_results",
    entity_resolution: bool = True,
) -> dict:
    """Load all saved ExtractionResult JSON files into Neo4j.

    Args:
        client: Connected Neo4jClient
        results_dir: Folder containing *.json ExtractionResult files
        entity_resolution: Run LLM entity resolution before loading

    Returns:
        Aggregate statistics dict
    """
    path = Path(results_dir)
    json_files = sorted(path.glob("*.json"))

    if not json_files:
        logger.warning(f"No extraction results in {path}. Run extract_and_save.py first.")
        return {}

    logger.info(f"Loading {len(json_files)} results into Neo4j...")

    # Deserialise all results
    results: list[ExtractionResult] = []
    for f in json_files:
        try:
            results.append(ExtractionResult.model_validate(json.loads(f.read_text())))
        except Exception as exc:
            logger.error(f"Failed to load {f.name}: {exc}")

    # Entity resolution — one Claude call for all vendor names
    if entity_resolution:
        vendor_names = [r.invoice.vendor_name for r in results if r.invoice.vendor_name]
        if vendor_names:
            canonical_map = resolve_batch(vendor_names)
            for r in results:
                if r.invoice.vendor_name and r.invoice.vendor_name in canonical_map:
                    r.invoice.vendor_name = canonical_map[r.invoice.vendor_name]

    # Load into graph
    totals: dict = {"suppliers": 0, "invoices": 0, "line_items": 0, "errors": 0}
    for r in results:
        try:
            stats = load_result(client, r)
            totals["suppliers"] += int(stats["supplier"])
            totals["invoices"] += int(stats["invoice"])
            totals["line_items"] += stats["line_items"]
        except Exception as exc:
            logger.error(f"Load error: {exc}")
            totals["errors"] += 1

    logger.info(f"Graph load complete: {totals}")
    return totals
