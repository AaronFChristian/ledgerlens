"""Confidence scoring for extracted invoice fields.

Produces ConfidenceScores (per-field + overall) and determines
whether a document needs human review.

Scoring philosophy:
  - Presence alone isn't enough: we cross-validate math, check plausibility
  - Threshold is configurable (default 0.75) via settings
  - Human review queue is the safety net, not an error state
"""

from __future__ import annotations

import re
from typing import Optional

from .schemas import ConfidenceScores, InvoiceExtraction

# Fields considered critical for auto-approval
CRITICAL_FIELDS = {"vendor_name", "invoice_number", "total_amount"}

# Date pattern: loose check for YYYY-MM-DD
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Invoice number: at least 2 chars
_INV_NUM_RE = re.compile(r"^.{2,}$")


def _score_string(value: Optional[str], min_len: int = 2) -> float:
    """Score a string field: 0 if missing, 0.3 if too short, 1.0 if valid."""
    if value is None:
        return 0.0
    cleaned = value.strip()
    if len(cleaned) == 0:
        return 0.0
    if len(cleaned) < min_len:
        return 0.3
    return 1.0


def _score_date(value: Optional[str]) -> float:
    """Score a date field: penalise non-standard formats."""
    if value is None:
        return 0.0
    if _DATE_RE.match(value.strip()):
        return 1.0
    # Non-standard format — extracted but possibly wrong
    return 0.5


def _score_amount(value: Optional[float]) -> float:
    """Score a monetary amount."""
    if value is None:
        return 0.0
    if value < 0:
        return 0.2  # Negative total is suspicious
    if value == 0:
        return 0.4  # Zero total is possible but unusual
    return 1.0


def _score_line_items(invoice: InvoiceExtraction) -> float:
    """Score line items as a group."""
    if not invoice.line_items:
        # Missing line items isn't always wrong (some invoices are just totals)
        # But penalise if subtotal exists, implying there should be items
        return 0.4 if invoice.subtotal else 0.7

    item_scores = []
    for item in invoice.line_items:
        # Each item must at least have a description
        desc_score = _score_string(item.description, min_len=3)
        total_score = _score_amount(item.total) if item.total is not None else 0.5
        item_scores.append((desc_score + total_score) / 2)

    return round(sum(item_scores) / len(item_scores), 3)


def _math_consistency(invoice: InvoiceExtraction) -> float:
    """Cross-validate: subtotal + tax (+ shipping - discount) ≈ total."""
    if invoice.subtotal is None or invoice.total_amount is None:
        return 1.0  # Can't validate, don't penalise

    expected = invoice.subtotal
    if invoice.tax is not None:
        expected += invoice.tax
    if invoice.shipping is not None:
        expected += invoice.shipping
    if invoice.discount is not None:
        expected -= invoice.discount

    diff = abs(expected - invoice.total_amount)

    # Allow small rounding differences
    if diff <= 0.10:
        return 1.0
    if diff <= 1.00:
        return 0.85
    if diff <= 5.00:
        return 0.70
    return 0.50  # Large discrepancy — likely extraction error


def compute_confidence_scores(invoice: InvoiceExtraction) -> ConfidenceScores:
    """Compute per-field and weighted overall confidence score.

    Weights reflect field importance for accounts-payable automation:
      - total_amount (30%) — highest: wrong amount = wrong payment
      - vendor_name (20%) — entity resolution depends on this
      - line_items (20%) — core structured data
      - invoice_number (15%) — deduplication key
      - invoice_date (15%) — payment timing
    """
    vendor_score = _score_string(invoice.vendor_name, min_len=2)
    inv_num_score = _score_string(invoice.invoice_number, min_len=2)
    date_score = _score_date(invoice.invoice_date)
    total_score = _score_amount(invoice.total_amount)
    items_score = _score_line_items(invoice)
    math_factor = _math_consistency(invoice)

    weighted = (
        vendor_score * 0.20
        + inv_num_score * 0.15
        + date_score * 0.15
        + total_score * 0.30
        + items_score * 0.20
    )

    overall = round(weighted * math_factor, 3)

    return ConfidenceScores(
        vendor_name=vendor_score,
        invoice_number=inv_num_score,
        invoice_date=date_score,
        total_amount=total_score,
        line_items=items_score,
        overall=overall,
    )


def get_low_confidence_fields(
    scores: ConfidenceScores,
    threshold: float,
) -> list[str]:
    """Return field names whose score falls below threshold."""
    checks = {
        "vendor_name": scores.vendor_name,
        "invoice_number": scores.invoice_number,
        "invoice_date": scores.invoice_date,
        "total_amount": scores.total_amount,
        "line_items": scores.line_items,
    }
    return [field for field, score in checks.items() if score < threshold]


def needs_human_review(scores: ConfidenceScores, threshold: float) -> bool:
    """True if overall confidence is below threshold OR any critical field is missing."""
    if scores.overall < threshold:
        return True
    # Any critical field with zero confidence → always escalate
    critical_scores = {
        "vendor_name": scores.vendor_name,
        "invoice_number": scores.invoice_number,
        "total_amount": scores.total_amount,
    }
    return any(s == 0.0 for s in critical_scores.values())
