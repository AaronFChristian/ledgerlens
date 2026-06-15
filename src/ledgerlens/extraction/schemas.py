"""Pydantic v2 schemas for invoice extraction.

All monetary values stored as float (sufficient for extraction accuracy).
Downstream accounting systems handle Decimal precision.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, model_validator


class LineItem(BaseModel):
    """Single line item from an invoice."""

    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total: Optional[float] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_math(self) -> "LineItem":
        """Flag mathematically inconsistent line items with lower confidence."""
        if self.quantity and self.unit_price and self.total:
            expected = round(self.quantity * self.unit_price, 2)
            if abs(expected - self.total) > 0.05:
                # Don't override if confidence was already set lower
                self.confidence = min(self.confidence, 0.6)
        return self


class ConfidenceScores(BaseModel):
    """Per-field and overall confidence scores (0.0 = missing/wrong, 1.0 = certain)."""

    vendor_name: float = Field(ge=0.0, le=1.0)
    invoice_number: float = Field(ge=0.0, le=1.0)
    invoice_date: float = Field(ge=0.0, le=1.0)
    total_amount: float = Field(ge=0.0, le=1.0)
    line_items: float = Field(ge=0.0, le=1.0)
    overall: float = Field(ge=0.0, le=1.0)


class InvoiceExtraction(BaseModel):
    """Fully structured invoice data extracted by Claude vision."""

    # Header fields
    vendor_name: Optional[str] = None
    vendor_address: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None        # YYYY-MM-DD
    due_date: Optional[str] = None            # YYYY-MM-DD
    purchase_order_number: Optional[str] = None

    # Line items
    line_items: list[LineItem] = Field(default_factory=list)

    # Totals
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    tax_rate: Optional[float] = None          # e.g. 0.08 for 8%
    shipping: Optional[float] = None
    discount: Optional[float] = None
    total_amount: Optional[float] = None
    currency: str = "USD"

    # Payment
    payment_terms: Optional[str] = None      # e.g. "Net 30"
    bank_details: Optional[str] = None

    # Set after extraction by the confidence module
    confidence_scores: Optional[ConfidenceScores] = None


class ExtractionResult(BaseModel):
    """Full result of one extraction pass — invoice + routing + telemetry."""

    invoice: InvoiceExtraction
    needs_human_review: bool
    low_confidence_fields: list[str]

    # Telemetry
    extraction_time_ms: float
    token_usage: dict                         # input_tokens, output_tokens, cost_usd
    image_path: str
    model_used: str

    @property
    def cost_usd(self) -> float:
        return self.token_usage.get("cost_usd", 0.0)

    @property
    def auto_approved(self) -> bool:
        return not self.needs_human_review


class HumanReviewItem(BaseModel):
    """An extraction routed to the human review queue."""

    result: ExtractionResult
    reason: str
    low_fields: list[str]
    priority: str = "normal"  # normal | high | urgent

    @model_validator(mode="after")
    def set_priority(self) -> "HumanReviewItem":
        score = self.result.invoice.confidence_scores
        if score and score.overall < 0.40:
            self.priority = "urgent"
        elif score and score.overall < 0.60:
            self.priority = "high"
        return self
