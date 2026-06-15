"""Day 1 test suite — extraction schemas, confidence scoring, review routing.

Run: pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ledgerlens.extraction.schemas import (
    ConfidenceScores,
    ExtractionResult,
    HumanReviewItem,
    InvoiceExtraction,
    LineItem,
)
from ledgerlens.extraction.confidence import (
    compute_confidence_scores,
    get_low_confidence_fields,
    needs_human_review,
)

THRESHOLD = 0.75


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def complete_invoice() -> InvoiceExtraction:
    return InvoiceExtraction(
        vendor_name="Acme Corporation",
        vendor_address="123 Market St, San Francisco, CA 94105",
        invoice_number="INV-2024-001",
        invoice_date="2024-01-15",
        due_date="2024-02-15",
        line_items=[
            LineItem(description="Professional Services — Jan 2024", quantity=10, unit_price=150.0, total=1500.0),
            LineItem(description="Software License (annual)", quantity=1, unit_price=499.0, total=499.0),
        ],
        subtotal=1999.0,
        tax=164.92,
        total_amount=2163.92,
        currency="USD",
        payment_terms="Net 30",
    )


@pytest.fixture
def empty_invoice() -> InvoiceExtraction:
    return InvoiceExtraction()


@pytest.fixture
def partial_invoice() -> InvoiceExtraction:
    return InvoiceExtraction(
        vendor_name="X",  # Too short
        invoice_number="INV-001",
        total_amount=0.0,  # Zero amount — suspicious
        line_items=[],
    )


# ── Schema tests ──────────────────────────────────────────────────────────────

class TestLineItem:
    def test_valid_line_item(self):
        item = LineItem(description="Widget A", quantity=5, unit_price=20.0, total=100.0)
        assert item.confidence == 1.0

    def test_math_inconsistency_reduces_confidence(self):
        item = LineItem(description="Widget A", quantity=5, unit_price=20.0, total=999.0)
        assert item.confidence < 1.0

    def test_missing_optional_fields_ok(self):
        item = LineItem(description="Consulting services")
        assert item.quantity is None
        assert item.total is None
        assert item.confidence == 1.0


class TestInvoiceExtraction:
    def test_full_invoice_validates(self, complete_invoice):
        assert complete_invoice.vendor_name == "Acme Corporation"
        assert complete_invoice.total_amount == 2163.92
        assert len(complete_invoice.line_items) == 2

    def test_empty_invoice_all_none(self, empty_invoice):
        assert empty_invoice.vendor_name is None
        assert empty_invoice.total_amount is None
        assert empty_invoice.line_items == []

    def test_confidence_scores_initially_none(self, complete_invoice):
        assert complete_invoice.confidence_scores is None


class TestHumanReviewItem:
    def test_priority_escalates_on_very_low_confidence(self):
        invoice = InvoiceExtraction(total_amount=0.01)
        scores = ConfidenceScores(
            vendor_name=0.0,
            invoice_number=0.0,
            invoice_date=0.0,
            total_amount=0.0,
            line_items=0.0,
            overall=0.1,
        )
        invoice.confidence_scores = scores

        result = ExtractionResult(
            invoice=invoice,
            needs_human_review=True,
            low_confidence_fields=["vendor_name", "total_amount"],
            extraction_time_ms=1000.0,
            token_usage={"input_tokens": 500, "output_tokens": 300, "cost_usd": 0.006},
            image_path="test.png",
            model_used="claude-sonnet-4-6",
        )
        review = HumanReviewItem(result=result, reason="low confidence", low_fields=["vendor_name"])
        assert review.priority == "urgent"


# ── Confidence tests ──────────────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_complete_invoice_high_confidence(self, complete_invoice):
        scores = compute_confidence_scores(complete_invoice)
        assert scores.overall >= 0.90
        assert scores.vendor_name == 1.0
        assert scores.total_amount == 1.0

    def test_empty_invoice_near_zero(self, empty_invoice):
        scores = compute_confidence_scores(empty_invoice)
        assert scores.overall < 0.50
        assert scores.vendor_name == 0.0
        assert scores.total_amount == 0.0

    def test_short_vendor_name_penalised(self):
        invoice = InvoiceExtraction(vendor_name="X")
        scores = compute_confidence_scores(invoice)
        assert scores.vendor_name < 1.0

    def test_math_inconsistency_reduces_overall(self):
        """subtotal + tax ≠ total should penalise overall confidence."""
        invoice = InvoiceExtraction(
            vendor_name="Test Corp",
            invoice_number="INV-001",
            invoice_date="2024-01-01",
            subtotal=100.0,
            tax=8.0,
            total_amount=999.0,  # Should be 108.00
            line_items=[LineItem(description="Item A", total=100.0)],
        )
        scores = compute_confidence_scores(invoice)
        assert scores.overall < 0.90  # Math penalty applied

    def test_math_consistent_no_penalty(self, complete_invoice):
        """subtotal + tax = total → no math penalty."""
        scores = compute_confidence_scores(complete_invoice)
        # 1999.00 + 164.92 = 2163.92 ✓ (within $0.10 tolerance)
        assert scores.overall >= 0.90

    def test_negative_total_penalised(self):
        invoice = InvoiceExtraction(vendor_name="Corp", total_amount=-50.0)
        scores = compute_confidence_scores(invoice)
        assert scores.total_amount < 1.0

    def test_zero_total_penalised(self):
        invoice = InvoiceExtraction(vendor_name="Corp", total_amount=0.0)
        scores = compute_confidence_scores(invoice)
        assert scores.total_amount < 1.0

    def test_non_standard_date_penalised(self):
        invoice = InvoiceExtraction(invoice_date="Jan 15, 2024")  # Not YYYY-MM-DD
        scores = compute_confidence_scores(invoice)
        assert scores.invoice_date < 1.0

    def test_iso_date_full_score(self):
        invoice = InvoiceExtraction(invoice_date="2024-01-15")
        scores = compute_confidence_scores(invoice)
        assert scores.invoice_date == 1.0

    def test_all_scores_in_range(self, complete_invoice):
        scores = compute_confidence_scores(complete_invoice)
        for field in ["vendor_name", "invoice_number", "invoice_date", "total_amount", "line_items", "overall"]:
            val = getattr(scores, field)
            assert 0.0 <= val <= 1.0, f"{field} = {val} out of range"


class TestLowConfidenceFields:
    def test_no_low_fields_for_complete(self, complete_invoice):
        scores = compute_confidence_scores(complete_invoice)
        low = get_low_confidence_fields(scores, THRESHOLD)
        assert low == []

    def test_missing_fields_flagged(self, empty_invoice):
        scores = compute_confidence_scores(empty_invoice)
        low = get_low_confidence_fields(scores, THRESHOLD)
        assert "vendor_name" in low
        assert "invoice_number" in low
        assert "total_amount" in low

    def test_threshold_respected(self):
        scores = ConfidenceScores(
            vendor_name=0.9,
            invoice_number=0.5,  # Below 0.75
            invoice_date=0.8,
            total_amount=0.3,    # Below 0.75
            line_items=0.9,
            overall=0.7,
        )
        low = get_low_confidence_fields(scores, THRESHOLD)
        assert "invoice_number" in low
        assert "total_amount" in low
        assert "vendor_name" not in low


class TestReviewRouting:
    def test_complete_invoice_no_review(self, complete_invoice):
        scores = compute_confidence_scores(complete_invoice)
        assert not needs_human_review(scores, THRESHOLD)

    def test_empty_invoice_needs_review(self, empty_invoice):
        scores = compute_confidence_scores(empty_invoice)
        assert needs_human_review(scores, THRESHOLD)

    def test_missing_critical_field_forces_review(self):
        """Even if overall is ok, a zero-confidence critical field must escalate."""
        scores = ConfidenceScores(
            vendor_name=0.0,   # Missing vendor = always review
            invoice_number=1.0,
            invoice_date=1.0,
            total_amount=1.0,
            line_items=1.0,
            overall=0.80,      # Above threshold
        )
        assert needs_human_review(scores, THRESHOLD)

    def test_overall_below_threshold_needs_review(self):
        scores = ConfidenceScores(
            vendor_name=0.8,
            invoice_number=0.8,
            invoice_date=0.8,
            total_amount=0.8,
            line_items=0.8,
            overall=0.60,  # Below THRESHOLD
        )
        assert needs_human_review(scores, THRESHOLD)


class TestExtractionResultProperties:
    def test_cost_property(self):
        invoice = InvoiceExtraction(vendor_name="Corp")
        scores = compute_confidence_scores(invoice)
        invoice.confidence_scores = scores
        result = ExtractionResult(
            invoice=invoice,
            needs_human_review=False,
            low_confidence_fields=[],
            extraction_time_ms=800.0,
            token_usage={"cost_usd": 0.0042},
            image_path="test.png",
            model_used="claude-sonnet-4-6",
        )
        assert result.cost_usd == 0.0042
        assert result.auto_approved is True
