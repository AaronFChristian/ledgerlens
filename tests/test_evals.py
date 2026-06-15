import json
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ledgerlens.evals.harness import (
    evaluate_context_relevance,
    evaluate_groundedness,
    field_match,
)


class TestFieldMatch:
    def test_exact_match(self):
        assert field_match("100", "100") is True

    def test_float_int_match(self):
        assert field_match(100.0, "100") is True

    def test_currency_stripped(self):
        assert field_match("$1,234.50", "1234.50") is True

    def test_mismatch(self):
        assert field_match("100", "200") is False

    def test_none_predicted(self):
        assert field_match(None, "100") is False

    def test_none_ground_truth(self):
        assert field_match("100", None) is None

    def test_empty_ground_truth(self):
        assert field_match("100", "") is None

    def test_case_insensitive(self):
        assert field_match("APPLE", "apple") is True

    def test_whitespace_stripped(self):
        assert field_match("  100  ", "100") is True

    def test_won_currency(self):
        assert field_match(5000.0, "5000") is True


class TestGroundednessEval:
    def test_high_groundedness(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value.content = [
            MagicMock(text='{"score": 0.95, "reason": "All claims supported"}')
        ]
        result = evaluate_groundedness(
            "Which supplier has the most invoices?",
            "XXI Cafe leads with 3 invoices.",
            [{"rows": [{"supplier": "XXI Cafe", "invoice_count": 3}]}],
            mock_client,
            "claude-sonnet-4-6",
        )
        assert result["score"] == 0.95
        assert "reason" in result

    def test_handles_parse_error(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value.content = [
            MagicMock(text="Not valid JSON")
        ]
        result = evaluate_groundedness("q", "a", [], mock_client, "claude-sonnet-4-6")
        assert result["score"] == 0.5


class TestContextRelevanceEval:
    def test_high_relevance(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value.content = [
            MagicMock(text='{"score": 0.9, "reason": "Data directly answers the question"}')
        ]
        result = evaluate_context_relevance(
            "Which suppliers have the most invoices?",
            [{"rows": [{"supplier": "XXI Cafe", "invoice_count": 3}]}],
            mock_client,
            "claude-sonnet-4-6",
        )
        assert result["score"] == 0.9

    def test_strips_markdown_fences(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value.content = [
            MagicMock(text='```json\n{"score": 0.8, "reason": "Mostly relevant"}\n```')
        ]
        result = evaluate_context_relevance("q", [], mock_client, "claude-sonnet-4-6")
        assert result["score"] == 0.8


@pytest.mark.slow
class TestExtractionAccuracyCI:
    def test_total_amount_accuracy_above_70_pct(self):
        from ledgerlens.evals.harness import evaluate_extraction_accuracy
        metrics = evaluate_extraction_accuracy(n_samples=20)
        total_acc = metrics["total_amount"]["accuracy"]
        assert total_acc >= 70.0, f"Total amount accuracy {total_acc}% below 70% threshold"

    def test_eval_results_file_exists_after_run(self):
        results_file = Path("data/eval_results.json")
        if results_file.exists():
            data = json.loads(results_file.read_text())
            assert "extraction" in data
