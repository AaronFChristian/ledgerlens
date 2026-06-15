"""Day 2 tests — graph layer and agent (offline where possible).

Tests that require Neo4j are marked with @pytest.mark.integration
and skipped by default. Run them with:
    pytest tests/test_graph.py -m integration
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ledgerlens.agent.graphrag_agent import AgentState, GraphRAGAgent, _synthesize, _understand
from ledgerlens.graph.entity_resolution import resolve_batch
from ledgerlens.graph.loader import _id, load_result


# ── ID helper ─────────────────────────────────────────────────────────────────

class TestIdHelper:
    def test_deterministic(self):
        assert _id("sup", "Apple Inc") == _id("sup", "Apple Inc")

    def test_different_prefixes(self):
        assert _id("sup", "Apple") != _id("inv", "Apple")

    def test_different_values(self):
        assert _id("sup", "Apple") != _id("sup", "Google")

    def test_format(self):
        result = _id("sup", "test")
        assert result.startswith("sup_")
        assert len(result) == 16  # "sup_" + 12 chars


# ── Entity resolution ─────────────────────────────────────────────────────────

class TestEntityResolution:
    def test_empty_list(self):
        result = resolve_batch([])
        assert result == {}

    def test_short_names_skipped(self):
        result = resolve_batch(["", "X", "  "])
        assert result == {}

    @patch("ledgerlens.graph.entity_resolution._get_client")
    def test_successful_resolution(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"Apple Inc": "Apple Inc.", "Apple Computer": "Apple Inc."}')]
        mock_get_client.return_value.messages.create.return_value = mock_response

        result = resolve_batch(["Apple Inc", "Apple Computer"])
        assert result["Apple Inc"] == "Apple Inc."
        assert result["Apple Computer"] == "Apple Inc."

    @patch("ledgerlens.graph.entity_resolution._get_client")
    def test_fallback_on_api_error(self, mock_get_client):
        mock_get_client.return_value.messages.create.side_effect = Exception("API error")
        result = resolve_batch(["Apple Inc", "Google"])
        # Falls back to identity mapping
        assert result["Apple Inc"] == "Apple Inc"
        assert result["Google"] == "Google"

    @patch("ledgerlens.graph.entity_resolution._get_client")
    def test_strips_markdown_fences(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='```json\n{"Corp A": "Corporation A"}\n```')]
        mock_get_client.return_value.messages.create.return_value = mock_response
        result = resolve_batch(["Corp A"])
        assert result["Corp A"] == "Corporation A"


# ── Graph loader ──────────────────────────────────────────────────────────────

class TestGraphLoader:
    def _make_result(self, vendor_name="Acme Corp", total=100.0, n_items=2):
        from ledgerlens.extraction.schemas import (
            ConfidenceScores, ExtractionResult, InvoiceExtraction, LineItem
        )
        invoice = InvoiceExtraction(
            vendor_name=vendor_name,
            invoice_number="INV-001",
            invoice_date="2024-01-15",
            total_amount=total,
            line_items=[
                LineItem(description=f"Item {i}", total=total / n_items)
                for i in range(n_items)
            ],
        )
        invoice.confidence_scores = ConfidenceScores(
            vendor_name=1.0, invoice_number=1.0, invoice_date=1.0,
            total_amount=1.0, line_items=1.0, overall=0.95,
        )
        return ExtractionResult(
            invoice=invoice,
            needs_human_review=False,
            low_confidence_fields=[],
            extraction_time_ms=800.0,
            token_usage={"cost_usd": 0.009},
            image_path="data/samples/cord/cord_0001.png",
            model_used="claude-sonnet-4-6",
        )

    def test_load_result_calls_neo4j(self):
        """load_result should call MERGE for supplier, invoice, and line items."""
        mock_client = MagicMock()
        mock_client.run_write = MagicMock()

        result = self._make_result(n_items=3)
        stats = load_result(mock_client, result)

        assert stats["supplier"] is True
        assert stats["invoice"] is True
        assert stats["line_items"] == 3
        # supplier + invoice + supplier→invoice + 3 items + 3 item relationships = 9 calls
        assert mock_client.run_write.call_count == 9

    def test_no_supplier_skips_supplier_node(self):
        mock_client = MagicMock()
        result = self._make_result()
        result.invoice.vendor_name = None

        stats = load_result(mock_client, result)
        assert stats["supplier"] is False

    def test_empty_line_items(self):
        mock_client = MagicMock()
        result = self._make_result(n_items=0)
        result.invoice.line_items = []
        stats = load_result(mock_client, result)
        assert stats["line_items"] == 0


# ── GraphRAG agent ────────────────────────────────────────────────────────────

class TestAgentState:
    def test_initial_state_structure(self):
        state: AgentState = {
            "question": "Which suppliers have the most invoices?",
            "cypher_queries": [],
            "graph_results": [],
            "traversal_path": [],
            "answer": "",
            "error": None,
        }
        assert state["question"] != ""
        assert state["answer"] == ""
        assert state["error"] is None

    @patch("ledgerlens.agent.graphrag_agent.anthropic.Anthropic")
    def test_understand_node_parses_queries(self, mock_anthropic):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='''{"cypher_queries": ["MATCH (s:Supplier) RETURN s"], "reasoning": "list all suppliers"}''')]
        mock_client.messages.create.return_value = mock_response

        state: AgentState = {
            "question": "List all suppliers",
            "cypher_queries": [], "graph_results": [],
            "traversal_path": [], "answer": "", "error": None,
        }
        result = _understand(state, mock_client)
        assert len(result["cypher_queries"]) == 1
        assert "MATCH" in result["cypher_queries"][0]
        assert result["error"] is None

    @patch("ledgerlens.agent.graphrag_agent.anthropic.Anthropic")
    def test_synthesize_extracts_answer(self, mock_anthropic):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"answer": "There are 5 suppliers.", "key_findings": ["5 unique suppliers"], "traversal_summary": "Supplier nodes queried"}')]
        mock_client.messages.create.return_value = mock_response

        state: AgentState = {
            "question": "How many suppliers?",
            "cypher_queries": ["MATCH (s:Supplier) RETURN count(s)"],
            "graph_results": [{"query": "...", "rows": [{"count(s)": 5}], "count": 1}],
            "traversal_path": ["Plan: count suppliers"],
            "answer": "", "error": None,
        }
        result = _synthesize(state, mock_client)
        assert result["answer"] == "There are 5 suppliers."

    @patch("ledgerlens.agent.graphrag_agent.anthropic.Anthropic")
    def test_error_state_skips_query(self, mock_anthropic):
        from ledgerlens.agent.graphrag_agent import _query_graph
        mock_neo4j = MagicMock()

        state: AgentState = {
            "question": "test", "cypher_queries": [],
            "graph_results": [], "traversal_path": [],
            "answer": "", "error": "something went wrong",
        }
        result = _query_graph(state, mock_neo4j)
        # Should pass through without calling Neo4j
        mock_neo4j.run.assert_not_called()
        assert result["error"] == "something went wrong"
