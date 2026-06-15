"""LangGraph GraphRAG agent for multi-hop invoice/supplier queries.

State machine:
  understand → query_graph → synthesize → END

The agent:
  1. Translates a natural-language question into Cypher queries
  2. Executes those queries against Neo4j
  3. Synthesises an answer + returns the full traversal path (auditability)
"""

from __future__ import annotations

import json
from typing import Optional, TypedDict

import anthropic
from langgraph.graph import END, StateGraph
from loguru import logger

from ..config import settings
from ..graph.neo4j_client import Neo4jClient


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    question: str
    cypher_queries: list[str]
    graph_results: list[dict]
    traversal_path: list[str]
    answer: str
    error: Optional[str]


# ── Prompts ───────────────────────────────────────────────────────────────────

GRAPH_SCHEMA = """
Neo4j graph schema:
  (:Supplier {id, name, canonical_name, address})
  (:Invoice  {id, invoice_number, date, due_date, total_amount, subtotal, tax, currency, payment_terms, confidence})
  (:LineItem {id, description, quantity, unit_price, total, position})

Relationships:
  (Supplier)-[:ISSUED]->(Invoice)
  (Invoice)-[:CONTAINS]->(LineItem)

Useful Cypher patterns:
  # Suppliers with most invoices
  MATCH (s:Supplier)-[:ISSUED]->(i:Invoice)
  RETURN s.canonical_name AS supplier, count(i) AS invoice_count
  ORDER BY invoice_count DESC LIMIT 10

  # Total spend per supplier
  MATCH (s:Supplier)-[:ISSUED]->(i:Invoice)
  RETURN s.canonical_name AS supplier, round(sum(i.total_amount)*100)/100 AS total_spend
  ORDER BY total_spend DESC

  # Invoices above a threshold
  MATCH (s:Supplier)-[:ISSUED]->(i:Invoice)
  WHERE i.total_amount > 100
  RETURN s.canonical_name, i.invoice_number, i.total_amount ORDER BY i.total_amount DESC

  # Line items containing a keyword
  MATCH (i:Invoice)-[:CONTAINS]->(l:LineItem)
  WHERE toLower(l.description) CONTAINS 'keyword'
  RETURN i.id, l.description, l.total

  # All suppliers
  MATCH (s:Supplier) RETURN s.canonical_name, s.address ORDER BY s.canonical_name
"""

UNDERSTAND_SYSTEM = f"""You are a GraphRAG query planner. Translate a natural-language question
about invoices and suppliers into 1-3 Cypher queries.

{GRAPH_SCHEMA}

Return ONLY a JSON object:
{{
  "cypher_queries": ["MATCH ...", "MATCH ..."],
  "reasoning": "one sentence explaining what you're querying"
}}
No explanation, no markdown fences."""

SYNTHESIZE_SYSTEM = """You are a business intelligence assistant analysing invoice data.
Given a question and the raw Neo4j query results, write a clear concise answer.

Return ONLY a JSON object:
{
  "answer": "direct answer with specific numbers from the data",
  "key_findings": ["finding 1", "finding 2"],
  "traversal_summary": "one sentence describing the graph path taken"
}
No markdown fences."""


# ── Node functions ─────────────────────────────────────────────────────────────

def _understand(state: AgentState, client: anthropic.Anthropic) -> AgentState:
    """Plan Cypher queries from the question."""
    logger.info(f"Planning queries for: {state['question']}")

    resp = client.messages.create(
        model=settings.claude_model,
        max_tokens=512,
        system=UNDERSTAND_SYSTEM,
        messages=[{"role": "user", "content": f"Question: {state['question']}"}],
    )

    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:-1]).strip()

    try:
        parsed = json.loads(raw)
        queries = parsed.get("cypher_queries", [])
        reasoning = parsed.get("reasoning", "")
        logger.info(f"Planned {len(queries)} queries — {reasoning}")
        return {
            **state,
            "cypher_queries": queries,
            "traversal_path": [f"Plan: {reasoning}"],
        }
    except json.JSONDecodeError as exc:
        logger.error(f"Query plan parse error: {exc}\nRaw: {raw[:300]}")
        return {**state, "error": f"Query planning failed: {exc}"}


def _query_graph(state: AgentState, neo4j: Neo4jClient) -> AgentState:
    """Execute all planned Cypher queries against Neo4j."""
    if state.get("error"):
        return state

    all_results: list[dict] = []
    path = list(state.get("traversal_path", []))

    for i, query in enumerate(state["cypher_queries"], 1):
        logger.info(f"Executing query {i}: {query[:80]}...")
        try:
            rows = neo4j.run(query)
            all_results.append({"query": query, "rows": rows, "count": len(rows)})
            path.append(f"Query {i} → {len(rows)} records")
        except Exception as exc:
            logger.error(f"Cypher error: {exc}")
            all_results.append({"query": query, "error": str(exc), "rows": []})
            path.append(f"Query {i} → error: {exc}")

    return {**state, "graph_results": all_results, "traversal_path": path}


def _synthesize(state: AgentState, client: anthropic.Anthropic) -> AgentState:
    """Synthesise final answer from graph results."""
    if state.get("error"):
        return {**state, "answer": f"Error: {state['error']}"}

    context = json.dumps(state["graph_results"], indent=2, default=str)

    resp = client.messages.create(
        model=settings.claude_model,
        max_tokens=1024,
        system=SYNTHESIZE_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Question: {state['question']}\n\nGraph results:\n{context}",
        }],
    )

    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:-1]).strip()

    try:
        parsed = json.loads(raw)
        answer = parsed.get("answer", "")
        traversal = list(state.get("traversal_path", []))
        traversal.append(f"Traversal: {parsed.get('traversal_summary', '')}")
        logger.info(f"Answer: {answer[:120]}")
        return {**state, "answer": answer, "traversal_path": traversal}
    except json.JSONDecodeError:
        return {**state, "answer": raw}


# ── Agent class ───────────────────────────────────────────────────────────────

class GraphRAGAgent:
    """LangGraph-based GraphRAG agent for multi-hop invoice intelligence.

    Usage:
        with Neo4jClient() as neo4j:
            agent = GraphRAGAgent(neo4j)
            result = agent.ask("Which suppliers appear on the most invoices?")
            print(result["answer"])
            print(result["traversal_path"])
    """

    def __init__(self, neo4j_client: Neo4jClient) -> None:
        self.neo4j = neo4j_client
        self._claude = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._graph = self._build()

    def _build(self):
        wf = StateGraph(AgentState)

        wf.add_node("understand",   lambda s: _understand(s,    self._claude))
        wf.add_node("query_graph",  lambda s: _query_graph(s,   self.neo4j))
        wf.add_node("synthesize",   lambda s: _synthesize(s,    self._claude))

        wf.set_entry_point("understand")
        wf.add_edge("understand",  "query_graph")
        wf.add_edge("query_graph", "synthesize")
        wf.add_edge("synthesize",  END)

        return wf.compile()

    def ask(self, question: str) -> dict:
        """Ask a multi-hop question about the invoice knowledge graph.

        Returns:
            {
              "question": str,
              "answer": str,
              "traversal_path": list[str],   # auditable graph path
              "cypher_queries": list[str],   # Cypher executed
              "graph_results": list[dict],   # raw Neo4j records
              "error": str | None
            }
        """
        initial: AgentState = {
            "question": question,
            "cypher_queries": [],
            "graph_results": [],
            "traversal_path": [],
            "answer": "",
            "error": None,
        }
        final = self._graph.invoke(initial)
        return {
            "question":       question,
            "answer":         final.get("answer", ""),
            "traversal_path": final.get("traversal_path", []),
            "cypher_queries": final.get("cypher_queries", []),
            "graph_results":  final.get("graph_results", []),
            "error":          final.get("error"),
        }
