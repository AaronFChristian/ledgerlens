"""Ask multi-hop questions about the invoice knowledge graph.

Single question:
    python scripts/ask_agent.py "Which suppliers appear most frequently?"

Interactive mode:
    python scripts/ask_agent.py
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ledgerlens.agent.graphrag_agent import GraphRAGAgent
from ledgerlens.graph.neo4j_client import Neo4jClient
from ledgerlens.utils.logging import setup_logging

console = Console()

EXAMPLES = [
    "Which suppliers appear on the most invoices?",
    "What is the total invoice value across all suppliers?",
    "Which invoices have the highest total amounts?",
    "List all unique suppliers in the graph",
    "What are the most common line item descriptions?",
    "Which suppliers have invoices with totals above 10000?",
]


def ask(agent: GraphRAGAgent, question: str) -> None:
    console.print(f"\n[bold blue]Question:[/bold blue] {question}")

    with console.status("[dim]Querying graph...[/dim]"):
        result = agent.ask(question)

    if result.get("error"):
        console.print(f"[red]Error: {result['error']}[/red]")
        return

    console.print(Panel(result["answer"], title="[bold green]Answer[/bold green]", border_style="green"))

    if result.get("traversal_path"):
        console.print("\n[bold]Audit trail (traversal path):[/bold]")
        for step in result["traversal_path"]:
            console.print(f"  [dim]→[/dim] {step}")

    if result.get("cypher_queries"):
        console.print("\n[bold]Cypher executed:[/bold]")
        for q in result["cypher_queries"]:
            console.print(f"  [dim]{q[:120]}[/dim]")


def main() -> None:
    setup_logging(level="WARNING")

    parser = argparse.ArgumentParser(description="Ask the LedgerLens GraphRAG agent")
    parser.add_argument("question", nargs="?", help="Question to ask (omit for interactive mode)")
    args = parser.parse_args()

    with Neo4jClient() as neo4j:
        if not neo4j.verify_connection():
            console.print("[red]Cannot connect to Neo4j. Check .env and run load_graph.py first.[/red]")
            return

        agent = GraphRAGAgent(neo4j)

        if args.question:
            ask(agent, args.question)
        else:
            console.print("\n[bold]LedgerLens GraphRAG — Interactive Mode[/bold]")
            console.print("[dim]Commands: 'examples' · 'exit'[/dim]\n")

            while True:
                try:
                    q = input("Question: ").strip()
                    if not q:
                        continue
                    if q.lower() == "exit":
                        break
                    if q.lower() == "examples":
                        for ex in EXAMPLES:
                            console.print(f"  [dim]• {ex}[/dim]")
                        continue
                    ask(agent, q)
                except (KeyboardInterrupt, EOFError):
                    break


if __name__ == "__main__":
    main()
