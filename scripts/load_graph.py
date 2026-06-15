"""Load saved extraction results into the Neo4j knowledge graph.

Run after extract_and_save.py:
    python scripts/load_graph.py
    python scripts/load_graph.py --no-entity-resolution   # skip Claude call
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ledgerlens.graph.neo4j_client import Neo4jClient
from ledgerlens.graph.loader import load_all
from ledgerlens.utils.logging import setup_logging

console = Console()


def main(entity_resolution: bool = True) -> None:
    setup_logging(level="INFO")

    console.print("\n[bold]LedgerLens — Loading knowledge graph[/bold]\n")

    with Neo4jClient() as client:
        if not client.verify_connection():
            console.print("[red]Cannot connect to Neo4j.[/red]")
            console.print("[yellow]Check NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env[/yellow]")
            return

        client.setup_schema()

        stats = load_all(
            client,
            results_dir="data/extraction_results",
            entity_resolution=entity_resolution,
        )

        if not stats:
            console.print("[yellow]No data loaded. Run extract_and_save.py first.[/yellow]")
            return

        graph_stats = client.get_stats()

        # Results table
        table = Table(title="Neo4j Graph — Node Counts", box=box.SIMPLE_HEAD)
        table.add_column("Node type", style="bold")
        table.add_column("Count", justify="right")

        table.add_row("Supplier",      str(graph_stats.get("Supplier", 0)))
        table.add_row("Invoice",       str(graph_stats.get("Invoice", 0)))
        table.add_row("LineItem",      str(graph_stats.get("LineItem", 0)))
        table.add_row("Relationships", str(graph_stats.get("relationships", 0)))

        console.print(table)
        console.print(
            "\n[dim]Visualise in Neo4j Browser → "
            "https://console.neo4j.io → Query → MATCH (n) RETURN n LIMIT 50[/dim]"
        )
        console.print("[dim]Next: python scripts/ask_agent.py[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Neo4j knowledge graph")
    parser.add_argument(
        "--no-entity-resolution",
        action="store_true",
        help="Skip LLM entity resolution (faster, no Claude API cost)",
    )
    args = parser.parse_args()
    main(entity_resolution=not args.no_entity_resolution)
