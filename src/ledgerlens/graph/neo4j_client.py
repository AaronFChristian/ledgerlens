"""Neo4j client — connection management, schema setup, Cypher helpers."""

from __future__ import annotations

from neo4j import GraphDatabase
from loguru import logger

from ..config import settings

SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT supplier_id IF NOT EXISTS FOR (s:Supplier) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT invoice_id IF NOT EXISTS FOR (i:Invoice) REQUIRE i.id IS UNIQUE",
    "CREATE CONSTRAINT lineitem_id IF NOT EXISTS FOR (l:LineItem) REQUIRE l.id IS UNIQUE",
    "CREATE INDEX supplier_canonical IF NOT EXISTS FOR (s:Supplier) ON (s.canonical_name)",
    "CREATE INDEX invoice_number IF NOT EXISTS FOR (i:Invoice) ON (i.invoice_number)",
    "CREATE INDEX invoice_date IF NOT EXISTS FOR (i:Invoice) ON (i.date)",
    "CREATE INDEX invoice_total IF NOT EXISTS FOR (i:Invoice) ON (i.total_amount)",
]


class Neo4jClient:
    """Thread-safe Neo4j driver wrapper.

    Use as context manager:
        with Neo4jClient() as client:
            client.run("MATCH (n) RETURN count(n)")
    """

    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )

    def close(self) -> None:
        self.driver.close()

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def verify_connection(self) -> bool:
        """Return True if Neo4j is reachable."""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j connection verified")
            return True
        except Exception as exc:
            logger.error(f"Neo4j connection failed: {exc}")
            return False

    def setup_schema(self) -> None:
        """Create constraints and indexes (idempotent)."""
        with self.driver.session() as session:
            for stmt in SCHEMA_STATEMENTS:
                try:
                    session.run(stmt)
                except Exception as exc:
                    logger.debug(f"Schema (already exists): {exc}")
        logger.info("Neo4j schema ready")

    def run(self, cypher: str, **params) -> list[dict]:
        """Execute a read Cypher query, return list of record dicts."""
        with self.driver.session() as session:
            result = session.run(cypher, **params)
            return [dict(r) for r in result]

    def run_write(self, cypher: str, **params) -> None:
        """Execute a write Cypher query."""
        with self.driver.session() as session:
            session.run(cypher, **params)

    def get_stats(self) -> dict:
        """Return node and relationship counts."""
        stats: dict = {}
        for label in ("Supplier", "Invoice", "LineItem"):
            rows = self.run(f"MATCH (n:{label}) RETURN count(n) AS count")
            stats[label] = rows[0]["count"] if rows else 0
        rows = self.run("MATCH ()-[r]->() RETURN count(r) AS count")
        stats["relationships"] = rows[0]["count"] if rows else 0
        return stats

    def clear_all(self) -> None:
        """Delete all nodes and relationships. Use carefully."""
        self.run_write("MATCH (n) DETACH DELETE n")
        logger.warning("All Neo4j data deleted")
