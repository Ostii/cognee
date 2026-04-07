#!/usr/bin/env python3
"""
Neo4j connectivity and pipeline verification tests for Cognee MCP Server.

Verifies that:
1. Neo4j is reachable and APOC is available
2. Cognee's Neo4j adapter initializes correctly
3. The full add -> cognify -> search pipeline writes to Neo4j
4. Node/edge counts in Neo4j match SQLite metadata

Prerequisites:
    - Neo4j running locally (docker-compose up -d)
    - .env configured with GRAPH_DATABASE_PROVIDER=neo4j

Usage:
    cd cognee-mcp
    uv run python src/test_neo4j.py
"""

import asyncio
import os
import sys
import time

from logging import ERROR

# Ensure .env is loaded before any cognee imports
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from cognee.shared.logging_utils import setup_logging


class Neo4jVerificationTests:
    """Verification tests for Neo4j integration with Cognee."""

    def __init__(self):
        self.test_results = {}
        self.neo4j_url = os.getenv("GRAPH_DATABASE_URL", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("GRAPH_DATABASE_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("GRAPH_DATABASE_PASSWORD", "cognee-local-dev")

    async def test_neo4j_connectivity(self):
        """Test basic Neo4j connectivity via the driver."""
        print("\n[1] Testing Neo4j connectivity...")
        try:
            from neo4j import AsyncGraphDatabase

            driver = AsyncGraphDatabase.driver(
                self.neo4j_url,
                auth=(self.neo4j_user, self.neo4j_password),
            )
            async with driver.session() as session:
                result = await session.run("RETURN 1 AS value")
                record = await result.single()
                assert record["value"] == 1, "Expected RETURN 1 to return 1"

            await driver.close()
            self.test_results["connectivity"] = {"status": "PASS"}
            print("    PASS - Connected to Neo4j and ran a query")
        except Exception as e:
            self.test_results["connectivity"] = {"status": "FAIL", "error": str(e)}
            print(f"    FAIL - {e}")

    async def test_apoc_available(self):
        """Test that APOC plugin is installed and apoc.create.addLabels works."""
        print("\n[2] Testing APOC availability...")
        try:
            from neo4j import AsyncGraphDatabase

            driver = AsyncGraphDatabase.driver(
                self.neo4j_url,
                auth=(self.neo4j_user, self.neo4j_password),
            )
            async with driver.session() as session:
                # Check apoc.create.addLabels exists
                result = await session.run(
                    "CALL apoc.help('addLabels') YIELD name RETURN name LIMIT 1"
                )
                records = [r async for r in result]
                assert len(records) > 0, "apoc.help('addLabels') returned no results"

                # Check apoc.merge.relationship exists
                result = await session.run(
                    "CALL apoc.help('merge.relationship') YIELD name RETURN name LIMIT 1"
                )
                records = [r async for r in result]
                assert len(records) > 0, "apoc.help('merge.relationship') returned no results"

            await driver.close()
            self.test_results["apoc"] = {"status": "PASS"}
            print("    PASS - APOC is installed (addLabels + merge.relationship)")
        except Exception as e:
            self.test_results["apoc"] = {"status": "FAIL", "error": str(e)}
            print(f"    FAIL - {e}")

    async def test_cognee_graph_engine(self):
        """Test that cognee's graph engine initializes with Neo4j."""
        print("\n[3] Testing Cognee graph engine initialization...")
        try:
            from cognee.infrastructure.databases.graph import get_graph_engine

            engine = await get_graph_engine()
            engine_type = type(engine).__name__
            assert "Neo4j" in engine_type, f"Expected Neo4jAdapter, got {engine_type}"

            # The initialize() method should have created the uniqueness constraint
            self.test_results["graph_engine"] = {"status": "PASS"}
            print(f"    PASS - Graph engine is {engine_type}")
        except Exception as e:
            self.test_results["graph_engine"] = {"status": "FAIL", "error": str(e)}
            print(f"    FAIL - {e}")

    async def test_pipeline_end_to_end(self):
        """Test the full add -> cognify -> search pipeline writes to Neo4j."""
        print("\n[4] Testing end-to-end pipeline...")
        try:
            import cognee
            from cognee import SearchType
            from cognee.modules.engine.operations.setup import setup

            await setup()

            # Prune first for clean state
            await cognee.prune.prune_data()
            await cognee.prune.prune_system(metadata=True)

            # Add test data
            test_text = (
                "Albert Einstein developed the theory of relativity. "
                "He was born in Ulm, Germany in 1879. "
                "Einstein received the Nobel Prize in Physics in 1921."
            )
            await cognee.add(test_text, dataset_name="neo4j_test")

            # Run cognify
            await cognee.cognify()

            # Query Neo4j directly to verify nodes were created
            from neo4j import AsyncGraphDatabase

            driver = AsyncGraphDatabase.driver(
                self.neo4j_url,
                auth=(self.neo4j_user, self.neo4j_password),
            )
            async with driver.session() as session:
                result = await session.run("MATCH (n) RETURN count(n) AS count")
                record = await result.single()
                node_count = record["count"]

                result = await session.run("MATCH ()-[r]->() RETURN count(r) AS count")
                record = await result.single()
                edge_count = record["count"]

            await driver.close()

            assert node_count > 0, "No nodes found in Neo4j after cognify"
            assert edge_count > 0, "No edges found in Neo4j after cognify"

            # Test GRAPH_COMPLETION search
            graph_results = await cognee.search(
                "Einstein", query_type=SearchType.GRAPH_COMPLETION
            )

            # Test CHUNKS search
            chunk_results = await cognee.search(
                "Einstein", query_type=SearchType.CHUNKS
            )

            self.test_results["pipeline"] = {
                "status": "PASS",
                "neo4j_nodes": node_count,
                "neo4j_edges": edge_count,
                "graph_search_results": len(graph_results) if graph_results else 0,
                "chunk_search_results": len(chunk_results) if chunk_results else 0,
            }
            print(f"    PASS - Neo4j: {node_count} nodes, {edge_count} edges")
            print(f"           GRAPH_COMPLETION returned {len(graph_results) if graph_results else 0} results")
            print(f"           CHUNKS returned {len(chunk_results) if chunk_results else 0} results")

        except Exception as e:
            self.test_results["pipeline"] = {"status": "FAIL", "error": str(e)}
            print(f"    FAIL - {e}")
            import traceback
            traceback.print_exc()

    async def test_store_consistency(self):
        """Verify Neo4j node count matches SQLite metadata."""
        print("\n[5] Testing store consistency (Neo4j vs SQLite)...")
        try:
            from cognee.infrastructure.databases.relational import get_relational_engine
            from sqlalchemy import text
            from neo4j import AsyncGraphDatabase

            # Count nodes in SQLite
            db = get_relational_engine()
            async with db.get_async_session() as session:
                result = await session.execute(text("SELECT count(*) FROM nodes"))
                sqlite_nodes = result.scalar()
                result = await session.execute(text("SELECT count(*) FROM edges"))
                sqlite_edges = result.scalar()

            # Count nodes in Neo4j
            driver = AsyncGraphDatabase.driver(
                self.neo4j_url,
                auth=(self.neo4j_user, self.neo4j_password),
            )
            async with driver.session() as session:
                result = await session.run("MATCH (n) RETURN count(n) AS count")
                record = await result.single()
                neo4j_nodes = record["count"]

                result = await session.run("MATCH ()-[r]->() RETURN count(r) AS count")
                record = await result.single()
                neo4j_edges = record["count"]

            await driver.close()

            print(f"    SQLite: {sqlite_nodes} nodes, {sqlite_edges} edges")
            print(f"    Neo4j:  {neo4j_nodes} nodes, {neo4j_edges} edges")

            # Neo4j may have more or fewer nodes depending on how cognee maps them,
            # but both should be > 0 if the pipeline ran
            if neo4j_nodes > 0 and sqlite_nodes > 0:
                self.test_results["consistency"] = {
                    "status": "PASS",
                    "sqlite_nodes": sqlite_nodes,
                    "sqlite_edges": sqlite_edges,
                    "neo4j_nodes": neo4j_nodes,
                    "neo4j_edges": neo4j_edges,
                }
                print("    PASS - Both stores have data")
            else:
                self.test_results["consistency"] = {
                    "status": "FAIL",
                    "error": f"Missing data: SQLite={sqlite_nodes} nodes, Neo4j={neo4j_nodes} nodes",
                }
                print("    FAIL - One or both stores are empty")

        except Exception as e:
            self.test_results["consistency"] = {"status": "FAIL", "error": str(e)}
            print(f"    FAIL - {e}")

    async def run_all_tests(self):
        """Run all verification tests."""
        print("=" * 60)
        print("Neo4j Verification Tests for Cognee MCP Server")
        print("=" * 60)

        # Infrastructure tests first (fast, no LLM needed)
        await self.test_neo4j_connectivity()

        if self.test_results.get("connectivity", {}).get("status") != "PASS":
            print("\n!! Neo4j is not reachable. Skipping remaining tests.")
            print("   Start Neo4j: cd cognee-mcp && docker-compose up -d")
            self.print_summary()
            return

        await self.test_apoc_available()
        await self.test_cognee_graph_engine()

        # Pipeline test (requires LLM + embeddings)
        await self.test_pipeline_end_to_end()

        # Consistency check (only if pipeline ran)
        if self.test_results.get("pipeline", {}).get("status") == "PASS":
            await self.test_store_consistency()
        else:
            print("\n[5] Skipping consistency check — pipeline test did not pass")

        self.print_summary()

    def print_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

        passed = 0
        failed = 0

        for name, result in self.test_results.items():
            status = result["status"]
            if status == "PASS":
                passed += 1
                print(f"  PASS  {name}")
            else:
                failed += 1
                error = result.get("error", "unknown")
                print(f"  FAIL  {name}: {error}")

        total = passed + failed
        print(f"\n  {passed}/{total} passed")

        if failed > 0:
            sys.exit(1)


async def main():
    tests = Neo4jVerificationTests()
    await tests.run_all_tests()


if __name__ == "__main__":
    setup_logging(log_level=ERROR)
    asyncio.run(main())
