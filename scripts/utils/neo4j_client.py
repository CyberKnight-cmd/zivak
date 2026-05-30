"""
exclusively-neo4j/neo4j_client.py
────────────────────────────────────────────────────────────────────────────────
Self-contained Neo4j connection and query runner for the exclusively-neo4j
pipeline. Reads credentials from .env via python-dotenv.
"""

import os
from dotenv import load_dotenv
import neo4j

load_dotenv()

_driver: neo4j.Driver | None = None


def get_driver() -> neo4j.Driver:
    global _driver
    if _driver is None:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        if not password:
            raise ValueError("NEO4J_PASSWORD is required but not set in environment")
        # neo4j+s:// verifies the server certificate which fails on Windows when
        # the Aura CA chain isn't in the system store. Switching to neo4j+ssc://
        # keeps the encrypted connection but skips cert verification.
        uri = uri.replace("neo4j+s://", "neo4j+ssc://").replace("bolt+s://", "bolt+ssc://")
        _driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
    return _driver


def run_query(query: str, params: dict = {}) -> list[dict]:
    try:
        driver = get_driver()
        with driver.session() as session:
            result = session.run(query, params)
            return [dict(record) for record in result]
    except Exception as e:
        raise RuntimeError(f"Neo4j read query failed: {e}") from e


def run_write_query(query: str, params: dict = {}) -> None:
    try:
        driver = get_driver()
        with driver.session() as session:
            session.execute_write(lambda tx: tx.run(query, params))
    except Exception as e:
        raise RuntimeError(f"Neo4j write query failed: {e}") from e


def run_write_batch(query: str, batch: list[dict]) -> None:
    try:
        driver = get_driver()
        with driver.session() as session:
            session.execute_write(lambda tx: tx.run(query, {"batch": batch}))
    except Exception as e:
        raise RuntimeError(f"Neo4j batch write failed: {e}") from e


def close_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def verify_connection() -> bool:
    try:
        driver = get_driver()
        with driver.session() as session:
            result = session.run("RETURN 1 AS x").single()
            return result["x"] == 1
    except Exception:
        return False
