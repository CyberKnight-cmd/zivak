"""
exclusively-neo4j/neo4j_queries.py
────────────────────────────────────────────────────────────────────────────────
Runtime query functions for the Zivak Bayesian diagnostic engine.
Part of the exclusively-neo4j pipeline — queries only Disease and Symptom nodes.
Test nodes (Phase 4b) are not loaded here; get_available_tests returns [].
"""

from neo4j_client import run_query


def get_initial_differential(hp_ids: list[str], limit: int = 20) -> list[dict]:
    if not hp_ids:
        return []
    try:
        records = run_query(
            """
            MATCH (d:Disease)-[r:PRESENTS_WITH]->(s:Symptom)
            WHERE s.id IN $hp_ids
            RETURN d.id AS disease_id, d.name AS name,
                   d.prevalence AS prevalence,
                   SUM(r.sensitivity) AS score,
                   COLLECT(s.id) AS matched_symptoms
            ORDER BY score DESC
            LIMIT $limit
            """,
            {"hp_ids": hp_ids, "limit": limit},
        )
        if not records:
            return []

        total = sum(r["score"] for r in records)
        if total == 0:
            return []

        results = [
            {
                "disease_id": r["disease_id"],
                "name": r["name"],
                "probability": r["score"] / total,
                "matched_symptoms": list(r["matched_symptoms"]),
            }
            for r in records
        ]

        # Rare disease floor: prevalence < 0.001 not already present
        present_ids = {d["disease_id"] for d in results}
        rare = run_query(
            """
            MATCH (d:Disease)
            WHERE d.prevalence < 0.001 AND NOT d.id IN $present_ids
            RETURN d.id AS disease_id, d.name AS name, d.prevalence AS prevalence
            LIMIT 50
            """,
            {"present_ids": list(present_ids)},
        )
        for r in rare:
            results.append(
                {
                    "disease_id": r["disease_id"],
                    "name": r["name"],
                    "probability": 0.02,
                    "matched_symptoms": [],
                }
            )

        # Renormalise
        total2 = sum(d["probability"] for d in results)
        for d in results:
            d["probability"] = d["probability"] / total2

        results.sort(key=lambda x: x["probability"], reverse=True)
        return results
    except Exception:
        return []


def get_lr_for_symptom(hp_id: str, disease_ids: list[str]) -> dict[str, dict]:
    output = {did: {"lr_positive": 1.0, "lr_negative": 1.0} for did in disease_ids}
    if not disease_ids:
        return output
    try:
        pos_rows = run_query(
            """
            MATCH (s:Symptom {id: $hp_id})-[r:RULES_IN]->(d:Disease)
            WHERE d.id IN $disease_ids
            RETURN d.id AS disease_id, r.likelihood_ratio AS lr
            """,
            {"hp_id": hp_id, "disease_ids": disease_ids},
        )
        for row in pos_rows:
            if row["disease_id"] in output:
                output[row["disease_id"]]["lr_positive"] = row["lr"]

        neg_rows = run_query(
            """
            MATCH (s:Symptom {id: $hp_id})-[r:RULES_OUT]->(d:Disease)
            WHERE d.id IN $disease_ids
            RETURN d.id AS disease_id, r.likelihood_ratio AS lr
            """,
            {"hp_id": hp_id, "disease_ids": disease_ids},
        )
        for row in neg_rows:
            if row["disease_id"] in output:
                output[row["disease_id"]]["lr_negative"] = row["lr"]
    except Exception:
        pass
    return output


def get_available_tests(disease_ids: list[str]) -> list[dict]:
    # Test nodes not loaded until Phase 4b — return empty list without raising
    return []


def get_disease_subgraph(disease_id: str) -> dict:
    try:
        nodes = run_query(
            "MATCH (d:Disease {id: $id}) RETURN d",
            {"id": disease_id},
        )
        if not nodes:
            return {}

        d = dict(nodes[0]["d"])
        symptoms = run_query(
            """
            MATCH (d:Disease {id: $id})-[r:PRESENTS_WITH]->(s:Symptom)
            RETURN s.id AS hp_id, s.name AS name, r.sensitivity AS sensitivity
            """,
            {"id": disease_id},
        )
        return {
            "disease_id": d.get("id", disease_id),
            "name": d.get("name", ""),
            "icd_codes": d.get("icd", []),
            "symptoms": [
                {
                    "hp_id": s["hp_id"],
                    "name": s["name"],
                    "sensitivity": s["sensitivity"],
                }
                for s in symptoms
            ],
            "tests": [],
        }
    except Exception:
        return {}


def verify_graph_counts() -> dict:
    try:
        counts = {}
        for label, key in [
            ("Disease", "diseases"),
            ("Symptom", "symptoms"),
        ]:
            r = run_query(f"MATCH (n:{label}) RETURN count(n) AS c")
            counts[key] = r[0]["c"] if r else 0

        for rel, key in [
            ("PRESENTS_WITH", "presents_with"),
            ("RULES_IN", "rules_in"),
            ("RULES_OUT", "rules_out"),
        ]:
            r = run_query(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c")
            counts[key] = r[0]["c"] if r else 0

        return counts
    except Exception:
        return {
            "diseases": 0,
            "symptoms": 0,
            "presents_with": 0,
            "rules_in": 0,
            "rules_out": 0,
        }
