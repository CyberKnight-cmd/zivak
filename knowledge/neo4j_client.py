"""
knowledge/neo4j_client.py
─────────────────────────────────────────────────────────────────────────────
Real Neo4j client for the ZIVAK diagnostic engine.

Implements the same three-method interface as MockNeo4jClient so the
orchestrator can swap clients without any other code changes.

Reads credentials from .env:  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

Graph schema (loaded by neo4j-setup/4_load_neo4j.py):
  (Disease {id, name, icd, omim_id, orpha_id, prevalence})
  (Symptom {id, name, synonyms, category})
  (Disease)-[:PRESENTS_WITH {sensitivity}]->(Symptom)
  (Symptom)-[:RULES_IN      {likelihood_ratio}]->(Disease)   LR+
  (Symptom)-[:RULES_OUT     {likelihood_ratio}]->(Disease)   LR-

HPO symptom terms are the diagnostic questions.
  question["test_id"] = "HP:0002875"  (not "test_fev1")
"""

import os
from typing import Union

import neo4j
from dotenv import load_dotenv

load_dotenv()

_RARE_DISEASE_PREVALENCE_THRESHOLD = 0.001
_RARE_DISEASE_FLOOR_PROBABILITY    = 0.02
_SENSITIVITY_THRESHOLD             = 0.10   # ignore Se < 10% co-occurrences
_RARE_DISEASE_LIMIT                = 50


class Neo4jClient:
    """
    Production Neo4j client.  Drop-in replacement for MockNeo4jClient.

    All queries use named parameters to prevent injection and enable
    Neo4j query-plan caching.
    """

    def __init__(self):
        uri      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user     = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        if not password:
            raise ValueError("NEO4J_PASSWORD is required — set it in .env")
        self._driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))

    # ------------------------------------------------------------------ #
    #  Public interface (matches MockNeo4jClient exactly)                  #
    # ------------------------------------------------------------------ #

    def get_initial_differential(
        self,
        symptom_id_or_ids: Union[str, list],
        limit: int = 20,
    ) -> list[dict]:
        """
        Seed the Bayesian differential from one or more presenting symptom HP IDs.

        Returns [{name, disease_id, specificity, prevalence}] sorted by specificity desc.

        DifferentialEngine.initialize() computes raw_score = specificity * prevalence.
        Since all disease prevalences are 0.01 (uniform in the pipeline), ranking is
        purely by specificity (= sum of PRESENTS_WITH sensitivities across matched symptoms).
        The extra disease_id key flows through state["differential"] for downstream
        Neo4j lookups — DifferentialEngine does not strip unknown keys.

        Also appends rare diseases (prevalence < 0.001) at the 2% probability floor
        so they are never entirely excluded before their specific symptoms are asked.
        """
        hp_ids = [symptom_id_or_ids] if isinstance(symptom_id_or_ids, str) else list(symptom_id_or_ids)
        if not hp_ids:
            return []

        with self._driver.session() as session:
            rows = session.run(
                """
                MATCH (d:Disease)-[r:PRESENTS_WITH]->(s:Symptom)
                WHERE s.id IN $hp_ids
                RETURN d.id          AS disease_id,
                       d.name        AS name,
                       d.prevalence  AS prevalence,
                       SUM(r.sensitivity) AS score
                ORDER BY score DESC
                LIMIT $limit
                """,
                hp_ids=hp_ids,
                limit=limit,
            ).data()

        results = [
            {
                "name":        r["name"],
                "disease_id":  r["disease_id"],
                "specificity": float(r["score"]),
                "prevalence":  float(r["prevalence"] or 0.01),
            }
            for r in rows
        ] if rows else []

        # Rare disease floor — always runs even when rows is empty so a
        # symptom that has no direct disease links still seeds the differential.
        present_ids = {d["disease_id"] for d in results}

        with self._driver.session() as session:
            rare_rows = session.run(
                """
                MATCH (d:Disease)
                WHERE d.prevalence < $threshold AND NOT d.id IN $present_ids
                RETURN d.id AS disease_id, d.name AS name, d.prevalence AS prevalence
                LIMIT $limit
                """,
                threshold=_RARE_DISEASE_PREVALENCE_THRESHOLD,
                present_ids=list(present_ids),
                limit=_RARE_DISEASE_LIMIT,
            ).data()

        for r in rare_rows:
            results.append({
                "name":        r["name"],
                "disease_id":  r["disease_id"],
                # Floor probability pre-set: specificity=floor, prevalence=1.0
                # so raw_score = floor * 1.0 = floor, which DifferentialEngine
                # will normalize back toward the 2% floor after re-normalization.
                "specificity": _RARE_DISEASE_FLOOR_PROBABILITY,
                "prevalence":  1.0,
            })

        return results

    def get_available_tests(self, disease_ids: list[str]) -> list[dict]:
        """
        Return HP symptom terms connected to the given diseases that are
        suitable as diagnostic questions.

        Filters applied in Cypher:
          category IN ['symptom', 'lab_finding']  — skip structural 'sign' terms
          sensitivity >= 0.10                     — skip rare co-occurrences

        Returns [{id, name, diseases}] where id is the HP ID that will become
        question["test_id"] when selected by QuestionSelectorAgent.

        disease_ids are DOID strings from state["differential"][i]["disease_id"].
        """
        if not disease_ids:
            return []

        with self._driver.session() as session:
            rows = session.run(
                """
                MATCH (d:Disease)-[r:PRESENTS_WITH]->(s:Symptom)
                WHERE d.id IN $disease_ids
                  AND s.category IN ['symptom', 'lab_finding']
                  AND r.sensitivity >= $threshold
                RETURN DISTINCT s.id   AS id,
                                s.name AS name,
                                collect(DISTINCT d.id) AS diseases
                ORDER BY s.id
                """,
                disease_ids=disease_ids,
                threshold=_SENSITIVITY_THRESHOLD,
            ).data()

        return [
            {"id": r["id"], "name": r["name"], "diseases": list(r["diseases"])}
            for r in rows
        ]

    def get_test_edges(self, hp_id: str) -> list[dict]:
        """
        Return LR edges for an HP symptom term.

        Returns [{disease, relationship, lr}] — exactly the format MockNeo4jClient
        returns, so EvidenceEvaluatorAgent._build_prompt() needs no changes.

          disease:      disease name string (EvidenceEvaluator matches by name)
          relationship: "RULES_IN"  (LR+) or "RULES_OUT" (LR-)
          lr:           likelihood_ratio float

        Two separate queries because UNION in Cypher requires identical column types
        and it is cleaner to combine in Python.
        """
        if not hp_id:
            return []

        with self._driver.session() as session:
            in_rows = session.run(
                """
                MATCH (s:Symptom {id: $hp_id})-[r:RULES_IN]->(d:Disease)
                RETURN d.name AS disease, r.likelihood_ratio AS lr
                """,
                hp_id=hp_id,
            ).data()

            out_rows = session.run(
                """
                MATCH (s:Symptom {id: $hp_id})-[r:RULES_OUT]->(d:Disease)
                RETURN d.name AS disease, r.likelihood_ratio AS lr
                """,
                hp_id=hp_id,
            ).data()

        return [
            {"disease": r["disease"], "relationship": "RULES_IN",  "lr": float(r["lr"])}
            for r in in_rows
        ] + [
            {"disease": r["disease"], "relationship": "RULES_OUT", "lr": float(r["lr"])}
            for r in out_rows
        ]

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def verify_connection(self) -> bool:
        try:
            with self._driver.session() as session:
                result = session.run("RETURN 1 AS x").single()
                return result["x"] == 1
        except Exception:
            return False

    def close(self) -> None:
        self._driver.close()
