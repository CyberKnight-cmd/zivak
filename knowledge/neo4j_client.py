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
        # neo4j+s:// verifies the server certificate which fails on Windows when
        # the Aura CA chain isn't in the system store. Switching to neo4j+ssc://
        # keeps the encrypted connection but skips cert verification.
        uri = uri.replace("neo4j+s://", "neo4j+ssc://").replace("bolt+s://", "bolt+ssc://")
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
                MATCH (d:Disease)-[r]-(s:Symptom)
                WHERE s.id IN $hp_ids
                RETURN d.id          AS disease_id,
                       d.name        AS name,
                       d.prevalence  AS prevalence,
                       SUM(COALESCE(r.sensitivity, 0.1)) AS score
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

        # Only append rare diseases that actually present with the given symptom(s).
        # A rare disease with no PRESENTS_WITH edge to the complaint is irrelevant
        # and dilutes the differential — exclude it.
        with self._driver.session() as session:
            rare_rows = session.run(
                """
                MATCH (d:Disease)-[r]-(s:Symptom)
                WHERE d.prevalence < $threshold
                  AND NOT d.id IN $present_ids
                  AND s.id IN $hp_ids
                RETURN DISTINCT d.id AS disease_id, d.name AS name, d.prevalence AS prevalence
                LIMIT $limit
                """,
                threshold=_RARE_DISEASE_PREVALENCE_THRESHOLD,
                present_ids=list(present_ids),
                hp_ids=hp_ids,
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

    def get_available_tests(self, disease_ids: list[str], limit: int = 40) -> list[dict]:
        """
        Return HP symptom terms connected to the given diseases that are
        suitable as diagnostic questions.

        Filters applied in Cypher:
          category IN ['symptom', 'lab_finding']  — skip structural 'sign' terms
          sensitivity >= 0.10                     — skip rare co-occurrences

        Ordered by discriminativeness: symptoms shared by the most candidate
        diseases come first (highest EIG potential), tie-broken by average
        sensitivity. The caller caps further via the EIG ranker.

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
                WITH s, collect(DISTINCT d.id) AS diseases, AVG(r.sensitivity) AS avg_se
                RETURN s.id   AS id,
                       s.name AS name,
                       diseases
                ORDER BY size(diseases) DESC, avg_se DESC
                LIMIT $limit
                """,
                disease_ids=disease_ids,
                threshold=_SENSITIVITY_THRESHOLD,
                limit=limit,
            ).data()

        return [
            {"id": r["id"], "name": r["name"], "diseases": list(r["diseases"])}
            for r in rows
        ]

    def get_test_edges(self, hp_id: str, disease_ids: list[str] | None = None) -> list[dict]:
        """
        Return LR edges for an HP symptom term.

        Returns [{disease, relationship, lr}] — exactly the format MockNeo4jClient
        returns, so EvidenceEvaluatorAgent._build_prompt() needs no changes.

          disease:      disease name string (EvidenceEvaluator matches by name)
          relationship: "RULES_IN"  (LR+) or "RULES_OUT" (LR-)
          lr:           likelihood_ratio float

        disease_ids: when supplied, only edges pointing to those diseases are
        returned, keeping the EvidenceEvaluator prompt scoped to the active
        differential. Pass None to get all edges (backward-compatible).

        Two separate queries because UNION in Cypher requires identical column types
        and it is cleaner to combine in Python.
        """
        if not hp_id:
            return []

        filter_clause = "AND d.id IN $disease_ids" if disease_ids else ""

        with self._driver.session() as session:
            in_rows = session.run(
                f"""
                MATCH (s:Symptom {{id: $hp_id}})-[r:RULES_IN]->(d:Disease)
                WHERE 1=1 {filter_clause}
                RETURN d.name AS disease, r.likelihood_ratio AS lr
                """,
                hp_id=hp_id,
                disease_ids=disease_ids or [],
            ).data()

            out_rows = session.run(
                f"""
                MATCH (s:Symptom {{id: $hp_id}})-[r:RULES_OUT]->(d:Disease)
                WHERE 1=1 {filter_clause}
                RETURN d.name AS disease, r.likelihood_ratio AS lr
                """,
                hp_id=hp_id,
                disease_ids=disease_ids or [],
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
