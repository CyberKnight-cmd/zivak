"""
knowledge/qdrant_client.py
──────────────────────────────────────────────────────────────────────────────
In-process semantic symptom search using sentence-transformers.
No external vector DB required.

Corpus loading (priority order):
  1. Neo4j  — queries all Symptom nodes at startup
  2. data/processed/symptom_index.json  — fallback if Neo4j is unavailable

Multi-symptom support: compound complaints like "chest pain and breathlessness"
are split into individual terms, searched separately, and deduplicated by HP ID.
Returns all matched HP IDs in symptom_ids so seed_node can union multiple
symptom clusters when building the initial differential.

search_symptom() always returns:
  {
    "clinical_term": str,       best-matching HPO name
    "symptom_id":    str,       primary HP ID  (highest-score match)
    "symptom_ids":   list[str], all matched HP IDs (1+ for compound inputs)
    "score":         float,     cosine similarity of best match [0, 1]
  }
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from knowledge.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

_MODEL_NAME           = "all-MiniLM-L6-v2"
_SIMILARITY_THRESHOLD = 0.30
_MAX_SYMPTOMS         = 5       # cap on distinct HP IDs returned per query
_MIN_TERM_LEN         = 4       # ignore split fragments shorter than this

_SYMPTOM_INDEX_JSON = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "symptom_index.json"
)

# ── Multi-symptom splitting ───────────────────────────────────────────────────
# Strips clinician/patient framing, then splits compound complaints.

_CLINICIAN_PATS = [
    r"^my patient (?:has|presents?\s+with|complains?\s+of|reports?|describes?|is\s+(?:experiencing|having|suffering\s+from))\s*",
    r"^(?:the\s+)?patient (?:has|presents?\s+with|complains?\s+of|reports?|describes?)\s*",
    r"^(?:he|she|they) (?:has|have|is|are|complains?\s+of|reports?|describes?|presents?\s+with)\s*",
    r"^presenting with\s*",
    r"^complaining of\s*",
    r"^(?:c/o|c\.o\.)\s*",
]
_PATIENT_PATS = [
    r"^i (?:have|am\s+having|am\s+experiencing|feel|keep|can't|cannot|suffer\s+from|notice|get)\s*",
    r"^i've\s+(?:been\s+)?(?:having|experiencing|feeling|getting|noticed)\s*",
    r"^i\s+(?:am|am\s+currently)\s*",
    r"^(?:been|feeling)\s*",
    r"^(?:my|there's)\s*(?:a\s+)?",
]
_SPECIAL = re.compile(r"[^\w\s,\.\/\-]")
_SPACES  = re.compile(r"\s{2,}")
# Preserves "FEV1/FVC" but splits "breathlessness / chest pain"
_SPLITTER = re.compile(
    r"\s*(?:,|;|\band\b|\bwith\b|\bplus\b|\balong\s+with\b|\bas\s+well\s+as\b|\balso\b|(?<!\w)\/(?!\w))\s*",
    flags=re.IGNORECASE,
)


def _normalise(text: str) -> str:
    text = text.strip()
    for pat in _CLINICIAN_PATS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    for pat in _PATIENT_PATS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return _SPACES.sub(" ", _SPECIAL.sub(" ", text)).strip().lower()


def _split(text: str) -> list[str]:
    parts = _SPLITTER.split(text)
    terms = [p.strip() for p in parts if len(p.strip()) >= _MIN_TERM_LEN]
    return terms if terms else [text.strip()]


# ── Client ────────────────────────────────────────────────────────────────────

class QdrantClient:
    """
    In-process semantic search over HPO symptom terms.

    The name "QdrantClient" is kept for interface compatibility — no Qdrant
    server is used. Embeddings are computed locally with all-MiniLM-L6-v2
    and cosine similarity is computed in NumPy.

    Class-level model and embeddings are shared across all instances so the
    ~80 MB model is loaded only once per process.
    """

    _model:      SentenceTransformer | None = None
    _embeddings: np.ndarray | None          = None
    _corpus:     list[tuple[str, str, str]] = []  # (text, clinical_term, hp_id)

    def __init__(self, neo4j_client: Neo4jClient | None = None):
        self._neo4j = neo4j_client
        if not QdrantClient._corpus:
            self._load_corpus()

    # ── Corpus loading ────────────────────────────────────────────────────────

    def _load_corpus(self) -> None:
        if self._neo4j is not None:
            try:
                self._load_from_neo4j()
                if QdrantClient._corpus:
                    return
            except Exception as exc:
                logger.warning("QdrantClient: Neo4j unavailable (%s) — trying JSON fallback", exc)
        self._load_from_json()

    def _load_from_neo4j(self) -> None:
        logger.info("QdrantClient: loading symptom corpus from Neo4j …")
        with self._neo4j._driver.session() as session:
            rows = session.run(
                "MATCH (s:Symptom) WHERE s.category = 'symptom' "
                "RETURN s.id AS id, s.name AS name, s.synonyms AS synonyms"
            ).data()
        self._build(rows)
        logger.info("QdrantClient: corpus ready — %d entries from Neo4j", len(QdrantClient._corpus))

    def _load_from_json(self) -> None:
        if not _SYMPTOM_INDEX_JSON.exists():
            logger.error("QdrantClient: symptom_index.json not found at %s", _SYMPTOM_INDEX_JSON)
            return
        logger.info("QdrantClient: loading symptom corpus from %s …", _SYMPTOM_INDEX_JSON)
        index = json.loads(_SYMPTOM_INDEX_JSON.read_text(encoding="utf-8"))
        rows  = [
            {"id": hp_id, "name": info.get("name", ""), "synonyms": info.get("synonyms", [])}
            for hp_id, info in index.items()
            if isinstance(info, dict)
        ]
        self._build(rows)
        logger.info("QdrantClient: corpus ready — %d entries from JSON", len(QdrantClient._corpus))

    def _build(self, rows: list[dict]) -> None:
        corpus: list[tuple[str, str, str]] = []
        for row in rows:
            hp_id = row.get("id", "")
            name  = row.get("name") or ""
            syns  = row.get("synonyms") or []
            if not hp_id or not name:
                continue
            corpus.append((name, name, hp_id))
            for syn in syns:
                if syn and syn != name:
                    corpus.append((syn, name, hp_id))
        if not corpus:
            return
        if QdrantClient._model is None:
            logger.info("QdrantClient: loading sentence-transformer '%s' …", _MODEL_NAME)
            QdrantClient._model = SentenceTransformer(_MODEL_NAME)
        QdrantClient._embeddings = QdrantClient._model.encode(
            [e[0] for e in corpus],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=256,
        )
        QdrantClient._corpus = corpus

    # ── Public interface ──────────────────────────────────────────────────────

    def search_symptom(self, user_text: str) -> dict:
        """
        Map a free-text patient complaint to one or more HPO symptom terms.

        Compound inputs ("chest pain and breathlessness") are split and searched
        separately. All matches above the similarity threshold are returned in
        symptom_ids so the orchestrator can union multiple symptom clusters.
        """
        fallback = {
            "clinical_term": "Unknown symptom",
            "symptom_id":    "HP:0000001",
            "symptom_ids":   ["HP:0000001"],
            "score":         0.0,
        }

        if not QdrantClient._corpus or QdrantClient._embeddings is None:
            logger.warning("QdrantClient: corpus is empty — cannot search")
            return fallback

        clean = _normalise(user_text)
        if not clean:
            return fallback
        terms = _split(clean)

        best_by_hp: dict[str, dict] = {}
        for term in terms:
            q      = QdrantClient._model.encode([term], normalize_embeddings=True)[0]
            scores = QdrantClient._embeddings @ q
            idx    = int(np.argmax(scores))
            score  = float(scores[idx])
            if score < _SIMILARITY_THRESHOLD:
                continue
            _text, clinical_term, hp_id = QdrantClient._corpus[idx]
            if hp_id not in best_by_hp or score > best_by_hp[hp_id]["score"]:
                best_by_hp[hp_id] = {
                    "clinical_term": clinical_term,
                    "hp_id":         hp_id,
                    "score":         score,
                }

        if not best_by_hp:
            return {**fallback, "score": float(
                (QdrantClient._embeddings @ QdrantClient._model.encode(
                    [clean], normalize_embeddings=True)[0]).max()
            )}

        ranked = sorted(best_by_hp.values(), key=lambda x: x["score"], reverse=True)
        ranked = ranked[:_MAX_SYMPTOMS]
        best   = ranked[0]

        logger.info(
            "QdrantClient: '%s' → %d HP ID(s) %s (top %.3f)",
            user_text[:60], len(ranked),
            [r["hp_id"] for r in ranked], best["score"],
        )
        return {
            "clinical_term": best["clinical_term"],
            "symptom_id":    best["hp_id"],
            "symptom_ids":   [r["hp_id"] for r in ranked],
            "score":         best["score"],
        }
