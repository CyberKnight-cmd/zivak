"""
knowledge/qdrant_client.py
Semantic symptom search backed by a real Qdrant vector database.

Model: NeuML/pubmedbert-base-embeddings (768d, PubMed fine-tuned)
Collection: "symptoms"  (populated by scripts/5_populate_qdrant.py)

search() return contract:
  Match (score >= threshold):
    {"clinical_term": "Exertional dyspnea", "symptom_id": "HP:0002875",
     "symptom_ids": ["HP:0002875", ...], "score": 0.87, "matched": True}
  No match (all scores below threshold or empty):
    {"clinical_term": None, "symptom_id": None,
     "symptom_ids": [], "score": 0.0, "matched": False}
"""

from __future__ import annotations

import os
import re
import threading

MATCH_THRESHOLD = float(os.getenv("QDRANT_MATCH_THRESHOLD", "0.60"))

FRAMING_PATTERNS = [
    r"^my patient (has|is)\s+",
    r"^patient (has|presents with|is complaining of|c/o)\s+",
    r"^the patient has\s+",
    r"^i('ve| have) (been |had |been having |been experiencing )?",
    r"^i am (having|experiencing)\s+",
    r"^i('m| am) feeling\s+",
    r"^(presenting|presents) with\s+",
    r"^complaining of\s+",
    r"^c/o\s+",
    r"^(suffering from|diagnosed with|history of)\s+",
    r"^i am (a )?(\d+[\-\s]year[\-\s]old)\s+(and\s+)?",
]

STOP_WORDS = {
    "the", "a", "an", "it", "this", "that", "also", "too", "as well",
    "i", "me", "my",
}

SPLIT_RE = re.compile(
    r',|;|\band\b|\bwith\b|\bplus\b|\balso\b|\bas well as\b'
    r'|\balong with\b|\bin addition to\b',
    re.IGNORECASE,
)
SENT_RE = re.compile(r'(?<=[.!?])\s+')


class QdrantClient:
    def __init__(self):
        self._qdrant    = None
        self._model     = None
        self._lock      = threading.Lock()
        self._url       = os.getenv("QDRANT_URL",            "http://localhost:6333")
        self._coll      = os.getenv("QDRANT_COLLECTION",     "symptoms")
        self._apikey    = os.getenv("QDRANT_API_KEY",        None) or None
        self._threshold = MATCH_THRESHOLD

    def _get_qdrant(self):
        if self._qdrant is None:
            with self._lock:
                if self._qdrant is None:
                    from qdrant_client import QdrantClient as _QC
                    self._qdrant = _QC(url=self._url, api_key=self._apikey)
        return self._qdrant

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self._model = SentenceTransformer(
                        os.getenv("EMBEDDING_MODEL", "NeuML/pubmedbert-base-embeddings"),
                        device=device,
                    )
        return self._model

    def _strip_framing(self, sentence: str) -> str:
        s = sentence.strip()
        for pat in FRAMING_PATTERNS:
            s = re.sub(pat, "", s, flags=re.IGNORECASE).strip()
        return s

    def _preprocess(self, query: str) -> list[str]:
        sentences = SENT_RE.split(query.strip())
        terms, seen = [], set()
        for sent in sentences:
            sent = self._strip_framing(sent)
            if not sent:
                continue
            for part in SPLIT_RE.split(sent):
                t = part.strip().strip(",.;")
                if len(t) < 2:
                    continue
                if re.fullmatch(r'[\d\s]+', t):
                    continue
                if t.lower() in STOP_WORDS:
                    continue
                key = t.lower()
                if key not in seen:
                    seen.add(key)
                    terms.append(t)
        return terms if terms else [query.strip()]

    def _embed(self, texts: list[str]) -> list:
        return self._get_model().encode(
            texts,
            batch_size=int(os.getenv("EMBED_BATCH_SIZE", "32")),
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def _search_one(self, vector, limit: int = 5) -> list[dict]:
        response = self._get_qdrant().query_points(
            collection_name=self._coll,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return [
            {
                "hp_id":    h.payload["hp_id"],
                "name":     h.payload["name"],
                "score":    h.score,
                "category": h.payload.get("category", "symptom"),
            }
            for h in response.points
        ]

    def _no_match(self) -> dict:
        return {
            "clinical_term": None,
            "symptom_id":    None,
            "symptom_ids":   [],
            "score":         0.0,
            "matched":       False,
        }

    def search(self, query: str) -> dict:
        terms   = self._preprocess(query)
        vectors = self._embed(terms)

        best: dict[str, dict] = {}
        for vec in vectors:
            for hit in self._search_one(vec, limit=5):
                hp = hit["hp_id"]
                if hp not in best or hit["score"] > best[hp]["score"]:
                    best[hp] = hit

        if not best:
            return self._no_match()

        confident = {hp: h for hp, h in best.items() if h["score"] >= self._threshold}
        if not confident:
            return self._no_match()

        ranked = sorted(confident.values(), key=lambda h: h["score"], reverse=True)
        top    = ranked[0]
        return {
            "clinical_term": top["name"],
            "symptom_id":    top["hp_id"],
            "symptom_ids":   [h["hp_id"] for h in ranked],
            "score":         top["score"],
            "matched":       True,
        }
