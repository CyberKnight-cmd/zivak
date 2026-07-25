"""
knowledge/qdrant_client.py
Semantic symptom search backed by a real Qdrant vector database.

Model: NeuML/pubmedbert-base-embeddings (768d, PubMed fine-tuned)
Collection: "symptoms"  (populated by scripts/5_populate_qdrant.py)
"""

from __future__ import annotations

import os
import threading
from typing import List, Union
from dotenv import load_dotenv

load_dotenv()

MATCH_THRESHOLD = float(os.getenv("QDRANT_MATCH_THRESHOLD", "0.60"))
MARGIN_THRESHOLD = float(os.getenv("QDRANT_MARGIN_THRESHOLD", "0.04"))

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
            "margin":        0.0,
            "ambiguous":     False,
            "matched":       False,
        }

    def search(self, terms: Union[str, List[str]]) -> dict:
        """
        Search Qdrant for a term or list of terms.
        Terms should be clean clinical concepts (e.g. from SymptomExtractorAgent),
        not raw conversational text.
        """
        if isinstance(terms, str):
            terms = [terms]
            
        if not terms:
            return self._no_match()

        all_confident = []
        ambiguous = False
        
        for term in terms:
            vec = self._embed([term])[0]
            hits = self._search_one(vec, limit=5)
            
            if not hits:
                continue
                
            # Margin check per term: if the top 2 hits are too close, the term match is ambiguous
            if len(hits) > 1:
                margin = hits[0]["score"] - hits[1]["score"]
                if margin < MARGIN_THRESHOLD:
                    ambiguous = True
                    
            if hits[0]["score"] >= self._threshold:
                all_confident.append(hits[0])

        if not all_confident:
            return self._no_match()

        # Sort all confident hits by score globally
        ranked = sorted(all_confident, key=lambda h: h["score"], reverse=True)
        top = ranked[0]
        
        # Collect unique symptom IDs across all confident terms
        unique_ids = []
        for h in ranked:
            if h["hp_id"] not in unique_ids:
                unique_ids.append(h["hp_id"])

        return {
            "clinical_term": top["name"], # Highest scoring name across all terms
            "symptom_id":    top["hp_id"],
            "symptom_ids":   unique_ids,  # All confident unique HP IDs found
            "score":         top["score"],
            "ambiguous":     ambiguous,
            "matched":       True,
        }
