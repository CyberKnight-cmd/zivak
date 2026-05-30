#!/usr/bin/env python3
"""
scripts/5_populate_qdrant.py
Embed all 19,389 symptoms and load into Qdrant.
Run once — idempotent (deletes and recreates the collection).
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

SCRIPT_DIR       = Path(__file__).resolve().parent
DATA_DIR         = Path(os.getenv("DATA_DIR", str(SCRIPT_DIR.parent / "data")))
PROCESSED_DIR    = DATA_DIR / "processed"

SYMPTOM_INDEX    = PROCESSED_DIR / "symptom_index.json"
SYMPTOM_CATS     = PROCESSED_DIR / "symptom_categories.json"
ID_MAP_PATH      = PROCESSED_DIR / "qdrant_id_map.json"

QDRANT_URL       = os.getenv("QDRANT_URL",        "http://localhost:6333")
QDRANT_API_KEY   = os.getenv("QDRANT_API_KEY",    None) or None
COLLECTION_NAME  = os.getenv("QDRANT_COLLECTION", "symptoms")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL",   "NeuML/pubmedbert-base-embeddings")
UPLOAD_BATCH     = 64   # smaller for cloud — reduces per-request payload over internet
VECTOR_SIZE      = 768
EXPECTED_COUNT   = 19389


def main():
    from qdrant_client import QdrantClient as _QC
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from sentence_transformers import SentenceTransformer
    import torch

    print(f"Connecting to Qdrant at {QDRANT_URL} ...")
    client = _QC(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=300)

    # Delete + recreate collection (idempotent)
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        print(f"  Deleting existing collection '{COLLECTION_NAME}' ...")
        client.delete_collection(COLLECTION_NAME)

    print(f"  Creating collection '{COLLECTION_NAME}' (size={VECTOR_SIZE}, distance=Cosine) ...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    # Load data
    print("  Loading symptom_index.json and symptom_categories.json ...")
    symptom_index = json.loads(SYMPTOM_INDEX.read_text(encoding="utf-8"))
    categories    = json.loads(SYMPTOM_CATS.read_text(encoding="utf-8"))

    # Build texts in stable order; keep hp_ids and texts aligned
    hp_ids    = []
    texts     = []
    names     = []
    syns_list = []
    cats_list = []
    for hp_id, info in symptom_index.items():
        if not isinstance(info, dict):
            continue
        name     = info.get("name", "")
        synonyms = info.get("synonyms", []) or []
        text     = name + (" " + " ".join(synonyms) if synonyms else "")
        hp_ids.append(hp_id)
        texts.append(text)
        names.append(name)
        syns_list.append(synonyms if isinstance(synonyms, list) else [])
        cats_list.append(categories.get(hp_id, "sign"))

    total = len(hp_ids)
    print(f"  Loaded {total} symptom entries.")

    # Embed in batches of EMBED_BATCH_SIZE
    print(f"  Embedding in batches of {EMBED_BATCH_SIZE} ...")
    t0          = time.time()
    all_vectors = []
    total_batches = (total + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    for b, i in enumerate(range(0, total, EMBED_BATCH_SIZE)):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        vecs  = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vectors.extend(vecs)
        print(f"  Embedded batch {b + 1}/{total_batches}")
    embed_time = time.time() - t0

    # Upload in batches of 256
    print(f"  Uploading {total} points in batches of {UPLOAD_BATCH} ...")
    t1 = time.time()
    for i in range(0, total, UPLOAD_BATCH):
        chunk_size = min(UPLOAD_BATCH, total - i)
        points = [
            PointStruct(
                id=i + j,
                vector=all_vectors[i + j].tolist(),
                payload={
                    "hp_id":    hp_ids[i + j],
                    "name":     names[i + j],
                    "synonyms": syns_list[i + j],
                    "category": cats_list[i + j],
                    "text":     texts[i + j],
                },
            )
            for j in range(chunk_size)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
    upload_time = time.time() - t1

    # Write qdrant_id_map.json — keys are strings (JSON requirement)
    id_map = {str(i): hp_ids[i] for i in range(total)}
    ID_MAP_PATH.write_text(json.dumps(id_map, indent=2), encoding="utf-8")
    print(f"  Written {ID_MAP_PATH}")

    # Assert point count
    info  = client.get_collection(COLLECTION_NAME)
    count = info.points_count
    assert count == EXPECTED_COUNT, f"FAIL: expected {EXPECTED_COUNT} points, got {count}"
    print(f"  Points count: {count} OK")

    # Test search: HP:0002875 must be in top 3 AND score >= 0.70
    print("  Test search: 'shortness of breath going up stairs' ...")
    test_vec = model.encode(
        ["shortness of breath going up stairs"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()
    response   = client.query_points(
        collection_name=COLLECTION_NAME,
        query=test_vec,
        limit=3,
        with_payload=True,
    )
    top_hp_ids = [h.payload["hp_id"] for h in response.points]
    top_score  = response.points[0].score if response.points else 0.0
    assert top_score >= 0.70, f"FAIL: top score {top_score:.3f} < 0.70 (top HP IDs: {top_hp_ids})"
    print(f"  Test PASS — top: {top_hp_ids[0]}  score={top_score:.3f}")

    print()
    print("=" * 50)
    print("  POPULATE COMPLETE")
    print("=" * 50)
    print(f"  Device:         {device}")
    print(f"  Points:         {count}")
    print(f"  Embed time:     {embed_time:.1f}s")
    print(f"  Upload time:    {upload_time:.1f}s")
    print(f"  Top test score: {top_score:.3f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
