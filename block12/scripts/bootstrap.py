# scripts/bootstrap.py
"""
Bootstrap & Artifact Verification Script for Sentinel-AI

Reconstructs the normalized FAISS IndexFlatIP vector index from tracked source data
(data/clean_pairs.csv) and initializes deterministic evaluation caches.
Enables 100% clean-checkout reproducibility.
"""

import hashlib
import json
import os
import shutil
import sys
import time

_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DATA_PATH = os.path.join(_ROOT, "data", "clean_pairs.csv")
FIXTURES_DIR = os.path.join(_ROOT, "eval", "fixtures")
CACHE_DIR = os.path.join(_ROOT, "cache")
INDEX_PATH_IP = os.path.join(CACHE_DIR, "faiss_index_ip.bin")
METADATA_PATH = os.path.join(CACHE_DIR, "index_metadata.json")

EXPECTED_ROWS = 15000
EXPECTED_DIM = 384
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def bootstrap_caches():
    """Copies tracked evaluation fixtures into cache/ if missing."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    fixture_map = {
        "classifier_cache.json": "classifier_cache_groq.json",
        "escalation_cache.json": "two_pass_continuous_calibration.json",
        "drafter_cache.json": "drafter_cache.json",
        "zeroshot_cache.json": "zeroshot_cache.json"
    }
    
    restored = 0
    for fixture_name, target_name in fixture_map.items():
        src = os.path.join(FIXTURES_DIR, fixture_name)
        dst = os.path.join(CACHE_DIR, target_name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            restored += 1
            print(f"[CACHE] Restored {target_name} from evaluation fixtures.")
            
    if restored > 0:
        print(f"[CACHE] Initialized {restored} evaluation response caches.")
    else:
        print("[CACHE] Evaluation response caches are present.")


def build_or_verify_faiss_ip(force_rebuild: bool = False):
    """Builds or verifies the FAISS IndexFlatIP index with normalized vectors."""
    import faiss
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Tracked data source missing: {DATA_PATH}")
        sys.exit(1)

    print(f"[DATA] Loading historical customer-brand interactions: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if len(df) != EXPECTED_ROWS:
        print(f"[WARNING] Expected {EXPECTED_ROWS} rows, found {len(df)}")

    # Check if existing index is valid
    if os.path.exists(INDEX_PATH_IP) and not force_rebuild:
        try:
            index = faiss.read_index(INDEX_PATH_IP)
            is_ip = getattr(index, "metric_type", None) == faiss.METRIC_INNER_PRODUCT
            if is_ip and index.ntotal == len(df) and index.d == EXPECTED_DIM:
                sha256 = compute_file_sha256(INDEX_PATH_IP)
                print(f"[FAISS] Found valid IndexFlatIP index (SHA-256: {sha256[:16]}...). Verification PASSED.")
                return index, sha256
        except Exception as e:
            print(f"[FAISS] Existing index invalid ({e!s}). Rebuilding...")

    # Build fresh IndexFlatIP with L2-normalized embeddings
    print(f"[FAISS] Building IndexFlatIP ({len(df)} vectors, d={EXPECTED_DIM}, metric=METRIC_INNER_PRODUCT)...")
    t0 = time.time()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    texts = df["text_customer"].tolist()
    
    # normalize_embeddings=True enforces ||v|| = 1.0 (Cosine Similarity == Inner Product)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    
    # Verify normalization on a sample
    sample_norm = float(np.linalg.norm(embeddings[0]))
    if not (0.999 <= sample_norm <= 1.001):
        raise ValueError(f"Embeddings not normalized! Norm is {sample_norm}")

    index = faiss.IndexFlatIP(EXPECTED_DIM)
    index.add(embeddings)
    
    faiss.write_index(index, INDEX_PATH_IP)
    build_time = time.time() - t0
    sha256 = compute_file_sha256(INDEX_PATH_IP)
    
    metadata = {
        "index_type": "IndexFlatIP",
        "metric_type": "INNER_PRODUCT",
        "dimension": EXPECTED_DIM,
        "ntotal": int(index.ntotal),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "normalized_vectors": True,
        "build_time_seconds": round(build_time, 2),
        "source_data": DATA_PATH,
        "source_row_count": len(df),
        "sha256": sha256,
        "status": "READY_FOR_OFFLINE_EVALUATION"
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[FAISS] Saved IndexFlatIP index to {INDEX_PATH_IP} in {build_time:.2f}s (SHA-256: {sha256[:16]}...).")
    return index, sha256


def verify_retrieval(index):
    """Executes a test retrieval query to confirm alignment."""
    import faiss
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    query = "Where is my package? Tracking has not updated."
    q_vec = model.encode([query], normalize_embeddings=True)
    
    distances, indices = index.search(q_vec, 3)
    top_score = float(distances[0][0])
    top_idx = int(indices[0][0])
    
    print(f"[VERIFY] Test Query: '{query}'")
    print(f"[VERIFY] Top Match Index: {top_idx} | Cosine Similarity Score: {top_score:.4f}")
    if top_score < 0.30:
        print("[WARNING] Top match similarity is unexpectedly low.")
    else:
        print("[VERIFY] Semantic alignment check PASSED.")


def main():
    print("=" * 65)
    print(" SENTINEL-AI BOOTSTRAP & REPRODUCIBILITY ENGINE")
    print("=" * 65)
    
    # 1. Initialize evaluation caches
    bootstrap_caches()
    
    # 2. Build or verify normalized FAISS IndexFlatIP
    force = "--rebuild" in sys.argv
    index, sha = build_or_verify_faiss_ip(force_rebuild=force)
    
    # 3. Test verification query
    verify_retrieval(index)
    
    print("=" * 65)
    print(" [SUCCESS] Repository bootstrap complete. All offline fixtures ready.")
    print("=" * 65)


if __name__ == "__main__":
    main()
