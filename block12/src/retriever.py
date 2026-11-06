# src/retriever.py
import os

import faiss
try:
    faiss.omp_set_num_threads(1)
except Exception:
    pass
import pandas as pd

# Anchor all paths to this file so they resolve correctly
# regardless of working directory (critical on HF Spaces).
_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.dirname(_HERE)   # block12/

# File paths
DATA_PATH     = os.path.join(_ROOT, "data", "clean_pairs.csv")
CACHE_DIR     = os.path.join(_ROOT, "cache")
INDEX_PATH_IP = os.path.join(CACHE_DIR, "faiss_index_ip.bin")
INDEX_PATH_L2 = os.path.join(CACHE_DIR, "faiss_index.bin")
INDEX_PATH    = INDEX_PATH_IP if os.path.exists(INDEX_PATH_IP) else INDEX_PATH_L2

# Lazy-loaded model — avoids triggering torch._C._cuda_init at import time,
# which would crash ZeroGPU before any @spaces.GPU context is active.
# Explicitly pinned to CPU since embeddings don't need GPU here.
_model = None

def _get_model():
    global _model
    if _model is None:
        import torch
        torch.set_num_threads(1)
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    return _model

def _build_or_load_index(df: pd.DataFrame, use_ip: bool = True):
    """Builds a FAISS index or loads it from disk if it already exists.
    
    Supports both IndexFlatIP (cosine similarity with normalized embeddings)
    and legacy IndexFlatL2 (Euclidean distance on unnormalized embeddings).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # 1. Prefer existing IndexFlatIP index
    if os.path.exists(INDEX_PATH_IP):
        return faiss.read_index(INDEX_PATH_IP)
        
    # 2. Fall back to existing legacy IndexFlatL2 index
    if os.path.exists(INDEX_PATH_L2):
        return faiss.read_index(INDEX_PATH_L2)
        
    # 3. Build new index (defaults to IndexFlatIP with normalized vectors)
    print("Building FAISS IndexFlatIP with normalized vectors. This may take a minute...")
    texts = df['text_customer'].tolist()
    
    if use_ip:
        # Inner Product corresponds to Cosine Similarity when vectors are L2-normalized
        embeddings = _get_model().encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        faiss.write_index(index, INDEX_PATH_IP)
        print(f"FAISS IndexFlatIP saved to {INDEX_PATH_IP}")
    else:
        # Legacy L2 distance index
        embeddings = _get_model().encode(texts, show_progress_bar=True, batch_size=64)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, INDEX_PATH_L2)
        print(f"FAISS IndexFlatL2 saved to {INDEX_PATH_L2}")
    
    return index

_DF = None
_INDEX = None

def get_index_metric_type() -> str:
    """Returns 'IP' (Inner Product / Cosine) or 'L2' (Euclidean) for loaded index."""
    global _INDEX, _DF
    if _DF is None and os.path.exists(DATA_PATH):
        _DF = pd.read_csv(DATA_PATH)
    if _INDEX is None and _DF is not None:
        _INDEX = _build_or_load_index(_DF)
    if _INDEX is not None:
        if getattr(_INDEX, 'metric_type', None) == faiss.METRIC_INNER_PRODUCT:
            return "IP"
        return "L2"
    return "UNKNOWN"

_EVAL_IDS = None
_EVAL_TEXTS = None

def _get_eval_exclusion_set():
    """Loads all 250 evaluation tweet IDs and normalized texts to prevent evaluation leakage."""
    global _EVAL_IDS, _EVAL_TEXTS
    if _EVAL_IDS is None:
        eval_path = os.path.join(_ROOT, "eval", "golden_test.json")
        if os.path.exists(eval_path):
            try:
                import json
                with open(eval_path, "r", encoding="utf-8") as f:
                    eval_data = json.load(f)
                _EVAL_IDS = set(int(r["tweet_id_customer"]) for r in eval_data if "tweet_id_customer" in r)
                _EVAL_TEXTS = set(str(r.get("text_customer", "")).strip().lower() for r in eval_data)
            except Exception:
                _EVAL_IDS = set()
                _EVAL_TEXTS = set()
        else:
            _EVAL_IDS = set()
            _EVAL_TEXTS = set()
    return _EVAL_IDS, _EVAL_TEXTS

def retrieve(
    text: str,
    k: int = 3,
    min_similarity: float = None,
    exclude_exact_match: bool = False,
    exclude_eval_set: bool = False,
    return_metadata: bool = False
) -> list:
    """Returns the top-k historical brand replies for a given customer text.
    
    Args:
        text: Customer message.
        k: Number of historical context items to retrieve.
        min_similarity: Minimum similarity score required (for IP index: cosine similarity; for L2: converted score).
        exclude_exact_match: When True, excludes cases where text_customer matches query or evaluation set.
        exclude_eval_set: When True, strictly excludes all 250 golden evaluation IDs and texts.
        return_metadata: If True, returns list of dicts with text and score instead of strings.
    """
    global _DF, _INDEX
    if not text or not str(text).strip():
        return []

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find {DATA_PATH}. Have you completed Block 1?")
        
    if _DF is None:
        _DF = pd.read_csv(DATA_PATH)
    if _INDEX is None:
        _INDEX = _build_or_load_index(_DF)
    
    df = _DF
    index = _INDEX
    is_ip = getattr(index, 'metric_type', None) == faiss.METRIC_INNER_PRODUCT
    
    # Embed the incoming customer text (normalize if index uses Inner Product)
    import torch
    with torch.inference_mode():
        query_vector = _get_model().encode([text.strip()], normalize_embeddings=is_ip)
    
    # Over-fetch slightly if filtering out evaluation cases
    fetch_k = k + 10 if (exclude_exact_match or exclude_eval_set) else k
    fetch_k = min(fetch_k, len(df))
    
    # Search the index for the closest matches
    distances, indices = index.search(query_vector, fetch_k)
    
    retrieved_results = []
    clean_query = text.strip().lower()
    eval_ids, eval_texts = _get_eval_exclusion_set()
    
    for score, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(df):
            continue
            
        row = df.iloc[idx]
        hist_cust = str(row['text_customer']).strip().lower()
        row_id = row.get('tweet_id_customer')
        
        # Golden-set evaluation leakage elimination:
        # Exclude exact query match, any row ID in the 250 golden test set, and any golden text
        if exclude_exact_match or exclude_eval_set:
            if row_id is not None and int(row_id) in eval_ids:
                continue
            if hist_cust in eval_texts:
                continue
            if hist_cust == clean_query:
                continue
            
        # Apply similarity threshold (default 0.35 for IP if not specified)
        effective_min = min_similarity if min_similarity is not None else (0.35 if is_ip else None)
        if effective_min is not None:
            if is_ip and score < effective_min:
                continue
            elif not is_ip and score > (1.0 - effective_min):
                continue
                
        # URL Sanitization: Replace obsolete external t.co links with official help link
        import re
        clean_brand_reply = re.sub(r'https?://t\.co/\S+', 'https://www.amazon.com/contact-us', str(row['text_brand']))

        if return_metadata:
            retrieved_results.append({
                "brand_reply": clean_brand_reply,
                "historical_query": row['text_customer'],
                "score": float(score)
            })
        else:
            retrieved_results.append(clean_brand_reply)
            
        if len(retrieved_results) >= k:
            break
            
    return retrieved_results

if __name__ == "__main__":
    # Test the retriever
    test_query = "My package says delivered but I can't find it anywhere!"
    print(f"Index metric type: {get_index_metric_type()}")
    print(f"Query: {test_query}\n")
    results = retrieve(test_query)
    for i, res in enumerate(results, 1):
        print(f"Retrieved Context {i}: {res}\n")