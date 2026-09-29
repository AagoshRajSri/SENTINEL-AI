# src/retriever.py
import os

import faiss
import pandas as pd

# Anchor all paths to this file so they resolve correctly
# regardless of working directory (critical on HF Spaces).
_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.dirname(_HERE)   # block12/

# File paths
DATA_PATH  = os.path.join(_ROOT, "data", "clean_pairs.csv")
CACHE_DIR  = os.path.join(_ROOT, "cache")
INDEX_PATH = os.path.join(CACHE_DIR, "faiss_index.bin")

# Lazy-loaded model — avoids triggering torch._C._cuda_init at import time,
# which would crash ZeroGPU before any @spaces.GPU context is active.
# Explicitly pinned to CPU since embeddings don't need GPU here.
_model = None

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    return _model

def _build_or_load_index(df: pd.DataFrame):
    """Builds a FAISS index or loads it from disk if it already exists."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    if os.path.exists(INDEX_PATH):
        # Load pre-built index to save time and compute
        index = faiss.read_index(INDEX_PATH)
    else:
        print("Building FAISS index for the first time. This may take a minute...")
        # Embed the customer's text to match incoming queries to historical queries
        texts = df['text_customer'].tolist()
        embeddings = _get_model().encode(texts, show_progress_bar=True)
        
        # Create a FAISS index (L2 distance)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        # Save to disk for instant loading next time
        faiss.write_index(index, INDEX_PATH)
        print(f"FAISS index saved to {INDEX_PATH}")
        
    return index

_DF = None
_INDEX = None

def retrieve(text: str, k: int = 3, severity: str = "MEDIUM") -> list:
    """Returns the top-k historical brand replies for a given customer text.

    Uses a two-pass approach when severity metadata is available:
      Pass 1: FAISS similarity search — over-fetch 20 candidates.
      Pass 2: Re-rank / filter by severity match so critical complaints
              don't retrieve cheerful low-severity templates.

    Args:
        text:     Customer text (should already be PII-sanitized).
        k:        Number of results to return (default 3).
        severity: Severity level of the incoming query — one of
                  'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'.
                  When HIGH or CRITICAL, only severity-matched examples
                  are returned (with fallback to unfiltered if too few).
    """
    global _DF, _INDEX
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find {DATA_PATH}. Have you completed Block 1?")

    if _DF is None:
        _DF = pd.read_csv(DATA_PATH)
    if _INDEX is None:
        _INDEX = _build_or_load_index(_DF)

    df = _DF
    index = _INDEX

    # Embed the incoming customer text
    query_vector = _get_model().encode([text])

    # Pass 1: Over-fetch to give the severity filter enough candidates to work with
    fetch_k = min(20, len(df))
    distances, indices = index.search(query_vector, fetch_k)

    candidates = [
        (df.iloc[idx], float(dist))
        for idx, dist in zip(indices[0], distances[0])
    ]

    # Pass 2: Severity filtering — only active when severity metadata exists
    # and the query is HIGH or CRITICAL severity.
    has_severity_col = "severity_level" in df.columns
    if has_severity_col and severity in ("HIGH", "CRITICAL"):
        severity_matched = [
            (row, dist) for row, dist in candidates
            if str(row.get("severity_level", "")).upper() in ("HIGH", "CRITICAL")
        ]
        # Only use filtered list if we have enough matches; otherwise fall back
        if len(severity_matched) >= k:
            candidates = severity_matched
        # else: fall through to unfiltered candidates

    # Return top-k brand replies
    return [c[0]["text_brand"] for c in candidates[:k]]

if __name__ == "__main__":
    # Test the retriever
    test_query = "My package says delivered but I can't find it anywhere!"
    print(f"Query: {test_query}\n")
    results = retrieve(test_query)
    for i, res in enumerate(results, 1):
        print(f"Retrieved Context {i}: {res}\n")