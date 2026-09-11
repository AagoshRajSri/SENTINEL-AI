# src/retriever.py
import os
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

# File paths
DATA_PATH = "data/clean_pairs.csv"
CACHE_DIR = "cache"
INDEX_PATH = os.path.join(CACHE_DIR, "faiss_index.bin")

# Load the local, free embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

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
        embeddings = model.encode(texts, show_progress_bar=True)
        
        # Create a FAISS index (L2 distance)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        # Save to disk for instant loading next time
        faiss.write_index(index, INDEX_PATH)
        print(f"FAISS index saved to {INDEX_PATH}")
        
    return index

def retrieve(text: str, k: int = 3) -> list:
    """Returns the top-k historical brand replies for a given customer text."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find {DATA_PATH}. Have you completed Block 1?")
        
    df = pd.read_csv(DATA_PATH)
    index = _build_or_load_index(df)
    
    # Embed the incoming customer text
    query_vector = model.encode([text])
    
    # Search the index for the closest matches
    distances, indices = index.search(query_vector, k)
    
    # Fetch the actual historical brand replies based on the matched indices
    retrieved_replies = []
    for idx in indices[0]:
        retrieved_replies.append(df.iloc[idx]['text_brand'])
        
    return retrieved_replies

if __name__ == "__main__":
    # Test the retriever
    test_query = "My package says delivered but I can't find it anywhere!"
    print(f"Query: {test_query}\n")
    results = retrieve(test_query)
    for i, res in enumerate(results, 1):
        print(f"Retrieved Context {i}: {res}\n")