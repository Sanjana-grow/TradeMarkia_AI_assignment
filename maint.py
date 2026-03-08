import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI, Body
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture

class VectorEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def prepare_data(self):
        # Filtering headers/footers to reduce noise as requested
        newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
        self.documents = [doc for doc in newsgroups.data if len(doc.strip()) > 50]
        return self.documents

    def create_index(self, documents):
        embeddings = self.model.encode(documents, show_progress_bar=True)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        return embeddings

    def search(self, query_vector, k=5):
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), k)
        return [self.documents[i] for i in indices[0]]

class FuzzyClusterer:
    def __init__(self, n_clusters=12): # 12 clusters to balance sub-topics
        self.n_clusters = n_clusters
        self.gmm = GaussianMixture(n_components=n_clusters, covariance_type='spherical')

    def fit_predict(self, embeddings):
        self.gmm.fit(embeddings)
        return self.gmm.predict_proba(embeddings)

class SemanticCache:
    def __init__(self, dimension, threshold=0.3):
        self.threshold = threshold
        self.cache_data = {}
        self.cache_index = faiss.IndexFlatL2(dimension)
        self.query_map = []
        self.hits = 0
        self.misses = 0

    def get(self, query_vector):
        if self.cache_index.ntotal == 0:
            return None
        
        # FAISS search for the closest previous query
        distances, indices = self.cache_index.search(query_vector.astype('float32'), 1)
        score = float(distances[0][0])
        
        # Lower score in FlatL2 means higher similarity
        if score < self.threshold:
            self.hits += 1
            matched_query = self.query_map[indices[0][0]]
            return {
                "matched_query": matched_query,
                "similarity_score": 1 - score, # Normalized for display
                "result": self.cache_data[matched_query]
            }
        
        self.misses += 1
        return None

    def set(self, query_text, query_vector, result):
        self.cache_index.add(query_vector.astype('float32'))
        self.query_map.append(query_text)
        self.cache_data[query_text] = result
    
    def clear(self):
        self.cache_data = {}
        self.query_map = []
        self.hits = 0
        self.misses = 0
        # Reset FAISS index
        self.cache_index = faiss.IndexFlatL2(self.cache_index.d)

app = FastAPI()

# Global State
engine = VectorEngine()
docs = engine.prepare_data()
embeddings = engine.create_index(docs)
clusterer = FuzzyClusterer()
soft_assignments = clusterer.fit_predict(embeddings)
cache = SemanticCache(dimension=embeddings.shape[1])

@app.post("/query")
async def query_endpoint(payload: dict = Body(...)):
    user_query = payload.get("query")
    query_vector = engine.model.encode([user_query])
    
    # Check Semantic Cache
    cache_result = cache.get(query_vector)
    
    # Get dominant cluster for the query
    cluster_probs = clusterer.gmm.predict_proba(query_vector)
    dominant_cluster = int(np.argmax(cluster_probs[0]))
    
    if cache_result:
        return {
            "query": user_query,
            "cache_hit": True,
            "matched_query": cache_result["matched_query"],
            "similarity_score": round(cache_result["similarity_score"], 2),
            "result": cache_result["result"],
            "dominant_cluster": dominant_cluster
        }
    
    # On Miss: Compute
    search_results = engine.search(query_vector, k=1)
    final_result = search_results[0]
    
    # Store in Cache
    cache.set(user_query, query_vector, final_result)
    
    return {
        "query": user_query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": 0.0,
        "result": final_result,
        "dominant_cluster": dominant_cluster
    }

@app.get("/cache/stats")
async def get_cache_stats():
    total = cache.hits + cache.misses
    hit_rate = cache.hits / total if total > 0 else 0
    return {
        "total_entries": len(cache.query_map),
        "hit_count": cache.hits,
        "miss_count": cache.misses,
        "hit_rate": round(hit_rate, 3)
    }

@app.delete("/cache")
async def flush_cache():
    cache.clear()
    return {"message": "Cache flushed and stats reset."}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)