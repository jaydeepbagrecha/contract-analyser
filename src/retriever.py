"""
Hybrid Retriever Module
Combines vector similarity search (semantic) with BM25 (keyword)
for better retrieval accuracy than either method alone.
 
Pattern: Query -> Vector Search + BM25 Search -> Reciprocal Rank Fusion -> Top-K
"""
 
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from typing import Optional
 
 
class HybridRetriever:
    """
    Combines vector similarity search with BM25 keyword search.
    Vector search catches semantic similarity ("ending the agreement" matches
    "termination"). BM25 catches exact keywords ("Section 4.2" matches
    "Section 4.2"). Together, they cover both cases.
    """
 
    def __init__(self, vector_store: Chroma, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k
        self._build_bm25_index()
 
    def _build_bm25_index(self):
        """Build a BM25 index from all documents in ChromaDB."""
        collection = self.vector_store._collection
        all_data = collection.get(include=["documents", "metadatas"])
 
        self.all_documents = all_data["documents"]
        self.all_metadatas = all_data["metadatas"]
 
        tokenized = [doc.lower().split() for doc in self.all_documents]
        self.bm25 = BM25Okapi(tokenized)
        print(f"BM25 index built with {len(self.all_documents)} documents")
 
    def vector_search(self, query: str, k: int = None) -> list[dict]:
        """Semantic similarity search using ChromaDB."""
        k = k or self.top_k
        results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
                "source": "vector",
            }
            for doc, score in results
        ]
 
    def bm25_search(self, query: str, k: int = None) -> list[dict]:
        """Keyword search using BM25."""
        k = k or self.top_k
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
 
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
 
        return [
            {
                "content": self.all_documents[i],
                "metadata": self.all_metadatas[i],
                "score": float(scores[i]),
                "source": "bm25",
            }
            for i in top_indices
            if scores[i] > 0
        ]
 
    def hybrid_search(self, query: str, k: int = None,
                      vector_weight: float = 0.6) -> list[dict]:
        """
        Hybrid search using Reciprocal Rank Fusion (RRF).
        RRF Formula: score = sum(1 / (rank + 60)) for each ranking list
        """
        k = k or self.top_k
        RRF_K = 60
 
        vector_results = self.vector_search(query, k=k * 2)
        bm25_results = self.bm25_search(query, k=k * 2)
 
        rrf_scores = {}
        content_map = {}
 
        for rank, result in enumerate(vector_results):
            key = hash(result["content"][:200])
            rrf_scores[key] = rrf_scores.get(key, 0) + vector_weight / (rank + RRF_K)
            content_map[key] = result
 
        bm25_weight = 1 - vector_weight
        for rank, result in enumerate(bm25_results):
            key = hash(result["content"][:200])
            rrf_scores[key] = rrf_scores.get(key, 0) + bm25_weight / (rank + RRF_K)
            if key not in content_map:
                content_map[key] = result
 
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[:k]
 
        results = []
        for i, key in enumerate(sorted_keys):
            result = content_map[key]
            result["rrf_score"] = rrf_scores[key]
            result["rank"] = i + 1
            result["source"] = "hybrid"
            results.append(result)
 
        return results
 
    def retrieve(self, query: str, method: str = "hybrid") -> list[dict]:
        """Main retrieval method. Returns top-k results."""
        if method == "hybrid":
            return self.hybrid_search(query)
        elif method == "vector":
            return self.vector_search(query)
        elif method == "bm25":
            return self.bm25_search(query)
        else:
            raise ValueError(f"Unknown method: {method}")
