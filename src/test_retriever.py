from src.ingestion import load_vector_store
from src.retriever import HybridRetriever
 
vector_store = load_vector_store()
retriever = HybridRetriever(vector_store, top_k=5)
 
queries = [
    "What are the termination clauses?",
    "What happens in case of breach of contract?",
    "What are the confidentiality obligations?",
    "Section 4.2",
]
 
for query in queries:
    print(f"\n{'=' * 50}")
    print(f"Query: {query}")
    print(f"{'=' * 50}")
    results = retriever.retrieve(query)
    for r in results:
        print(f"  [{r['rank']}] Score: {r['rrf_score']:.4f}")
        print(f"      Source: {r['metadata'].get('source', '?')} | Page: {r['metadata'].get('page', '?')}")
        print(f"      Text: {r['content'][:120]}...")
