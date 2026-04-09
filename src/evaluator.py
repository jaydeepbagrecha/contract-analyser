"""
RAG Evaluation Module
Uses RAGAS 0.4.3 framework to evaluate retrieval and generation quality.
Produces scores for faithfulness, relevancy, precision, and recall.

Key implementation notes (ragas 0.4.3 quirks):
─────────────────────────────────────────────────
1. Metrics must be imported from ragas.metrics.collections (not ragas.metrics)
2. Metrics must be instantiated as class objects: Faithfulness(llm=...)
3. ragas.evaluate() is broken (BaseMetric ≠ Metric isinstance check), so we
   call metric.score(**kwargs) directly on each sample
4. score() internally runs async code → llm_factory needs AsyncOpenAI client
5. Each metric accepts different kwargs for score():
     faithfulness       → user_input, response, retrieved_contexts
     answer_relevancy   → user_input, response
     context_precision  → user_input, reference, retrieved_contexts
     context_recall     → user_input, retrieved_contexts, reference
"""

import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# ── RAGAS imports ──────────────────────────────────────────────────────────
import ragas
print(f"RAGAS version: {ragas.__version__}")

from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

# ── Project imports ────────────────────────────────────────────────────────
from src.ingestion import load_vector_store
from src.retriever import HybridRetriever
from src.generator import generate_answer


# ═══════════════════════════════════════════════════════════════════════════
# METRIC SETUP
# ═══════════════════════════════════════════════════════════════════════════

def _build_metrics() -> dict:
    """
    Create RAGAS metric instances.
    Uses AsyncOpenAI because score() internally calls agenerate().
    """
    api_key = os.getenv("OPENAI_API_KEY")
    async_client = AsyncOpenAI(api_key=api_key)

    ragas_llm = llm_factory("gpt-4o-mini", client=async_client)
    ragas_emb = embedding_factory("openai", model="text-embedding-3-small", client=async_client)

    print("Configured RAGAS LLM and embeddings via AsyncOpenAI client")

    return {
        "faithfulness": Faithfulness(llm=ragas_llm),
        "answer_relevancy": AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        "context_precision": ContextPrecision(llm=ragas_llm),
        "context_recall": ContextRecall(llm=ragas_llm),
    }


def _get_metric_kwargs(metric_name: str, sample: dict) -> dict:
    """
    Return the exact kwargs each metric expects (confirmed via inspect).
    """
    q = sample["question"]
    a = sample["answer"]
    c = sample["contexts"]
    r = sample["ground_truth"]

    kwargs_map = {
        "faithfulness":      {"user_input": q, "response": a, "retrieved_contexts": c},
        "answer_relevancy":  {"user_input": q, "response": a},
        "context_precision": {"user_input": q, "reference": r, "retrieved_contexts": c},
        "context_recall":    {"user_input": q, "retrieved_contexts": c, "reference": r},
    }
    return kwargs_map[metric_name]


# ═══════════════════════════════════════════════════════════════════════════
# TEST SET & RAG PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def load_test_set(filepath: str = "eval/test_set.json") -> list[dict]:
    """Load test question-answer pairs from JSON."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return data["questions"]


def run_rag_pipeline(question: str, retriever: HybridRetriever) -> dict:
    """Run the full RAG pipeline for a single question."""
    chunks = retriever.retrieve(question)
    contexts = [c["content"] for c in chunks]
    result = generate_answer(question, chunks)
    return {
        "question": question,
        "answer": result["answer"],
        "contexts": contexts,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_rag(test_set_path: str = "eval/test_set.json") -> dict:
    """
    Run RAGAS evaluation on the full test set.
    Returns a dict with overall scores and per-question details.
    """
    print("=" * 50)
    print("RAGAS EVALUATION")
    print("=" * 50)

    # ── 1. Build metrics ──
    print("\nConfiguring RAGAS metrics...")
    metrics = _build_metrics()

    # ── 2. Load test set ──
    print("\nLoading test set...")
    test_questions = load_test_set(test_set_path)
    print(f"  {len(test_questions)} questions loaded")

    # ── 3. Load vector store & retriever ──
    print("\nLoading vector store...")
    vector_store = load_vector_store()
    retriever = HybridRetriever(vector_store)

    # ── 4. Run RAG pipeline for each question ──
    print("\nRunning RAG pipeline...")
    samples = []
    for i, item in enumerate(test_questions):
        print(f"  Processing {i+1}/{len(test_questions)}: {item['question'][:60]}...")
        try:
            result = run_rag_pipeline(item["question"], retriever)
            samples.append({
                "question": result["question"],
                "answer": result["answer"],
                "contexts": result["contexts"],
                "ground_truth": item["ground_truth"],
            })
        except Exception as e:
            print(f"    ERROR: {e}")

    # ── 5. Score each sample with each metric ──
    print(f"\nScoring {len(samples)} samples across {len(metrics)} metrics...")
    all_scores = {name: [] for name in metrics}

    for i, sample in enumerate(samples):
        print(f"\n  Sample {i+1}/{len(samples)}:")
        for name, metric in metrics.items():
            try:
                kwargs = _get_metric_kwargs(name, sample)
                result = metric.score(**kwargs)
                score = float(result)
                all_scores[name].append(score)
                print(f"    {name}: {score:.3f}")
            except Exception as e:
                print(f"    {name}: FAILED - {e}")
                all_scores[name].append(float("nan"))

    # ── 6. Compute averages (ignoring NaN) ──
    averages = {}
    for name, scores in all_scores.items():
        valid = [s for s in scores if s == s]  # NaN != NaN
        averages[name] = sum(valid) / len(valid) if valid else 0.0

    # ── 7. Print results ──
    print("\n" + "=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    for name, score in averages.items():
        print(f"  {name}: {score:.3f}")
    print("=" * 50)

    # ── 8. Interpretation guide ──
    print("\nInterpretation:")
    print("  faithfulness > 0.8  → answers grounded in context (not hallucinating)")
    print("  answer_relevancy > 0.7  → answers address the question asked")
    print("  context_precision > 0.7  → retrieved chunks are relevant")
    print("  context_recall > 0.7  → retriever finds all needed information")

    # ── 9. Save results ──
    output = {
        "overall": {k: round(v, 4) for k, v in averages.items()},
        "per_question": [
            {"question": s["question"], "answer": s["answer"]}
            for s in samples
        ],
        "per_sample_scores": {
            k: [round(s, 4) if s == s else None for s in v]
            for k, v in all_scores.items()
        },
    }

    os.makedirs("eval", exist_ok=True)
    with open("eval/ragas_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to eval/ragas_results.json")

    return output


if __name__ == "__main__":
    evaluate_rag()