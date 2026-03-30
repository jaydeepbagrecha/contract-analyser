"""
RAG Evaluation Module
Uses RAGAS framework to evaluate retrieval and generation quality.
Produces scores for faithfulness, relevancy, precision, and recall.
"""
 
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
 
from src.ingestion import load_vector_store
from src.retriever import HybridRetriever
from src.generator import generate_answer
 
 
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
 
 
def evaluate_rag(test_set_path: str = "eval/test_set.json") -> dict:
    """Run RAGAS evaluation on the full test set."""
    print("Loading test set...")
    test_questions = load_test_set(test_set_path)
 
    print("Loading vector store...")
    vector_store = load_vector_store()
    retriever = HybridRetriever(vector_store)
 
    questions = []
    answers = []
    contexts = []
    ground_truths = []
 
    for i, item in enumerate(test_questions):
        print(f"  Processing {i+1}/{len(test_questions)}: {item['question'][:60]}...")
        result = run_rag_pipeline(item["question"], retriever)
        questions.append(result["question"])
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        ground_truths.append(item["ground_truth"])
 
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
 
    print("\nRunning RAGAS evaluation (this takes a few minutes)...")
    results = evaluate(
        eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
 
    print("\n" + "=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    for metric, score in results.items():
        print(f"  {metric}: {score:.3f}")
    print("=" * 50)
 
    return {
        "overall": dict(results),
        "per_question": [
            {"question": q, "answer": a}
            for q, a in zip(questions, answers)
        ],
    }
 
 
if __name__ == "__main__":
    evaluate_rag()
