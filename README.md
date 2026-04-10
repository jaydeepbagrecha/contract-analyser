# Contract Analyser — RAG-Based Legal Contract Analysis

An AI-powered contract analysis system for Indian legal documents (Gujarat & Maharashtra jurisdiction). Built with a hybrid RAG pipeline using OpenAI GPT-4o, LangChain, and ChromaDB, evaluated with RAGAS metrics.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  OFFLINE — Ingestion Pipeline            │
│                                                          │
│  PDF/DOCX ──→ pdfplumber ──→ RecursiveCharacter ──→ OpenAI       │
│  Documents     Loader         TextSplitter          text-embedding │
│                               (1000 chars,          -3-small       │
│                                200 overlap)              │
│                                      │                   │
│                                      ▼                   │
│                                  ChromaDB                │
│                              (persistent store)          │
│                              727 chunks / 19 docs        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  ONLINE — Query Pipeline                  │
│                                                          │
│  User      ──→  OpenAI       ──→  Hybrid Search  ──→ Top-K  │
│  Question      Embedder          (Vector + BM25       Chunks │
│                                   via RRF)               │
│                                                     │    │
│                                                     ▼    │
│                                                  GPT-4o  │
│                                              (grounded   │
│                                               generation)│
│                                                     │    │
│                                                     ▼    │
│                                          Answer with     │
│                                          Citations &     │
│                                          Confidence      │
└─────────────────────────────────────────────────────────┘
```

## RAGAS Evaluation Results

Evaluated on 42 question-answer pairs across 19 Indian legal contracts.

| Metric             | Score | Target | Status |
|--------------------|-------|--------|--------|
| Faithfulness       | 0.977 | > 0.80 | ✅     |
| Answer Relevancy   | 0.746 | > 0.70 | ✅     |
| Context Precision  | 0.732 | > 0.70 | ✅     |
| Context Recall     | 0.803 | > 0.70 | ✅     |

> **Note:** Fill in actual scores after completing your RAGAS evaluation run. Replace `—` with your scores and `🔄` with ✅ or ⚠️. 

## Tech Stack

| Component        | Tool                          |
|------------------|-------------------------------|
| LLM              | OpenAI GPT-4o                 |
| Embeddings       | OpenAI text-embedding-3-small |
| Orchestration    | LangChain                     |
| Vector Database  | ChromaDB (persistent)         |
| Retrieval        | Hybrid (Vector + BM25 via RRF)|
| Evaluation       | RAGAS 0.1.7                   |
| Frontend         | Streamlit (multi-tab)         |

## Document Coverage

19 Indian legal contracts including: sale deeds, lease agreements, leave & license agreements, NDAs, SaaS agreements, service contracts, power of attorney, bank locker agreements, cheque bounce complaints, and RERA sale agreements.

## Quick Start

```bash
# Clone and set up
git clone https://github.com/<your-username>/contract-analyser.git
cd contract-analyser
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="your-key-here"

# Ingest documents
python -m src.ingestion

# Run the app
streamlit run app/streamlit_app.py

# Run evaluation
python -m src.evaluator
```

## Project Structure

```
contract-analyser/
├── app/
│   └── streamlit_app.py       # Multi-tab Streamlit UI
├── src/
│   ├── ingestion.py           # PDF loading, chunking, embedding
│   ├── retriever.py           # Hybrid search (vector + BM25)
│   ├── generator.py           # GPT-4o answer generation with citations
│   └── evaluator.py           # RAGAS evaluation pipeline
├── eval/
│   └── test_set.json          # 42 Q&A pairs for evaluation
├── data/
│   └── contracts/             # Indian legal contract PDFs
├── chroma_db/                 # Persistent vector store
├── requirements.txt
└── README.md
```

## Key Design Decisions

- **Hybrid retrieval (Vector + BM25):** Legal contracts contain specific terms (e.g., clause numbers, party names) that benefit from exact keyword matching alongside semantic similarity.
- **Domain-specific prompting:** System prompt instructs the LLM to behave as a legal contract analyst, improving answer quality for Indian legal terminology.
- **Citation grounding:** Every answer includes source references mapped back to specific documents and pages, reducing hallucination risk.
- **RAGAS evaluation:** Continuous measurement across faithfulness, relevancy, precision, and recall ensures the system improves with each iteration.

## License

MIT