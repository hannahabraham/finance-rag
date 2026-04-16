# Financial Research Assistant

A multi-agent RAG system that answers natural language questions over SEC financial filing documents. Built entirely with free, open-source models that run locally.

---

## The Problem

SEC filings (10-K, 10-Q) can be 300+ pages long. Finding a specific figure, risk statement, or disclosure manually takes hours. Existing LLM-based solutions hallucinate figures or fail to cite sources, making them unreliable for financial use cases.

## The Solution

This system retrieves the exact passage from the correct document, verifies the evidence, generates a grounded answer, and cites the source page — all through a self-correcting five-agent pipeline.

---

## Demo

```
Question:  What was Amazon's operating income in 2022?

Answer:    Amazon reported an operating loss of $2.5 billion in 2022, compared
           to operating income of $24.9 billion in 2021.

Source:    AMAZON_2022_10K.pdf, page 38
Confidence: High
```

---

## Architecture

The pipeline is built as a LangGraph state machine. Five agents execute in sequence, with the critic agent able to loop back to retrieval if the answer quality is insufficient.

```
User Question
      │
      ▼
┌──────────────────────┐
│  Query Understanding │   Extracts company name, year, and financial intent
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Retrieval Agent    │   Hybrid search (dense + BM25) with company filter
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Evidence Verifier   │   Removes irrelevant chunks, keeps grounded evidence
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Answer Writer      │   Generates cited answer strictly from evidence
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    Critic Agent      │   Accepts answer or triggers retry (max 2 retries)
└──────────┬───────────┘
           │
    ┌──────┴──────┐
  ACCEPT        RETRY ──► back to Retrieval
    │
    ▼
  Structured output: answer + explanation + source + confidence
```

---

## Key Design Decisions

**Why multi-agent instead of a single chain?**
Each agent has one responsibility. This makes failures debuggable — if an answer is wrong, you can inspect each agent's output to find where it broke down. A single chain gives you no visibility into intermediate steps.

**Why hybrid retrieval?**
Dense retrieval handles paraphrasing and synonyms ("earnings" vs "net income"). BM25 handles exact terms (ticker symbols, specific dollar figures). Combining them with Reciprocal Rank Fusion gives better coverage than either alone.

**Why a critic agent with retry?**
Financial answers need to be grounded. The critic checks whether the answer is actually supported by the retrieved chunks. If not, it broadens the search and tries again — reducing hallucination without requiring a more expensive model.

---

## Tech Stack

| Layer | Tool | Reason |
|---|---|---|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Free, runs locally, no API key |
| Vector search | FAISS | Fast in-memory search, Apple MPS supported |
| Keyword search | rank-bm25 | Exact term matching for financial figures |
| Language model | Mistral-7B GGUF via llama-cpp-python | 4-bit quantised, runs on 8 GB RAM, Metal-accelerated on Apple Silicon |
| Agent framework | LangGraph | Typed shared state, conditional routing |
| PDF extraction | PyMuPDF | Page-accurate text with metadata |
| Backend API | FastAPI + Uvicorn | Streaming SSE endpoint for real-time pipeline progress |
| Frontend | React + TypeScript + Tailwind | Professional UI with step-by-step pipeline visualisation |
| Evaluation | rouge-score | Standard metric, no external API needed |
| Dataset | FinanceBench | 150 annotated Q&A pairs over real SEC filings |

---

## Project Structure

```
finance-rag/
├── app.py                          FastAPI entry point (uvicorn)
├── requirements.txt
│
├── frontend/                       React + TypeScript + Tailwind SPA
│   ├── package.json
│   ├── vite.config.ts              Dev proxy → FastAPI :7860
│   ├── src/
│   │   ├── App.tsx                 Root component + pipeline state machine
│   │   ├── lib/api.ts              SSE streaming client
│   │   ├── types.ts                Shared TypeScript types
│   │   └── components/
│   │       ├── Header.tsx
│   │       ├── QuestionForm.tsx    Mode + retrieval selectors
│   │       ├── PipelineTracker.tsx Horizontal stepper + step cards
│   │       ├── StepCard.tsx        Per-agent detail panel
│   │       └── AnswerCard.tsx      Final answer + sources + evidence
│   └── dist/                       Built assets (served by FastAPI)
│
├── src/
│   ├── config.py                   All settings loaded from .env
│   ├── ingestion/
│   │   ├── pdf_loader.py           Page-level PDF extraction with metadata
│   │   ├── chunker.py              Recursive, fixed, and page chunking strategies
│   │   └── dataset_loader.py       FinanceBench JSONL loader and merger
│   ├── retrieval/
│   │   ├── embedder.py             HuggingFace embedding model wrapper
│   │   ├── vector_store.py         FAISS index build, save, load
│   │   └── retriever.py            Dense, BM25, and hybrid retrieval
│   ├── agents/
│   │   ├── state.py                Shared TypedDict state for LangGraph
│   │   ├── llm.py                  Local GGUF LLM loader (llama-cpp-python)
│   │   ├── nodes.py                Five agent node functions
│   │   └── graph.py                LangGraph assembly and run_pipeline()
│   ├── evaluation/
│   │   └── metrics.py              Retrieval, answer, and grounding metrics
│   └── app/
│       ├── server.py               FastAPI server (API + SPA static files)
│       ├── pipeline_service.py     Streaming SSE wrapper for the pipeline
│       └── baseline_rag.py         Single-chain baseline for comparison
│
├── scripts/
│   ├── build_index.py              One-time PDF ingestion and index build
│   ├── download_model.py           Downloads Mistral-7B GGUF from HuggingFace
│   └── run_evaluation.py           Runs evaluation against FinanceBench questions
│
└── results/                        Saved evaluation JSON outputs
```

---

## Evaluation

Evaluated against 150 annotated FinanceBench questions with gold answers, evidence text, and source page numbers.

| Metric | What It Measures |
|---|---|
| Document Match Rate | Was the correct source document retrieved? |
| Page Match Rate | Was the correct page retrieved? |
| MRR | How highly was the correct document ranked? |
| ROUGE-L | How closely does the answer match the gold answer? |
| Exact Match Rate | Does the answer contain the gold answer text? |
| Grounding Rate | Is the answer supported by retrieved evidence? |

Six experiments compare baseline single-chain RAG against multi-agent RAG across three retrieval strategies (dense, BM25, hybrid) and three chunking strategies (recursive, fixed, page-based).

---

## Quickstart

**Requirements:** Python 3.11+, Node.js 18+, 8 GB RAM minimum, ~6 GB disk space

```bash
# 1. Clone and install (Python)
git clone https://github.com/YOUR_USERNAME/finance-rag.git
cd finance-rag
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Build the React frontend
cd frontend && npm install && npm run build && cd ..

# 3. Get the dataset
git clone https://github.com/patronus-ai/financebench.git /tmp/financebench
mkdir -p data/pdfs
cp /tmp/financebench/data/*.jsonl data/
cp /tmp/financebench/pdfs/*.pdf data/pdfs/

# 4. Download the LLM (~4 GB, one time)
python scripts/download_model.py

# 5. Build the vector index (one time, ~10–30 min)
python scripts/build_index.py

# 6. Run
python app.py
# Open http://localhost:7860
```

**Development (hot reload):**

```bash
# Terminal 1 — API server
python app.py

# Terminal 2 — React dev server (proxies /api to :7860)
cd frontend && npm run dev
# Open http://localhost:5173
```

---

---

## Running Evaluation

```bash
# Baseline
python scripts/run_evaluation.py --mode baseline --retrieval dense

# Multi-agent with hybrid retrieval
python scripts/run_evaluation.py --mode multiagent --retrieval hybrid

# Quick smoke test (20 questions)
python scripts/run_evaluation.py --limit 20 --mode multiagent --retrieval hybrid
```

Results are saved to `results/eval_MODE_RETRIEVAL_TIMESTAMP.json`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: faiss` | `pip install faiss-cpu` |
| LLM model not found | `python scripts/download_model.py` |
| Vector store not found | `python scripts/build_index.py` |
| Metal / MPS out of memory | Set `n_gpu_layers=0` in `src/agents/llm.py` |
| PDF pages return no text | PDF is a scanned image — OCR not supported in this version |

---

## Dataset

[FinanceBench](https://github.com/patronus-ai/financebench) by Patronus AI. Open-source sample contains 150 annotated questions over real SEC filings from companies including Amazon, Apple, Adobe, 3M, and others. Each question includes the gold answer, evidence text, and the exact page number where the answer appears.

---
