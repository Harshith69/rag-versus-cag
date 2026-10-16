# RAG vs CAG: A Comparative Study

A hands-on comparison of **Retrieval Augmented Generation (RAG)** and **Cache Augmented Generation (CAG)** for question-answering over PDF documents.

## What is RAG?

**Retrieval Augmented Generation** retrieves only the most relevant chunks of a knowledge base at query time:

```
Query → Embed → Retrieve top-K chunks → LLM generates answer from chunks
```

- Splits documents into small chunks and stores embeddings in a vector database
- At query time, finds the most similar chunks via semantic search
- Sends only those chunks as context to the LLM

## What is CAG?

**Cache Augmented Generation** loads the *entire* knowledge base into the LLM's context window:

```
Query → Prepend full knowledge base → LLM generates answer from complete context
```

- No chunking, no embedding, no retrieval step
- The full document text is included in every prompt
- Relies on the LLM's large context window to process all information at once

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      PDF Documents                      │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
    ┌───────────┐            ┌───────────┐
    │    RAG    │            │    CAG    │
    │ Pipeline  │            │ Pipeline  │
    ├───────────┤            ├───────────┤
    │ 1. Chunk  │            │ 1. Load   │
    │ 2. Embed  │            │    full   │
    │ 3. Store  │            │    text   │
    │    in     │            │ 2. Inject │
    │  ChromaDB │            │    into   │
    │ 4. Query: │            │    prompt │
    │  retrieve │            │ 3. Query  │
    │  top-K →  │            │    LLM    │
    │  LLM      │            │           │
    └─────┬─────┘            └─────┬─────┘
          │                        │
          └──────────┬─────────────┘
                     ▼
           ┌──────────────────┐
           │   Evaluation     │
           │  (LLM-as-Judge)  │
           │  Latency, Quality│
           │  Comparison      │
           └──────────────────┘
```

## When to Use What?

| Criteria | RAG | CAG |
|----------|-----|-----|
| **Knowledge base size** | Large (millions of docs) | Small-medium (fits in context window) |
| **Update frequency** | Frequently updated data | Mostly static content |
| **Query latency** | Retrieval adds latency | Single LLM call |
| **Token cost** | Lower (only relevant chunks) | Higher (full context every query) |
| **Answer completeness** | May miss context across chunks | Access to all information |
| **Source attribution** | Easy (retrieved chunks are known) | Harder (full context, no retrieval) |
| **Setup complexity** | Needs vector DB, embeddings | Minimal setup |
| **Accuracy risk** | Retrieval may miss relevant info | Context window limit may truncate |

### Choose RAG when:
- Your knowledge base is large (>100K tokens)
- Documents are updated frequently
- You need source attribution (which document answered the query)
- Cost efficiency matters at scale

### Choose CAG when:
- Your knowledge base is small enough to fit in the context window
- You need holistic understanding across the entire corpus
- Simplicity and fast setup are priorities
- Queries require synthesizing information from multiple sections

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Groq API (Llama 3.1 8B) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| Framework | LangChain |
| Evaluation | LLM-as-Judge |
| Language | Python 3.11+ |

## Project Structure

```
rag-versus-cag/
├── main.py                      # CLI entry point
├── requirements.txt
├── .env.example
├── data/
│   └── sample_pdfs/             # Place your PDFs here
├── src/
│   ├── config.py                # Settings and environment
│   ├── document_loader.py       # PDF loading and processing
│   ├── rag/
│   │   └── pipeline.py          # RAG pipeline (chunk → embed → retrieve → answer)
│   ├── cag/
│   │   └── pipeline.py          # CAG pipeline (full context → answer)
│   └── evaluation/
│       ├── metrics.py           # LLM-as-Judge quality scoring
│       └── comparator.py        # Side-by-side comparison runner
└── results/                     # Comparison output
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Harshith69/rag-versus-cag.git
cd rag-versus-cag
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

### 5. Add PDF documents

Place your PDF files in `data/sample_pdfs/`.

## Usage

### Ingest documents (RAG)

```bash
python main.py ingest
```

### Query with RAG

```bash
python main.py query-rag "What are the key findings?"
```

### Query with CAG

```bash
python main.py query-cag "What are the key findings?"
```

### Run full comparison

```bash
python main.py compare
```

This runs a set of sample questions through both pipelines, evaluates quality using an LLM judge, and outputs a comparison table.

### Interactive mode

```bash
python main.py interactive
```

Ask questions and see both RAG and CAG answers side by side in real time.

## Evaluation Metrics

Each answer is scored on four dimensions (1-5 scale) using an LLM judge:

- **Relevance** — Does the answer address the question?
- **Completeness** — Does it cover all aspects?
- **Accuracy** — Is the information factually correct?
- **Conciseness** — Is it appropriately concise?

Additional metrics collected:
- **Latency** — Time to generate the answer
- **Chunks retrieved** (RAG) — Number of context chunks used
- **Context tokens** (CAG) — Approximate tokens in full context

## Sample Output

```
┌─────────────────────── RAG vs CAG Comparison Results ───────────────────────┐
│ Question              │ RAG Latency │ CAG Latency │ RAG Quality │ CAG Quality │ Winner │
├───────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┼────────┤
│ What is the main ...  │ 1.234s      │ 2.567s      │ 4.2/5       │ 4.5/5       │ CAG    │
│ What are the key ...  │ 1.100s      │ 2.890s      │ 3.8/5       │ 4.8/5       │ CAG    │
│ What methodology ...  │ 0.980s      │ 2.340s      │ 4.5/5       │ 4.0/5       │ RAG    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Key Takeaways

1. **CAG generally produces more complete answers** because it has access to the entire document context
2. **RAG is faster and cheaper per query** because it only processes a few relevant chunks
3. **RAG scales better** to large knowledge bases that exceed the context window
4. **CAG is simpler to implement** — no vector database, no embedding pipeline, no retrieval tuning
5. The right choice depends on your **knowledge base size**, **query patterns**, and **cost constraints**

## License

MIT
