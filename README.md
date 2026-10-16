# RAG vs CAG: A Comparative Study

A hands-on comparison of **Retrieval Augmented Generation (RAG)** and **Cache Augmented Generation (CAG)** for question-answering over PDF documents. Both pipelines are built end-to-end, evaluated on the same queries using an LLM-as-judge, and benchmarked on latency, answer quality, and token efficiency.

---

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

---

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

---

## Real Benchmark Results

Tested on a 5-page research paper: *"Artificial Intelligence in Banking and Finance"* (~6,500 tokens), with 7 domain-specific questions evaluated by an LLM judge (1-5 scale on relevance, completeness, accuracy, conciseness).

### Summary Table

| Question | RAG Latency | CAG Latency | RAG Quality | CAG Quality | Winner |
|----------|-------------|-------------|-------------|-------------|--------|
| Main applications of AI in banking? | 0.51s | 1.47s | 4.8/5 | 5.0/5 | **CAG** |
| ML for fraud detection? | 3.39s | 50.14s | 4.8/5 | 4.8/5 | **Tie** |
| Risks of AI in financial services? | 3.58s | 50.37s | 4.8/5 | 5.0/5 | **CAG** |
| AI improving customer experience? | 3.58s | 50.76s | 4.8/5 | 5.0/5 | **CAG** |
| Regulatory considerations? | 3.76s | 49.60s | 5.0/5 | 4.8/5 | **RAG** |
| Role of NLP in finance? | 3.44s | 50.04s | 5.0/5 | 5.0/5 | **Tie** |
| Chatbots transforming banking? | 3.64s | 49.47s | 4.8/5 | 4.8/5 | **Tie** |

### Aggregate Metrics

| Metric | RAG | CAG |
|--------|-----|-----|
| **Average Latency** | **3.13s** | 43.12s |
| **Average Quality** | 4.82/5 | **4.89/5** |
| **Context Tokens per Query** | ~500 (4 chunks) | ~6,572 (full doc) |
| **Wins** | 1 | 3 |
| **Ties** | 3 | 3 |

### Key Observations

1. **CAG produces more complete answers** — with access to the full document, CAG consistently identified more details, real-world examples (Citibank, JPMorgan COIN, Wells Fargo's Erika), and cross-section context that RAG's chunk retrieval missed.

2. **RAG is 14x faster** — averaging 3.13s vs 43.12s per query. RAG sends only ~500 tokens of context while CAG sends the full ~6,500 tokens every time.

3. **RAG can miss relevant context** — on the NLP question, RAG retrieved chunks that didn't mention NLP at all and correctly said "the context does not contain information about NLP." CAG found the answer in a different section of the document that RAG's retrieval didn't surface.

4. **RAG wins on focused, well-localized questions** — when the answer lives in a single section (like regulatory considerations), RAG retrieves exactly the right chunk and gives a focused answer.

5. **Quality gap is small but consistent** — CAG scored 4.89/5 vs RAG's 4.82/5. The difference shows up mainly in completeness, where CAG's full-context access lets it synthesize across sections.

---

## Pros and Cons

### RAG — Retrieval Augmented Generation

| Pros | Cons |
|------|------|
| Fast response times (retrieves only relevant chunks) | May miss relevant context in non-retrieved chunks |
| Low token cost per query | Requires embedding pipeline + vector DB setup |
| Scales to massive knowledge bases (millions of docs) | Chunk quality heavily affects answer quality |
| Easy source attribution (know exactly which chunks were used) | Multi-hop reasoning across distant sections is weak |
| Supports incremental updates (re-embed changed docs only) | Retrieval accuracy depends on embedding model quality |
| Works within any model's context window | Tuning chunk size, overlap, and top-K requires experimentation |

### CAG — Cache Augmented Generation

| Pros | Cons |
|------|------|
| More complete answers (model sees everything) | High token cost per query (full KB sent every time) |
| No retrieval errors — nothing gets missed | Limited by model's context window (can't handle huge KBs) |
| Simpler architecture (no vector DB, no embeddings) | Slower for large documents (more tokens to process) |
| Excellent for cross-section synthesis and multi-hop reasoning | No inherent source attribution |
| Zero setup time — just load the text | Every query re-processes the entire context |
| No tuning needed (no chunk size, overlap, or top-K to optimize) | Cost grows linearly with knowledge base size |

---

## When to Use Each Approach

### Use RAG When:

- **Knowledge base is large** — hundreds or thousands of documents exceeding the context window
- **Documents update frequently** — only re-embed changed documents, no need to reload everything
- **Cost matters at scale** — sending only relevant chunks keeps per-query token costs low
- **You need source citations** — retrieved chunks provide direct traceability to source documents
- **Questions are specific and localized** — answers likely exist in a single section or document
- **Production workloads** — high query volume where latency and cost per query compound

### Use CAG When:

- **Knowledge base fits in context** — small to medium document sets (under ~100K tokens)
- **Questions require cross-document reasoning** — answers span multiple sections or need holistic understanding
- **Simplicity is a priority** — no infrastructure to manage (no vector DB, no embedding pipeline)
- **Accuracy is critical over speed** — when missing context is more costly than higher latency
- **Prototyping or exploration** — quick setup for testing and validating an idea
- **Static content** — documentation, policies, or manuals that rarely change

### Avoid RAG When:

- Your knowledge base is small enough to fit in context (CAG will be simpler and more complete)
- Questions routinely require synthesizing information from distant parts of the corpus
- You don't have time to tune chunk size, overlap, embedding model, and retrieval parameters

### Avoid CAG When:

- Your knowledge base exceeds the model's context window
- You're running high-volume production queries where token cost adds up
- You need sub-second response times
- Documents change frequently and you need incremental updates without reloading everything

---

## Decision Flowchart

```
Does your knowledge base fit in the model's context window?
├── NO → Use RAG
└── YES
    ├── Do questions require cross-section reasoning? → Use CAG
    ├── Is per-query cost a concern at scale? → Use RAG
    ├── Do you need source attribution? → Use RAG
    ├── Is setup simplicity the priority? → Use CAG
    └── Default → Start with CAG, switch to RAG when you outgrow it
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Groq API (Qwen 3.8 27B) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| Framework | LangChain |
| Evaluation | LLM-as-Judge (4-dimension scoring) |
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
└── results/                     # Comparison output (JSON)
```

---

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

---

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

Runs a set of sample questions through both pipelines, evaluates quality using an LLM judge, and outputs a comparison table.

### Interactive mode

```bash
python main.py interactive
```

Ask questions and see both RAG and CAG answers side by side in real time.

---

## Evaluation Methodology

Each answer is scored on four dimensions (1-5 scale) using an LLM-as-judge:

| Dimension | What It Measures |
|-----------|-----------------|
| **Relevance** | Does the answer address the question asked? |
| **Completeness** | Does it cover all aspects of the question? |
| **Accuracy** | Is the information factually correct vs. the source? |
| **Conciseness** | Is it appropriately brief without losing information? |

Additional metrics:
- **Latency** — end-to-end time from question to answer
- **Chunks retrieved** (RAG) — number of context chunks used
- **Context tokens** (CAG) — approximate tokens in full context

---

## Configuration

Edit `src/config.py` to customize:

```python
GROQ_MODEL = "qwen/qwen3.8-27b"                        # Groq model ID
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
CHUNK_SIZE = 500                                         # Characters per RAG chunk
CHUNK_OVERLAP = 50                                       # Overlap between chunks
RETRIEVAL_TOP_K = 4                                      # Number of chunks to retrieve
```

---

## License

MIT
