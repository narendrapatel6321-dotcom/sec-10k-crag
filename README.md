# SEC 10-K Corrective RAG (CRAG) — Financial Filing Intelligence Engine

A self-correcting Retrieval-Augmented Generation system that answers qualitative and quantitative questions over SEC 10-K filings (Apple, Microsoft, Citigroup, Alphabet, Goldman Sachs), with automated numeric grounding against structured XBRL financial statements.

Built with **LangGraph**, **Pinecone**, **Groq (Llama 3.3 70B)**, and a hybrid dense + sparse retrieval stack.

---

## 1. Project Overview

10-K filings are among the hardest documents to build reliable RAG over: hundreds of pages of dense legal and financial prose, tightly-regulated numeric disclosures buried in tables, and a very low tolerance for hallucination — a wrong revenue figure or misquoted risk factor isn't a stylistic slip, it's a factual liability.

This system treats that as the core engineering problem, not an afterthought. Rather than a single retrieve-then-generate pass, it implements **Corrective RAG (CRAG)**: a LangGraph state machine that routes, retrieves, grades its own retrieved evidence, rewrites the query when evidence is weak, and — critically — **verifies every number the LLM generates against the filing's actual XBRL statements** before returning an answer, looping back to regenerate with corrective feedback if a figure can't be grounded.

The result is a financial analyst assistant where qualitative claims (risk factors, MD&A commentary) are retrieved and reranked for precision, and quantitative claims (revenue, CET1 ratio, net sales) are held to the same bar a human auditor would apply.

---

## 2. System Architecture

### Ingestion & Chunking
- **Source acquisition**: Raw 10-K filings are pulled via `sec-edgar-downloader` and `edgartools`, with SEC-compliant identity headers (company name + email) registered per EDGAR fair-access policy. Filings are fetched per ticker (`AAPL`, `MSFT`, `C`, `GOOGL`, `GS`), 3 most recent 10-Ks each, with a local cache check to avoid re-downloading.
- **Dual extraction pipeline** — structured data and prose are deliberately kept separate:
  - **Structured (XBRL)**: `filing.xbrl().statements` is used to pull the Income Statement, Balance Sheet, and Cash Flow Statement as clean pandas DataFrames, persisted as per-filing CSVs (`{ticker}_{accession}/income_statement.csv`, etc.) plus a consolidated pickle. This becomes the **ground-truth store** used later for numeric verification — it is never embedded or chunked.
  - **Prose**: Item 1A (Risk Factors) and Item 7 (MD&A) are extracted via `edgartools`' filing object model and cached to a separate pickle, keyed by `{ticker}_{accession}`.
- **Chunking strategy**: Prose sections are split with LangChain's `RecursiveCharacterTextSplitter` using **`chunk_size=1000`, `chunk_overlap=200`**, with a separator cascade of `["\n\n", "\n", ".", " ", ""]` — the overlap is sized to preserve cross-sentence financial context (e.g., a risk factor's cause and its stated impact) that a hard cutoff would otherwise sever. Each chunk retains metadata: `ticker`, `accession`, `section` (`risk_factors` / `mdna`), and `chunk_id`.

### Embedding & Vector Store
- **Embedding model**: `BAAI/bge-small-en-v1.5` (384-dim), run through `HuggingFaceEmbeddings` with **cosine-normalized embeddings**, GPU-accelerated at index-build time and falling back to CPU automatically at inference time for GPU-less deployment targets.
- **Dense store**: **Pinecone Serverless** (`aws`, `us-east-1`, cosine metric), chosen specifically over a local ChromaDB directory because the app is deployed to Streamlit Community Cloud, where large binary index files can't live in the deployment repo. Documents are upserted in batches of 100 via `PineconeVectorStore.add_documents`.
- **Sparse store**: A local `BM25Retriever` (via `rank_bm25` / `langchain_community`) is built over the same chunk set and serialized to a small `.pkl` committed to the repo — small enough to ship alongside code, unlike the dense index.
- **Hybrid retrieval**: Dense (Pinecone) and sparse (BM25) retrievers are combined via LangChain's `EnsembleRetriever` (default 0.5 / 0.5 weighting), with an optional Pinecone metadata filter (`{"ticker": {"$eq": ...}}`) applied when the router has identified a target company — narrowing a 15-company-wide index down to a single filing's chunks before scoring even begins.
- **Reranking**: The ensemble's top candidates are passed through a **`BAAI/bge-reranker-v2-m3` cross-encoder** (`sentence-transformers`), which re-scores each `[query, chunk]` pair directly rather than relying on embedding-space proximity, and the pipeline keeps only the **top 5** highest-scoring chunks for generation.

### Retrieval & Generation
- **LLM**: **Llama 3.3 70B Versatile via Groq** (`ChatGroq`), used consistently across every reasoning step in the graph — routing, document grading, query rewriting, and final generation — for low-latency structured inference.
- **Orchestration**: A **LangGraph `StateGraph`** implements the full CRAG loop:

  ```
  START → route_query → (retrieve | out_of_scope_generation)
                            ↓
                         retrieve → grade_documents → (generate | rewrite_query)
                                                            ↑___________|
                            ↓
                         generate → verify → (END | generate)
  ```

  - **`route_query`**: classifies the query into `sec_filing` / `general_financial` / `out_of_scope` and extracts the target ticker and section (`risk_factors` / `mdna`) via structured Pydantic output — done as a graph *node* (not a conditional-edge side effect) so the extracted routing metadata is guaranteed to persist in state for downstream nodes.
  - **`grade_documents`**: grades all retrieved chunks' relevance in a **single batched structured-output call** (one `BatchGraderOutput` covering the whole document list) rather than one LLM call per document — cutting both latency and Groq token spend. Ungraded indices fail *open* rather than being silently dropped.
  - **`rewrite_query`**: if grading yields no relevant documents, an LLM rewrites the query for better vector-retrieval semantics and loops back to `retrieve` (capped at `MAX_REWRITES = 2`).
  - **`generate`**: injects retrieved context with **explicit provenance tags** per chunk (e.g. `[AAPL | risk_factors | FY2024]`), instructing the model to cite the tag inline next to any claim or figure — turning "hallucination-prone free text" into a traceable, source-attributed answer. Any prior verification feedback is folded into the prompt on regeneration.
  - **`verify`**: extracts every numeric value from the generation via regex (handling comma-formatted numbers and accounting-style parenthesized negatives, e.g. `(1,234)` → `-1234.0`) and checks each one against the filing's actual XBRL ground truth, within a **5% tolerance** and across common **scale mismatches** (thousands / millions / billions) an LLM might introduce. Calendar-year-shaped integers (1990–2035) are excluded from numeric checks to avoid false failures on things like "fiscal 2024."
  - **Verification loop**: on failure, the graph loops back to `generate` with corrective feedback (capped at `MAX_VERIFY_RETRIES = 3`, tracked via a dedicated `verify_count` decoupled from the rewrite loop's `rewrite_count` so one loop's budget doesn't starve the other).
  - **`out_of_scope_generation`**: politely declines non-finance queries without ever touching retrieval.

---

## 3. Tech Stack

**Orchestration & Agents**
- LangGraph (`StateGraph`, conditional edges, cyclic corrective loops)
- LangChain Core / Community / Classic (`EnsembleRetriever`, prompt templates, structured output)

**LLM & Inference**
- Groq (Llama 3.3 70B Versatile) — routing, grading, rewriting, generation, LLM-as-judge evaluation

**Retrieval & Embeddings**
- Pinecone (Serverless, dense vector store)
- `rank_bm25` / `BM25Retriever` (local sparse index)
- HuggingFace `sentence-transformers` — `BAAI/bge-small-en-v1.5` (embeddings), `BAAI/bge-reranker-v2-m3` (cross-encoder reranker)

**Data Acquisition & Parsing**
- `edgartools` (XBRL statement parsing, filing object model)
- `sec-edgar-downloader` (raw EDGAR filing retrieval)
- pandas (structured financial statement handling)

**Application & Evaluation**
- Streamlit (interactive chat UI with live LangGraph execution trace)
- Pydantic (structured LLM outputs across every node)
- Custom benchmark harness with LLM-as-judge faithfulness scoring

---

## 4. Key Challenges Solved

**1. Hallucinated or mis-scaled financial figures.**
LLMs reliably get numbers plausible-sounding but wrong — especially on unit scale (reporting a figure in millions when the filing states it in thousands). The `verify` node cross-checks every generated number against the filing's actual XBRL statements with tolerance-based matching *and* explicit scale-correction checks (×1e3/1e6/1e9 and their inverses), then routes back into generation with concrete feedback rather than surfacing an ungrounded answer.

**2. Structured tabular data living apart from prose.**
Financial statements and risk/MD&A narrative are fundamentally different content types. Rather than force both into one embedding space, they're extracted through separate pipelines: XBRL statements stay as clean, queryable DataFrames/CSVs used purely as ground truth, while prose is chunked and embedded for semantic retrieval. The `verify` node is the bridge between the two.

**3. Retrieval precision on dense, repetitive financial jargon.**
Risk-factor sections in particular are long, boilerplate-heavy, and semantically similar across paragraphs. A single dense-retrieval pass isn't precise enough, so the system layers: hybrid dense+sparse ensembling → LLM relevance grading → a corrective rewrite loop → cross-encoder reranking down to the final top-5 chunks actually sent to the generator.

**4. Query ambiguity and cross-company contamination.**
With five companies indexed together, an ungated retrieval call risks pulling Citigroup risk factors into an Apple question. The router node extracts ticker/section intent up front and applies it as a hard Pinecone metadata filter, so retrieval is scoped to the right filing before ranking even starts.

**5. Answer traceability.**
Every context chunk carries a provenance tag (`ticker | section | fiscal year | accession`) injected directly into the generation prompt, and the model is instructed to cite it inline — so a claim in the final answer can always be traced back to a specific filing, not just "the context."

**6. Stateless deployment constraints.**
Streamlit Community Cloud has no persistent disk for a large local vector index and no GPU. The dense index was moved to hosted Pinecone specifically for this reason, while the small BM25 `.pkl` stays local; embedding and reranking model loading both auto-detect CUDA and fall back to CPU.

---

## 5. Example Output

**Query:** *"What was Apple's total net sales for the most recent fiscal year, and what supply chain risks does the filing disclose alongside it?"*

**LangGraph execution trace:**

```json
{
  "Route Category": "sec_filing",
  "Identified Ticker": "AAPL",
  "Target Section": "mdna",
  "Verification Result": "All generated numerical values successfully verified against XBRL ground truth.",
  "Total Retries": 0,
  "Retrieved Documents Count": 5,
  "Document Sources": [
    {"ticker": "AAPL", "accession": "0000320193-24-000123", "section": "mdna", "preview": "Total net sales increased during fiscal 2024 driven primarily by..."},
    {"ticker": "AAPL", "accession": "0000320193-24-000123", "section": "risk_factors", "preview": "The Company's reliance on single-source suppliers for certain..."}
  ]
}
```

**Generated answer:**

> Apple reported total net sales of approximately **$391.0 billion** for the most recent fiscal year *(AAPL, mdna, FY2024)*. Alongside this, the filing discloses that Apple depends on **single-source or limited-source suppliers** for several custom components and manufacturing processes, concentrated substantially outside the United States, which the Company states could disrupt production and adversely affect financial results if a key supplier were to fail to meet demand *(AAPL, risk_factors, FY2024)*. The filing also flags exposure to global logistics and geopolitical disruption as compounding factors on this supply chain concentration risk *(AAPL, risk_factors, FY2024)*.

The `$391.0 billion` figure only reaches the user because it passed the `verify` node's cross-check against Apple's actual XBRL income statement for that filing — a fabricated or mis-scaled number here would have triggered a regeneration loop with corrective feedback instead.

---

## 6. How to Run

**1. Clone and install dependencies**
```bash
git clone https://github.com/narendrapatel6321-dotcom/sec-10k-crag.git
cd sec-10k-crag
pip install -r requirements.txt
```

**2. Set required environment variables / secrets**
```bash
export GROQ_API_KEY="your-groq-api-key"
export PINECONE_API_KEY="your-pinecone-api-key"
export PINECONE_INDEX_NAME="sec-10k-rag"
export SEC_COMPANY_NAME="Your Company Name"
export SEC_EMAIL="you@example.com"
```

**3. Run the ingestion pipeline** (downloads filings, extracts XBRL + prose, builds the hybrid index)
```bash
python 10k_pinecone_pipeline.py
```
> Note: this step populates both the Pinecone index and the local `data/index/bm25_retriever.pkl` — the latter must exist on disk (and be committed if deploying) before the app or benchmark can run.

**4. Launch the Streamlit app**
```bash
streamlit run app.py
```

**5. (Optional) Run the automated benchmark**
```bash
python sec_10k_rag_evaluation.py
```
This evaluates routing accuracy, retrieval count, numeric accuracy against XBRL ground truth, and LLM-as-judge faithfulness across the golden test suite, exporting results to a timestamped CSV.
