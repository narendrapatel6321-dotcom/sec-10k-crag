"""
LangGraph Nodes

This module defines the individual nodes (functions) that manipulate the AgentState.
It implements the Corrective RAG (CRAG) logic, including document grading, query rewriting,
and the specialized quantitative risk analytics verification step.
"""

import os
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from src.graph.state import AgentState
from src.graph.verifier import verify_numbers
from src.retrieval.pinecone_client import get_hybrid_retriever
from src.retrieval.reranker import get_reranker_model, rerank_documents
from src.retrieval.router import get_query_router

# Initialize LLM (Ensure GROQ_API_KEY is in your environment variables)
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)

# --- Module-level singletons ---
# The retriever (embeddings + Pinecone connection), reranker (cross-encoder), and
# router are expensive to construct and were previously being rebuilt on every
# single `retrieve` call — including every pass through the rewrite loop within
# one request. Load them once per process instead.
_retriever_cache: Dict[Any, Any] = {}
_reranker_model = None
_route_query_func = None


def _get_cached_retriever(ticker_filter: Dict[str, Any] | None):
    """Returns a hybrid retriever, cached per distinct filter shape.

    A plain (unfiltered) retriever and a per-ticker-filtered retriever are
    different EnsembleRetriever instances, so we key the cache on the filter
    itself rather than caching a single global instance.
    """
    cache_key = tuple(sorted(ticker_filter.items())) if ticker_filter else None
    if cache_key not in _retriever_cache:
        search_kwargs = {"filter": ticker_filter} if ticker_filter else {}
        _retriever_cache[cache_key] = get_hybrid_retriever(
            index_dir="./data/index",
            top_k=15,
            search_kwargs=search_kwargs,
            index_name=os.environ.get("PINECONE_INDEX_NAME"),
        )
    return _retriever_cache[cache_key]


def _get_cached_reranker():
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = get_reranker_model()
    return _reranker_model


def _get_cached_router():
    global _route_query_func
    if _route_query_func is None:
        _route_query_func = get_query_router()
    return _route_query_func


# --- Schema Definitions for Structured Output ---

class GraderOutput(BaseModel):
    """Binary score for relevance check."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class DocGrade(BaseModel):
    """Relevance grade for a single document, identified by its index."""
    doc_index: int = Field(description="Index of the document in the provided list, 0-based.")
    binary_score: str = Field(description="'yes' if relevant to the question, 'no' otherwise.")


class BatchGraderOutput(BaseModel):
    """Relevance grades for an entire batch of documents in a single call."""
    grades: List[DocGrade] = Field(description="One grade per input document, in any order.")


class RewriterOutput(BaseModel):
    """Rewritten query."""
    question: str = Field(
        description="The optimized and rewritten user query."
    )


# --- Node Functions ---

def route_query(state: AgentState) -> Dict[str, Any]:
    """
    Node: Classifies the query and extracts routing metadata (ticker, section).

    This is a plain node (not a conditional-edge function) specifically so its
    output goes through the normal state-merge path. Conditional edge functions
    only decide which edge to take — they aren't guaranteed to have in-place
    state mutations persisted, and this pipeline was previously assigning to
    `state[...]` directly inside the edge function, which silently failed to
    stick for `ticker` / `route_category` / `target_section` downstream.
    """
    print("---NODE: ROUTE QUERY---")
    router = _get_cached_router()
    decision = router(state["question"])

    print(f"  -> category={decision.category} ticker={decision.ticker} section={decision.section}")

    return {
        "route_category": decision.category,
        "ticker": decision.ticker,
        "target_section": decision.section,
    }


def retrieve(state: AgentState) -> Dict[str, Any]:
    """
    Node: Retrieves documents using hybrid search (Pinecone dense + local BM25 sparse)
    and reranks them.
    """
    print("---NODE: RETRIEVE---")
    question = state["question"]
    ticker = state.get("ticker")

    ticker_filter = {"ticker": ticker} if ticker else None

    retriever = _get_cached_retriever(ticker_filter)
    reranker = _get_cached_reranker()

    # 1. Hybrid Search
    initial_docs = retriever.invoke(question)

    # 2. Cross-Encoder Reranking
    ranked_docs = rerank_documents(question, initial_docs, reranker, top_k=5)

    return {"documents": ranked_docs}


def grade_documents(state: AgentState) -> Dict[str, Any]:
    """
    Node: Grades the retrieved documents in a single batched LLM call and filters
    out irrelevant ones.

    Previously this issued one Groq call per document (5+ calls per retrieve pass,
    more across rewrite loops). Batching into one structured-output call with a
    list schema cuts both latency and token cost.
    """
    print("---NODE: GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]

    if not documents:
        return {"documents": []}

    structured_llm = llm.with_structured_output(BatchGraderOutput)

    system = """You are a grader assessing relevance of retrieved documents to a user question.
For EACH numbered document below, decide if it contains keyword(s) or semantic meaning related
to the question. Return one grade per document (indices 0-based, matching the input order),
with a binary score of 'yes' or 'no'. You must return exactly one grade per document provided."""

    formatted_docs = "\n\n".join(
        f"[Document {i}]\n{doc.page_content}" for i, doc in enumerate(documents)
    )

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "User question: {question}\n\nDocuments:\n\n{documents}")
    ])

    batch_grader = grade_prompt | structured_llm
    result = batch_grader.invoke({"question": question, "documents": formatted_docs})

    relevant_indices = {
        g.doc_index for g in result.grades
        if g.binary_score.strip().lower() == "yes" and 0 <= g.doc_index < len(documents)
    }
    # Fail-open on indices the model didn't return, rather than silently dropping docs
    # it forgot to grade.
    graded_indices = {g.doc_index for g in result.grades}
    missing_indices = set(range(len(documents))) - graded_indices
    relevant_indices |= missing_indices

    relevant_docs = [documents[i] for i in sorted(relevant_indices)]

    return {"documents": relevant_docs}


def generate(state: AgentState) -> Dict[str, Any]:
    """
    Node: Generates the final answer using the relevant documents and any prior verification feedback.
    """
    print("---NODE: GENERATE---")
    question = state["question"]
    documents = state["documents"]
    feedback = state.get("verification_feedback", "")

    # Tag each chunk with its provenance (ticker / section / fiscal year) so the
    # model can produce citations that are actually traceable back to a filing,
    # instead of being asked to "cite sources" against undifferentiated text.
    context_parts = []
    for doc in documents:
        meta = doc.metadata or {}
        tag_bits = [
            meta.get("ticker"),
            meta.get("section"),
            f"FY{meta.get('fiscal_year')}" if meta.get("fiscal_year") else None,
            f"accession {meta.get('accession')}" if meta.get("accession") else None,
        ]
        tag = " | ".join(b for b in tag_bits if b)
        header = f"[{tag}]" if tag else "[source]"
        context_parts.append(f"{header}\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    system = """You are a financial analyst assistant answering questions based strictly on the provided SEC 10-K context.
    If you don't know the answer, just say that you don't know. Do not make up numbers.
    Each context chunk is prefixed with a source tag like [AAPL | risk_factors | FY2024]. Cite the
    relevant tag(s) inline (e.g. "(AAPL, risk_factors, FY2024)") next to claims and figures you draw from them.
    
    Previous Verification Feedback (if any, use this to correct your numbers):
    {feedback}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Context: {context} \n\n Question: {question}")
    ])

    rag_chain = prompt | llm
    response = rag_chain.invoke({"context": context, "question": question, "feedback": feedback})

    return {"generation": response.content}


def verify(state: AgentState) -> Dict[str, Any]:
    """
    Node: Verifies generated numbers against ground truth XBRL data.
    """
    print("---NODE: VERIFY NUMBERS---")
    generation = state["generation"]
    ticker = state.get("ticker")
    documents = state.get("documents", [])

    # If we don't have a specific ticker or it's a general question, skip verification
    if not ticker or state.get("route_category") != "sec_filing":
        return {"verification_feedback": "Passed (No numeric verification required for this query route)."}

    # Extract the accession number from the retrieved documents' metadata to locate the right XBRL tables
    accession = None
    if documents:
        accession = documents[0].metadata.get("accession")

    if not accession:
        return {"verification_feedback": "Passed (Could not determine accession number for XBRL verification)."}

    # Call the verifier logic
    is_valid, feedback = verify_numbers(
        generation=generation,
        ticker=ticker,
        accession=accession,
        xbrl_dir="./data/xbrl_financials"
    )

    # This loop's own counter — decoupled from rewrite_count so a query that
    # already used its rewrite budget on retrieval doesn't get starved of
    # numeric-correction attempts.
    current_verify_retries = state.get("verify_count", 0) + 1

    return {
        "verification_feedback": feedback if not is_valid else "Passed",
        "verify_count": current_verify_retries,
    }


def rewrite_query(state: AgentState) -> Dict[str, Any]:
    """
    Node: Rewrites the user query to optimize for better vector retrieval if grading failed.
    """
    print("---NODE: REWRITE QUERY---")
    question = state["question"]

    system = """You a question re-writer that converts an input user question to a better version that is optimized 
    for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question.")
    ])

    structured_llm = llm.with_structured_output(RewriterOutput)
    question_rewriter = re_write_prompt | structured_llm

    new_query = question_rewriter.invoke({"question": question})

    current_rewrite_retries = state.get("rewrite_count", 0) + 1

    return {"question": new_query.question, "rewrite_count": current_rewrite_retries}
