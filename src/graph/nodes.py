"""
LangGraph Nodes

This module defines the individual nodes (functions) that manipulate the AgentState.
It implements the Corrective RAG (CRAG) logic, including document grading, query rewriting,
and the specialized quantitative risk analytics verification step.
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from src.graph.state import AgentState
from src.graph.verifier import verify_numbers
from src.retrieval.chroma_client import get_hybrid_retriever
from src.retrieval.reranker import get_reranker_model, rerank_documents

# Initialize LLM (Ensure GROQ_API_KEY is in your environment variables)
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)

# --- Schema Definitions for Structured Output ---

class GraderOutput(BaseModel):
    """Binary score for relevance check."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class RewriterOutput(BaseModel):
    """Rewritten query."""
    question: str = Field(
        description="The optimized and rewritten user query."
    )

# --- Node Functions ---

def retrieve(state: AgentState) -> Dict[str, Any]:
    """
    Node: Retrieves documents using hybrid search and reranks them.
    """
    print("---NODE: RETRIEVE---")
    question = state["question"]
    ticker = state.get("ticker")
    
    # Optional metadata filtering based on the router
    search_kwargs = {}
    if ticker:
        search_kwargs["filter"] = {"ticker": ticker}
        
    # In a production environment, you might load this once globally or pass via config
    retriever = get_hybrid_retriever(index_dir="./data/index", top_k=15, search_kwargs=search_kwargs)
    reranker = get_reranker_model()
    
    # 1. Hybrid Search
    initial_docs = retriever.invoke(question)
    
    # 2. Cross-Encoder Reranking
    ranked_docs = rerank_documents(question, initial_docs, reranker, top_k=5)
    
    return {"documents": ranked_docs, "retry_count": state.get("retry_count", 0)}


def grade_documents(state: AgentState) -> Dict[str, Any]:
    """
    Node: Grades the retrieved documents. Filters out irrelevant ones.
    """
    print("---NODE: GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    
    structured_llm = llm.with_structured_output(GraderOutput)
    
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
    ])
    
    retrieval_grader = grade_prompt | structured_llm
    
    relevant_docs = []
    for doc in documents:
        score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        if score.binary_score == "yes":
            relevant_docs.append(doc)
            
    return {"documents": relevant_docs}


def generate(state: AgentState) -> Dict[str, Any]:
    """
    Node: Generates the final answer using the relevant documents and any prior verification feedback.
    """
    print("---NODE: GENERATE---")
    question = state["question"]
    documents = state["documents"]
    feedback = state.get("verification_feedback", "")
    
    context = "\n\n".join(doc.page_content for doc in documents)
    
    system = """You are a financial analyst assistant answering questions based strictly on the provided SEC 10-K context.
    If you don't know the answer, just say that you don't know. Do not make up numbers.
    Always cite your sources based on the context provided.
    
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
    
    # Increment retry count to prevent infinite loops if verification keeps failing
    current_retries = state.get("retry_count", 0) + 1
    
    return {"verification_feedback": feedback if not is_valid else "Passed", "retry_count": current_retries}


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
    
    current_retries = state.get("retry_count", 0) + 1
    
    return {"question": new_query.question, "retry_count": current_retries}
