"""
Cross-Encoder Reranker

This module applies the BAAI/bge-reranker-v2-m3 cross-encoder to re-score
and re-rank the documents retrieved by the hybrid search pipeline, drastically
improving the precision of the context window.
"""

from typing import List, Optional
import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


def get_reranker_model(
    model_name: str = "BAAI/bge-reranker-v2-m3",
    device: Optional[str] = None
) -> CrossEncoder:
    """
    Loads and returns the cross-encoder reranking model.
    Dynamically falls back to CPU if a GPU (CUDA) is not available.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Reranker: {model_name} on {device}...")
    return CrossEncoder(model_name, device=device)


def rerank_documents(
    query: str,
    documents: List[Document],
    reranker: CrossEncoder,
    top_k: int = 5
) -> List[Document]:
    """
    Reranks a list of documents against the user query using the cross-encoder.

    Example:
        >>> reranker_model = get_reranker_model()
        >>> top_docs = rerank_documents("What are the credit risks?", retrieved_docs, reranker_model)

    Args:
        query: The user's specific question.
        documents: The initial list of LangChain Documents from the hybrid retriever.
        reranker: The loaded CrossEncoder model instance.
        top_k: The final number of highest-scoring documents to return.

    Returns:
        A list of the top_k Documents, sorted by cross-encoder relevance score.
    """
    if not documents:
        return []

    # Format input for the cross-encoder: List of [query, text] pairs
    pairs = [[query, doc.page_content] for doc in documents]

    # Generate relevance scores
    scores = reranker.predict(pairs)

    # Pair documents with their scores and sort descending
    doc_score_pairs = list(zip(documents, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

    # Extract and return the top_k sorted documents
    best_docs = [doc for doc, score in doc_score_pairs[:top_k]]

    return best_docs
