"""
Hybrid Search Retriever Client

This module provides the connection to the persisted ChromaDB and BM25 index.
It configures and returns an EnsembleRetriever for hybrid (dense + sparse) search capabilities,
combining semantic search and keyword matching.
"""

import torch
import pickle
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings


def get_hybrid_retriever(
    index_dir: str | Path,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    top_k: int = 10,
    search_kwargs: Optional[dict] = None
) -> EnsembleRetriever:
    """
    Initializes and returns a hybrid EnsembleRetriever.

    Example:
        >>> from src.retrieval.chroma_client import get_hybrid_retriever
        >>> retriever = get_hybrid_retriever(index_dir="./data/index", top_k=5)
        >>> results = retriever.invoke("What are the primary cybersecurity risks?")

    Args:
        index_dir: Path to the directory containing 'chroma_db' and 'bm25_retriever.pkl'.
        dense_weight: Weight assigned to the semantic search results (ChromaDB). Defaults to 0.5.
        sparse_weight: Weight assigned to the exact keyword match results (BM25). Defaults to 0.5.
        top_k: The number of documents each individual retriever should return before ensembling.
        search_kwargs: Additional arguments for the Chroma vector store search (e.g., metadata filters).

    Returns:
        An EnsembleRetriever configured with the specified weights and search parameters.
    """
    index_path = Path(index_dir)
    chroma_path = str(index_path / "chroma_db")
    bm25_path = index_path / "bm25_retriever.pkl"

    if not index_path.exists():
        raise FileNotFoundError(f"Index directory not found at: {index_path}")

    # 1. Initialize Dense Retriever (ChromaDB) dynamically
    # Automatically uses GPU if available, otherwise falls back to CPU (essential for Streamlit Cloud)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    vectorstore = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings
    )
    
    default_search_kwargs = {"k": top_k}
    if search_kwargs:
        default_search_kwargs.update(search_kwargs)
        
    chroma_retriever = vectorstore.as_retriever(search_kwargs=default_search_kwargs)

    # 2. Initialize Sparse Retriever (BM25)
    with open(bm25_path, "rb") as f:
        bm25_retriever: BM25Retriever = pickle.load(f)
    
    bm25_retriever.k = top_k

    # 3. Combine into Ensemble Retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[dense_weight, sparse_weight]
    )

    return ensemble_retriever
