"""
Hybrid Search Retriever Client (Pinecone Edition)

Replaces the local ChromaDB dense index with a hosted Pinecone index so the
retrieval layer works on stateless deployment targets like Streamlit
Community Cloud, where large binary index directories can't be committed to
Git. The sparse BM25 index remains a small local .pkl file and is combined
with Pinecone via an EnsembleRetriever.
"""

import os
import pickle
import torch
from pathlib import Path
from typing import Optional

from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone


def _build_pinecone_filter(search_kwargs: Optional[dict]) -> Optional[dict]:
    """Translates a simple {'ticker': 'AAPL'} filter into Pinecone's operator syntax."""
    if not search_kwargs or "filter" not in search_kwargs:
        return None

    raw_filter = search_kwargs["filter"]
    pinecone_filter = {}
    for key, value in raw_filter.items():
        # Already using an operator (e.g. {"$eq": ...}) -> pass through unchanged
        if isinstance(value, dict):
            pinecone_filter[key] = value
        else:
            pinecone_filter[key] = {"$eq": value}
    return pinecone_filter


def get_hybrid_retriever(
    index_dir: str | Path,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    top_k: int = 10,
    search_kwargs: Optional[dict] = None,
    index_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> EnsembleRetriever:
    """
    Initializes and returns a hybrid EnsembleRetriever backed by Pinecone (dense)
    and a local BM25Retriever (sparse).

    Example:
        >>> from src.retrieval.pinecone_client import get_hybrid_retriever
        >>> retriever = get_hybrid_retriever(index_dir="./data/index", top_k=5)
        >>> results = retriever.invoke("What are the primary cybersecurity risks?")

    Args:
        index_dir: Path to the directory containing 'bm25_retriever.pkl'.
        dense_weight: Weight assigned to the semantic search results (Pinecone). Defaults to 0.5.
        sparse_weight: Weight assigned to the exact keyword match results (BM25). Defaults to 0.5.
        top_k: The number of documents each individual retriever should return before ensembling.
        search_kwargs: Additional arguments for the Pinecone vector store search
            (e.g. {"filter": {"ticker": "AAPL"}}).
        index_name: Pinecone index name. Falls back to the PINECONE_INDEX_NAME env var.
        api_key: Pinecone API key. Falls back to the PINECONE_API_KEY env var
            (the Pinecone client also picks this up automatically if unset).

    Returns:
        An EnsembleRetriever configured with the specified weights and search parameters.
    """
    index_path = Path(index_dir)
    bm25_path = index_path / "bm25_retriever.pkl"

    resolved_index_name = index_name or os.environ.get("PINECONE_INDEX_NAME")
    if not resolved_index_name:
        raise ValueError(
            "No Pinecone index name provided. Pass index_name explicitly or set "
            "the PINECONE_INDEX_NAME environment variable."
        )

    if not bm25_path.exists():
        raise FileNotFoundError(f"BM25 index not found at: {bm25_path}")

    # 1. Initialize Dense Retriever (Pinecone)
    # Automatically uses GPU if available, otherwise falls back to CPU (essential for
    # Streamlit Community Cloud, which has no CUDA device).
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    pc_client = Pinecone(api_key=api_key or os.environ.get("PINECONE_API_KEY"))
    pinecone_index = pc_client.Index(resolved_index_name)

    vectorstore = PineconeVectorStore(index=pinecone_index, embedding=embeddings)

    default_search_kwargs = {"k": top_k}
    pinecone_filter = _build_pinecone_filter(search_kwargs)
    if pinecone_filter:
        default_search_kwargs["filter"] = pinecone_filter

    pinecone_retriever = vectorstore.as_retriever(search_kwargs=default_search_kwargs)

    # 2. Initialize Sparse Retriever (BM25) — still a small local .pkl, committed to Git
    with open(bm25_path, "rb") as f:
        bm25_retriever: BM25Retriever = pickle.load(f)

    bm25_retriever.k = top_k

    # 3. Combine into Ensemble Retriever
    # NOTE: imported from langchain_classic.retrievers, not langchain.retrievers,
    # since Streamlit Cloud installs LangChain 1.0+.
    ensemble_retriever = EnsembleRetriever(
        retrievers=[pinecone_retriever, bm25_retriever],
        weights=[dense_weight, sparse_weight],
    )

    return ensemble_retriever
