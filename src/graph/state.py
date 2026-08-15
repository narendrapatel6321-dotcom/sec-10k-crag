"""
LangGraph State Schema

Defines the state structure that flows through the SEC 10-K CRAG pipeline.
"""

from typing import TypedDict, List, Optional
from langchain_core.documents import Document


class AgentState(TypedDict):
    """
    Represents the internal state of the LangGraph execution.
    Each node in the graph will read from and write updates to this state.
    """
    # User Input
    question: str
    
    # Router Extracted Context
    route_category: Optional[str]
    ticker: Optional[str]
    target_section: Optional[str]
    
    # Retrieval Phase
    documents: List[Document]
    
    # Generation and Verification Phase
    generation: str
    verification_feedback: str
    
    # Orchestration Tracking
    # NOTE: kept separate so the query-rewrite loop (grading-driven) and the
    # numeric-correction loop (verification-driven) don't share one budget —
    # a query that needs 2 rewrites shouldn't leave verification with only
    # one shot left before check_verification force-halts.
    rewrite_count: int
    verify_count: int
    retry_count: int  # retained for backwards compatibility / total-attempts logging
