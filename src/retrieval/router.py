"""
Query Router & Intent Classifier

Classifies user queries into routing categories (SEC filing retrieval, 
general finance explanation, or out-of-scope) and extracts target metadata 
(ticker, year) to constrain downstream search.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


class RouteDecision(BaseModel):
    """Routing classification schema and extracted metadata."""
    category: Literal["sec_filing", "general_financial", "out_of_scope"] = Field(
        ...,
        description="The routing path for the query based on content and intent."
    )
    ticker: Optional[str] = Field(
        default=None,
        description="Ticker symbol if specified or inferred from company name (e.g. AAPL, C, GOOGL, GS, MSFT)."
    )
    section: Optional[Literal["risk_factors", "mdna"]] = Field(
        default=None,
        description="Target 10-K section if query explicitly targets risk factors or MD&A / management discussion."
    )
    reasoning: str = Field(
        ...,
        description="Brief justification for the chosen route."
    )


ROUTER_SYSTEM_PROMPT = """You are an expert query router for a financial analysis system specializing in SEC 10-K filings.

Classify the user's query into one of three routes:
1. 'sec_filing': The question asks about specific company metrics, performance, risks, disclosures, or comparisons involving Apple (AAPL), Citigroup (C), Alphabet/Google (GOOGL), Goldman Sachs (GS), or Microsoft (MSFT).
2. 'general_financial': The question asks about financial concepts, definitions, formulas, or standard accounting practices without needing specific 10-K document retrieval (e.g., "What is Basel III CET1 ratio?", "Define operating cash flow").
3. 'out_of_scope': The query is chitchat, coding, completely unrelated to finance, or asks about unsupported topics.

Extract any mentioned or implied ticker from: AAPL, C, GOOGL, GS, MSFT.
Extract target section ('risk_factors' or 'mdna') if relevant."""


def get_query_router(api_key: Optional[str] = None, model_name: str = "llama-3.3-70b-versatile") -> callable:
    """
    Initializes and returns a callable query routing function.

    Example:
        >>> route_query = get_query_router()
        >>> decision = route_query("What were Microsoft's major risk factors in 2024?")
        >>> print(decision.category, decision.ticker)
        sec_filing MSFT
    """
    llm = ChatGroq(
        model_name=model_name,
        temperature=0.0,
        api_key=api_key
    )
    
    structured_llm = llm.with_structured_output(RouteDecision)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_SYSTEM_PROMPT),
        ("human", "{query}")
    ])
    
    chain = prompt | structured_llm

    def route(query: str) -> RouteDecision:
        return chain.invoke({"query": query})

    return route
