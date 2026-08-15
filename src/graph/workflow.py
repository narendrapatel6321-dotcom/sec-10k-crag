"""
LangGraph Workflow Orchestrator

Wires together the nodes and conditional edges to form the final CRAG state machine.
Handles query routing, relevance grading loops, and numerical verification fallbacks.
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END

from src.graph.state import AgentState
from src.graph.nodes import retrieve, grade_documents, generate, verify, rewrite_query
from src.retrieval.router import get_query_router

from src.graph.nodes import llm
from langchain_core.messages import HumanMessage

# Initialize the Groq-powered query router
route_query_func = get_query_router()

# --- Conditional Edge Functions ---

def route_initial_query(state: AgentState) -> Literal["retrieve", "out_of_scope_generation"]:
    """Routes the query based on the initial Groq classification."""
    print("---ROUTING QUERY---")
    decision = route_query_func(state["question"])
    
    # Store routing metadata in state
    state["route_category"] = decision.category
    state["ticker"] = decision.ticker
    state["target_section"] = decision.section
    
    if decision.category == "sec_filing":
        print(f"  -> Routing to SEC 10-K Retrieval (Ticker: {decision.ticker})")
        return "retrieve"
    else:
        print(f"  -> Routing to standard generation (Category: {decision.category})")
        return "out_of_scope_generation"

def check_relevance(state: AgentState) -> Literal["generate", "rewrite_query"]:
    """Determines whether to generate an answer or rewrite the query based on document grading."""
    print("---CHECKING RELEVANCE---")
    
    if state["retry_count"] >= 2:
        print("  -> Max retries reached. Forcing generation.")
        return "generate"
        
    if not state.get("documents"):
        print("  -> No relevant documents found. Rewriting query.")
        return "rewrite_query"
        
    print("  -> Documents are relevant. Proceeding to generation.")
    return "generate"

def check_verification(state: AgentState) -> Literal["END", "generate"]:
    """Determines whether the graph is finished or needs to regenerate due to bad numbers."""
    print("---CHECKING VERIFICATION---")
    feedback = state.get("verification_feedback", "")
    
    if "Passed" in feedback or state["retry_count"] >= 3:
        if state["retry_count"] >= 3:
            print("  -> Max verification retries reached. Halting.")
        else:
            print("  -> Verification passed. Ending workflow.")
        return "END"
        
    print("  -> Verification failed. Looping back to generator with feedback.")
    return "generate"


# --- Out of Scope / General Finance Fallback Node ---

def out_of_scope_generation(state: AgentState) -> dict:
    prompt = (
        f"The user asked: '{state['question']}'. "
        "You are an SEC 10-K financial analysis assistant. Politely decline "
        "to answer any questions unrelated to finance, corporate disclosures, or SEC filings."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"generation": response.content, "verification_feedback": "Passed (No RAG)."} 
    
# --- Graph Construction ---

def build_graph() -> StateGraph:
    """Assembles and compiles the LangGraph StateMachine."""
    # 1. Initialize Graph with State Schema
    workflow = StateGraph(AgentState)
    
    # 2. Add Nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("verify", verify)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("out_of_scope_generation", out_of_scope_generation)
    
    # 3. Define the Flow (Edges)
    
    # Start -> Router
    workflow.add_conditional_edges(
        START,
        route_initial_query,
        {
            "retrieve": "retrieve",
            "out_of_scope_generation": "out_of_scope_generation"
        }
    )
    
    # Fallback -> END
    workflow.add_edge("out_of_scope_generation", END)
    
    # Retrieve -> Grade
    workflow.add_edge("retrieve", "grade_documents")
    
    # Grade -> Generate OR Rewrite
    workflow.add_conditional_edges(
        "grade_documents",
        check_relevance,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query"
        }
    )
    
    # Rewrite -> Retrieve (Loop back)
    workflow.add_edge("rewrite_query", "retrieve")
    
    # Generate -> Verify
    workflow.add_edge("generate", "verify")
    
    # Verify -> END OR Generate (Correction loop)
    workflow.add_conditional_edges(
        "verify",
        check_verification,
        {
            "END": END,
            "generate": "generate"
        }
    )
    
    # 4. Compile the application
    app = workflow.compile()
    
    return app

# Expose the compiled app
crag_app = build_graph()
