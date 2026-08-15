"""
LangGraph Workflow Orchestrator

Wires together the nodes and conditional edges to form the final CRAG state machine.
Handles query routing, relevance grading loops, and numerical verification fallbacks.
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END

from src.graph.state import AgentState
from src.graph.nodes import route_query, retrieve, grade_documents, generate, verify, rewrite_query
from src.graph.nodes import llm
from langchain_core.messages import HumanMessage

MAX_REWRITES = 2
MAX_VERIFY_RETRIES = 3

# --- Conditional Edge Functions ---
#
# NOTE: routing classification itself now happens in the `route_query` NODE
# (src/graph/nodes.py), not here. Conditional edge functions only choose which
# edge to follow; they are not a reliable place to attach state updates, since
# only a node's *returned* dict is guaranteed to be merged into the graph's
# state. The previous version assigned to `state["route_category"]` etc.
# directly inside this function, which meant `ticker` / `route_category` /
# `target_section` could silently fail to persist to the rest of the graph.
# This function now just reads state that `route_query` already wrote.

def route_after_classification(state: AgentState) -> Literal["retrieve", "out_of_scope_generation"]:
    """Routes to retrieval or the fallback generator based on the classification already in state."""
    print("---ROUTING QUERY---")
    category = state.get("route_category")

    if category == "sec_filing":
        print(f"  -> Routing to SEC 10-K Retrieval (Ticker: {state.get('ticker')})")
        return "retrieve"
    else:
        print(f"  -> Routing to standard generation (Category: {category})")
        return "out_of_scope_generation"


def check_relevance(state: AgentState) -> Literal["generate", "rewrite_query"]:
    """Determines whether to generate an answer or rewrite the query based on document grading."""
    print("---CHECKING RELEVANCE---")

    if state.get("rewrite_count", 0) >= MAX_REWRITES:
        print("  -> Max rewrite retries reached. Forcing generation.")
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

    if "Passed" in feedback or state.get("verify_count", 0) >= MAX_VERIFY_RETRIES:
        if state.get("verify_count", 0) >= MAX_VERIFY_RETRIES:
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
    workflow.add_node("route_query", route_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("verify", verify)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("out_of_scope_generation", out_of_scope_generation)

    # 3. Define the Flow (Edges)

    # Start -> Route (classification node, writes route_category/ticker/target_section to state)
    workflow.add_edge(START, "route_query")

    # Route -> Retrieve OR Fallback, based on the classification just written to state
    workflow.add_conditional_edges(
        "route_query",
        route_after_classification,
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
