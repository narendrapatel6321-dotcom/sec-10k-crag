"""
SEC 10-K Corrective RAG (CRAG) Streamlit Dashboard

Provides an interactive user interface to query SEC 10-K filings with real-time
LangGraph execution traces, hybrid retrieval inspections, and XBRL ground-truth checks.
"""

import os
import streamlit as st
import pandas as pd
from pathlib import Path

# Streamlit Page Config
st.set_page_config(
    page_title="SEC 10-K CRAG Financial Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Financial Terminal Aesthetic
st.markdown("""
<style>
    .main-header { font-size: 1.8rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 0.95rem; color: #6B7280; margin-bottom: 1.5rem; }
    .status-badge {
        display: inline-block; padding: 2px 8px; border-radius: 4px;
        font-size: 0.8rem; font-weight: 600; background-color: #E0E7FF; color: #3730A3;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. Sidebar Setup & Key Management
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ System Config")
    
    # API Key Configuration
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Provide your Groq API key for Llama 3.3 70B inference."
    )
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key

    st.divider()
    st.subheader("🏢 Monitored Entities")
    st.markdown("""
    - **AAPL**: Apple Inc.
    - **C**: Citigroup Inc.
    - **GOOGL**: Alphabet Inc.
    - **GS**: Goldman Sachs Group
    - **MSFT**: Microsoft Corporation
    """)

    st.divider()
    st.subheader("🛠️ Pipeline Components")
    st.caption("• **Vector Store:** ChromaDB (Dense) + BM25 (Sparse)")
    st.caption("• **Embeddings:** BAAI/bge-small-en-v1.5")
    st.caption("• **Reranker:** BAAI/bge-reranker-v2-m3")
    st.caption("• **Orchestration:** LangGraph CRAG State Machine")
    st.caption("• **Verifier:** Float tolerance against XBRL Statements")

# -----------------------------------------------------------------------------
# 2. Main Interface Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">SEC 10-K Quantitative & Qualitative Risk Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Self-reflective financial query engine with automated XBRL numeric verification</div>', unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "trace" in message:
            with st.expander("🔍 View LangGraph Execution Trace & Grounding"):
                st.json(message["trace"])

# -----------------------------------------------------------------------------
# 3. Query Execution & Graph Invocation
# -----------------------------------------------------------------------------
query = st.chat_input("Ask a question about risks, MD&A, or financials (e.g., 'What were Microsoft's key risk factors in 2024?')...")

if query:
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Please enter a valid Groq API Key in the sidebar to proceed.")
        st.stop()

    # Append and render user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Execute LangGraph Pipeline
    with st.chat_message("assistant"):
        status_placeholder = st.status("Initializing CRAG pipeline...", expanded=True)
        
        try:
            from src.graph.workflow import crag_app

            status_placeholder.write("🧭 Routing query and evaluating intent...")
            
            # Initial state setup
            initial_state = {
                "question": query,
                "route_category": None,
                "ticker": None,
                "target_section": None,
                "documents": [],
                "generation": "",
                "verification_feedback": "",
                "retry_count": 0
            }
            
            status_placeholder.write("⚙️ Executing retrieval, grading, and verification nodes...")
            final_state = crag_app.invoke(initial_state)
            
            status_placeholder.update(label="Pipeline execution completed!", state="complete", expanded=False)
            
            # Render response
            generation = final_state.get("generation", "No response generated.")
            st.markdown(generation)
            
            # Prepare Trace Data
            trace_data = {
                "Route Category": final_state.get("route_category"),
                "Identified Ticker": final_state.get("ticker"),
                "Target Section": final_state.get("target_section"),
                "Verification Result": final_state.get("verification_feedback"),
                "Total Retries": final_state.get("retry_count"),
                "Retrieved Documents Count": len(final_state.get("documents", [])),
                "Document Sources": [
                    {
                        "ticker": doc.metadata.get("ticker"),
                        "accession": doc.metadata.get("accession"),
                        "section": doc.metadata.get("section"),
                        "preview": doc.page_content[:200] + "..."
                    }
                    for doc in final_state.get("documents", [])
                ]
            }
            
            with st.expander("🔍 View LangGraph Execution Trace & Grounding"):
                st.json(trace_data)
                
            # Persist message history
            st.session_state.messages.append({
                "role": "assistant",
                "content": generation,
                "trace": trace_data
            })
            
        except Exception as e:
            status_placeholder.update(label="Execution Failed", state="error", expanded=True)
            st.error(f"Error during graph execution: {e}")
