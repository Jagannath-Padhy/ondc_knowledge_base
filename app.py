#!/usr/bin/env python3
"""
ONDC Knowledge Base POC - Streamlit Frontend
Interactive interface for querying ONDC documentation
"""

import streamlit as st
import requests
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# API configuration
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="ONDC Knowledge Base Assistant",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "contexts" not in st.session_state:
    st.session_state.contexts = []


def call_api(endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict[str, Any]:
    """Make API call to FastAPI backend"""
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return {}


def check_health() -> Dict[str, bool]:
    """Check system health via API"""
    result = call_api("/health")
    return result.get("services", {}) if result else {}


def get_stats() -> Dict[str, Any]:
    """Get knowledge base statistics via API"""
    return call_api("/stats")


def query_knowledge_base(question: str, domain: str = "retail", limit: int = 5) -> List[Dict[str, Any]]:
    """Query the knowledge base via API"""
    data = {
        "question": question,
        "domain": domain,
        "limit": limit
    }
    result = call_api("/query", method="POST", data=data)
    return result.get("results", []) if result else []


def chat_with_knowledge(question: str, domain: str = "retail", include_context: bool = True) -> Dict[str, Any]:
    """Chat with the knowledge base via API"""
    data = {
        "question": question,
        "domain": domain,
        "include_context": include_context
    }
    return call_api("/chat", method="POST", data=data)

# Header
st.title("🔍 ONDC Knowledge Base Assistant")
st.markdown("Ask questions about ONDC API specifications and get instant answers powered by AI")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Status
    with st.expander("API Status", expanded=True):
        if st.button("Check Health"):
            health = check_health()
            if health:
                for service, status in health.items():
                    if status:
                        st.success(f"✅ {service}")
                    else:
                        st.error(f"❌ {service}")
            else:
                st.error("Failed to connect to API")
    
    # Statistics
    with st.expander("Knowledge Base Stats"):
        if st.button("Get Statistics"):
            stats = get_stats()
            if stats and "error" not in stats:
                st.metric("Total Documents", stats.get("total_documents", 0))
                st.metric("Vector Size", stats.get("vector_size", 0))
                st.metric("Status", stats.get("status", "Unknown"))
            else:
                st.error("Failed to get statistics")
    
    # Search Settings
    st.header("🔧 Search Settings")
    
    domain = st.selectbox(
        "Domain",
        ["retail", "logistics", "services"],
        index=0,
        help="Select the ONDC domain to search in"
    )
    
    search_limit = st.slider(
        "Number of Results",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of search results to retrieve"
    )
    
    include_sources = st.checkbox(
        "Show Sources",
        value=True,
        help="Display source documents with answers"
    )

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "🔍 Search", "📚 Ingestion"])

# Chat Tab
with tab1:
    st.header("Chat with ONDC Knowledge Base")
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📄 Sources"):
                    for source in message["sources"]:
                        st.write(f"**{source['title']}** (Score: {source.get('score', 0):.3f})")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about ONDC..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_with_knowledge(prompt, domain, include_sources)
                
                if response and "answer" in response:
                    st.markdown(response["answer"])
                    
                    # Add to messages
                    message_data = {
                        "role": "assistant",
                        "content": response["answer"]
                    }
                    
                    if include_sources and "sources" in response:
                        message_data["sources"] = response["sources"]
                        with st.expander("📄 Sources"):
                            for source in response["sources"]:
                                st.write(f"**{source['title']}** (Score: {source.get('score', 0):.3f})")
                    
                    st.session_state.messages.append(message_data)
                else:
                    error_msg = "I couldn't process your question. Please try again."
                    st.markdown(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

# Search Tab
with tab2:
    st.header("Search ONDC Documentation")
    
    # Search form
    with st.form("search_form"):
        search_query = st.text_input(
            "Enter your search query",
            placeholder="e.g., What is the order cancellation process?"
        )
        
        col1, col2 = st.columns([3, 1])
        with col2:
            search_button = st.form_submit_button("🔍 Search", use_container_width=True)
    
    # Display search results
    if search_button and search_query:
        with st.spinner("Searching..."):
            results = query_knowledge_base(search_query, domain, search_limit)
            
            if results:
                st.success(f"Found {len(results)} results")
                
                for i, result in enumerate(results, 1):
                    with st.expander(f"Result {i} - Score: {result['score']:.3f}"):
                        # Metadata
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Source:** {result['metadata'].get('doc_title', 'Unknown')}")
                            st.write(f"**Type:** {result['metadata'].get('chunk_type', 'Unknown')}")
                        with col2:
                            st.write(f"**Section:** {result['metadata'].get('section', 'N/A')}")
                            st.write(f"**Version:** {result['metadata'].get('version', 'N/A')}")
                        
                        # Content
                        st.markdown("**Content:**")
                        st.text(result['content'][:1000] + "..." if len(result['content']) > 1000 else result['content'])
            else:
                st.warning("No results found. Try a different query.")

# Ingestion Tab
with tab3:
    st.header("Document Ingestion")
    st.info("Use the API endpoint `/ingest` to add new documents to the knowledge base.")
    
    # Show example curl command
    st.subheader("Example: Ingest from Google Docs")
    st.code("""
curl -X POST "http://localhost:8000/ingest" \\
  -H "Content-Type: application/json" \\
  -d '{
    "sources": [
      {
        "type": "google_docs",
        "url": "https://docs.google.com/document/d/YOUR_DOC_ID",
        "options": {"follow_links": true}
      }
    ],
    "domain": "retail",
    "version": "1.2.0",
    "recreate": false
  }'
    """, language="bash")
    
    st.subheader("Example: Ingest from GitHub")
    st.code("""
curl -X POST "http://localhost:8000/ingest" \\
  -H "Content-Type: application/json" \\
  -d '{
    "sources": [
      {
        "type": "github_markdown",
        "url": "https://github.com/ONDC-Official/ONDC-Protocol-Specs",
        "options": {
          "is_repository": true,
          "doc_paths": ["docs", "api-specs"]
        }
      }
    ],
    "domain": "retail",
    "version": "1.2.0"
  }'
    """, language="bash")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ for ONDC")
st.sidebar.markdown("[View API Docs](http://localhost:8000/docs)")