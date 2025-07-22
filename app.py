"""
Enhanced ONDC Knowledge Base Query Interface with Source References
Supports comprehensive querying with detailed source attribution
"""

import streamlit as st
import json
from typing import Dict, List, Any
from src.rag.enhanced_rag_chain import EnhancedONDCRAGChain
from src.utils.config import Config
from loguru import logger

# Initialize configuration and RAG chain
@st.cache_resource
def init_rag_chain():
    """Initialize the enhanced RAG chain"""
    config = Config()
    return EnhancedONDCRAGChain(config)

def format_response_with_references(response: dict) -> str:
    """Format the RAG response with proper source references"""
    if not response:
        return "No response generated."
    
    # Extract components
    answer = response.get('answer', 'No answer available')
    references = response.get('references', [])
    
    # Check if answer contains JSON examples
    import re
    has_json = bool(re.search(r'```json', answer))
    
    # Build formatted response
    formatted = f"{answer}\n\n"
    
    # If no JSON examples in answer, try to extract from raw results
    if not has_json and response.get('raw_results'):
        json_examples = []
        for result in response.get('raw_results', [])[:5]:
            content = result.get('content', '')
            # Look for JSON in content
            json_matches = re.findall(r'```json\s*(.+?)\s*```', content, re.DOTALL)
            for match in json_matches:
                try:
                    import json
                    parsed = json.loads(match)
                    json_examples.append(parsed)
                except:
                    pass
        
        if json_examples:
            formatted += "\n\n## Additional JSON Examples Found:\n\n"
            for i, example in enumerate(json_examples[:3]):
                formatted += f"### Example {i+1}:\n```json\n"
                formatted += json.dumps(example, indent=2)
                formatted += "\n```\n\n"
    
    # Add references section
    if references:
        formatted += "---\n\n**📚 References:**\n"
        for ref in references:
            ref_num = ref.get('number', '')
            ref_text = ref.get('reference', '')
            ref_type = ref.get('type', '')
            score = ref.get('score', 0)
            
            # Add emoji based on type
            emoji = {
                'api_endpoint': '🔌',
                'schema': '📐',
                'example': '💡',
                'json_example': '📋',
                'section': '📄',
                'enumeration': '🔢',
                'enumeration_group': '🔢',
                'payload': '📦'
            }.get(ref_type, '📄')
            
            formatted += f"\n[{ref_num}] {emoji} {ref_text} (relevance: {score:.2f})\n"
    
    return formatted

def display_search_filters(col):
    """Display search filter options"""
    with col.expander("🔧 Advanced Search Options", expanded=False):
        filters = {}
        
        # Document type filter
        doc_type = st.selectbox(
            "Document Type",
            ["All", "API Endpoints", "Schemas", "Examples", "Enumerations", "Sections"],
            help="Filter results by document type"
        )
        if doc_type != "All":
            type_map = {
                "API Endpoints": "api_endpoint",
                "Schemas": "schema",
                "Examples": "json_example",
                "Enumerations": "enumeration_group",
                "Sections": "section"
            }
            filters['type'] = type_map.get(doc_type)
        
        # Source filter
        source = st.selectbox(
            "Document Source",
            ["All", "PDF Contract", "YAML Specification", "HTML Contract"],
            help="Filter by document source"
        )
        if source != "All":
            source_map = {
                "PDF Contract": "pdf_contract",
                "YAML Specification": "yaml_spec",
                "HTML Contract": "html_contract"
            }
            filters['source'] = source_map.get(source)
        
        # Max results
        filters['max_results'] = st.slider("Max Results", 5, 20, 10)
        
        return filters

def display_grouped_results(grouped_results: Dict[str, List[Dict]]):
    """Display results grouped by type"""
    st.subheader("📊 Search Results by Type")
    
    type_info = {
        'api_endpoints': ('🔌 API Endpoints', 'primary'),
        'schemas': ('📐 Schemas', 'info'),
        'examples': ('💡 Examples', 'success'),
        'enumerations': ('🔢 Enumerations', 'warning'),
        'pdf_contract': ('📄 PDF Contract', 'secondary'),
        'yaml_spec': ('📋 YAML Spec', 'secondary')
    }
    
    for group_key, (label, color) in type_info.items():
        results = grouped_results.get(group_key, [])
        if results:
            with st.expander(f"{label} ({len(results)} results)", expanded=False):
                for i, result in enumerate(results[:3]):
                    metadata = result.get('metadata', {})
                    content_preview = result.get('content', '')[:200] + "..."
                    
                    # Display based on type
                    if group_key == 'api_endpoints':
                        st.write(f"**{metadata.get('method', '')} {metadata.get('path', '')}**")
                    elif group_key == 'schemas':
                        st.write(f"**Schema: {metadata.get('schema_name', 'Unknown')}**")
                    elif metadata.get('source') == 'pdf_contract':
                        st.write(f"**Page {metadata.get('page_num', 'N/A')}**")
                    
                    st.write(content_preview)
                    st.divider()

def main():
    st.set_page_config(
        page_title="ONDC Knowledge Base - Enhanced",
        page_icon="🔍",
        layout="wide"
    )
    
    # Header
    st.title("🔍 ONDC Knowledge Base - Enhanced Query Interface")
    st.markdown("""
    Query the comprehensive ONDC documentation including API Contract (PDF), YAML specifications, and more.
    All responses include detailed source references.
    
    **Example queries:**
    - "What is the payload structure for /on_search?"
    - "How to implement force cancellation?"
    - "Explain the Context schema"
    - "What are the fulfillment states in ONDC?"
    - "Show payment settlement process with examples"
    """)
    
    # Initialize RAG chain
    rag_chain = init_rag_chain()
    
    # Create layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Query input
        query = st.text_area(
            "Enter your query:",
            placeholder="e.g., What is the payload structure for /on_search? Include examples and explain the Context schema.",
            height=100,
            key="query_input"
        )
        
        # Search options
        search_col1, search_col2, search_col3 = st.columns([1, 1, 2])
        
        with search_col1:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        
        with search_col2:
            # Add option for detailed mode
            detailed_mode = st.checkbox("Detailed mode", value=True, help="Get comprehensive explanations with all examples")
        
        # Advanced filters
        filters = display_search_filters(search_col3)
        
        # Process search
        if search_button:
            if query:
                with st.spinner("Searching comprehensive knowledge base..."):
                    try:
                        # Enhance query for detailed mode
                        enhanced_query = query
                        if detailed_mode:
                            if 'payload' in query.lower() or '/' in query:
                                enhanced_query = f"{query} - provide complete documentation with full JSON payload examples and detailed field explanations"
                            else:
                                enhanced_query = f"{query} - provide a detailed explanation with examples"
                        
                        # Determine query type and search
                        if filters:
                            response = rag_chain.search_with_filters(enhanced_query, filters)
                        elif query.lower().startswith('/') or 'payload' in query.lower():
                            response = rag_chain.answer_api_query(query)
                        else:
                            response = rag_chain.answer_query(enhanced_query, max_results=20 if detailed_mode else 10)
                        
                        # Store in session state
                        st.session_state['last_response'] = response
                        st.session_state['last_query'] = query
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        logger.error(f"Query error: {e}")
            else:
                st.warning("Please enter a query.")
    
    with col2:
        st.subheader("📊 Search Analytics")
        
        if 'last_response' in st.session_state:
            response = st.session_state['last_response']
            
            # Display metrics
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                total_results = len(response.get('raw_results', []))
                st.metric("Total Results", total_results)
            
            with col_m2:
                num_refs = len(response.get('references', []))
                st.metric("References", num_refs)
            
            # Source distribution
            if 'grouped_results' in response:
                st.subheader("📈 Result Distribution")
                grouped = response['grouped_results']
                
                source_counts = {
                    'PDF': len(grouped.get('pdf_contract', [])),
                    'YAML': len(grouped.get('yaml_spec', [])),
                    'HTML': len(grouped.get('html_contract', []))
                }
                
                # Show as progress bars
                for source, count in source_counts.items():
                    if count > 0:
                        st.progress(count / total_results if total_results > 0 else 0, 
                                   text=f"{source}: {count}")
        
        # Help section
        with st.expander("💡 Search Tips", expanded=False):
            st.markdown("""
            - Start API queries with "/" (e.g., "/on_search")
            - Use "payload" to get JSON examples
            - Ask for "with examples" to get code samples
            - Filter by source for specific documentation
            - References show [number] for citation tracking
            """)
    
    # Display results
    if 'last_response' in st.session_state:
        response = st.session_state['last_response']
        
        # Main response area
        st.divider()
        st.subheader(f"📝 Results for: *{st.session_state.get('last_query', '')}*")
        
        # Display formatted answer with references
        formatted_response = format_response_with_references(response)
        st.markdown(formatted_response)
        
        # Display grouped results if available
        if 'grouped_results' in response and st.checkbox("Show detailed results by type"):
            display_grouped_results(response['grouped_results'])
        
        # Raw JSON view
        if st.checkbox("Show raw response data"):
            # Create a safe version without the full content
            safe_response = {
                'answer': response.get('answer', '')[:500] + '...',
                'references': response.get('references', []),
                'result_count': len(response.get('raw_results', [])),
                'result_types': list(response.get('grouped_results', {}).keys())
            }
            st.json(safe_response)

if __name__ == "__main__":
    main()