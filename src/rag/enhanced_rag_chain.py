"""
Enhanced RAG Chain for ONDC Knowledge Base with Source References
"""

from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from ..utils.config import Config
from ..utils.gemini import get_gemini_llm
from .enhanced_retriever import EnhancedRetriever

class EnhancedONDCRAGChain:
    """
    Enhanced RAG Chain that provides detailed source references and handles multiple document types
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.retriever = EnhancedRetriever(config)
        self.llm = get_gemini_llm(config)
    
    def _format_source_reference(self, result: Dict) -> str:
        """Format a source reference for citation"""
        metadata = result.get('metadata', {})
        source_type = metadata.get('source', 'unknown')
        doc_type = metadata.get('type', '')
        
        if source_type == 'text_contract':
            if doc_type == 'api_endpoint_master':
                endpoint = metadata.get('endpoint', 'Unknown')
                lines = f"Lines {metadata.get('line_start', '?')}-{metadata.get('line_end', '?')}"
                return f"Text Contract - {endpoint} API Documentation ({lines})"
            elif doc_type == 'json_example':
                endpoint = metadata.get('endpoint', 'Unknown')
                scenario = metadata.get('scenario', 'Example')
                return f"Text Contract - {endpoint} JSON Example: {scenario}"
            elif doc_type == 'special_topic':
                topic = metadata.get('topic', 'Unknown Topic')
                return f"Text Contract - Special Topic: {topic}"
            elif doc_type == 'cross_reference':
                ref_range = metadata.get('ref_range', 'Unknown')
                return f"Text Contract - Cross References {ref_range}"
            else:
                return f"Text Contract - {doc_type}"
        
        elif source_type == 'pdf_contract':
            page_num = metadata.get('page_num', 'N/A')
            section = metadata.get('section_title', result.get('section', 'Unknown'))
            return f"PDF Contract - Page {page_num}, Section: {section}"
        
        elif source_type == 'html_contract':
            section = result.get('section', metadata.get('section_title', 'Unknown'))
            path = result.get('path', metadata.get('path', ''))
            return f"HTML Contract - {section} ({path})"
        
        elif source_type == 'yaml_spec':
            if metadata.get('type') == 'api_endpoint':
                return f"YAML Spec - {metadata.get('method', '')} {metadata.get('path', '')}"
            elif metadata.get('type') == 'schema':
                return f"YAML Spec - Schema: {metadata.get('schema_name', '')}"
            else:
                return f"YAML Spec - {metadata.get('type', 'unknown')}"
        
        else:
            # Fallback for legacy data
            if metadata.get('type') == 'api_endpoint':
                return f"API Endpoint - {metadata.get('method', '')} {metadata.get('path', '')}"
            else:
                return f"{metadata.get('type', 'Document')} - {result.get('section', 'Unknown')}"
    
    def _group_results_by_source(self, results: List[Dict]) -> Dict[str, List[Dict]]:
        """Group results by their source type"""
        grouped = {
            'pdf_contract': [],
            'html_contract': [],
            'yaml_spec': [],
            'api_endpoints': [],
            'schemas': [],
            'examples': [],
            'enumerations': []
        }
        
        for result in results:
            metadata = result.get('metadata', {})
            source = metadata.get('source', '')
            doc_type = metadata.get('type', '')
            
            # Group by source
            if source == 'pdf_contract':
                grouped['pdf_contract'].append(result)
            elif source == 'html_contract':
                grouped['html_contract'].append(result)
            elif source == 'yaml_spec':
                grouped['yaml_spec'].append(result)
            
            # Also group by type
            if doc_type == 'api_endpoint_master':
                grouped['api_endpoint_master'] = grouped.get('api_endpoint_master', [])
                grouped['api_endpoint_master'].append(result)
            elif doc_type == 'json_example':
                grouped['json_example'] = grouped.get('json_example', [])
                grouped['json_example'].append(result)
            elif doc_type == 'special_topic':
                grouped['special_topic'] = grouped.get('special_topic', [])
                grouped['special_topic'].append(result)
            elif doc_type == 'cross_reference':
                grouped['cross_reference'] = grouped.get('cross_reference', [])
                grouped['cross_reference'].append(result)
            elif doc_type == 'api_endpoint':
                grouped['api_endpoints'].append(result)
            elif doc_type == 'api_endpoint_complete':
                grouped['api_endpoint_complete'] = grouped.get('api_endpoint_complete', [])
                grouped['api_endpoint_complete'].append(result)
            elif doc_type == 'request_payload':
                grouped['request_payload'] = grouped.get('request_payload', [])
                grouped['request_payload'].append(result)
            elif doc_type == 'response_payload':
                grouped['response_payload'] = grouped.get('response_payload', [])
                grouped['response_payload'].append(result)
            elif doc_type == 'schema':
                grouped['schemas'].append(result)
            elif doc_type in ['example']:
                grouped['examples'].append(result)
            elif doc_type in ['enumeration', 'enumeration_group', 'enum_definition']:
                grouped['enumerations'].append(result)
        
        return grouped
    
    def answer_query(self, query: str, max_results: int = 15, include_sources: bool = True) -> Dict[str, Any]:
        """
        Answer a query using RAG approach with detailed source references
        """
        try:
            # Step 1: Retrieve relevant documents
            results = self.retriever.semantic_search(query)[:max_results]
            
            if not results:
                return {
                    'answer': "I couldn't find relevant information to answer your question.",
                    'sources': [],
                    'raw_results': [],
                    'references': []
                }
            
            # Group results by source
            grouped_results = self._group_results_by_source(results)
            
            # Step 2: Prepare context with source attribution
            context_parts = []
            references = []
            
            # Prioritize certain types of documents
            priority_order = [
                ('api_endpoint_master', "API Master Documentation"),
                ('json_example', "JSON Examples"),
                ('api_endpoints', "API Documentation"),
                ('api_endpoint_complete', "Complete API Documentation"),
                ('special_topic', "Special Topics"),
                ('request_payload', "Request Payloads"),
                ('response_payload', "Response Payloads"),
                ('schemas', "Schema Definitions"),
                ('examples', "Examples"),
                ('pdf_contract', "Contract Documentation"),
                ('cross_reference', "Cross References"),
                ('enumerations', "Enumeration Definitions")
            ]
            
            ref_num = 1
            for group_key, group_label in priority_order:
                group_results = grouped_results.get(group_key, [])
                for result in group_results[:5]:  # Increase limit per group for more context
                    ref = self._format_source_reference(result)
                    content = result.get('content', '')
                    
                    context_parts.append(f"[{ref_num}. {ref}]:\n{content}\n")
                    
                    references.append({
                        'number': ref_num,
                        'reference': ref,
                        'type': result.get('metadata', {}).get('type', 'unknown'),
                        'score': result.get('score', 0)
                    })
                    
                    ref_num += 1
            
            context = "\n---\n".join(context_parts)
            
            # Step 3: Generate answer with source citations
            prompt = f"""You are an expert on ONDC (Open Network for Digital Commerce) API specifications. 
Based on the following context from various ONDC documentation sources, please provide a comprehensive answer to the user's question.

IMPORTANT: When referencing information, cite the source using [number] format corresponding to the references provided.

Context with References:
{context}

User Question: {query}

Instructions:
1. Provide a DETAILED and COMPREHENSIVE answer based on the context
2. ALWAYS cite your sources using [1], [2], etc. when stating facts from the context
3. **CRITICAL: If there are JSON payload examples in the context, you MUST include the COMPLETE JSON payload in your response. Do not truncate, summarize, or show only parts of the JSON. Include the entire JSON structure exactly as provided.**
4. For API endpoints, provide:
   - Purpose and usage
   - Complete request structure with all fields explained
   - Response structure with field descriptions
   - **COMPLETE JSON examples** with field-by-field explanations (show the entire payload)
5. If there are enumeration values mentioned, explain their meanings with citations
6. Structure your response with clear sections using headers (##, ###)
7. Use code blocks (```json) for all JSON examples - show the FULL JSON, not abbreviated versions
8. Be extremely detailed - explain every field, every concept thoroughly
9. **MANDATORY: If the context contains JSON examples, you MUST include them COMPLETELY in your answer - never truncate or abbreviate**
10. When showing JSON payloads, always include the complete structure from opening brace to closing brace

Provide a COMPREHENSIVE answer with all available details and COMPLETE examples:"""

            answer = self.llm(prompt)
            
            return {
                'answer': answer,
                'sources': results[:5] if include_sources else [],
                'raw_results': results,
                'references': references,
                'grouped_results': grouped_results
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced RAG chain: {e}")
            return {
                'answer': f"An error occurred while processing your query: {str(e)}",
                'sources': [],
                'raw_results': [],
                'references': []
            }
    
    def answer_api_query(self, endpoint: str) -> Dict[str, Any]:
        """
        Answer queries about specific API endpoints with detailed references
        """
        # Clean endpoint name
        endpoint_name = endpoint.replace('payload', '').strip()
        
        # Search for comprehensive endpoint information
        queries = [
            f"{endpoint_name} API endpoint complete documentation",
            f"{endpoint_name} request payload structure",
            f"{endpoint_name} response format",
            f"{endpoint_name} JSON example"
        ]
        
        all_results = []
        for q in queries:
            results = self.retriever.semantic_search(q)[:5]
            all_results.extend(results)
        
        # Deduplicate
        seen_contents = set()
        results = []
        for r in all_results:
            content_hash = hash(r.get('content', ''))
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                results.append(r)
        
        results = results[:20]  # Limit to 20 unique results
        
        if not results:
            return {
                'answer': f"I couldn't find information about the {endpoint_name} endpoint.",
                'sources': [],
                'raw_results': [],
                'references': []
            }
        
        # Group and prioritize results
        grouped = self._group_results_by_source(results)
        
        # Build context with references
        context_parts = []
        references = []
        ref_num = 1
        
        # Add API endpoint documentation
        for result in grouped['api_endpoints'][:3]:
            if endpoint_name in result.get('content', ''):
                ref = self._format_source_reference(result)
                context_parts.append(f"[{ref_num}. {ref}]:\n{result['content']}\n")
                references.append({
                    'number': ref_num,
                    'reference': ref,
                    'type': 'api_endpoint'
                })
                ref_num += 1
        
        # Add related schemas
        for result in grouped['schemas'][:2]:
            ref = self._format_source_reference(result)
            context_parts.append(f"[{ref_num}. {ref}]:\n{result['content'][:1000]}\n")
            references.append({
                'number': ref_num,
                'reference': ref,
                'type': 'schema'
            })
            ref_num += 1
        
        # Add examples
        for result in grouped['examples'][:2]:
            if endpoint_name in result.get('content', ''):
                ref = self._format_source_reference(result)
                context_parts.append(f"[{ref_num}. {ref}]:\n{result['content']}\n")
                references.append({
                    'number': ref_num,
                    'reference': ref,
                    'type': 'example'
                })
                ref_num += 1
        
        context = "\n---\n".join(context_parts)
        
        prompt = f"""You are an expert on ONDC API specifications. 
Explain the {endpoint_name} API endpoint based on the following documentation with references:

{context}

Please provide a COMPREHENSIVE explanation:

1. **Purpose and Overview** of the {endpoint_name} endpoint (with citation)
2. **HTTP Method and Tags** (with citation)
3. **Complete Request Payload Structure**:
   - Include the FULL JSON structure
   - Explain EVERY field with its purpose, type, and requirements
   - Show nested objects and arrays
   - Cite sources for each section
4. **Response Structure**:
   - Include the complete response format
   - Explain all response fields
   - Show success and error responses if available
5. **Referenced Schemas** (Context, Order, etc.):
   - Explain what each schema contains
   - Show the schema structure if available in context
6. **Complete JSON Examples**:
   - **CRITICAL: Include COMPLETE, FULL JSON examples from the context**
   - **Include FULL request examples - show the entire JSON payload**
   - **Include FULL response examples - show the entire JSON payload**
   - **DO NOT truncate, summarize, or abbreviate - show complete payloads from opening brace {{ to closing brace }}**
   - **Never use "..." or ellipsis to shorten JSON examples**

MANDATORY REQUIREMENTS: 
- You MUST include all JSON examples found in the context COMPLETELY
- Use ```json code blocks for all JSON and include the ENTIRE payload
- Explain every field in detail after showing the complete JSON
- This should be a complete reference documentation with full payload examples
- **If JSON examples exist in the context, they MUST be included in their entirety**"""

        answer = self.llm(prompt)
        
        return {
            'answer': answer,
            'sources': results[:5],
            'raw_results': results,
            'references': references,
            'grouped_results': grouped
        }
    
    def search_with_filters(self, query: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search with specific filters like document type, source, etc.
        """
        # Use enhanced retriever to search with filters
        results = self.retriever.semantic_search(query)
        
        # Apply filters
        filtered_results = []
        for result in results:
            metadata = result.get('metadata', {})
            
            # Check document type filter
            if 'type' in filters and metadata.get('type') != filters['type']:
                continue
            
            # Check source filter
            if 'source' in filters and metadata.get('source') != filters['source']:
                continue
            
            # Check page range filter (for PDF)
            if 'page_range' in filters:
                page_num = metadata.get('page_num', 0)
                min_page, max_page = filters['page_range']
                if not (min_page <= page_num <= max_page):
                    continue
            
            filtered_results.append(result)
        
        # Limit results
        filtered_results = filtered_results[:filters.get('max_results', 10)]
        
        # Generate answer with filtered results
        return self._generate_answer_from_results(query, filtered_results)
    
    def _generate_answer_from_results(self, query: str, results: List[Dict]) -> Dict[str, Any]:
        """Generate answer from a specific set of results"""
        if not results:
            return {
                'answer': "No results found matching your filters.",
                'sources': [],
                'raw_results': [],
                'references': []
            }
        
        # Prepare context with references
        context_parts = []
        references = []
        
        for i, result in enumerate(results):
            ref = self._format_source_reference(result)
            content = result.get('content', '')
            
            context_parts.append(f"[{i+1}. {ref}]:\n{content}\n")
            
            references.append({
                'number': i+1,
                'reference': ref,
                'type': result.get('metadata', {}).get('type', 'unknown'),
                'score': result.get('score', 0)
            })
        
        context = "\n---\n".join(context_parts)
        
        prompt = f"""Based on the following filtered search results, answer the user's question.
Remember to cite sources using [number] format.

Context:
{context}

Question: {query}

Answer with citations:"""

        answer = self.llm(prompt)
        
        return {
            'answer': answer,
            'sources': results[:5],
            'raw_results': results,
            'references': references
        }