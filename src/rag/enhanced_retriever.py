"""
Enhanced retriever that works with the new ONDC HTML ingestion
"""

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from typing import List, Dict, Any, Optional
from loguru import logger
from ..utils.config import Config
from ..utils.database import QdrantDB
from ..utils.gemini import get_gemini_embed_fn

class EnhancedRetriever:
    """
    Enhanced retriever for ONDC knowledge base with support for:
    - Semantic search
    - Section-based filtering
    - Enumeration-aware search
    - API endpoint search
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.db = QdrantDB(self.config)
        self.embed_fn = get_gemini_embed_fn(self.config)
        self.collection_name = self.config.get('qdrant.collection_name', 'ondc_knowledge')
        self.similarity_threshold = self.config.rag.get("similarity_threshold", 0.7)
        self.max_results = self.config.rag.get("max_results", 10)
    
    def semantic_search(self, query: str, filter_section: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Enhanced semantic search with optional section filtering
        """
        try:
            query_vec = self.embed_fn(query)
            
            # Build filter if section specified
            search_filter = None
            if filter_section:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="section",
                            match=MatchValue(value=filter_section)
                        )
                    ]
                )
            
            hits = self.db.client.search(
                collection_name=self.collection_name,
                query_vector=query_vec,
                limit=self.max_results,
                score_threshold=self.similarity_threshold,
                query_filter=search_filter,
                with_payload=True
            )
            
            results = []
            for hit in hits:
                # Handle both old and new metadata formats
                section = hit.payload.get('section_title') or hit.payload.get('section', 'Unknown')
                path = hit.payload.get('section_path') or hit.payload.get('path', 'Unknown')
                
                # Extract enumeration info
                enum_info = []
                if 'linked_enumerations' in hit.payload:
                    enum_info = hit.payload['linked_enumerations']
                elif 'enum' in hit.payload:
                    enum_info = [{
                        'field': 'Unknown',
                        'value': hit.payload['enum'],
                        'definition': hit.payload.get('def', '')
                    }]
                
                result = {
                    'id': hit.id,
                    'score': hit.score,
                    'content': hit.payload.get('content', ''),
                    'section': section,
                    'path': path,
                    'enumerations': enum_info,
                    'has_enums': len(enum_info) > 0,
                    'metadata': hit.payload
                }
                results.append(result)
            
            logger.info(f"Semantic search found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []
    
    
    def search_by_enum_value(self, enum_value: str) -> List[Dict[str, Any]]:
        """
        Search for documents containing a specific enumeration value
        """
        # First try semantic search
        results = self.semantic_search(f"enumeration {enum_value}")
        
        # Then filter for actual enum presence
        filtered_results = []
        for result in results:
            content = result['content'].lower()
            if enum_value.lower() in content:
                # Check if it has enum definition
                for enum in result['enumerations']:
                    if enum['value'] == enum_value:
                        result['matched_enum'] = enum
                        filtered_results.append(result)
                        break
                else:
                    # Even if no exact enum match, include if value is in content
                    filtered_results.append(result)
        
        return filtered_results
    
    def get_all_sections(self) -> List[str]:
        """
        Get all unique section names in the collection
        """
        sections = set()
        offset = None
        
        while True:
            scroll_result = self.db.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=['section', 'section_title'],
                with_vectors=False
            )
            
            points, next_offset = scroll_result
            
            for point in points:
                section = point.payload.get('section_title') or point.payload.get('section')
                if section:
                    sections.add(section)
            
            if next_offset is None:
                break
            
            offset = next_offset
        
        return sorted(list(sections))