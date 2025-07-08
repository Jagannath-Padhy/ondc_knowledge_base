import re
import json
import uuid
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

# Import DocumentChunk from vector_store
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vector_store import DocumentChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for document chunks"""
    doc_id: str
    doc_title: str
    version: str
    chunk_type: str  # 'api_endpoint', 'schema', 'enum', 'definition', 'content'
    section: Optional[str] = None
    api_path: Optional[str] = None
    http_method: Optional[str] = None
    parent_chunk_id: Optional[str] = None


class DocumentChunker:
    """Intelligent chunking for ONDC API documentation"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_bytes = 30000  # Conservative limit under 36k bytes
        
    def _get_byte_size(self, text: str) -> int:
        """Get byte size of text"""
        return len(text.encode('utf-8'))
        
    def chunk_document(
        self, 
        content: str,
        doc_id: str,
        metadata: Dict[str, Any] = None
    ) -> List[DocumentChunk]:
        """Chunk a document into smaller pieces"""
        if metadata is None:
            metadata = {}
            
        # Simple text chunking for now
        chunks = []
        text = content.strip()
        
        if not text:
            return chunks
            
        # Split into paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # Check if adding this paragraph would exceed size limits
            test_chunk = current_chunk + "\n\n" + para if current_chunk else para
            
            if self._get_byte_size(test_chunk) > self.max_bytes or len(test_chunk) > self.chunk_size:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(DocumentChunk(
                        doc_id=doc_id,
                        content=current_chunk,
                        metadata={**metadata, 'chunk_index': chunk_index},
                        chunk_index=chunk_index,
                        total_chunks=1  # Will update later
                    ))
                    chunk_index += 1
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    # Take last part of current chunk as overlap
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    current_chunk = para
            else:
                current_chunk = test_chunk
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(DocumentChunk(
                doc_id=doc_id,
                content=current_chunk,
                metadata={**metadata, 'chunk_index': chunk_index},
                chunk_index=chunk_index,
                total_chunks=1
            ))
        
        # Update total chunks count
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        
        logger.info(f"Created {len(chunks)} chunks for document {doc_id}")
        return chunks
    
    def chunk_document_advanced(
        self, 
        doc_content: Dict[str, Any],
        domain: str = "retail"
    ) -> List[Dict[str, Any]]:
        """Advanced chunking method for structured documents (legacy)"""
        chunks = []
        
        # Extract metadata
        base_metadata = ChunkMetadata(
            doc_id=doc_content['doc_id'],
            doc_title=doc_content['title'],
            version=doc_content['version'],
            chunk_type='content'
        )
        
        # 1. Chunk API endpoints
        api_chunks = self._extract_api_endpoints(doc_content['main_content'], base_metadata)
        chunks.extend(api_chunks)
        
        # 2. Chunk schemas/models
        schema_chunks = self._extract_schemas(doc_content['main_content'], base_metadata)
        chunks.extend(schema_chunks)
        
        # 3. Chunk enums as separate entities
        enum_chunks = self._chunk_enums(doc_content.get('enums', {}), base_metadata)
        chunks.extend(enum_chunks)
        
        # 4. Chunk definitions
        def_chunks = self._chunk_definitions(doc_content.get('definitions', {}), base_metadata)
        chunks.extend(def_chunks)
        
        # 5. Chunk regular content
        content_chunks = self._chunk_content(doc_content['main_content'], base_metadata)
        chunks.extend(content_chunks)
        
        # 6. Chunk footer content if significant
        if doc_content.get('footer_content'):
            footer_chunks = self._chunk_content(
                doc_content['footer_content'], 
                base_metadata,
                chunk_type_override='footer'
            )
            chunks.extend(footer_chunks)
        
        # Add domain metadata to all chunks
        for chunk in chunks:
            chunk['metadata']['domain'] = domain
        
        logger.info(f"Created {len(chunks)} chunks for document {doc_content['doc_id']}")
        return chunks
    
    def _extract_api_endpoints(
        self, 
        content: str, 
        base_metadata: ChunkMetadata
    ) -> List[Dict[str, Any]]:
        """Extract and chunk API endpoint documentation"""
        chunks = []
        
        # Patterns for API endpoints
        endpoint_patterns = [
            r'(POST|GET|PUT|DELETE|PATCH)\s+(/[^\s\n]+)',
            r'endpoint:\s*(/[^\s\n]+)',
            r'path:\s*(/[^\s\n]+)'
        ]
        
        # Split content by API sections
        api_sections = re.split(r'\n(?=(?:POST|GET|PUT|DELETE|PATCH)\s+/)', content)
        
        for section in api_sections:
            for pattern in endpoint_patterns:
                match = re.search(pattern, section, re.IGNORECASE)
                if match:
                    method = match.group(1) if len(match.groups()) > 1 else 'POST'
                    path = match.group(2) if len(match.groups()) > 1 else match.group(1)
                    
                    # Extract request/response schemas
                    request_schema = self._extract_section(section, ['request', 'payload', 'body'])
                    response_schema = self._extract_section(section, ['response', 'output'])
                    
                    # Truncate section content to respect byte limits
                    section_content = section
                    while self._get_byte_size(f"{method} {path}\n\n{section_content}") > self.max_bytes:
                        section_content = section_content[:len(section_content)//2]
                    
                    chunk_content = f"{method} {path}\n\n{section_content}"
                    
                    chunk = {
                        'chunk_id': str(uuid.uuid4()),
                        'content': chunk_content,
                        'metadata': {
                            **base_metadata.__dict__,
                            'chunk_type': 'api_endpoint',
                            'api_path': path,
                            'http_method': method,
                            'has_request_schema': bool(request_schema),
                            'has_response_schema': bool(response_schema)
                        }
                    }
                    chunks.append(chunk)
                    
                    # Create separate chunks for schemas if they're large
                    if request_schema and len(request_schema) > self.min_chunk_size:
                        schema_chunk = {
                            'chunk_id': str(uuid.uuid4()),
                            'content': f"Request Schema for {method} {path}:\n{request_schema}",
                            'metadata': {
                                **base_metadata.__dict__,
                                'chunk_type': 'schema',
                                'schema_type': 'request',
                                'api_path': path,
                                'http_method': method,
                                'parent_chunk_id': chunk['chunk_id']
                            }
                        }
                        chunks.append(schema_chunk)
                    
                    if response_schema and len(response_schema) > self.min_chunk_size:
                        schema_chunk = {
                            'chunk_id': str(uuid.uuid4()),
                            'content': f"Response Schema for {method} {path}:\n{response_schema}",
                            'metadata': {
                                **base_metadata.__dict__,
                                'chunk_type': 'schema',
                                'schema_type': 'response',
                                'api_path': path,
                                'http_method': method,
                                'parent_chunk_id': chunk['chunk_id']
                            }
                        }
                        chunks.append(schema_chunk)
                    
                    break
        
        return chunks
    
    def _extract_schemas(
        self, 
        content: str, 
        base_metadata: ChunkMetadata
    ) -> List[Dict[str, Any]]:
        """Extract JSON schemas and data models"""
        chunks = []
        
        # Find JSON blocks
        json_pattern = r'```json\n(.*?)\n```'
        json_blocks = re.findall(json_pattern, content, re.DOTALL)
        
        for i, json_block in enumerate(json_blocks):
            try:
                # Try to parse as JSON to validate
                parsed = json.loads(json_block)
                
                # Identify schema type
                schema_type = 'unknown'
                if '$schema' in parsed:
                    schema_type = 'json_schema'
                elif 'properties' in parsed:
                    schema_type = 'object_schema'
                elif isinstance(parsed, list):
                    schema_type = 'array_example'
                
                chunk = {
                    'chunk_id': str(uuid.uuid4()),
                    'content': json_block,
                    'metadata': {
                        **base_metadata.__dict__,
                        'chunk_type': 'schema',
                        'schema_type': schema_type,
                        'schema_index': i
                    }
                }
                chunks.append(chunk)
            except json.JSONDecodeError:
                # Not valid JSON, skip
                pass
        
        return chunks
    
    def _chunk_enums(
        self, 
        enums: Dict[str, List[str]], 
        base_metadata: ChunkMetadata
    ) -> List[Dict[str, Any]]:
        """Create chunks for enum definitions"""
        chunks = []
        
        for enum_name, values in enums.items():
            content = f"Enum: {enum_name}\nValues: {', '.join(values)}"
            
            chunk = {
                'chunk_id': str(uuid.uuid4()),
                'content': content,
                'metadata': {
                    **base_metadata.__dict__,
                    'chunk_type': 'enum',
                    'enum_name': enum_name,
                    'enum_values': values,
                    'value_count': len(values)
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_definitions(
        self, 
        definitions: Dict[str, str], 
        base_metadata: ChunkMetadata
    ) -> List[Dict[str, Any]]:
        """Create chunks for term definitions"""
        chunks = []
        
        # Group related definitions
        grouped_defs = {}
        for term, definition in definitions.items():
            # Simple grouping by common prefixes
            prefix = term.split('_')[0].lower()
            if prefix not in grouped_defs:
                grouped_defs[prefix] = []
            grouped_defs[prefix].append((term, definition))
        
        # Create chunks for grouped definitions
        for group, defs in grouped_defs.items():
            if len(defs) > 3:
                # Create a group chunk
                content = f"Definitions for {group.upper()} terms:\n"
                for term, definition in defs:
                    content += f"\n{term}: {definition}"
                
                chunk = {
                    'chunk_id': str(uuid.uuid4()),
                    'content': content[:self.chunk_size],
                    'metadata': {
                        **base_metadata.__dict__,
                        'chunk_type': 'definition',
                        'definition_group': group,
                        'term_count': len(defs)
                    }
                }
                chunks.append(chunk)
            else:
                # Create individual chunks
                for term, definition in defs:
                    chunk = {
                        'chunk_id': str(uuid.uuid4()),
                        'content': f"{term}: {definition}",
                        'metadata': {
                            **base_metadata.__dict__,
                            'chunk_type': 'definition',
                            'term': term
                        }
                    }
                    chunks.append(chunk)
        
        return chunks
    
    def _chunk_content(
        self, 
        content: str, 
        base_metadata: ChunkMetadata,
        chunk_type_override: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Chunk regular content with overlap"""
        chunks = []
        
        # Split by sections first
        sections = self._split_by_sections(content)
        
        for section_title, section_content in sections:
            if (len(section_content) <= self.chunk_size and 
                self._get_byte_size(section_content) <= self.max_bytes):
                # Small section, keep as single chunk
                chunk = {
                    'chunk_id': str(uuid.uuid4()),
                    'content': section_content,
                    'metadata': {
                        **base_metadata.__dict__,
                        'chunk_type': chunk_type_override or 'content',
                        'section': section_title
                    }
                }
                chunks.append(chunk)
            else:
                # Large section, split with overlap
                text_chunks = self._split_text_with_overlap(section_content)
                for i, text_chunk in enumerate(text_chunks):
                    chunk = {
                        'chunk_id': str(uuid.uuid4()),
                        'content': text_chunk,
                        'metadata': {
                            **base_metadata.__dict__,
                            'chunk_type': chunk_type_override or 'content',
                            'section': section_title,
                            'chunk_index': i,
                            'total_chunks': len(text_chunks)
                        }
                    }
                    chunks.append(chunk)
        
        return chunks
    
    def _split_by_sections(self, content: str) -> List[Tuple[str, str]]:
        """Split content by section headers"""
        sections = []
        
        # Pattern for section headers
        header_pattern = r'\n(#{1,6}\s+.+?)\n'
        
        # Split by headers
        parts = re.split(header_pattern, content)
        
        current_title = "Introduction"
        current_content = ""
        
        for i, part in enumerate(parts):
            if re.match(r'^#{1,6}\s+', part):
                # This is a header
                if current_content.strip():
                    sections.append((current_title, current_content.strip()))
                current_title = part.strip('#').strip()
                current_content = ""
            else:
                current_content += part
        
        # Add last section
        if current_content.strip():
            sections.append((current_title, current_content.strip()))
        
        return sections
    
    def _split_text_with_overlap(self, text: str) -> List[str]:
        """Split text into chunks with overlap, respecting byte limits"""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        for sentence in sentences:
            test_chunk = current_chunk + sentence + " "
            if (len(test_chunk) <= self.chunk_size and 
                self._get_byte_size(test_chunk) <= self.max_bytes):
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + sentence + " "
                
                # If single sentence is too large, truncate it
                if self._get_byte_size(current_chunk) > self.max_bytes:
                    # Find a safe truncation point
                    while self._get_byte_size(current_chunk) > self.max_bytes and len(current_chunk) > 100:
                        current_chunk = current_chunk[:-100]
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _extract_section(self, content: str, keywords: List[str]) -> Optional[str]:
        """Extract a section based on keywords"""
        for keyword in keywords:
            pattern = rf'{keyword}[:\s]*\n(.*?)(?=\n[A-Z]|\n\n|\Z)'
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return None


