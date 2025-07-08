"""
JSON Schema parser for ONDC schema files
Parses JSON schema files and converts them to Document format
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger

from .base import BaseParser
from ...core.models import Document, DocumentMetadata, DocumentChunk, ChunkType
from ...core.exceptions import IngestionError


class JSONSchemaParser(BaseParser):
    """Parser for JSON schema files"""
    
    def __init__(self, settings):
        super().__init__(settings)
    
    async def parse(self, source: str, context: Dict[str, Any] = None) -> Document:
        """
        Parse JSON schema file(s)
        
        Args:
            source: Path to JSON schema file or directory
            context: Additional context (domain, version, etc.)
            
        Returns:
            Document: Parsed document with schema chunks
        """
        
        try:
            path = Path(source)
            
            if not path.exists():
                raise IngestionError(f"Schema file not found: {source}")
            
            # Load schemas
            schemas = []
            if path.is_file():
                with open(path, 'r') as f:
                    data = json.load(f)
                    schemas.append(data)
            else:
                # Load all JSON files from directory
                for json_file in path.glob('**/*.json'):
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        schemas.append({'file': json_file.name, 'data': data})
            
            if not schemas:
                raise IngestionError(f"No schemas found in: {source}")
            
            # Create document
            doc_metadata = DocumentMetadata(
                title=f"ONDC API Schemas - {path.name}",
                version=context.get('version', 'unknown') if context else 'unknown',
                domain=context.get('domain', 'all') if context else 'all',
                source_url=str(path.absolute()),
                tags=['schema', 'api', 'payload']
            )
            
            document = Document(
                doc_type='json_schema',
                metadata=doc_metadata,
                content=f"Collection of {len(schemas)} ONDC API schemas"
            )
            
            # Convert schemas to chunks
            chunks = []
            for idx, schema_data in enumerate(schemas):
                # Handle both file data and direct data
                if isinstance(schema_data, dict) and 'data' in schema_data:
                    schema = schema_data['data']
                    filename = schema_data.get('file', 'unknown')
                else:
                    schema = schema_data
                    filename = path.name
                
                # Extract metadata from schema
                action = schema.get('context', {}).get('action', 'unknown')
                domain = schema.get('context', {}).get('domain', 'unknown')
                version = schema.get('context', {}).get('core_version', 'unknown')
                
                # Main schema chunk
                main_chunk = DocumentChunk(
                    content=json.dumps(schema, indent=2),
                    chunk_type=ChunkType.PAYLOAD_EXAMPLE,
                    chunk_index=idx,
                    total_chunks=len(schemas),
                    section_title=f"{action} - {domain} v{version}",
                    metadata={
                        'filename': filename,
                        'action': action,
                        'domain': domain,
                        'version': version
                    }
                )
                chunks.append(main_chunk)
                
                # Store payload examples for easy access
                document.payload_examples.append({
                    'action': action,
                    'domain': domain,
                    'version': version,
                    'schema': schema,
                    'filename': filename
                })
            
            document.chunks = chunks
            
            logger.info(f"Parsed {len(schemas)} schemas into {len(chunks)} chunks")
            return document
            
        except Exception as e:
            logger.error(f"Failed to parse schema file: {e}")
            raise IngestionError(f"Schema parsing failed: {e}")
    
    def supported_formats(self) -> List[str]:
        """Get supported file formats"""
        return ['.json']
    
    def can_handle(self, source_url: str) -> bool:
        """
        Check if parser can handle the given source URL
        
        Args:
            source_url: Source URL or path
            
        Returns:
            bool: True if parser can handle the source
        """
        # Handle local file paths
        if os.path.exists(source_url):
            path = Path(source_url)
            return path.suffix.lower() == '.json' or path.is_dir()
        
        # Handle URLs pointing to JSON files
        return source_url.lower().endswith('.json')