"""
Unified Ingestion Pipeline for ONDC Knowledge Base
Supports Google Docs, GitHub Markdown, and Google Sheets
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from enum import Enum

from docs_scraper import GoogleDocsScraper, DocumentContent
from github_scraper import GitHubMarkdownScraper, GitHubContent
from sheets_scraper import GoogleSheetsScreener, SheetContent
from document_chunker import DocumentChunker
from embeddings import GeminiEmbeddings, EmbeddingCache
from vector_store import QdrantVectorStore, DocumentChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SourceType(Enum):
    GOOGLE_DOCS = "google_docs"
    GITHUB_MARKDOWN = "github_markdown"
    GOOGLE_SHEETS = "google_sheets"


class UnifiedIngestionPipeline:
    """Unified pipeline for ingesting multiple document types into vector database"""
    
    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        qdrant_config: Optional[Dict[str, Any]] = None,
        github_token: Optional[str] = None,
        google_credentials_path: Optional[str] = None,
        use_cache: bool = True
    ):
        # Initialize scrapers
        self.docs_scraper = GoogleDocsScraper(credentials_path=google_credentials_path)
        self.github_scraper = GitHubMarkdownScraper(github_token=github_token)
        self.sheets_scraper = GoogleSheetsScreener(credentials_path=google_credentials_path)
        
        # Initialize processing components
        self.chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
        self.embedder = GeminiEmbeddings(api_key=gemini_api_key)
        
        # Initialize vector store
        qdrant_config = qdrant_config or {}
        self.vector_store = QdrantVectorStore(**qdrant_config)
        
        # Initialize cache
        self.embedding_cache = EmbeddingCache() if use_cache else None
        
        # Statistics
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'chunks_stored': 0,
            'errors': []
        }
    
    def ingest_from_sources(
        self,
        sources: List[Dict[str, Any]],
        domain: str = "retail",
        version: str = "1.2.0",
        recreate_collection: bool = False,
        save_scraped: bool = True,
        scraped_dir: str = "scraped_data"
    ) -> Dict[str, Any]:
        """
        Ingest documents from multiple sources
        
        Args:
            sources: List of source configurations, each containing:
                - type: SourceType (google_docs, github_markdown, google_sheets)
                - url: Source URL
                - options: Source-specific options
            domain: Domain for the documents
            version: Version of the documentation
            recreate_collection: Whether to recreate the vector collection
            save_scraped: Whether to save scraped content
            scraped_dir: Directory to save scraped content
        """
        logger.info(f"Starting unified ingestion for {len(sources)} sources")
        
        # Setup vector store
        self.vector_store.create_collection(recreate=recreate_collection)
        
        # Process each source
        all_scraped_content = {}
        
        for source in sources:
            source_type = SourceType(source['type'])
            url = source['url']
            options = source.get('options', {})
            
            try:
                logger.info(f"Processing {source_type.value} source: {url}")
                
                if source_type == SourceType.GOOGLE_DOCS:
                    content = self._process_google_docs(url, options)
                elif source_type == SourceType.GITHUB_MARKDOWN:
                    content = self._process_github_markdown(url, options)
                elif source_type == SourceType.GOOGLE_SHEETS:
                    content = self._process_google_sheets(url, options)
                else:
                    raise ValueError(f"Unsupported source type: {source_type}")
                
                all_scraped_content[url] = content
                
            except Exception as e:
                logger.error(f"Failed to process source {url}: {e}")
                self.stats['errors'].append({
                    'source': url,
                    'type': source_type.value,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Save scraped content if requested
        if save_scraped and all_scraped_content:
            self._save_scraped_content(all_scraped_content, scraped_dir, domain, version)
        
        # Process and store in vector database
        for url, content_list in all_scraped_content.items():
            for content in content_list:
                self._process_and_store_content(content, domain, version)
        
        # Return statistics
        return {
            'stats': self.stats,
            'sources_processed': len(sources),
            'errors': len(self.stats['errors']),
            'timestamp': datetime.now().isoformat()
        }
    
    def _process_google_docs(self, url: str, options: Dict[str, Any]) -> List[DocumentContent]:
        """Process Google Docs source"""
        follow_links = options.get('follow_links', True)
        follow_depth = options.get('follow_depth', 2) if follow_links else 0
        
        # Scrape document
        doc_content = self.docs_scraper.scrape_document(url, follow_links_depth=follow_depth)
        
        # Return all scraped documents (main + linked)
        return list(self.docs_scraper.scraped_docs.values())
    
    def _process_github_markdown(self, url: str, options: Dict[str, Any]) -> List[GitHubContent]:
        """Process GitHub markdown source"""
        if options.get('is_repository', False):
            # Scrape entire repository documentation
            doc_paths = options.get('doc_paths', None)
            return self.github_scraper.scrape_repository_docs(url, doc_paths)
        else:
            # Scrape single file
            return [self.github_scraper.scrape_markdown_file(url)]
    
    def _process_google_sheets(self, url: str, options: Dict[str, Any]) -> List[SheetContent]:
        """Process Google Sheets source"""
        sheet_names = options.get('sheet_names', None)
        return self.sheets_scraper.scrape_sheet(url, sheet_names)
    
    def _process_and_store_content(
        self,
        content: Union[DocumentContent, GitHubContent, SheetContent],
        domain: str,
        version: str
    ):
        """Process content and store in vector database"""
        try:
            # Convert content to chunks based on type
            if isinstance(content, DocumentContent):
                chunks = self._chunk_document_content(content, domain, version)
            elif isinstance(content, GitHubContent):
                chunks = self._chunk_github_content(content, domain, version)
            elif isinstance(content, SheetContent):
                chunks = self._chunk_sheet_content(content, domain, version)
            else:
                raise ValueError(f"Unknown content type: {type(content)}")
            
            self.stats['chunks_created'] += len(chunks)
            
            # Generate embeddings
            for chunk in chunks:
                # Check cache first
                if self.embedding_cache:
                    embedding = self.embedding_cache.get(chunk.content)
                    if embedding is None:
                        embedding = self.embedder.generate_embedding(chunk.content)
                        self.embedding_cache.set(chunk.content, embedding)
                else:
                    embedding = self.embedder.generate_embedding(chunk.content)
                
                chunk.embedding = embedding
                self.stats['embeddings_generated'] += 1
            
            # Store in vector database
            self.vector_store.add_documents(chunks)
            self.stats['chunks_stored'] += len(chunks)
            self.stats['documents_processed'] += 1
            
        except Exception as e:
            logger.error(f"Failed to process content: {e}")
            raise
    
    def _chunk_document_content(
        self,
        content: DocumentContent,
        domain: str,
        version: str
    ) -> List[DocumentChunk]:
        """Chunk Google Docs content"""
        # Combine main and footer content
        full_text = f"{content.main_content}\n\n{content.footer_content}".strip()
        
        # Create chunks
        text_chunks = self.chunker.chunk_text(full_text)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = DocumentChunk(
                content=chunk_text,
                metadata={
                    'doc_id': content.doc_id,
                    'doc_title': content.title,
                    'version': version,
                    'chunk_type': 'content',
                    'chunk_index': i,
                    'total_chunks': len(text_chunks),
                    'domain': domain,
                    'source_url': content.source_url,
                    'has_enums': bool(content.enums),
                    'has_definitions': bool(content.definitions)
                }
            )
            chunks.append(chunk)
        
        # Add enum chunks
        for enum_name, values in content.enums.items():
            chunk = DocumentChunk(
                content=f"Enum: {enum_name}\nValues: {', '.join(values)}",
                metadata={
                    'doc_id': content.doc_id,
                    'doc_title': content.title,
                    'version': version,
                    'chunk_type': 'enum',
                    'enum_name': enum_name,
                    'enum_values': values,
                    'value_count': len(values),
                    'domain': domain
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_github_content(
        self,
        content: GitHubContent,
        domain: str,
        version: str
    ) -> List[DocumentChunk]:
        """Chunk GitHub markdown content"""
        # Create chunks from main content
        text_chunks = self.chunker.chunk_text(content.content)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = DocumentChunk(
                content=chunk_text,
                metadata={
                    'doc_id': content.commit_sha[:12],
                    'doc_title': content.title,
                    'file_path': content.file_path,
                    'version': version,
                    'chunk_type': 'markdown',
                    'chunk_index': i,
                    'total_chunks': len(text_chunks),
                    'domain': domain,
                    'source_url': content.source_url,
                    'has_code_blocks': len(content.code_blocks) > 0,
                    'header_count': sum(len(headers) for headers in content.headers.values())
                }
            )
            chunks.append(chunk)
        
        # Add code block chunks
        for code_block in content.code_blocks:
            chunk = DocumentChunk(
                content=f"Code ({code_block['language']}):\n{code_block['code']}",
                metadata={
                    'doc_id': content.commit_sha[:12],
                    'doc_title': content.title,
                    'file_path': content.file_path,
                    'version': version,
                    'chunk_type': 'code',
                    'language': code_block['language'],
                    'context': code_block.get('context', ''),
                    'domain': domain
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_sheet_content(
        self,
        content: SheetContent,
        domain: str,
        version: str
    ) -> List[DocumentChunk]:
        """Chunk Google Sheets content"""
        chunks = []
        
        # Create schema chunk
        schema = self.sheets_scraper.extract_schema_info(content)
        schema_chunk = DocumentChunk(
            content=f"Schema for {content.sheet_name}:\n{json.dumps(schema, indent=2)}",
            metadata={
                'doc_id': content.sheet_id,
                'doc_title': f"Sheet: {content.sheet_name}",
                'version': version,
                'chunk_type': 'schema',
                'domain': domain,
                'source_url': content.source_url,
                'column_count': len(content.headers),
                'row_count': len(content.rows)
            }
        )
        chunks.append(schema_chunk)
        
        # Create data chunks (group rows)
        rows_per_chunk = 50
        for i in range(0, len(content.rows), rows_per_chunk):
            batch_rows = content.rows[i:i + rows_per_chunk]
            
            # Convert to structured text
            chunk_text = f"Data from {content.sheet_name} (rows {i+1}-{i+len(batch_rows)}):\n"
            chunk_text += json.dumps(batch_rows, indent=2)
            
            chunk = DocumentChunk(
                content=chunk_text,
                metadata={
                    'doc_id': content.sheet_id,
                    'doc_title': f"Sheet: {content.sheet_name}",
                    'version': version,
                    'chunk_type': 'data',
                    'chunk_index': i // rows_per_chunk,
                    'row_start': i + 1,
                    'row_end': i + len(batch_rows),
                    'domain': domain,
                    'headers': content.headers
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _save_scraped_content(
        self,
        content_dict: Dict[str, List[Any]],
        scraped_dir: str,
        domain: str,
        version: str
    ):
        """Save scraped content to disk"""
        os.makedirs(scraped_dir, exist_ok=True)
        
        # Save metadata
        metadata = {
            'scrape_timestamp': datetime.now().isoformat(),
            'domain': domain,
            'version': version,
            'sources': list(content_dict.keys()),
            'total_documents': sum(len(docs) for docs in content_dict.values())
        }
        
        with open(os.path.join(scraped_dir, '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save each document
        doc_count = 0
        for source_url, documents in content_dict.items():
            for doc in documents:
                doc_count += 1
                
                # Convert to dict for JSON serialization
                if hasattr(doc, '__dict__'):
                    doc_data = asdict(doc) if hasattr(doc, '__dataclass_fields__') else doc.__dict__
                else:
                    doc_data = doc
                
                # Generate filename
                if isinstance(doc, DocumentContent):
                    filename = f"gdoc_{doc.doc_id}.json"
                elif isinstance(doc, GitHubContent):
                    filename = f"github_{doc.commit_sha[:12]}_{os.path.basename(doc.file_path)}.json"
                elif isinstance(doc, SheetContent):
                    filename = f"sheet_{doc.sheet_id}_{doc.sheet_name}.json"
                else:
                    filename = f"doc_{doc_count}.json"
                
                with open(os.path.join(scraped_dir, filename), 'w') as f:
                    json.dump(doc_data, f, indent=2)
        
        logger.info(f"Saved {doc_count} documents to {scraped_dir}")