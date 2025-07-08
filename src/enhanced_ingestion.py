"""
Enhanced Ingestion Pipeline with Recursive Scraping and Update Checking
"""

import os
import json
import logging
import hashlib
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, urljoin
from enum import Enum
import re
import asyncio

from docs_scraper import GoogleDocsScraper, DocumentContent
from github_scraper import GitHubMarkdownScraper, GitHubContent
from sheets_scraper import GoogleSheetsScreener, SheetContent
from document_chunker import DocumentChunker
from embeddings import GeminiEmbeddings, EmbeddingCache
from vector_store import QdrantVectorStore, DocumentChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Supported document types"""
    GOOGLE_DOCS = "google_docs"
    GITHUB_MARKDOWN = "github_markdown"
    GOOGLE_SHEETS = "google_sheets"
    WEB_PAGE = "web_page"
    JSON_SCHEMA = "json_schema"


class DocumentMetadata:
    """Metadata for tracking document updates"""
    
    def __init__(self, url: str, content_hash: str, last_updated: str, 
                 title: str, doc_type: str, version: str = None):
        self.url = url
        self.content_hash = content_hash
        self.last_updated = last_updated
        self.title = title
        self.doc_type = doc_type
        self.version = version
        self.linked_documents = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'url': self.url,
            'content_hash': self.content_hash,
            'last_updated': self.last_updated,
            'title': self.title,
            'doc_type': self.doc_type,
            'version': self.version,
            'linked_documents': self.linked_documents
        }


class EnhancedIngestionPipeline:
    """Enhanced pipeline with recursive scraping and update checking"""
    
    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        qdrant_config: Optional[Dict[str, Any]] = None,
        github_token: Optional[str] = None,
        google_credentials_path: Optional[str] = None,
        use_cache: bool = True,
        metadata_dir: str = "document_metadata"
    ):
        # Initialize scrapers
        self.docs_scraper = GoogleDocsScraper(credentials_path=google_credentials_path)
        self.github_scraper = GitHubMarkdownScraper(github_token=github_token)
        self.sheets_scraper = GoogleSheetsScreener(credentials_path=google_credentials_path)
        
        # Initialize processing components
        self.chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
        self.embedder = GeminiEmbeddings(api_key=gemini_api_key)
        
        # Initialize vector store with deduplication support
        qdrant_config = qdrant_config or {}
        self.vector_store = QdrantVectorStore(**qdrant_config)
        
        # Initialize cache
        self.embedding_cache = EmbeddingCache() if use_cache else None
        
        # Metadata management
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(exist_ok=True)
        self.metadata_file = self.metadata_dir / "documents_metadata.json"
        self.document_metadata = self._load_metadata()
        
        # Track processed URLs in current session to avoid cycles
        self.processed_urls = set()
        self.scraped_content_cache = {}
        
        # Statistics
        self.stats = {
            'documents_processed': 0,
            'documents_skipped': 0,
            'documents_updated': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'chunks_stored': 0,
            'duplicates_removed': 0,
            'errors': []
        }
    
    def _load_metadata(self) -> Dict[str, DocumentMetadata]:
        """Load document metadata from file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                return {
                    url: DocumentMetadata(**meta) 
                    for url, meta in data.items()
                }
        return {}
    
    def _save_metadata(self):
        """Save document metadata to file"""
        data = {
            url: meta.to_dict() 
            for url, meta in self.document_metadata.items()
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate hash of content for change detection"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _detect_document_type(self, url: str) -> DocumentType:
        """Detect document type from URL"""
        if 'docs.google.com/document' in url:
            return DocumentType.GOOGLE_DOCS
        elif 'github.com' in url and ('.md' in url or '/blob/' in url):
            return DocumentType.GITHUB_MARKDOWN
        elif 'docs.google.com/spreadsheets' in url:
            return DocumentType.GOOGLE_SHEETS
        elif url.endswith('.json') or 'schema.json' in url:
            return DocumentType.JSON_SCHEMA
        else:
            return DocumentType.WEB_PAGE
    
    def _should_update_document(self, url: str, content: str) -> bool:
        """Check if document needs updating based on content hash"""
        content_hash = self._calculate_content_hash(content)
        
        if url not in self.document_metadata:
            return True
        
        existing_meta = self.document_metadata[url]
        return existing_meta.content_hash != content_hash
    
    def _extract_hyperlinks(self, content: Any, doc_type: DocumentType) -> List[str]:
        """Extract hyperlinks from document content"""
        links = []
        
        if doc_type == DocumentType.GOOGLE_DOCS and hasattr(content, 'hyperlinks'):
            links = [link['url'] for link in content.hyperlinks]
        elif doc_type == DocumentType.GITHUB_MARKDOWN and hasattr(content, 'links'):
            links = content.links
        elif isinstance(content, str):
            # Extract URLs from text content
            url_pattern = re.compile(
                r'https?://(?:www\.)?[a-zA-Z0-9-._~:/?#[\]@!$&\'()*+,;=]+\.[a-zA-Z]{2,}'
            )
            links = url_pattern.findall(content)
        
        # Filter and normalize links
        filtered_links = []
        for link in links:
            # Skip email links, anchors, and already processed URLs
            if link.startswith('mailto:') or link.startswith('#'):
                continue
            if link in self.processed_urls:
                continue
            filtered_links.append(link)
        
        return filtered_links
    
    async def ingest_document_recursive(
        self,
        url: str,
        domain: str = "general",
        version: str = None,
        max_depth: int = 3,
        current_depth: int = 0,
        follow_links: bool = True,
        check_updates: bool = True
    ) -> Dict[str, Any]:
        """
        Recursively ingest a document and all its linked documents
        
        Args:
            url: Document URL to ingest
            domain: Domain for categorization
            version: Version identifier
            max_depth: Maximum recursion depth for following links
            current_depth: Current recursion depth
            follow_links: Whether to follow hyperlinks
            check_updates: Whether to check for updates before re-ingesting
        """
        # Check if already processed in this session
        if url in self.processed_urls:
            logger.info(f"Skipping already processed URL: {url}")
            self.stats['documents_skipped'] += 1
            return {'status': 'skipped', 'reason': 'already_processed'}
        
        # Mark as processed
        self.processed_urls.add(url)
        
        try:
            # Detect document type
            doc_type = self._detect_document_type(url)
            logger.info(f"Processing {doc_type.value} document: {url}")
            
            # Scrape content
            content = await self._scrape_document(url, doc_type)
            if not content:
                raise ValueError(f"Failed to scrape content from {url}")
            
            # Convert content to string for hashing
            content_str = self._content_to_string(content, doc_type)
            
            # Check if update needed
            if check_updates and not self._should_update_document(url, content_str):
                logger.info(f"Document unchanged, skipping: {url}")
                self.stats['documents_skipped'] += 1
                
                # Still process linked documents if needed
                if follow_links and current_depth < max_depth:
                    links = self._extract_hyperlinks(content, doc_type)
                    for link in links:
                        await self.ingest_document_recursive(
                            link, domain, version, max_depth, 
                            current_depth + 1, follow_links, check_updates
                        )
                
                return {'status': 'skipped', 'reason': 'no_changes'}
            
            # Store scraped content in cache
            self.scraped_content_cache[url] = {
                'content': content,
                'doc_type': doc_type,
                'domain': domain,
                'version': version
            }
            
            # Process and store in vector database
            await self._process_and_store_content(content, doc_type, domain, version, url)
            
            # Update metadata
            metadata = DocumentMetadata(
                url=url,
                content_hash=self._calculate_content_hash(content_str),
                last_updated=datetime.now().isoformat(),
                title=self._extract_title(content, doc_type),
                doc_type=doc_type.value,
                version=version
            )
            
            # Process linked documents recursively
            if follow_links and current_depth < max_depth:
                links = self._extract_hyperlinks(content, doc_type)
                metadata.linked_documents = links
                
                for link in links:
                    await self.ingest_document_recursive(
                        link, domain, version, max_depth, 
                        current_depth + 1, follow_links, check_updates
                    )
            
            self.document_metadata[url] = metadata
            self._save_metadata()
            
            self.stats['documents_processed'] += 1
            
            return {
                'status': 'success',
                'doc_type': doc_type.value,
                'chunks_created': self.stats['chunks_created'],
                'linked_documents': len(metadata.linked_documents)
            }
            
        except Exception as e:
            logger.error(f"Failed to process document {url}: {e}")
            self.stats['errors'].append({
                'url': url,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return {'status': 'error', 'error': str(e)}
    
    async def _scrape_document(self, url: str, doc_type: DocumentType) -> Any:
        """Scrape document based on type"""
        if doc_type == DocumentType.GOOGLE_DOCS:
            return self.docs_scraper.scrape_document(url, follow_links_depth=0)
        elif doc_type == DocumentType.GITHUB_MARKDOWN:
            # Assuming github_scraper.scrape_markdown is async
            if hasattr(self.github_scraper, 'scrape_markdown'):
                return self.github_scraper.scrape_markdown(url)
            else:
                # Fallback for non-existing method
                return None
        elif doc_type == DocumentType.GOOGLE_SHEETS:
            return self.sheets_scraper.scrape_sheet(url)
        elif doc_type == DocumentType.JSON_SCHEMA:
            # Handle JSON schema files
            import requests
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        else:
            # Generic web scraping
            import requests
            from bs4 import BeautifulSoup
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup.get_text()
    
    def _content_to_string(self, content: Any, doc_type: DocumentType) -> str:
        """Convert content to string for hashing"""
        if isinstance(content, str):
            return content
        elif hasattr(content, 'main_content'):
            return content.main_content
        elif isinstance(content, dict):
            return json.dumps(content, sort_keys=True)
        else:
            return str(content)
    
    def _extract_title(self, content: Any, doc_type: DocumentType) -> str:
        """Extract title from content"""
        if hasattr(content, 'title'):
            return content.title
        elif isinstance(content, dict) and 'title' in content:
            return content['title']
        else:
            return f"Document from {doc_type.value}"
    
    async def _process_and_store_content(
        self, 
        content: Any, 
        doc_type: DocumentType,
        domain: str,
        version: str,
        url: str
    ):
        """Process content and store in vector database with deduplication"""
        # Create chunks based on document type
        if doc_type == DocumentType.JSON_SCHEMA:
            # Special handling for JSON schemas
            chunks = self._chunk_json_schema(content, url)
        else:
            # Convert to text and chunk
            text_content = self._content_to_string(content, doc_type)
            chunks = self.chunker.chunk_document(
                text_content,
                doc_id=url,
                metadata={
                    'doc_type': doc_type.value,
                    'domain': domain,
                    'version': version,
                    'source_url': url
                }
            )
        
        # Generate embeddings with caching
        for chunk in chunks:
            if self.embedding_cache:
                embedding = self.embedding_cache.get_or_create(
                    chunk.content,
                    lambda: self.embedder.embed_text(chunk.content)
                )
            else:
                embedding = self.embedder.embed_text(chunk.content)
            
            chunk.embedding = embedding
            self.stats['embeddings_generated'] += 1
        
        # Store in vector database with deduplication
        stored_count = await self._store_with_deduplication(chunks, domain)
        self.stats['chunks_stored'] += stored_count
        self.stats['chunks_created'] += len(chunks)
    
    def _chunk_json_schema(self, schema: Dict[str, Any], url: str) -> List[DocumentChunk]:
        """Special chunking for JSON schemas"""
        chunks = []
        
        # Create a chunk for the entire schema
        chunks.append(DocumentChunk(
            doc_id=url,
            content=json.dumps(schema, indent=2),
            metadata={
                'chunk_type': 'json_schema',
                'schema_type': schema.get('$schema', 'unknown'),
                'title': schema.get('title', 'JSON Schema'),
                'source_url': url
            },
            chunk_index=0,
            total_chunks=1
        ))
        
        # Create chunks for individual properties if it's an object schema
        if schema.get('type') == 'object' and 'properties' in schema:
            for prop_name, prop_schema in schema['properties'].items():
                chunks.append(DocumentChunk(
                    doc_id=url,
                    content=f"Property: {prop_name}\n{json.dumps(prop_schema, indent=2)}",
                    metadata={
                        'chunk_type': 'json_schema_property',
                        'property_name': prop_name,
                        'parent_schema': url
                    },
                    chunk_index=len(chunks),
                    total_chunks=len(schema['properties']) + 1
                ))
        
        return chunks
    
    async def _store_with_deduplication(
        self, 
        chunks: List[DocumentChunk], 
        domain: str
    ) -> int:
        """Store chunks with deduplication based on content hash"""
        stored_count = 0
        
        for chunk in chunks:
            # Create a unique identifier based on content
            content_hash = self._calculate_content_hash(chunk.content)
            chunk.metadata['content_hash'] = content_hash
            chunk.metadata['domain'] = domain
            
            # Check if similar chunk exists (make sync call async)
            loop = asyncio.get_event_loop()
            existing = await loop.run_in_executor(
                None,
                lambda: self.vector_store.search_by_metadata({
                    'content_hash': content_hash,
                    'domain': domain
                }, limit=1)
            )
            
            if existing and len(existing) > 0:
                # Update existing chunk if metadata changed
                logger.debug(f"Updating existing chunk with hash {content_hash}")
                await loop.run_in_executor(
                    None,
                    self.vector_store.update_chunk,
                    existing[0]['id'],
                    chunk
                )
                self.stats['duplicates_removed'] += 1
            else:
                # Store new chunk
                await loop.run_in_executor(
                    None,
                    self.vector_store.store_chunks,
                    [chunk]
                )
                stored_count += 1
        
        return stored_count
    
    async def ingest_from_config(
        self, 
        documents: List[Dict[str, Any]],
        force_update: bool = False
    ) -> Dict[str, Any]:
        """
        Ingest documents from configuration
        
        Args:
            documents: List of document configurations
            force_update: Force re-ingestion even if no changes detected
        """
        logger.info(f"Starting ingestion of {len(documents)} configured documents")
        
        results = []
        for doc_config in documents:
            url = doc_config['url']
            domain = doc_config.get('domain', 'general')
            version = doc_config.get('version')
            follow_links = doc_config.get('follow_links', True)
            check_updates = False if force_update else doc_config.get('check_updates', True)
            max_depth = doc_config.get('max_depth', 3)
            
            result = await self.ingest_document_recursive(
                url=url,
                domain=domain,
                version=version,
                max_depth=max_depth,
                follow_links=follow_links,
                check_updates=check_updates
            )
            
            results.append({
                'url': url,
                'result': result
            })
        
        # Save all scraped content
        if self.scraped_content_cache:
            self._save_scraped_content()
        
        return {
            'stats': self.stats,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_scraped_content(self):
        """Save all scraped content to disk"""
        scraped_dir = Path("scraped_docs")
        scraped_dir.mkdir(exist_ok=True)
        
        for url, data in self.scraped_content_cache.items():
            # Generate filename from URL
            doc_id = hashlib.md5(url.encode()).hexdigest()[:16]
            filename = scraped_dir / f"{data['doc_type'].value}_{doc_id}.json"
            
            # Save content
            with open(filename, 'w') as f:
                json.dump({
                    'url': url,
                    'doc_type': data['doc_type'].value,
                    'domain': data['domain'],
                    'version': data['version'],
                    'content': self._serialize_content(data['content']),
                    'scraped_at': datetime.now().isoformat()
                }, f, indent=2)
        
        logger.info(f"Saved {len(self.scraped_content_cache)} scraped documents")
    
    def _serialize_content(self, content: Any) -> Any:
        """Serialize content for saving"""
        if hasattr(content, '__dict__'):
            return content.__dict__
        elif isinstance(content, (str, dict, list)):
            return content
        else:
            return str(content)