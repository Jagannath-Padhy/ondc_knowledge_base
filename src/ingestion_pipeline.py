import os
import json
import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from docs_scraper import GoogleDocsScraper, DocumentContent
from document_chunker import DocumentChunker
from embeddings import GeminiEmbeddings, EmbeddingCache
from vector_store import QdrantVectorStore, DocumentChunk
from footnote_extractor import FootnoteExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Main pipeline for ingesting ONDC documentation into Qdrant"""
    
    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        qdrant_config: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ):
        # Initialize components
        self.scraper = GoogleDocsScraper()
        self.chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
        self.embedder = GeminiEmbeddings(api_key=gemini_api_key)
        self.footnote_extractor = FootnoteExtractor()
        
        # Initialize Qdrant
        qdrant_config = qdrant_config or {}
        self.vector_store = QdrantVectorStore(**qdrant_config)
        
        # Initialize cache
        self.embedding_cache = EmbeddingCache() if use_cache else None
        
        # Pipeline state
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'chunks_stored': 0,
            'footnotes_extracted': 0,
            'footnote_chunks_created': 0,
            'errors': []
        }
        
        # Cache for footnote mappings
        self.footnote_mappings = {}
    
    def ingest_documents(
        self,
        doc_urls: List[str],
        domain: str = "retail",
        follow_links: bool = True,
        recreate_collection: bool = False,
        save_scraped_docs: bool = True,
        scraped_docs_dir: str = "scraped_docs"
    ) -> Dict[str, Any]:
        """Main method to ingest documents with two-phase approach"""
        logger.info(f"Starting ingestion of {len(doc_urls)} documents")
        
        # Phase 1: Scrape all documents first
        logger.info("Phase 1: Scraping documents...")
        scraped_count = 0
        for url in doc_urls:
            try:
                doc_content = self.scraper.scrape_document(
                    url, 
                    follow_links_depth=2 if follow_links else 0
                )
                scraped_count += 1
                logger.info(f"Scraped document {scraped_count}/{len(doc_urls)}: {doc_content.title}")
            except Exception as e:
                logger.error(f"Failed to scrape document {url}: {e}")
                self.stats['errors'].append({
                    'url': url,
                    'error': str(e),
                    'phase': 'scraping',
                    'timestamp': datetime.now().isoformat()
                })
        
        # Save scraped documents
        if save_scraped_docs and self.scraper.scraped_docs:
            logger.info(f"Saving {len(self.scraper.scraped_docs)} scraped documents to {scraped_docs_dir}...")
            self.scraper.save_scraped_content(scraped_docs_dir)
            
            # Create metadata file
            metadata_file = os.path.join(scraped_docs_dir, "_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump({
                    'total_documents': len(self.scraper.scraped_docs),
                    'scrape_timestamp': datetime.now().isoformat(),
                    'domain': domain,
                    'follow_links': follow_links,
                    'document_ids': list(self.scraper.scraped_docs.keys())
                }, f, indent=2)
        
        # Phase 1.5: Extract footnotes from all documents
        logger.info("Phase 1.5: Extracting footnotes from documents...")
        if save_scraped_docs and os.path.exists(scraped_docs_dir):
            self.footnote_mappings = self.footnote_extractor.process_all_documents(scraped_docs_dir)
            self.footnote_extractor.save_footnote_mappings(self.footnote_mappings, scraped_docs_dir)
            
            # Update stats
            for doc_id, footnote_map in self.footnote_mappings.items():
                self.stats['footnotes_extracted'] += len(footnote_map.get('mappings', {}))
            
            logger.info(f"Extracted {self.stats['footnotes_extracted']} footnotes from {len(self.footnote_mappings)} documents")
        
        # Phase 2: Process and ingest into vector store
        logger.info("Phase 2: Processing and ingesting documents...")
        
        # Setup vector store
        self.vector_store.create_collection(recreate=recreate_collection)
        
        # Process each scraped document
        for doc_id, doc_content in self.scraper.scraped_docs.items():
            try:
                self._process_scraped_document(doc_content, domain)
            except Exception as e:
                logger.error(f"Failed to process document {doc_id}: {e}")
                self.stats['errors'].append({
                    'doc_id': doc_id,
                    'error': str(e),
                    'phase': 'processing',
                    'timestamp': datetime.now().isoformat()
                })
        
        # Log final stats
        logger.info(f"Ingestion complete. Stats: {self.stats}")
        return self.stats
    
    def _process_scraped_document(
        self,
        doc_content: DocumentContent,
        domain: str
    ):
        """Process a scraped document"""
        logger.info(f"Processing document: {doc_content.title} (ID: {doc_content.doc_id})")
        
        # Check for duplicates
        if self._is_duplicate(doc_content.doc_id):
            logger.info(f"Skipping duplicate document: {doc_content.doc_id}")
            return
        
        self.stats['documents_processed'] += 1
        
        # Convert to dict format
        doc_dict = {
            'doc_id': doc_content.doc_id,
            'title': doc_content.title,
            'version': doc_content.version,
            'source_url': doc_content.source_url,
            'main_content': doc_content.main_content,
            'footer_content': doc_content.footer_content,
            'enums': doc_content.enums,
            'definitions': doc_content.definitions,
            'hyperlinks': doc_content.hyperlinks
        }
        
        # Chunk document
        chunks = self.chunker.chunk_document(doc_dict, domain)
        self.stats['chunks_created'] += len(chunks)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Get footnote mapping for this document
        footnote_map = self.footnote_mappings.get(doc_content.doc_id, {})
        
        # Enrich chunks with footnotes
        enriched_chunks = []
        for chunk in chunks:
            enriched_content, refs = self.footnote_extractor.enrich_chunk_with_footnotes(
                chunk['content'], 
                footnote_map
            )
            chunk['content'] = enriched_content
            chunk['metadata']['has_footnotes'] = len(refs) > 0
            chunk['metadata']['footnote_refs'] = refs
            enriched_chunks.append(chunk)
        
        # Create additional footnote reference chunks
        if footnote_map and footnote_map.get('mappings'):
            footnote_chunks = self.footnote_extractor.create_footnote_chunks(footnote_map)
            for fn_chunk in footnote_chunks:
                # Format as regular chunk
                chunk = {
                    'chunk_id': str(uuid.uuid4()),
                    'content': fn_chunk['content'],
                    'metadata': {
                        'doc_id': fn_chunk['doc_id'],
                        'doc_title': fn_chunk['doc_title'],
                        'version': fn_chunk['version'],
                        'chunk_type': fn_chunk['chunk_type'],
                        'domain': domain,
                        'footnote_prefix': fn_chunk.get('footnote_prefix', '')
                    }
                }
                enriched_chunks.append(chunk)
                self.stats['footnote_chunks_created'] += 1
        
        # Generate embeddings
        embedded_chunks = self._generate_embeddings(enriched_chunks)
        self.stats['embeddings_generated'] += len(embedded_chunks)
        
        # Store in Qdrant
        self._store_chunks(embedded_chunks)
        self.stats['chunks_stored'] += len(embedded_chunks)
    
    def _generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """Generate embeddings for chunks"""
        embedded_chunks = []
        
        for chunk in chunks:
            # Check cache if enabled
            if self.embedding_cache:
                cached_embedding = self.embedding_cache.get(chunk['content'])
                if cached_embedding:
                    embedding = cached_embedding
                else:
                    # Generate new embedding
                    embedding = self.embedder.generate_embedding(chunk['content'])
                    self.embedding_cache.set(chunk['content'], embedding)
            else:
                embedding = self.embedder.generate_embedding(chunk['content'])
            
            # Create DocumentChunk object
            doc_chunk = DocumentChunk(
                chunk_id=chunk['chunk_id'],
                doc_id=chunk['metadata']['doc_id'],
                content=chunk['content'],
                metadata=chunk['metadata'],
                embedding=embedding
            )
            embedded_chunks.append(doc_chunk)
        
        return embedded_chunks
    
    def _store_chunks(self, chunks: List[DocumentChunk]):
        """Store chunks in Qdrant"""
        self.vector_store.upsert_chunks(chunks, batch_size=50)
    
    def _is_duplicate(self, doc_id: str) -> bool:
        """Check if document already exists in vector store"""
        try:
            # Check if any chunks exist for this doc_id
            results = self.vector_store.get_chunks_by_doc_id(doc_id, limit=1)
            return len(results) > 0
        except:
            return False
    
    def query_documents(
        self,
        query: str,
        domain: Optional[str] = None,
        version: Optional[str] = None,
        limit: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Query the vector store"""
        logger.info(f"[IngestionPipeline] Querying for: '{query}' with filters: domain={domain}, version={version}, threshold={score_threshold}")
        
        # Generate query embedding
        query_embedding = self.embedder.generate_query_embedding(query)
        logger.info(f"[IngestionPipeline] Generated query embedding of length: {len(query_embedding)}")
        
        # Build filters
        filters = {}
        if domain:
            filters['domain'] = domain
        if version:
            filters['version'] = version
        logger.info(f"[IngestionPipeline] Search filters: {filters}")
        
        # Search
        results = self.vector_store.search(
            query_embedding=query_embedding,
            limit=limit,
            filters=filters,
            score_threshold=score_threshold
        )
        logger.info(f"[IngestionPipeline] Vector store returned {len(results)} results")
        
        return results
    
    def update_document(self, doc_url: str, domain: str = "retail"):
        """Update a specific document"""
        # Extract doc_id
        doc_id = self.scraper.extract_doc_id(doc_url)
        
        # Delete existing chunks
        self.vector_store.delete_document(doc_id)
        
        # Re-process document
        self._process_document(doc_url, domain, follow_links=False)
        
        logger.info(f"Updated document: {doc_id}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        collection_info = self.vector_store.get_collection_info()
        
        return {
            'pipeline_stats': self.stats,
            'vector_store': collection_info,
            'health': {
                'qdrant': self.vector_store.health_check(),
                'gemini': self._check_gemini_health()
            }
        }
    
    def _check_gemini_health(self) -> bool:
        """Check if Gemini API is accessible"""
        try:
            test_embedding = self.embedder.generate_embedding("test")
            return len(test_embedding) > 0
        except:
            return False
    
    def export_knowledge_base(self, output_dir: str):
        """Export the knowledge base for backup"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export scraped documents
        self.scraper.save_scraped_content(str(output_path / "scraped_docs"))
        
        # Export pipeline stats
        with open(output_path / "pipeline_stats.json", 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Exported knowledge base to {output_dir}")
    
    def load_from_scraped_docs(self, scraped_docs_dir: str, domain: str = "retail"):
        """Load and process documents from previously scraped folder"""
        logger.info(f"Loading documents from {scraped_docs_dir}")
        
        # Load metadata
        metadata_file = os.path.join(scraped_docs_dir, "_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                logger.info(f"Found {metadata['total_documents']} documents scraped on {metadata['scrape_timestamp']}")
        
        # Load each document
        doc_files = [f for f in os.listdir(scraped_docs_dir) if f.endswith('.json') and not f.startswith('_')]
        
        for doc_file in doc_files:
            doc_path = os.path.join(scraped_docs_dir, doc_file)
            with open(doc_path, 'r') as f:
                doc_data = json.load(f)
                
            # Convert back to DocumentContent
            doc_content = DocumentContent(
                doc_id=doc_data['doc_id'],
                title=doc_data['title'],
                main_content=doc_data['main_content'],
                footer_content=doc_data['footer_content'],
                enums=doc_data['enums'],
                definitions=doc_data['definitions'],
                hyperlinks=doc_data['hyperlinks'],
                version=doc_data['version'],
                source_url=doc_data['source_url']
            )
            
            # Process the document
            try:
                self._process_scraped_document(doc_content, domain)
            except Exception as e:
                logger.error(f"Failed to process {doc_file}: {e}")
        
        return self.stats


