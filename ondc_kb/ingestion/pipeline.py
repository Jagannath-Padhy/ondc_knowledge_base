"""
Main document ingestion pipeline
Orchestrates document parsing, chunking, enrichment and storage
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger
import time

from ..core.config import Settings
from ..core.models import (
    Document, DocumentType, IngestionJob, DocumentStatus,
    DocumentMetadata, DocumentChunk
)
from ..core.exceptions import IngestionError, ConfigurationError

from .parsers import (
    GoogleDocsParser, GitHubMarkdownParser, 
    GoogleSheetsParser, ManualUploadParser, JSONSchemaParser
)
from .chunkers import DocumentChunker, ChunkingStrategy
from .enrichers import DocumentEnricher


class DocumentIngestionPipeline:
    """
    Main document ingestion pipeline
    Handles end-to-end document processing from source to storage
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.jobs: Dict[str, IngestionJob] = {}
        
        # Initialize parsers
        self.parsers = {
            DocumentType.GOOGLE_DOC: GoogleDocsParser(settings),
            DocumentType.GITHUB_MARKDOWN: GitHubMarkdownParser(settings),
            DocumentType.GOOGLE_SHEET: GoogleSheetsParser(settings),
            DocumentType.MANUAL_UPLOAD: ManualUploadParser(settings),
            DocumentType.JSON_SCHEMA: JSONSchemaParser(settings)
        }
        
        # Initialize processors
        self.chunker = DocumentChunker(settings)
        self.enricher = DocumentEnricher(settings)
        
        logger.info("Document ingestion pipeline initialized")
    
    async def ingest_document(self, source_url: str, doc_type: DocumentType,
                            domain: str = None, version: str = None) -> IngestionJob:
        """
        Ingest a single document
        
        Args:
            source_url: Source URL or path
            doc_type: Type of document
            domain: ONDC domain (retail, logistics, etc.)
            version: Document version
            
        Returns:
            IngestionJob: Job tracking information
        """
        
        # Create ingestion job
        job = IngestionJob(
            source_url=source_url,
            doc_type=doc_type,
            status=DocumentStatus.PENDING,
            total_documents=1
        )
        
        self.jobs[job.job_id] = job
        logger.info(f"Created ingestion job {job.job_id} for {source_url}")
        
        # Start processing asynchronously
        asyncio.create_task(self._process_document(job, domain, version))
        
        return job
    
    async def _process_document(self, job: IngestionJob, domain: str = None, 
                               version: str = None):
        """
        Process a document through the ingestion pipeline
        
        Args:
            job: Ingestion job
            domain: ONDC domain
            version: Document version
        """
        
        start_time = time.time()
        
        try:
            # Update job status
            job.status = DocumentStatus.PROCESSING
            job.started_at = datetime.now()
            job.progress = 10
            
            logger.info(f"Starting document processing: {job.source_url}")
            
            # Step 1: Parse document
            parser = self.parsers.get(job.doc_type)
            if not parser:
                raise IngestionError(f"No parser available for {job.doc_type}")
            
            document = await parser.parse(job.source_url)
            job.progress = 30
            
            # Step 2: Set metadata
            document.metadata.domain = domain
            document.metadata.version = version
            job.progress = 40
            
            # Step 3: Chunk document
            chunks = await self.chunker.chunk_document(document)
            document.chunks = chunks
            job.progress = 60
            
            # Step 4: Enrich document
            enriched_document = await self.enricher.enrich_document(document)
            job.progress = 80
            
            # Step 5: Store document
            await self._store_document(enriched_document)
            job.progress = 100
            
            # Update job completion
            job.status = DocumentStatus.COMPLETED
            job.completed_at = datetime.now()
            job.processed_documents = 1
            
            processing_time = time.time() - start_time
            logger.info(f"Document processing completed in {processing_time:.2f}s: {job.source_url}")
            
        except Exception as e:
            # Handle errors
            job.status = DocumentStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            
            logger.error(f"Document processing failed: {e}")
            raise IngestionError(f"Document processing failed: {e}")
    
    async def _store_document(self, document: Document):
        """
        Store document in vector store and metadata store
        
        Args:
            document: Processed document
        """
        
        try:
            # Import here to avoid circular imports
            from ..storage import VectorStore, MetadataStore
            
            # Store in metadata store
            metadata_store = MetadataStore(self.settings)
            await metadata_store.store_document(document)
            
            # Store in vector store
            vector_store = VectorStore(self.settings)
            await vector_store.store_document(document)
            
            logger.info(f"Document stored successfully: {document.doc_id}")
            
        except Exception as e:
            logger.error(f"Failed to store document: {e}")
            raise IngestionError(f"Storage failed: {e}")
    
    async def ingest_documents_batch(self, sources: List[Dict[str, Any]]) -> List[IngestionJob]:
        """
        Ingest multiple documents in batch
        
        Args:
            sources: List of source configurations
            
        Returns:
            List[IngestionJob]: Job tracking information
        """
        
        jobs = []
        
        for source in sources:
            job = await self.ingest_document(
                source_url=source['url'],
                doc_type=DocumentType(source['type']),
                domain=source.get('domain'),
                version=source.get('version')
            )
            jobs.append(job)
        
        logger.info(f"Batch ingestion started: {len(jobs)} jobs created")
        return jobs
    
    async def get_job_status(self, job_id: str) -> Optional[IngestionJob]:
        """
        Get job status by ID
        
        Args:
            job_id: Job ID
            
        Returns:
            IngestionJob or None
        """
        
        return self.jobs.get(job_id)
    
    async def list_jobs(self, status: DocumentStatus = None) -> List[IngestionJob]:
        """
        List all jobs with optional status filter
        
        Args:
            status: Optional status filter
            
        Returns:
            List[IngestionJob]: List of jobs
        """
        
        jobs = list(self.jobs.values())
        
        if status:
            jobs = [job for job in jobs if job.status == status]
        
        return jobs
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job
        
        Args:
            job_id: Job ID
            
        Returns:
            bool: Success status
        """
        
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status == DocumentStatus.PROCESSING:
            job.status = DocumentStatus.FAILED
            job.error_message = "Job cancelled by user"
            job.completed_at = datetime.now()
            
            logger.info(f"Job cancelled: {job_id}")
            return True
        
        return False
    
    async def retry_job(self, job_id: str) -> bool:
        """
        Retry a failed job
        
        Args:
            job_id: Job ID
            
        Returns:
            bool: Success status
        """
        
        job = self.jobs.get(job_id)
        if not job or job.status != DocumentStatus.FAILED:
            return False
        
        # Reset job status
        job.status = DocumentStatus.PENDING
        job.error_message = None
        job.progress = 0
        job.started_at = None
        job.completed_at = None
        
        # Restart processing
        asyncio.create_task(self._process_document(job))
        
        logger.info(f"Job retried: {job_id}")
        return True
    
    async def clear_completed_jobs(self) -> int:
        """
        Clear completed jobs from memory
        
        Returns:
            int: Number of jobs cleared
        """
        
        completed_jobs = [
            job_id for job_id, job in self.jobs.items()
            if job.status in [DocumentStatus.COMPLETED, DocumentStatus.FAILED]
        ]
        
        for job_id in completed_jobs:
            del self.jobs[job_id]
        
        logger.info(f"Cleared {len(completed_jobs)} completed jobs")
        return len(completed_jobs)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get ingestion pipeline statistics
        
        Returns:
            Dict[str, Any]: Statistics
        """
        
        total_jobs = len(self.jobs)
        completed_jobs = len([j for j in self.jobs.values() if j.status == DocumentStatus.COMPLETED])
        failed_jobs = len([j for j in self.jobs.values() if j.status == DocumentStatus.FAILED])
        processing_jobs = len([j for j in self.jobs.values() if j.status == DocumentStatus.PROCESSING])
        
        return {
            'total_jobs': total_jobs,
            'completed_jobs': completed_jobs,
            'failed_jobs': failed_jobs,
            'processing_jobs': processing_jobs,
            'success_rate': (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0,
            'active_parsers': list(self.parsers.keys()),
            'supported_formats': [doc_type.value for doc_type in DocumentType]
        }