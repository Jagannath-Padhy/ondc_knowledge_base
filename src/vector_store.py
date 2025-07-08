import os
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest
)
from qdrant_client.http import models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of document for vector storage"""
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    chunk_index: int = 0
    total_chunks: int = 1
    embedding: Optional[List[float]] = None
    chunk_id: Optional[str] = None
    
    def __post_init__(self):
        if self.chunk_id is None:
            self.chunk_id = str(uuid.uuid4())


class QdrantVectorStore:
    """Manages Qdrant vector database operations"""
    
    def __init__(
        self, 
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        use_cloud: bool = False,
        collection_name: str = "ondc_docs"
    ):
        self.collection_name = collection_name
        self.client = self._initialize_client(host, port, api_key, use_cloud)
        self.vector_size = 768  # Gemini embedding size (text-embedding-004)
        
    def _initialize_client(
        self, 
        host: Optional[str], 
        port: Optional[int], 
        api_key: Optional[str],
        use_cloud: bool
    ) -> QdrantClient:
        """Initialize Qdrant client for local or cloud deployment"""
        if use_cloud:
            if not host or not api_key:
                raise ValueError("Cloud deployment requires host and api_key")
            logger.info(f"Connecting to Qdrant Cloud at {host}")
            return QdrantClient(
                url=host,
                api_key=api_key,
                https=True
            )
        else:
            # Local deployment
            host = host or "localhost"
            port = port or 6333
            logger.info(f"Connecting to local Qdrant at {host}:{port}")
            return QdrantClient(host=host, port=port)
    
    def create_collection(self, recreate: bool = False):
        """Create or recreate the collection"""
        collections = self.client.get_collections().collections
        exists = any(col.name == self.collection_name for col in collections)
        
        if exists and recreate:
            logger.info(f"Deleting existing collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
            exists = False
        
        if not exists:
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            
            # Create payload indices for efficient filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="doc_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="version",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="domain",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="chunk_type",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            # Add content_hash index for deduplication
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="content_hash",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
        else:
            logger.info(f"Collection {self.collection_name} already exists")
    
    def upsert_chunks(self, chunks: List[DocumentChunk], batch_size: int = 100):
        """Insert or update document chunks in Qdrant"""
        total_chunks = len(chunks)
        logger.info(f"Upserting {total_chunks} chunks to Qdrant")
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            points = []
            
            for chunk in batch:
                if not chunk.embedding:
                    logger.warning(f"Skipping chunk {chunk.chunk_id} - no embedding")
                    continue
                
                point = PointStruct(
                    id=chunk.chunk_id,
                    vector=chunk.embedding,
                    payload={
                        "doc_id": chunk.doc_id,
                        "content": chunk.content,
                        **chunk.metadata
                    }
                )
                points.append(point)
            
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"Upserted batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}")
    
    def search(
        self, 
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = 0.3
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        logger.info(f"[VectorStore] Searching with embedding length: {len(query_embedding)}, limit: {limit}, filters: {filters}, threshold: {score_threshold}")
        
        # Check collection status first
        try:
            collection_info = self.client.get_collection(self.collection_name)
            # Handle different response formats
            points_count = getattr(collection_info, 'points_count', None)
            if points_count is None and hasattr(collection_info, 'result'):
                points_count = getattr(collection_info.result, 'points_count', 0)
            logger.info(f"[VectorStore] Collection '{self.collection_name}' has {points_count or 0} points")
        except Exception as e:
            logger.debug(f"[VectorStore] Could not get detailed collection info: {e}")
        
        search_params = {
            "collection_name": self.collection_name,
            "query_vector": query_embedding,
            "limit": limit,
            "with_payload": True
        }
        
        # Add filters if provided
        if filters:
            conditions = []
            for field, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=value)
                    )
                )
            
            if conditions:
                search_params["query_filter"] = Filter(must=conditions)
        
        # Add score threshold
        if score_threshold is not None:
            search_params["score_threshold"] = score_threshold
            logger.info(f"[VectorStore] Using score threshold: {score_threshold}")
        else:
            # Use default threshold of 0.3
            search_params["score_threshold"] = 0.3
            logger.info(f"[VectorStore] Using default score threshold: 0.3")
        
        logger.info(f"[VectorStore] Executing search with params: collection={self.collection_name}, filters={filters}, threshold={score_threshold}")
        results = self.client.search(**search_params)
        logger.info(f"[VectorStore] Search returned {len(results)} raw results")
        
        # Format results
        formatted_results = []
        for hit in results:
            logger.debug(f"[VectorStore] Hit - ID: {hit.id}, Score: {hit.score}")
            formatted_results.append({
                "id": hit.id,
                "score": hit.score,
                "content": hit.payload.get("content", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "content"}
            })
        
        logger.info(f"[VectorStore] Returning {len(formatted_results)} formatted results")
        if formatted_results:
            logger.info(f"[VectorStore] Top result score: {formatted_results[0]['score']}")
        
        return formatted_results
    
    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a specific document"""
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id)
                    )
                ]
            ),
            with_payload=True,
            with_vectors=False,
            limit=1000
        )
        
        chunks = []
        for point in results[0]:
            chunks.append({
                "id": point.id,
                "content": point.payload.get("content", ""),
                "metadata": {k: v for k, v in point.payload.items() if k != "content"}
            })
        
        return chunks
    
    def get_chunks_by_doc_id(self, doc_id: str, limit: int = 1) -> List[Dict[str, Any]]:
        """Get chunks by document ID with limit (for duplicate checking)"""
        return self.get_document_chunks(doc_id)[:limit]
    
    def delete_document(self, doc_id: str):
        """Delete all chunks for a specific document"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_id)
                        )
                    ]
                )
            )
        )
        logger.info(f"Deleted all chunks for document: {doc_id}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": getattr(info, 'vectors_count', 0),
                "points_count": getattr(info, 'points_count', 0),
                "status": "available"
            }
        except Exception as e:
            logger.warning(f"Could not get detailed collection info: {e}")
            return {
                "name": self.collection_name,
                "vectors_count": 0,
                "points_count": 0,
                "status": "unknown"
            }
    
    def health_check(self) -> bool:
        """Check if Qdrant is healthy and accessible"""
        try:
            collections = self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
    
    def search_by_metadata(
        self,
        metadata_filters: Dict[str, Any],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for documents by metadata fields only"""
        conditions = []
        for field, value in metadata_filters.items():
            conditions.append(
                FieldCondition(
                    key=field,
                    match=MatchValue(value=value)
                )
            )
        
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=conditions),
                with_payload=True,
                with_vectors=False,
                limit=limit
            )
            
            formatted_results = []
            for point in results[0]:
                formatted_results.append({
                    "id": point.id,
                    "content": point.payload.get("content", ""),
                    "metadata": {k: v for k, v in point.payload.items() if k != "content"}
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []
    
    def update_chunk(self, chunk_id: str, new_chunk: 'DocumentChunk'):
        """Update an existing chunk with new data"""
        try:
            # Update the point with new embedding and payload
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=chunk_id,
                        vector=new_chunk.embedding,
                        payload={
                            "doc_id": new_chunk.doc_id,
                            "content": new_chunk.content,
                            **new_chunk.metadata
                        }
                    )
                ]
            )
            logger.info(f"Updated chunk {chunk_id}")
        except Exception as e:
            logger.error(f"Error updating chunk {chunk_id}: {e}")
    
    def store_chunks(self, chunks: List['DocumentChunk']):
        """Store chunks (wrapper for upsert_chunks)"""
        self.upsert_chunks(chunks)
    
    def delete_by_metadata(self, metadata_filters: Dict[str, Any]):
        """Delete chunks matching metadata filters"""
        conditions = []
        for field, value in metadata_filters.items():
            conditions.append(
                FieldCondition(
                    key=field,
                    match=MatchValue(value=value)
                )
            )
        
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=Filter(must=conditions)
                )
            )
            logger.info(f"Deleted chunks matching filters: {metadata_filters}")
        except Exception as e:
            logger.error(f"Error deleting by metadata: {e}")
    
    def create_payload_index_if_not_exists(self, field_name: str):
        """Create a payload index if it doesn't exist"""
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info(f"Created payload index for field: {field_name}")
        except Exception as e:
            # Index might already exist
            logger.debug(f"Payload index creation note: {e}")

