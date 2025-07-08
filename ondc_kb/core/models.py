"""
Core data models for ONDC Knowledge Base
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class DocumentType(str, Enum):
    """Supported document types"""
    GOOGLE_DOC = "google_doc"
    GITHUB_MARKDOWN = "github_markdown"
    GOOGLE_SHEET = "google_sheet"
    MANUAL_UPLOAD = "manual_upload"
    ENRICHED_DOC = "enriched_doc"
    JSON_SCHEMA = "json_schema"


class DocumentStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    OUTDATED = "outdated"


class ChunkType(str, Enum):
    """Types of document chunks"""
    CONTENT = "content"
    HEADER = "header"
    FOOTER = "footer"
    TABLE = "table"
    CODE = "code"
    FOOTNOTE = "footnote"
    DEFINITION = "definition"
    PAYLOAD_EXAMPLE = "payload_example"


class DocumentMetadata(BaseModel):
    """Document metadata"""
    title: str
    author: Optional[str] = None
    version: Optional[str] = None
    domain: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    source_url: Optional[str] = None
    file_size: Optional[int] = None
    checksum: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    
class DocumentChunk(BaseModel):
    """A chunk of document content"""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    chunk_type: ChunkType = ChunkType.CONTENT
    chunk_index: int = 0
    total_chunks: int = 1
    section_title: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    
class Document(BaseModel):
    """Document model"""
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_type: DocumentType
    status: DocumentStatus = DocumentStatus.PENDING
    metadata: DocumentMetadata
    content: str = ""
    chunks: List[DocumentChunk] = Field(default_factory=list)
    footnotes: Dict[str, Any] = Field(default_factory=dict)
    definitions: Dict[str, str] = Field(default_factory=dict)
    payload_examples: List[Dict[str, Any]] = Field(default_factory=list)
    enrichment_data: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('doc_id')
    def validate_doc_id(cls, v):
        if not v:
            return str(uuid.uuid4())
        return v


class QueryRequest(BaseModel):
    """Query request model"""
    query: str = Field(..., min_length=1, max_length=1000)
    domain: Optional[str] = None
    version: Optional[str] = None
    doc_type: Optional[DocumentType] = None
    limit: int = Field(default=10, ge=1, le=100)
    score_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    include_metadata: bool = True
    include_chunks: bool = False
    
    
class QueryResult(BaseModel):
    """Individual query result"""
    doc_id: str
    chunk_id: Optional[str] = None
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_info: Optional[Dict[str, Any]] = None
    
    
class QueryResponse(BaseModel):
    """Query response model"""
    query: str
    results: List[QueryResult]
    total_results: int
    query_time: float
    generated_answer: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    
    
class ValidationRequest(BaseModel):
    """Payload validation request"""
    payload: Dict[str, Any]
    schema_name: str
    domain: str
    version: str
    action: str
    
    
class ValidationResult(BaseModel):
    """Validation result"""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    schema_used: str
    validation_time: float
    
    
class IngestionJob(BaseModel):
    """Document ingestion job"""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_url: str
    doc_type: DocumentType
    status: DocumentStatus = DocumentStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress: int = 0
    total_documents: int = 0
    processed_documents: int = 0
    
    
class SystemHealth(BaseModel):
    """System health status"""
    status: str = "healthy"
    components: Dict[str, bool] = Field(default_factory=dict)
    uptime: float = 0.0
    last_check: datetime = Field(default_factory=datetime.now)
    version: str = "2.0.0"
    
    
class MCPToolRequest(BaseModel):
    """MCP tool request"""
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    
    
class MCPToolResponse(BaseModel):
    """MCP tool response"""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0