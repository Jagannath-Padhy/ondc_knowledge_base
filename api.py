#!/usr/bin/env python3
"""
ONDC Knowledge Base - FastAPI Service
RESTful API for querying and managing ONDC documentation
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from unified_ingestion import UnifiedIngestionPipeline, SourceType
from vector_store import QdrantVectorStore
from embeddings import GeminiEmbeddings
import google.generativeai as genai

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Pydantic models for API
class IngestionSource(BaseModel):
    type: str = Field(..., description="Source type: google_docs, github_markdown, google_sheets")
    url: str = Field(..., description="URL of the source")
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional options for the source")


class IngestionRequest(BaseModel):
    sources: List[IngestionSource] = Field(..., description="List of sources to ingest")
    domain: str = Field(default="retail", description="Domain for the documents")
    version: str = Field(default="1.2.0", description="Version of specifications")
    recreate: bool = Field(default=False, description="Whether to recreate the collection")


class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to search for")
    domain: str = Field(default="retail", description="Domain to search in")
    version: Optional[str] = Field(None, description="Specific version to search")
    limit: int = Field(default=5, ge=1, le=20, description="Number of results to return")


class ChatRequest(BaseModel):
    question: str = Field(..., description="Question to ask")
    domain: str = Field(default="retail", description="Domain for context")
    include_context: bool = Field(default=True, description="Include source context in response")


class HealthResponse(BaseModel):
    status: str
    services: Dict[str, bool]
    timestamp: str


class StatsResponse(BaseModel):
    collection_name: str
    total_documents: int
    vector_size: int
    status: str


class QueryResult(BaseModel):
    content: str
    score: float
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    question: str
    results: List[QueryResult]
    total: int


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None


class IngestionResponse(BaseModel):
    status: str
    stats: Dict[str, Any]
    message: str


# Global instance
kb_instance = None


class ONDCKnowledgeBase:
    """Main class for ONDC Knowledge Base operations"""
    
    def __init__(self, config_path: str = "config/config.json"):
        """Initialize with configuration"""
        self.config = self._load_config(config_path)
        
        # Initialize components
        self._init_components()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            "gemini_api_key": os.getenv("GEMINI_API_KEY"),
            "github_token": os.getenv("GITHUB_TOKEN"),
            "google_credentials_path": os.getenv("GOOGLE_CREDENTIALS_PATH"),
            "qdrant": {
                "host": os.getenv("QDRANT_HOST", "localhost"),
                "port": int(os.getenv("QDRANT_PORT", 6333)),
                "collection_name": "ondc_docs"
            },
            "embedding": {
                "cache_enabled": True,
                "cache_dir": "embedding_cache"
            }
        }
    
    def _init_components(self):
        """Initialize knowledge base components"""
        # Vector store
        self.vector_store = QdrantVectorStore(
            host=self.config["qdrant"]["host"],
            port=self.config["qdrant"]["port"],
            collection_name=self.config["qdrant"]["collection_name"]
        )
        
        # Embeddings
        self.embedder = GeminiEmbeddings(
            api_key=self.config["gemini_api_key"]
        )
        
        # Ingestion pipeline
        self.ingestion_pipeline = UnifiedIngestionPipeline(
            gemini_api_key=self.config["gemini_api_key"],
            qdrant_config={
                "host": self.config["qdrant"]["host"],
                "port": self.config["qdrant"]["port"],
                "collection_name": self.config["qdrant"]["collection_name"]
            },
            github_token=self.config.get("github_token"),
            google_credentials_path=self.config.get("google_credentials_path"),
            use_cache=self.config.get("embedding", {}).get("cache_enabled", True)
        )
        
        # Configure Gemini
        genai.configure(api_key=self.config["gemini_api_key"])
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp')
    
    def ingest(
        self,
        sources: List[Dict[str, Any]],
        domain: str = "retail",
        version: str = "1.2.0",
        recreate: bool = False
    ) -> Dict[str, Any]:
        """Ingest documents from multiple sources"""
        logger.info(f"Starting ingestion for domain: {domain}, version: {version}")
        
        return self.ingestion_pipeline.ingest_from_sources(
            sources=sources,
            domain=domain,
            version=version,
            recreate_collection=recreate,
            save_scraped=True,
            scraped_dir=f"scraped_data/{domain}/{version}"
        )
    
    def query(
        self,
        question: str,
        domain: str = "retail",
        version: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Query the knowledge base"""
        logger.info(f"Querying: '{question}' in domain: {domain}")
        
        # Generate embedding for query
        query_embedding = self.embedder.generate_embedding(question)
        
        # Build filters
        filters = {"domain": domain}
        if version:
            filters["version"] = version
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            limit=limit,
            filters=filters
        )
        
        logger.info(f"Found {len(results)} results")
        return results
    
    def chat(self, question: str, domain: str = "retail") -> Dict[str, Any]:
        """Interactive chat with context"""
        # Get relevant documents
        results = self.query(question, domain=domain, limit=3)
        
        if results:
            # Combine top results
            context_text = "\n\n".join([
                f"[{r['metadata'].get('chunk_type', 'content')}] {r['content']}"
                for r in results
            ])
            
            # Create prompt for Gemini
            prompt = f"""Based on the following ONDC documentation context, please answer the user's question.

Context:
{context_text}

Question: {question}

Please provide a clear and concise answer based on the provided context. If the context doesn't contain enough information to fully answer the question, mention what information is missing."""

            # Use Gemini to generate answer
            try:
                response = self.gemini_model.generate_content(prompt)
                return {
                    "question": question,
                    "answer": response.text,
                    "sources": [
                        {
                            "title": r['metadata'].get('doc_title', 'Unknown'),
                            "type": r['metadata'].get('chunk_type', 'content'),
                            "score": r.get('score', 0)
                        }
                        for r in results
                    ]
                }
            except Exception as e:
                logger.error(f"Gemini generation failed: {e}")
                # Fallback to simple response
                return {
                    "question": question,
                    "answer": results[0].get('content', '')[:500] + "...",
                    "sources": [{"title": results[0].get('metadata', {}).get('doc_title', 'Unknown')}]
                }
        else:
            return {
                "question": question,
                "answer": "I couldn't find relevant information in the ONDC documentation to answer your question.",
                "sources": []
            }
    
    def check_health(self) -> Dict[str, bool]:
        """Check health of all components"""
        health = {}
        
        # Check Qdrant
        try:
            collections = self.vector_store.client.get_collections()
            health["qdrant"] = True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            health["qdrant"] = False
        
        # Check Gemini
        try:
            test_embedding = self.embedder.generate_embedding("test")
            health["gemini"] = len(test_embedding) == 768
        except Exception as e:
            logger.error(f"Gemini health check failed: {e}")
            health["gemini"] = False
        
        return health
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        try:
            collection_info = self.vector_store.client.get_collection(
                self.config["qdrant"]["collection_name"]
            )
            
            return {
                "collection_name": self.config["qdrant"]["collection_name"],
                "total_documents": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "status": collection_info.status
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}


# FastAPI app lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global kb_instance
    logger.info("Initializing ONDC Knowledge Base...")
    kb_instance = ONDCKnowledgeBase()
    logger.info("Knowledge Base initialized successfully")
    yield
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="ONDC Knowledge Base API",
    description="API for querying and managing ONDC documentation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "ONDC Knowledge Base API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "stats": "/stats",
            "query": "/query",
            "chat": "/chat",
            "ingest": "/ingest"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check system health"""
    health = kb_instance.check_health()
    
    return HealthResponse(
        status="healthy" if all(health.values()) else "unhealthy",
        services=health,
        timestamp=datetime.now().isoformat()
    )


@app.get("/stats", response_model=StatsResponse, tags=["Stats"])
async def get_stats():
    """Get knowledge base statistics"""
    stats = kb_instance.get_stats()
    
    if "error" in stats:
        raise HTTPException(status_code=500, detail=stats["error"])
    
    return StatsResponse(**stats)


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base for relevant documents"""
    try:
        results = kb_instance.query(
            question=request.question,
            domain=request.domain,
            version=request.version,
            limit=request.limit
        )
        
        return QueryResponse(
            question=request.question,
            results=[
                QueryResult(
                    content=r.get('content', ''),
                    score=r.get('score', 0.0),
                    metadata=r.get('metadata', {})
                )
                for r in results
            ],
            total=len(results)
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_with_knowledge(request: ChatRequest):
    """Chat with the knowledge base using AI"""
    try:
        result = kb_instance.chat(
            question=request.question,
            domain=request.domain
        )
        
        return ChatResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result.get("sources") if request.include_context else None
        )
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestionResponse, tags=["Ingestion"])
async def ingest_documents(request: IngestionRequest):
    """Ingest documents from various sources"""
    try:
        # Convert Pydantic models to dicts
        sources = [source.dict() for source in request.sources]
        
        result = kb_instance.ingest(
            sources=sources,
            domain=request.domain,
            version=request.version,
            recreate=request.recreate
        )
        
        return IngestionResponse(
            status="success",
            stats=result.get("stats", {}),
            message=f"Successfully ingested documents for domain: {request.domain}, version: {request.version}"
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )