import os
import time
import logging
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiEmbeddings:
    """Handles embedding generation using Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        
        genai.configure(api_key=self.api_key)
        
        # Using the embedding model
        self.embedding_model = "models/text-embedding-004"
        self.batch_size = 100  # Gemini supports batch embeddings
        self.max_retries = 3
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate_embedding(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        """Generate embedding for a single text"""
        try:
            # Limit text size to prevent payload size errors (Gemini limit ~30KB)
            max_chars = 25000  # Safe limit well below 30KB
            if len(text) > max_chars:
                logger.warning(f"Text truncated from {len(text)} to {max_chars} characters")
                text = text[:max_chars] + "..."
            
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type=task_type
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(
        self, 
        texts: List[str], 
        task_type: str = "RETRIEVAL_DOCUMENT",
        show_progress: bool = True
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches"""
        embeddings = []
        total_texts = len(texts)
        
        logger.info(f"Generating embeddings for {total_texts} texts")
        
        for i in range(0, total_texts, self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = []
            
            for text in batch:
                try:
                    embedding = self.generate_embedding(text, task_type)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Failed to generate embedding for text: {text[:50]}...")
                    # Use zero vector as fallback
                    batch_embeddings.append([0.0] * 768)  # Gemini embedding dimension
            
            embeddings.extend(batch_embeddings)
            
            if show_progress:
                progress = (i + len(batch)) / total_texts * 100
                logger.info(f"Progress: {progress:.1f}% ({i + len(batch)}/{total_texts})")
            
            # Rate limiting
            time.sleep(0.1)
        
        return embeddings
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a search query"""
        return self.generate_embedding(query, task_type="RETRIEVAL_QUERY")
    
    def embed_text(self, text: str) -> List[float]:
        """Alias for generate_embedding for compatibility"""
        return self.generate_embedding(text)
    
    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Embed a list of document chunks"""
        embedded_docs = []
        
        # Extract texts
        texts = []
        for doc in documents:
            # Create enriched text for better embeddings
            enriched_text = self._create_enriched_text(doc)
            texts.append(enriched_text)
        
        # Generate embeddings
        embeddings = self.generate_embeddings_batch(texts)
        
        # Add embeddings to documents
        for doc, embedding in zip(documents, embeddings):
            doc_with_embedding = doc.copy()
            doc_with_embedding['embedding'] = embedding
            embedded_docs.append(doc_with_embedding)
        
        return embedded_docs
    
    def _create_enriched_text(self, doc: Dict[str, Any]) -> str:
        """Create enriched text for better embeddings"""
        content = doc['content']
        metadata = doc.get('metadata', {})
        
        # Add contextual information based on chunk type
        chunk_type = metadata.get('chunk_type', 'content')
        
        if chunk_type == 'api_endpoint':
            # Add API-specific context
            method = metadata.get('http_method', '')
            path = metadata.get('api_path', '')
            enriched = f"API Endpoint: {method} {path}\n{content}"
            
        elif chunk_type == 'schema':
            # Add schema context
            schema_type = metadata.get('schema_type', 'schema')
            enriched = f"JSON Schema ({schema_type}):\n{content}"
            
        elif chunk_type == 'enum':
            # Add enum context
            enum_name = metadata.get('enum_name', '')
            enriched = f"Enumeration {enum_name}:\n{content}"
            
        elif chunk_type == 'definition':
            # Add definition context
            enriched = f"Term Definition:\n{content}"
            
        else:
            # Regular content
            section = metadata.get('section', '')
            if section:
                enriched = f"Section: {section}\n{content}"
            else:
                enriched = content
        
        # Add version info if available
        version = metadata.get('version', '')
        if version:
            enriched = f"[Version {version}] {enriched}"
        
        return enriched
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings"""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_similar_chunks(
        self, 
        query_embedding: List[float], 
        chunk_embeddings: List[Dict[str, Any]], 
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find most similar chunks to a query"""
        similarities = []
        
        for chunk in chunk_embeddings:
            if 'embedding' not in chunk:
                continue
                
            similarity = self.compute_similarity(query_embedding, chunk['embedding'])
            
            if similarity >= threshold:
                similarities.append({
                    'chunk': chunk,
                    'similarity': similarity
                })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]


class EmbeddingCache:
    """Simple cache for embeddings to avoid recomputation"""
    
    def __init__(self, cache_dir: str = "embedding_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        import hashlib
        return hashlib.sha256(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get cached embedding if exists"""
        cache_key = self.get_cache_key(text)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.npy")
        
        if os.path.exists(cache_path):
            return np.load(cache_path).tolist()
        
        return None
    
    def set(self, text: str, embedding: List[float]):
        """Cache an embedding"""
        cache_key = self.get_cache_key(text)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.npy")
        
        np.save(cache_path, np.array(embedding))
    
    def clear(self):
        """Clear all cached embeddings"""
        import shutil
        shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)


