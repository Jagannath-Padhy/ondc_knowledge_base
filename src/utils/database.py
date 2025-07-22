from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from loguru import logger
from typing import Optional
from .config import Config, ConfigError

class QdrantDBError(Exception):
    """Custom exception for Qdrant database errors."""
    pass

class QdrantDB:
    """
    Handles Qdrant database connection, collection setup, and provides a client instance.
    """
    def __init__(self, config: Optional[Config] = None, force_recreate: bool = False):
        self.config = config or Config()
        self.client = self._connect()
        self._ensure_collection(force_recreate=force_recreate)

    def _connect(self) -> QdrantClient:
        try:
            host = self.config.qdrant["host"]
            port = self.config.qdrant["port"]
            logger.info(f"Connecting to Qdrant at {host}:{port}")
            client = QdrantClient(host=host, port=port)
            logger.info("Qdrant connection established.")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise QdrantDBError(f"Failed to connect to Qdrant: {e}")

    def _ensure_collection(self, force_recreate: bool = False):
        collection_name = self.config.qdrant["collection_name"]
        vector_size = 768  # Gemini embedding size
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if force_recreate and collection_name in collections:
                self.client.delete_collection(collection_name=collection_name)
                logger.info(f"Deleted existing Qdrant collection: {collection_name}")
            if collection_name not in [c.name for c in self.client.get_collections().collections]:
                logger.info(f"Creating Qdrant collection: {collection_name}")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )
                logger.info(f"Collection '{collection_name}' created.")
            else:
                logger.info(f"Collection '{collection_name}' already exists.")
        except Exception as e:
            if "already exists" in str(e) or "409" in str(e):
                logger.warning(f"Collection '{collection_name}' already exists. Proceeding.")
            else:
                logger.error(f"Error ensuring Qdrant collection: {e}")
                raise QdrantDBError(f"Error ensuring Qdrant collection: {e}")

    def get_client(self) -> QdrantClient:
        """
        Returns the Qdrant client instance for further operations.
        """
        return self.client 