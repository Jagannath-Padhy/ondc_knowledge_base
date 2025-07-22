"""
Clean up Qdrant database for fresh start
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models
from loguru import logger
import sys

def cleanup_qdrant():
    """Clean up Qdrant database"""
    try:
        # Connect to Qdrant
        client = QdrantClient(host='localhost', port=6333)
        logger.info("✅ Connected to Qdrant")
        
        collection_name = 'ondc_knowledge'
        
        # Check if collection exists
        try:
            collections = client.get_collections()
            collection_exists = any(c.name == collection_name for c in collections.collections)
            
            if collection_exists:
                # Get collection info
                try:
                    collection_info = client.get_collection(collection_name)
                    point_count = collection_info.points_count
                    logger.info(f"📊 Collection '{collection_name}' exists with {point_count} points")
                except:
                    logger.info(f"📊 Collection '{collection_name}' exists")
                    point_count = "unknown"
                
                # Delete the collection
                client.delete_collection(collection_name)
                logger.info(f"🗑️ Deleted collection '{collection_name}'")
            else:
                logger.info(f"Collection '{collection_name}' does not exist")
            
        except Exception as e:
            logger.error(f"Error checking collection: {e}")
        
        # Recreate the collection with proper configuration
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=768,  # Gemini embedding size
                distance=models.Distance.COSINE
            ),
            # Increase payload size limits
            hnsw_config=models.HnswConfigDiff(
                payload_m=16,
                m=16
            )
        )
        logger.info(f"✨ Created fresh collection '{collection_name}'")
        
        # Verify collection is empty
        try:
            collection_info = client.get_collection(collection_name)
            logger.info(f"✅ Collection now has {collection_info.points_count} points (should be 0)")
        except Exception as e:
            # Just verify it exists
            collections = client.get_collections()
            if any(c.name == collection_name for c in collections.collections):
                logger.info(f"✅ Collection '{collection_name}' created successfully")
            else:
                logger.error(f"Failed to create collection")
        
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")
        sys.exit(1)

def main():
    """Main entry point"""
    logger.info("🧹 Starting Qdrant cleanup...")
    
    # Confirm with user
    response = input("\n⚠️  This will DELETE all data in the 'ondc_knowledge' collection. Continue? (yes/no): ")
    if response.lower() != 'yes':
        logger.info("❌ Cleanup cancelled")
        return
    
    cleanup_qdrant()
    logger.info("\n✅ Cleanup complete! Database is ready for fresh ingestion.")

if __name__ == "__main__":
    main()