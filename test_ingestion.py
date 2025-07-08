#!/usr/bin/env python3
"""
Test script for the enhanced ingestion pipeline
"""

import asyncio
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_single_url_ingestion():
    """Test ingesting a single URL with recursive scraping"""
    from main import ONDCKnowledgeBase
    
    logger.info("=== Testing Single URL Ingestion ===")
    
    # Initialize knowledge base
    kb = ONDCKnowledgeBase()
    
    # Test URL
    test_url = "https://docs.google.com/document/d/1brvcltG_DagZ3kGr1ZZQk4hG4tze3zvcxmGV4NMTzr8/edit"
    
    # Run ingestion
    result = await kb.ingest(
        url=test_url,
        domain="retail",
        version="1.2.0",
        follow_links=True,
        check_updates=True,
        max_depth=2
    )
    
    logger.info(f"Ingestion result: {json.dumps(result, indent=2)}")
    
    # Test query
    logger.info("\n=== Testing Query ===")
    results = kb.query("what is force cancellation?", domain="retail", limit=3)
    
    if results:
        logger.info(f"Found {len(results)} results")
        for i, result in enumerate(results, 1):
            logger.info(f"\nResult {i}:")
            logger.info(f"Score: {result.get('score', 0):.3f}")
            logger.info(f"Content: {result.get('content', '')[:200]}...")
    else:
        logger.info("No results found")
    
    return result

async def test_config_ingestion():
    """Test ingesting from startup documents in config"""
    from main import ONDCKnowledgeBase
    
    logger.info("\n=== Testing Config-based Ingestion ===")
    
    # Initialize knowledge base
    kb = ONDCKnowledgeBase()
    
    # Run ingestion from config
    result = await kb.ingest(
        domain="retail",
        check_updates=True
    )
    
    logger.info(f"Config ingestion result: {json.dumps(result, indent=2)}")
    
    return result

async def test_deduplication():
    """Test that deduplication is working"""
    from main import ONDCKnowledgeBase
    
    logger.info("\n=== Testing Deduplication ===")
    
    # Initialize knowledge base
    kb = ONDCKnowledgeBase()
    
    # Get initial stats
    initial_stats = kb.get_stats()
    logger.info(f"Initial document count: {initial_stats.get('total_documents', 0)}")
    
    # Run ingestion twice with the same URL
    test_url = "https://docs.google.com/document/d/1E2OyVXh34YNEVOnS4rO3nPVoqTO3RsG2uh-BqnSrqgY/edit"
    
    # First ingestion
    result1 = await kb.ingest(
        url=test_url,
        domain="retail",
        version="1.2.5",
        follow_links=False,
        check_updates=False  # Force update
    )
    
    stats1 = kb.get_stats()
    logger.info(f"After first ingestion: {stats1.get('total_documents', 0)} documents")
    
    # Second ingestion (should detect no changes if check_updates=True)
    result2 = await kb.ingest(
        url=test_url,
        domain="retail",
        version="1.2.5",
        follow_links=False,
        check_updates=True  # Should skip unchanged content
    )
    
    stats2 = kb.get_stats()
    logger.info(f"After second ingestion: {stats2.get('total_documents', 0)} documents")
    
    # Check deduplication stats
    if 'stats' in result2:
        logger.info(f"Documents skipped: {result2['stats'].get('documents_skipped', 0)}")
        logger.info(f"Duplicates removed: {result2['stats'].get('duplicates_removed', 0)}")

async def main():
    """Run all tests"""
    try:
        # Test 1: Single URL ingestion
        await test_single_url_ingestion()
        
        # Test 2: Config-based ingestion
        await test_config_ingestion()
        
        # Test 3: Deduplication
        await test_deduplication()
        
        logger.info("\n=== All tests completed ===")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())