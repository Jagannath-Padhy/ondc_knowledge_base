#!/usr/bin/env python3
"""
ONDC Knowledge Base - Main Application
Clean implementation with unified ingestion support
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from enhanced_ingestion import EnhancedIngestionPipeline, DocumentType
from vector_store import QdrantVectorStore
from embeddings import GeminiEmbeddings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        
        # Enhanced ingestion pipeline
        self.ingestion_pipeline = EnhancedIngestionPipeline(
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
        
        # Run startup ingestion if configured
        if self.config.get("startup_documents"):
            logger.info("Running startup document ingestion...")
            asyncio.create_task(self._startup_ingestion())
    
    async def _startup_ingestion(self):
        """Run ingestion for startup documents from config"""
        try:
            startup_docs = self.config.get("startup_documents", [])
            if startup_docs:
                result = await self.ingestion_pipeline.ingest_from_config(
                    documents=startup_docs,
                    force_update=False
                )
                logger.info(f"Startup ingestion completed: {result['stats']}")
        except Exception as e:
            logger.error(f"Startup ingestion failed: {e}")
    
    async def ingest(
        self,
        url: str = None,
        sources: List[Dict[str, Any]] = None,
        domain: str = "retail",
        version: str = "1.2.0",
        recreate: bool = False,
        follow_links: bool = True,
        check_updates: bool = True,
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Unified ingestion endpoint supporting single URL or multiple sources
        
        Args:
            url: Single URL to ingest (with recursive scraping)
            sources: List of source configurations (alternative to url)
            domain: Domain for the documents
            version: Version of specifications
            recreate: Whether to recreate the collection
            follow_links: Whether to follow hyperlinks recursively
            check_updates: Whether to check for updates before re-ingesting
            max_depth: Maximum depth for recursive link following
        
        Returns:
            Ingestion statistics
        """
        logger.info(f"Starting ingestion for domain: {domain}, version: {version}")
        
        # Handle single URL ingestion
        if url:
            logger.info(f"Ingesting single URL: {url}")
            if recreate:
                self.vector_store.create_collection(recreate=True)
            
            result = await self.ingestion_pipeline.ingest_document_recursive(
                url=url,
                domain=domain,
                version=version,
                max_depth=max_depth,
                follow_links=follow_links,
                check_updates=check_updates
            )
            
            return {
                'url': url,
                'result': result,
                'stats': self.ingestion_pipeline.stats
            }
        
        # Handle multiple sources from config
        elif sources:
            return await self.ingestion_pipeline.ingest_from_config(
                documents=sources,
                force_update=not check_updates
            )
        
        # Use startup documents from config if no url or sources provided
        else:
            startup_docs = self.config.get("startup_documents", [])
            if not startup_docs:
                raise ValueError("No URL, sources, or startup documents provided")
            
            return await self.ingestion_pipeline.ingest_from_config(
                documents=startup_docs,
                force_update=not check_updates
            )
    
    def query(
        self,
        question: str,
        domain: str = "retail",
        version: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query the knowledge base
        
        Args:
            question: User's question
            domain: Domain to search in
            version: Specific version (optional)
            limit: Number of results to return
        
        Returns:
            List of relevant document chunks with metadata
        """
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


def run_async(coro):
    """Helper to run async functions in sync context"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


def main():
    """CLI interface for the knowledge base"""
    parser = argparse.ArgumentParser(
        description="ONDC Knowledge Base Management"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument(
        "--url",
        help="Single URL to ingest (with recursive scraping)"
    )
    ingest_parser.add_argument(
        "--sources-file",
        help="JSON file containing source configurations"
    )
    ingest_parser.add_argument(
        "--google-docs",
        nargs="+",
        help="Google Docs URLs to ingest"
    )
    ingest_parser.add_argument(
        "--github-repos",
        nargs="+",
        help="GitHub repository URLs to ingest"
    )
    ingest_parser.add_argument(
        "--google-sheets",
        nargs="+",
        help="Google Sheets URLs to ingest"
    )
    ingest_parser.add_argument(
        "--follow-links",
        action="store_true",
        default=True,
        help="Follow hyperlinks recursively"
    )
    ingest_parser.add_argument(
        "--no-follow-links",
        dest="follow_links",
        action="store_false",
        help="Don't follow hyperlinks"
    )
    ingest_parser.add_argument(
        "--check-updates",
        action="store_true",
        default=True,
        help="Check for updates before re-ingesting"
    )
    ingest_parser.add_argument(
        "--force-update",
        dest="check_updates",
        action="store_false",
        help="Force re-ingestion even if no changes"
    )
    ingest_parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum depth for recursive link following"
    )
    ingest_parser.add_argument(
        "--domain",
        default="retail",
        help="Domain (default: retail)"
    )
    ingest_parser.add_argument(
        "--version",
        default="1.2.0",
        help="Version (default: 1.2.0)"
    )
    ingest_parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate collection"
    )
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query knowledge base")
    query_parser.add_argument(
        "question",
        help="Question to search for"
    )
    query_parser.add_argument(
        "--domain",
        default="retail",
        help="Domain to search in"
    )
    query_parser.add_argument(
        "--version",
        help="Specific version to search"
    )
    query_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results"
    )
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat mode")
    chat_parser.add_argument(
        "-q", "--question",
        help="Question to ask (for single query mode)"
    )
    chat_parser.add_argument(
        "--domain",
        default="retail",
        help="Domain for chat context"
    )
    
    # Health command
    health_parser = subparsers.add_parser("health", help="Check system health")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get statistics")
    
    args = parser.parse_args()
    
    # Initialize knowledge base
    kb = ONDCKnowledgeBase()
    
    if args.command == "ingest":
        # Handle single URL ingestion
        if args.url:
            result = run_async(kb.ingest(
                url=args.url,
                domain=args.domain,
                version=args.version,
                recreate=args.recreate,
                follow_links=args.follow_links,
                check_updates=args.check_updates,
                max_depth=args.max_depth
            ))
        else:
            # Build sources list
            sources = []
            
            if args.sources_file:
                # Load from file
                with open(args.sources_file, 'r') as f:
                    sources = json.load(f)
            else:
                # Build from command line arguments
                if args.google_docs:
                    for url in args.google_docs:
                        sources.append({
                            "url": url,
                            "domain": args.domain,
                            "version": args.version,
                            "follow_links": args.follow_links,
                            "check_updates": args.check_updates,
                            "max_depth": args.max_depth
                        })
                
                if args.github_repos:
                    for url in args.github_repos:
                        sources.append({
                            "url": url,
                            "domain": args.domain,
                            "version": args.version,
                            "follow_links": args.follow_links,
                            "check_updates": args.check_updates
                        })
                
                if args.google_sheets:
                    for url in args.google_sheets:
                        sources.append({
                            "url": url,
                            "domain": args.domain,
                            "version": args.version,
                            "check_updates": args.check_updates
                        })
            
            if not sources:
                # Use startup documents from config
                print("No sources specified. Using startup documents from config...")
                result = run_async(kb.ingest(
                    domain=args.domain,
                    version=args.version,
                    recreate=args.recreate,
                    check_updates=args.check_updates
                ))
            else:
                # Run ingestion with sources
                result = run_async(kb.ingest(
                    sources=sources,
                    recreate=args.recreate
                ))
        
        print("\nIngestion completed!")
        print(json.dumps(result.get("stats", result), indent=2))
        
    elif args.command == "query":
        # Run query
        results = kb.query(
            question=args.question,
            domain=args.domain,
            version=args.version,
            limit=args.limit
        )
        
        print(f"\nFound {len(results)} results for: {args.question}\n")
        
        for i, result in enumerate(results, 1):
            print(f"--- Result {i} ---")
            print(f"Score: {result.get('score', 0):.3f}")
            print(f"Source: {result.get('metadata', {}).get('doc_title', 'Unknown')}")
            print(f"Type: {result.get('metadata', {}).get('chunk_type', 'Unknown')}")
            print(f"\nContent:\n{result.get('content', '')[:500]}...")
            print()
    
    elif args.command == "chat":
        # Single question mode
        if args.question:
            print(f"\nQuestion: {args.question}\n")
            results = kb.query(args.question, domain=args.domain, limit=3)
            
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

Question: {args.question}

Please provide a clear and concise answer based on the provided context. If the context doesn't contain enough information to fully answer the question, mention what information is missing."""

                # Use Gemini to generate answer
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=kb.config["gemini_api_key"])
                    model = genai.GenerativeModel('gemini-2.5-pro')
                    response = model.generate_content(prompt)
                    print(f"Answer: {response.text}")
                except Exception as e:
                    # Fallback to simple response
                    print(f"Answer: {results[0].get('content', '')[:500]}...")
                    print(f"\n(Found in: {results[0].get('metadata', {}).get('doc_title', 'Unknown')})")
            else:
                print("Answer: I couldn't find relevant information in the ONDC documentation to answer your question.")
            return
        
        # Interactive mode
        print(f"Interactive chat mode (domain: {args.domain})")
        print("Type 'exit' to quit\n")
        
        while True:
            try:
                question = input("You: ").strip()
                if question.lower() in ['exit', 'quit']:
                    break
                
                if not question:
                    continue
                
                # Get results
                results = kb.query(question, domain=args.domain, limit=3)
                
                if results:
                    print("\nAssistant: Based on the ONDC documentation:\n")
                    
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
                        import google.generativeai as genai
                        genai.configure(api_key=kb.config["gemini_api_key"])
                        model = genai.GenerativeModel('gemini-2.5-pro')
                        response = model.generate_content(prompt)
                        print(response.text)
                    except Exception as e:
                        # Fallback to simple response
                        print(results[0].get('content', '')[:500] + "...")
                        print(f"\n(Found in: {results[0].get('metadata', {}).get('doc_title', 'Unknown')})")
                else:
                    print("\nAssistant: I couldn't find relevant information about that.")
                
                print()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    
    elif args.command == "health":
        health = kb.check_health()
        print("\nSystem Health Check:")
        for service, status in health.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {service}: {'OK' if status else 'Failed'}")
    
    elif args.command == "stats":
        stats = kb.get_stats()
        print("\nKnowledge Base Statistics:")
        print(json.dumps(stats, indent=2))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()