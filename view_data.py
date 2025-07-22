"""
View ingested data - chunks, metadata, and search functionality
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger

class DataViewer:
    def __init__(self):
        self.data_dir = Path("ingested_data")
        self.chunks_dir = self.data_dir / "chunks"
        self.metadata_dir = self.data_dir / "metadata"
        
        if not self.data_dir.exists():
            logger.error(f"Data directory not found: {self.data_dir}")
            logger.info("Run 'python ingest.py' first to create data")
            return
    
    def load_chunks(self, source: str = None) -> List[Dict]:
        """Load chunks from JSON files"""
        chunks = []
        
        if source:
            chunk_file = self.chunks_dir / f"{source}_chunks.json"
            if chunk_file.exists():
                with open(chunk_file) as f:
                    chunks.extend(json.load(f))
        else:
            # Load all chunk files
            for chunk_file in self.chunks_dir.glob("*_chunks.json"):
                try:
                    with open(chunk_file) as f:
                        file_chunks = json.load(f)
                        chunks.extend(file_chunks)
                except:
                    logger.warning(f"Could not load {chunk_file}")
        
        return chunks
    
    def show_summary(self):
        """Show ingestion summary"""
        summary_file = self.metadata_dir / "ingestion_summary.json"
        
        if not summary_file.exists():
            logger.error("No ingestion summary found. Run 'python ingest.py' first.")
            return
        
        with open(summary_file) as f:
            summary = json.load(f)
        
        logger.info("\\n📊 INGESTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Timestamp: {summary['ingestion_timestamp']}")
        logger.info(f"Duration: {summary['time_elapsed_seconds']:.1f} seconds")
        logger.info(f"Text chunks: {summary['text_chunks_created']}")
        logger.info(f"YAML chunks: {summary['yaml_chunks_created']}")
        logger.info(f"Total chunks: {summary['total_chunks_created']}")
        logger.info(f"Uploaded: {summary['chunks_uploaded']}")
        logger.info(f"API endpoints: {summary['api_endpoints_found']}")
        
        # Show chunk breakdown
        chunks = self.load_chunks()
        if chunks:
            logger.info("\\n📦 CHUNK TYPES")
            logger.info("-" * 20)
            type_counts = {}
            for chunk in chunks:
                chunk_type = chunk['metadata']['type']
                type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
            
            for chunk_type, count in sorted(type_counts.items()):
                logger.info(f"{chunk_type}: {count}")
    
    def show_chunks(self, chunk_type: Optional[str] = None, limit: int = 5):
        """Show chunk details"""
        chunks = self.load_chunks()
        
        if chunk_type:
            chunks = [c for c in chunks if c['metadata']['type'] == chunk_type]
        
        if not chunks:
            logger.warning(f"No chunks found" + (f" of type '{chunk_type}'" if chunk_type else ""))
            return
        
        logger.info(f"\\n📦 CHUNKS" + (f" ({chunk_type})" if chunk_type else ""))
        logger.info("=" * 50)
        logger.info(f"Showing {min(limit, len(chunks))} of {len(chunks)} chunks\\n")
        
        for i, chunk in enumerate(chunks[:limit]):
            meta = chunk['metadata']
            logger.info(f"[{i+1}] {meta['type']}")
            
            if meta['type'] == 'api_endpoint_master':
                logger.info(f"    Endpoint: {meta['endpoint']}")
                logger.info(f"    JSON examples: {meta.get('json_example_count', 0)}")
                logger.info(f"    Lines: {meta.get('line_start', '?')}-{meta.get('line_end', '?')}")
            
            elif meta['type'] == 'json_example':
                logger.info(f"    Endpoint: {meta['endpoint']}")
                logger.info(f"    Scenario: {meta['scenario']}")
                logger.info(f"    Domain: {meta.get('domain', 'N/A')}")
            
            elif meta['type'] == 'special_topic':
                logger.info(f"    Topic: {meta['topic']}")
                logger.info(f"    Related APIs: {', '.join(meta.get('related_endpoints', []))}")
            
            elif meta['type'] in ['api_endpoint', 'schema']:
                if 'path' in meta:
                    logger.info(f"    {meta['method']} {meta['path']}")
                elif 'schema_name' in meta:
                    logger.info(f"    Schema: {meta['schema_name']}")
            
            logger.info(f"    Source: {meta['source']}")
            logger.info(f"    Content: {len(chunk['content'])} characters")
            logger.info()
    
    def search_chunks(self, search_term: str, limit: int = 10):
        """Search for chunks containing specific term"""
        chunks = self.load_chunks()
        
        matches = []
        for chunk in chunks:
            content = chunk['content'].lower()
            if search_term.lower() in content:
                matches.append(chunk)
        
        logger.info(f"\\n🔍 SEARCH RESULTS for '{search_term}'")
        logger.info("=" * 50)
        logger.info(f"Found {len(matches)} matches\\n")
        
        for i, chunk in enumerate(matches[:limit]):
            meta = chunk['metadata']
            logger.info(f"[{i+1}] {meta['type']}")
            
            if meta['type'] in ['api_endpoint_master', 'json_example']:
                logger.info(f"    Endpoint: {meta.get('endpoint', 'N/A')}")
            elif meta['type'] == 'special_topic':
                logger.info(f"    Topic: {meta.get('topic', 'N/A')}")
            
            # Show context around search term
            content = chunk['content']
            idx = content.lower().find(search_term.lower())
            if idx != -1:
                start = max(0, idx - 100)
                end = min(len(content), idx + len(search_term) + 100)
                context = content[start:end]
                logger.info(f"    Context: ...{context}...")
            
            logger.info()
    
    def show_endpoint_data(self, endpoint: str):
        """Show all data for a specific endpoint"""
        chunks = self.load_chunks()
        
        endpoint_chunks = []
        for chunk in chunks:
            meta = chunk['metadata']
            if (meta.get('endpoint') == endpoint or 
                meta.get('path') == endpoint or
                endpoint in chunk['content']):
                endpoint_chunks.append(chunk)
        
        if not endpoint_chunks:
            logger.warning(f"No data found for endpoint '{endpoint}'")
            return
        
        logger.info(f"\\n🔌 DATA for {endpoint}")
        logger.info("=" * 50)
        logger.info(f"Found {len(endpoint_chunks)} related chunks\\n")
        
        # Group by type
        by_type = {}
        for chunk in endpoint_chunks:
            chunk_type = chunk['metadata']['type']
            if chunk_type not in by_type:
                by_type[chunk_type] = []
            by_type[chunk_type].append(chunk)
        
        # Show each type
        for chunk_type, type_chunks in by_type.items():
            logger.info(f"📋 {chunk_type.upper()} ({len(type_chunks)})")
            logger.info("-" * 30)
            
            for i, chunk in enumerate(type_chunks):
                meta = chunk['metadata']
                
                if chunk_type == 'json_example':
                    logger.info(f"  Example {i+1}: {meta.get('scenario', 'N/A')}")
                    # Show if it contains complete JSON
                    has_json = '```json' in chunk['content']
                    logger.info(f"    Has JSON block: {has_json}")
                    
                    if has_json:
                        # Count lines in JSON
                        json_lines = chunk['content'].count('\\n')
                        logger.info(f"    Content size: {json_lines} lines, {len(chunk['content'])} chars")
                
                elif chunk_type == 'api_endpoint_master':
                    logger.info(f"  Master documentation")
                    logger.info(f"    JSON examples: {meta.get('json_example_count', 0)}")
                    logger.info(f"    Content: {len(chunk['content'])} characters")
                
                elif chunk_type == 'special_topic':
                    logger.info(f"  Topic: {meta.get('topic', 'N/A')}")
                
            logger.info()
    
    def show_files(self):
        """Show available data files"""
        logger.info("\\n📁 AVAILABLE DATA FILES")
        logger.info("=" * 50)
        
        # Chunk files
        logger.info("Chunk files:")
        chunk_files = list(self.chunks_dir.glob("*.json"))
        if chunk_files:
            for f in sorted(chunk_files):
                size_mb = f.stat().st_size / (1024 * 1024)
                logger.info(f"  {f.name} ({size_mb:.1f} MB)")
        else:
            logger.info("  No chunk files found")
        
        # Metadata files
        logger.info("\\nMetadata files:")
        meta_files = list(self.metadata_dir.glob("*.json"))
        if meta_files:
            for f in sorted(meta_files):
                size_kb = f.stat().st_size / 1024
                logger.info(f"  {f.name} ({size_kb:.1f} KB)")
        else:
            logger.info("  No metadata files found")

def main():
    parser = argparse.ArgumentParser(description='View ONDC ingested data')
    parser.add_argument('--summary', action='store_true', help='Show ingestion summary')
    parser.add_argument('--chunks', type=str, nargs='?', const='', help='Show chunks (optionally filter by type)')
    parser.add_argument('--search', type=str, help='Search for term in chunks')
    parser.add_argument('--endpoint', type=str, help='Show data for specific endpoint')
    parser.add_argument('--files', action='store_true', help='Show available files')
    parser.add_argument('--limit', type=int, default=5, help='Limit results (default: 5)')
    
    args = parser.parse_args()
    
    viewer = DataViewer()
    
    if not viewer.data_dir.exists():
        return
    
    # Default: show summary
    if not any([args.summary, args.chunks is not None, args.search, args.endpoint, args.files]):
        viewer.show_summary()
        return
    
    if args.summary:
        viewer.show_summary()
    
    if args.chunks is not None:
        chunk_type = args.chunks if args.chunks else None
        viewer.show_chunks(chunk_type, args.limit)
    
    if args.search:
        viewer.search_chunks(args.search, args.limit)
    
    if args.endpoint:
        viewer.show_endpoint_data(args.endpoint)
    
    if args.files:
        viewer.show_files()

if __name__ == "__main__":
    main()