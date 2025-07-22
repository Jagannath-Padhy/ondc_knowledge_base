"""
Unified ONDC Knowledge Base Ingestion
Combines text and YAML ingestion with data saving
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import yaml
from loguru import logger
from tqdm import tqdm
from src.utils.config import Config
from src.utils.database import QdrantDB
from src.utils.gemini import get_gemini_embed_fn
from qdrant_client.models import PointStruct

@dataclass
class APISection:
    """Represents an API endpoint section"""
    endpoint: str
    line_start: int
    line_end: int
    content: str
    json_examples: List[Dict] = field(default_factory=list)
    scenarios: List[str] = field(default_factory=list)
    cross_references: List[int] = field(default_factory=list)

@dataclass
class SpecialTopic:
    """Represents a special topic section"""
    title: str
    line_start: int
    line_end: int
    content: str
    related_endpoints: List[str] = field(default_factory=list)

@dataclass
class CrossReference:
    """Represents a cross-reference"""
    ref_num: int
    definition: str
    line_num: int
    related_sections: List[str] = field(default_factory=list)

class ONDCIngester:
    def __init__(self):
        self.config = Config()
        self.db = QdrantDB(self.config)
        self.embed_fn = get_gemini_embed_fn(self.config)
        self.collection_name = self.config.get('qdrant.collection_name', 'ondc_knowledge')
        
        # Data directories
        self.data_dir = Path("ingested_data")
        self.chunks_dir = self.data_dir / "chunks"
        self.metadata_dir = self.data_dir / "metadata"
        
        # Document structure
        self.api_sections: Dict[str, APISection] = {}
        self.special_topics: List[SpecialTopic] = []
        self.cross_references: Dict[int, CrossReference] = {}
        
        # Patterns
        self.api_pattern = re.compile(r'^(/[a-z_]+(?:\s+.*)?)$', re.MULTILINE)
        self.json_pattern = re.compile(r'\{[\s\S]*?\}', re.MULTILINE)
        self.ref_pattern = re.compile(r'\[(\d+)\]')
        self.section_separator = '________________'
        
    def ingest_text_contract(self, file_path: str) -> List[Dict]:
        """Ingest text contract file"""
        logger.info("📄 Ingesting text contract...")
        
        # Parse document
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            full_text = ''.join(lines)
        
        # Find API endpoints
        api_matches = []
        for i, line in enumerate(lines):
            if line.strip() and line[0] == '/' and not line.startswith('//'):
                endpoint = line.strip().split()[0]
                api_matches.append((i, endpoint, line.strip()))
        
        logger.info(f"Found {len(api_matches)} API endpoints")
        
        # Create sections
        for i in range(len(api_matches)):
            line_num, endpoint, full_marker = api_matches[i]
            
            # Find section end
            if i < len(api_matches) - 1:
                next_line = api_matches[i + 1][0]
            else:
                next_line = len(lines)
                for j in range(line_num + 1, len(lines)):
                    if self.section_separator in lines[j]:
                        next_line = j
                        break
            
            # Extract content
            section_lines = lines[line_num:next_line]
            content = ''.join(section_lines)
            
            # Create section
            key = endpoint.split('(')[0].strip()
            if key not in self.api_sections:
                self.api_sections[key] = APISection(
                    endpoint=key,
                    line_start=line_num + 1,
                    line_end=next_line,
                    content=content
                )
        
        # Extract JSON examples and create chunks
        chunks = []
        
        for endpoint, section in self.api_sections.items():
            # Extract JSON examples
            json_examples = self._extract_json_examples(section)
            section.json_examples = json_examples
            
            # Create master chunk
            master_chunk = {
                'content': f"# {endpoint} API Documentation\\n\\n{section.content}",
                'metadata': {
                    'type': 'api_endpoint_master',
                    'endpoint': endpoint,
                    'line_start': section.line_start,
                    'line_end': section.line_end,
                    'json_example_count': len(json_examples),
                    'source': 'text_contract',
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            chunks.append(master_chunk)
            
            # Create JSON example chunks
            for i, example in enumerate(json_examples):
                json_chunk = {
                    'content': f"{endpoint} - {example['scenario']}\\n\\n```json\\n{json.dumps(example['json'], indent=2)}\\n```",
                    'metadata': {
                        'type': 'json_example',
                        'endpoint': endpoint,
                        'scenario': example['scenario'],
                        'example_index': i + 1,
                        'domain': example['json'].get('context', {}).get('domain', 'unknown'),
                        'action': example['json'].get('context', {}).get('action', endpoint.strip('/')),
                        'source': 'text_contract',
                        'timestamp': datetime.utcnow().isoformat()
                    }
                }
                chunks.append(json_chunk)
        
        # Extract special topics
        special_topics = self._extract_special_topics(full_text, lines)
        for topic in special_topics:
            topic_chunk = {
                'content': f"# {topic['title']}\\n\\n{topic['content']}",
                'metadata': {
                    'type': 'special_topic',
                    'topic': topic['title'],
                    'related_endpoints': topic['related_endpoints'],
                    'source': 'text_contract',
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            chunks.append(topic_chunk)
        
        logger.info(f"Created {len(chunks)} chunks from text contract")
        return chunks
    
    def _extract_json_examples(self, section: APISection) -> List[Dict]:
        """Extract JSON examples from a section"""
        json_examples = []
        content = section.content
        
        # Find complete JSON objects using brace matching
        brace_stack = []
        json_start = -1
        
        for i, char in enumerate(content):
            if char == '{':
                if not brace_stack:
                    json_start = i
                brace_stack.append('{')
            elif char == '}':
                if brace_stack:
                    brace_stack.pop()
                    if not brace_stack and json_start != -1:
                        try:
                            json_str = content[json_start:i+1]
                            obj = json.loads(json_str)
                            # Verify it's an ONDC payload
                            if 'context' in obj or 'message' in obj:
                                # Find scenario description
                                pre_text = content[max(0, json_start-200):json_start]
                                scenario_match = re.search(r'(\\d+\\.\\s+[^\\n]+)', pre_text)
                                scenario = scenario_match.group(1) if scenario_match else "Example"
                                
                                json_examples.append({
                                    'scenario': scenario,
                                    'json': obj,
                                    'position': json_start
                                })
                        except json.JSONDecodeError:
                            pass
                        json_start = -1
        
        return json_examples
    
    def _extract_special_topics(self, full_text: str, lines: List[str]) -> List[Dict]:
        """Extract special topics"""
        topics = []
        
        topic_patterns = [
            (r'Seller NP.*collecting payment', 'Seller NP collecting payment'),
            (r'Force cancellation', 'Force cancellation'),
            (r'Authorization \\(OTP\\)', 'Authorization (OTP)'),
        ]
        
        for pattern, title in topic_patterns:
            for i, line in enumerate(lines):
                if re.search(pattern, line, re.IGNORECASE):
                    # Find end of topic
                    end_line = i + 1
                    for j in range(i + 1, min(i + 200, len(lines))):
                        if self.section_separator in lines[j]:
                            end_line = j
                            break
                    
                    content = ''.join(lines[i:end_line])
                    
                    # Find related endpoints
                    related_endpoints = []
                    for endpoint in self.api_sections.keys():
                        if endpoint in content:
                            related_endpoints.append(endpoint)
                    
                    topics.append({
                        'title': title,
                        'content': content,
                        'related_endpoints': related_endpoints
                    })
                    break
        
        return topics
    
    def ingest_yaml_spec(self, file_path: str) -> List[Dict]:
        """Ingest YAML specification file"""
        logger.info("📋 Ingesting YAML specification...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            spec = yaml.safe_load(f)
        
        chunks = []
        
        # Process API paths
        if 'paths' in spec:
            for path, path_spec in tqdm(spec['paths'].items(), desc="Processing API paths"):
                for method, method_spec in path_spec.items():
                    if isinstance(method_spec, dict):
                        # Create API endpoint chunk
                        content = f"# {method.upper()} {path}\\n\\n"
                        if 'summary' in method_spec:
                            content += f"**Summary:** {method_spec['summary']}\\n\\n"
                        if 'description' in method_spec:
                            content += f"**Description:** {method_spec['description']}\\n\\n"
                        
                        # Add request body schema
                        if 'requestBody' in method_spec:
                            content += "## Request Body\\n"
                            request_schema = method_spec['requestBody'].get('content', {}).get('application/json', {}).get('schema', {})
                            content += f"```json\\n{json.dumps(request_schema, indent=2)}\\n```\\n\\n"
                        
                        # Add responses
                        if 'responses' in method_spec:
                            content += "## Responses\\n"
                            for status, response in method_spec['responses'].items():
                                content += f"### {status}\\n"
                                if 'description' in response:
                                    content += f"{response['description']}\\n"
                                if 'content' in response:
                                    response_schema = response['content'].get('application/json', {}).get('schema', {})
                                    content += f"```json\\n{json.dumps(response_schema, indent=2)}\\n```\\n"
                            content += "\\n"
                        
                        chunk = {
                            'content': content,
                            'metadata': {
                                'type': 'api_endpoint',
                                'path': path,
                                'method': method.upper(),
                                'source': 'yaml_spec',
                                'timestamp': datetime.utcnow().isoformat()
                            }
                        }
                        chunks.append(chunk)
        
        # Process schemas
        if 'components' in spec and 'schemas' in spec['components']:
            for schema_name, schema_spec in tqdm(spec['components']['schemas'].items(), desc="Processing schemas"):
                content = f"# {schema_name} Schema\\n\\n"
                if 'description' in schema_spec:
                    content += f"**Description:** {schema_spec['description']}\\n\\n"
                
                content += f"```json\\n{json.dumps(schema_spec, indent=2)}\\n```\\n"
                
                chunk = {
                    'content': content,
                    'metadata': {
                        'type': 'schema',
                        'schema_name': schema_name,
                        'source': 'yaml_spec',
                        'timestamp': datetime.utcnow().isoformat()
                    }
                }
                chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from YAML spec")
        return chunks
    
    def save_chunks(self, chunks: List[Dict], source_type: str) -> None:
        """Save chunks to JSON files"""
        filename = f"{source_type}_chunks.json"
        filepath = self.chunks_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"💾 Saved {len(chunks)} chunks to {filepath}")
    
    def upload_chunks(self, chunks: List[Dict]) -> int:
        """Upload chunks to vector database"""
        batch_size = 10
        uploaded = 0
        
        # Get starting ID
        try:
            count = self.db.client.count(self.collection_name).count
            start_id = count + 1
        except:
            start_id = 1
        
        with tqdm(total=len(chunks), desc="Uploading chunks") as pbar:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                points = []
                
                for j, chunk in enumerate(batch):
                    try:
                        # Check payload size
                        payload_size = len(json.dumps(chunk).encode('utf-8'))
                        if payload_size > 35000:
                            chunk['content'] = chunk['content'][:15000]
                        
                        embedding = self.embed_fn(chunk['content'])
                        point = PointStruct(
                            id=start_id + uploaded + j,
                            vector=embedding,
                            payload={
                                'content': chunk['content'],
                                **chunk['metadata']
                            }
                        )
                        points.append(point)
                    except Exception as e:
                        logger.error(f"Error creating embedding: {e}")
                
                if points:
                    try:
                        self.db.client.upsert(
                            collection_name=self.collection_name,
                            points=points
                        )
                        uploaded += len(points)
                        pbar.update(len(points))
                    except Exception as e:
                        logger.error(f"Upload error: {e}")
                
                time.sleep(0.1)  # Rate limiting
        
        return uploaded
    
    def run_ingestion(self, text_file: str = None, yaml_file: str = None) -> Dict[str, Any]:
        """Run complete ingestion process"""
        logger.info("🚀 Starting ONDC knowledge base ingestion")
        start_time = time.time()
        
        # Default file paths
        if not text_file:
            text_file = "knowledge_sources/ONDC - API Contract for Retail (v1.2.0).txt"
        if not yaml_file:
            yaml_file = "knowledge_sources/build.yaml"
        
        all_chunks = []
        results = {}
        
        # Ingest text contract
        if Path(text_file).exists():
            text_chunks = self.ingest_text_contract(text_file)
            all_chunks.extend(text_chunks)
            self.save_chunks(text_chunks, "text_contract")
            results['text_chunks'] = len(text_chunks)
        else:
            logger.warning(f"Text file not found: {text_file}")
            results['text_chunks'] = 0
        
        # Ingest YAML spec
        if Path(yaml_file).exists():
            yaml_chunks = self.ingest_yaml_spec(yaml_file)
            all_chunks.extend(yaml_chunks)
            self.save_chunks(yaml_chunks, "yaml_spec")
            results['yaml_chunks'] = len(yaml_chunks)
        else:
            logger.warning(f"YAML file not found: {yaml_file}")
            results['yaml_chunks'] = 0
        
        # Upload all chunks
        logger.info("📤 Uploading all chunks to vector database...")
        uploaded = self.upload_chunks(all_chunks)
        
        # Save summary
        summary = {
            'ingestion_timestamp': datetime.utcnow().isoformat(),
            'time_elapsed_seconds': time.time() - start_time,
            'text_file': text_file,
            'yaml_file': yaml_file,
            'text_chunks_created': results.get('text_chunks', 0),
            'yaml_chunks_created': results.get('yaml_chunks', 0),
            'total_chunks_created': len(all_chunks),
            'chunks_uploaded': uploaded,
            'api_endpoints_found': len(self.api_sections),
            'special_topics_found': len(self.special_topics)
        }
        
        summary_file = self.metadata_dir / "ingestion_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Ingestion complete in {elapsed:.2f} seconds")
        logger.info(f"📊 Summary: {results['text_chunks']} text + {results['yaml_chunks']} YAML = {len(all_chunks)} total chunks")
        logger.info(f"📁 Data saved in: {self.data_dir}")
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='ONDC Knowledge Base Ingestion')
    parser.add_argument('--text', help='Path to text contract file')
    parser.add_argument('--yaml', help='Path to YAML specification file')
    parser.add_argument('--clean', action='store_true', help='Clean database before ingestion')
    
    args = parser.parse_args()
    
    ingester = ONDCIngester()
    
    if args.clean:
        logger.info("🧹 Cleaning database...")
        try:
            ingester.db.client.delete_collection(ingester.collection_name)
            logger.info("Database cleaned")
        except:
            logger.info("Database was already clean")
    
    # Run ingestion
    summary = ingester.run_ingestion(args.text, args.yaml)
    
    # Show results
    logger.info("\\n📋 Ingestion Summary:")
    logger.info(f"  - Text chunks: {summary['text_chunks_created']}")
    logger.info(f"  - YAML chunks: {summary['yaml_chunks_created']}")
    logger.info(f"  - Total uploaded: {summary['chunks_uploaded']}")
    logger.info(f"  - API endpoints: {summary['api_endpoints_found']}")

if __name__ == "__main__":
    main()