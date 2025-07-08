import re
import json
import logging
from typing import Dict, List, Tuple, Set
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FootnoteExtractor:
    """Extract and manage footnote references and definitions from ONDC documents"""
    
    def __init__(self):
        # Common footnote patterns
        self.ref_pattern = re.compile(r'\[([a-z]{1,3})\]')
        self.def_pattern = re.compile(r'^\[([a-z]{1,3})\](.+?)(?=^\[|$)', re.MULTILINE | re.DOTALL)
        
    def extract_footnotes_from_document(self, doc_data: Dict) -> Dict[str, Dict]:
        """Extract all footnotes from a document"""
        doc_id = doc_data.get('doc_id', '')
        doc_title = doc_data.get('title', '')
        version = doc_data.get('version', '')
        
        # Find all references in main content
        main_content = doc_data.get('main_content', '')
        references = self.find_references(main_content)
        
        # Extract definitions from footer
        footer_content = doc_data.get('footer_content', '')
        definitions = self.extract_definitions(footer_content)
        
        # Also check if any sections have footnote-like content
        for section in doc_data.get('sections', []):
            section_content = section.get('content', '')
            section_refs = self.find_references(section_content)
            references.update(section_refs)
            
            # Some sections might contain definitions too
            if any(keyword in section.get('title', '').lower() 
                   for keyword in ['footnote', 'note', 'reference', 'appendix']):
                section_defs = self.extract_definitions(section_content)
                definitions.update(section_defs)
        
        # Create footnote mapping
        footnote_map = {
            'doc_id': doc_id,
            'doc_title': doc_title,
            'version': version,
            'total_references': len(references),
            'total_definitions': len(definitions),
            'mappings': {}
        }
        
        # Match references with definitions
        for ref in references:
            if ref in definitions:
                footnote_map['mappings'][ref] = {
                    'reference': f'[{ref}]',
                    'definition': definitions[ref],
                    'found_in_footer': ref in self.extract_definitions(footer_content)
                }
            else:
                footnote_map['mappings'][ref] = {
                    'reference': f'[{ref}]',
                    'definition': None,
                    'found_in_footer': False
                }
                
        # Log unmatched definitions (might be useful)
        unmatched_defs = set(definitions.keys()) - references
        if unmatched_defs:
            logger.warning(f"Document {doc_id} has definitions without references: {unmatched_defs}")
            
        return footnote_map
    
    def find_references(self, content: str) -> Set[str]:
        """Find all footnote references in content"""
        references = set()
        matches = self.ref_pattern.findall(content)
        references.update(matches)
        return references
    
    def extract_definitions(self, content: str) -> Dict[str, str]:
        """Extract footnote definitions from content"""
        definitions = {}
        
        # Method 1: Look for [ref] at start of line
        matches = self.def_pattern.findall(content)
        for ref, definition in matches:
            definitions[ref] = definition.strip()
        
        # Method 2: Look for inline patterns like "[a]definition text"
        lines = content.split('\n')
        for line in lines:
            match = re.match(r'^\[([a-z]{1,3})\](.+)', line)
            if match:
                ref, definition = match.groups()
                if ref not in definitions:  # Don't overwrite if already found
                    definitions[ref] = definition.strip()
                    
        return definitions
    
    def enrich_chunk_with_footnotes(
        self, 
        chunk_content: str, 
        footnote_map: Dict[str, Dict]
    ) -> Tuple[str, List[str]]:
        """Enrich a chunk with its footnote definitions"""
        # Find references in this chunk
        references = self.find_references(chunk_content)
        
        if not references:
            return chunk_content, []
        
        # Collect relevant footnotes
        footnotes = []
        mappings = footnote_map.get('mappings', {})
        
        for ref in references:
            if ref in mappings and mappings[ref]['definition']:
                footnotes.append(f"[{ref}] {mappings[ref]['definition']}")
        
        # Append footnotes to chunk
        if footnotes:
            enriched_content = chunk_content + "\n\n--- Footnotes ---\n" + "\n".join(footnotes)
        else:
            enriched_content = chunk_content
            
        return enriched_content, list(references)
    
    def create_footnote_chunks(self, footnote_map: Dict[str, Dict]) -> List[Dict]:
        """Create separate chunks for footnote mappings"""
        chunks = []
        
        # Create a summary chunk
        summary_content = f"Footnote Reference Guide for {footnote_map['doc_title']}\n"
        summary_content += f"Version: {footnote_map['version']}\n"
        summary_content += f"Total References: {footnote_map['total_references']}\n\n"
        
        # Group footnotes by prefix (a-z, aa-az, etc.)
        grouped = {}
        for ref, mapping in footnote_map['mappings'].items():
            prefix = ref[0]
            if prefix not in grouped:
                grouped[prefix] = []
            grouped[prefix].append((ref, mapping))
        
        # Create chunks for each group
        for prefix, footnotes in grouped.items():
            chunk_content = f"Footnotes [{prefix}*]:\n\n"
            for ref, mapping in sorted(footnotes):
                if mapping['definition']:
                    chunk_content += f"[{ref}] {mapping['definition']}\n\n"
            
            if len(chunk_content) > 50:  # Only create chunk if meaningful content
                chunks.append({
                    'doc_id': footnote_map['doc_id'],
                    'content': chunk_content.strip(),
                    'chunk_type': 'footnote_reference',
                    'doc_title': footnote_map['doc_title'],
                    'version': footnote_map['version'],
                    'footnote_prefix': prefix
                })
        
        return chunks
    
    def process_all_documents(self, scraped_docs_dir: str) -> Dict[str, Dict]:
        """Process all documents and extract footnotes"""
        all_footnotes = {}
        docs_path = Path(scraped_docs_dir)
        
        for json_file in docs_path.glob("*.json"):
            if json_file.name == "_metadata.json":
                continue
                
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)
                
                footnote_map = self.extract_footnotes_from_document(doc_data)
                doc_id = doc_data.get('doc_id', json_file.stem)
                all_footnotes[doc_id] = footnote_map
                
                logger.info(f"Extracted {len(footnote_map['mappings'])} footnotes from {doc_id}")
                
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
                
        return all_footnotes
    
    def save_footnote_mappings(self, footnotes: Dict[str, Dict], output_path: str):
        """Save footnote mappings to file"""
        output_file = Path(output_path) / "footnote_mappings.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(footnotes, f, indent=2)
        logger.info(f"Saved footnote mappings to {output_file}")
        
    def load_footnote_mappings(self, input_path: str) -> Dict[str, Dict]:
        """Load footnote mappings from file"""
        input_file = Path(input_path) / "footnote_mappings.json"
        if input_file.exists():
            with open(input_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}


if __name__ == "__main__":
    # Test the footnote extractor
    extractor = FootnoteExtractor()
    
    # Process all documents
    scraped_docs_dir = "/Users/jagannath/Documents/MCP_validator/ondc-kb-poc/scraped_docs"
    all_footnotes = extractor.process_all_documents(scraped_docs_dir)
    
    # Save mappings
    extractor.save_footnote_mappings(all_footnotes, scraped_docs_dir)
    
    # Print summary
    total_refs = sum(fm['total_references'] for fm in all_footnotes.values())
    total_defs = sum(fm['total_definitions'] for fm in all_footnotes.values())
    print(f"\nProcessed {len(all_footnotes)} documents")
    print(f"Total footnote references: {total_refs}")
    print(f"Total footnote definitions: {total_defs}")