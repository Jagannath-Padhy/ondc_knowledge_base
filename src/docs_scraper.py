import os
import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs
import requests
from bs4 import BeautifulSoup
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentContent:
    """Represents scraped document content with metadata"""
    doc_id: str
    title: str
    main_content: str
    footer_content: str
    enums: Dict[str, List[str]]
    definitions: Dict[str, str]
    hyperlinks: List[Dict[str, str]]
    version: str
    source_url: str


class GoogleDocsScraper:
    """Scrapes Google Docs including hyperlinks and footer content"""
    
    def __init__(self, credentials_path: Optional[str] = None):
        self.service = self._initialize_service(credentials_path)
        self.scraped_docs = {}
        
    def _initialize_service(self, credentials_path: Optional[str] = None):
        """Initialize Google Docs API service"""
        if credentials_path and os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://www.googleapis.com/auth/documents.readonly']
            )
            return build('docs', 'v1', credentials=credentials)
        else:
            # For public documents, we can use requests + BeautifulSoup
            logger.info("No credentials provided, will use public scraping method")
            return None
    
    def extract_doc_id(self, url: str) -> str:
        """Extract Google Doc ID from URL"""
        patterns = [
            r'/document/d/([a-zA-Z0-9-_]+)',
            r'id=([a-zA-Z0-9-_]+)',
            r'/d/([a-zA-Z0-9-_]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError(f"Could not extract document ID from URL: {url}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def scrape_public_doc(self, doc_url: str) -> str:
        """Scrape public Google Doc as HTML"""
        doc_id = self.extract_doc_id(doc_url)
        export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=html"
        
        response = requests.get(export_url)
        response.raise_for_status()
        
        return response.text
    
    def parse_html_content(self, html_content: str) -> Tuple[str, List[Dict[str, str]], str]:
        """Parse HTML content to extract text, hyperlinks, and footer"""
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Extract hyperlinks
        hyperlinks = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True)
            if href and not href.startswith('#'):  # Skip internal anchors
                hyperlinks.append({
                    'text': text,
                    'url': href,
                    'context': self._get_link_context(link)
                })
        
        # Extract main content
        main_content = []
        footer_content = []
        is_footer = False
        
        # Look for footer indicators
        footer_indicators = ['footnote', 'endnote', 'references', 'appendix', 'glossary']
        
        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'table']):
            text = element.get_text(strip=True)
            if not text:
                continue
                
            # Check if we've reached footer section
            if any(indicator in text.lower() for indicator in footer_indicators):
                is_footer = True
            
            if is_footer:
                footer_content.append(text)
            else:
                main_content.append(text)
        
        return '\n'.join(main_content), hyperlinks, '\n'.join(footer_content)
    
    def _get_link_context(self, link_element, context_length=100):
        """Get surrounding context for a hyperlink"""
        parent = link_element.parent
        if parent:
            text = parent.get_text(strip=True)
            link_text = link_element.get_text(strip=True)
            link_pos = text.find(link_text)
            
            if link_pos != -1:
                start = max(0, link_pos - context_length)
                end = min(len(text), link_pos + len(link_text) + context_length)
                return text[start:end]
        
        return ""
    
    def extract_enums_and_definitions(self, content: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """Extract enum values and definitions from content"""
        enums = {}
        definitions = {}
        
        # Pattern for enum-like structures
        enum_patterns = [
            r'(?:enum|values?|options?)[:\s]+([^:]+)[:\s]+\[([^\]]+)\]',
            r'([A-Z_]+)\s*:\s*\[([^\]]+)\]',
            r'Valid values?[:\s]+([^:]+)[:\s]+(.+?)(?:\n|$)'
        ]
        
        for pattern in enum_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                name = match.group(1).strip()
                values_str = match.group(2) if len(match.groups()) > 1 else match.group(1)
                values = [v.strip().strip('"\'') for v in re.split(r'[,|]', values_str)]
                enums[name] = values
        
        # Pattern for definitions
        definition_patterns = [
            r'([A-Za-z_]+)\s*:\s*"([^"]+)"',
            r'([A-Za-z_]+)\s*means?\s+(.+?)(?:\n|$)',
            r'Definition[:\s]+([A-Za-z_]+)[:\s]+(.+?)(?:\n|$)'
        ]
        
        for pattern in definition_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                term = match.group(1).strip()
                definition = match.group(2).strip()
                definitions[term] = definition
        
        return enums, definitions
    
    def scrape_linked_documents(self, hyperlinks: List[Dict[str, str]], depth: int = 1) -> Dict[str, DocumentContent]:
        """Recursively scrape linked Google Docs"""
        linked_docs = {}
        
        if depth <= 0:
            return linked_docs
        
        for link in hyperlinks:
            url = link['url']
            if 'docs.google.com' in url and '/document/' in url:
                try:
                    doc_id = self.extract_doc_id(url)
                    if doc_id not in self.scraped_docs:
                        logger.info(f"Scraping linked document: {url}")
                        linked_doc = self.scrape_document(url, depth - 1)
                        linked_docs[doc_id] = linked_doc
                except Exception as e:
                    logger.error(f"Failed to scrape linked document {url}: {e}")
        
        return linked_docs
    
    def scrape_document(self, doc_url: str, follow_links_depth: int = 1) -> DocumentContent:
        """Main method to scrape a Google Doc"""
        doc_id = self.extract_doc_id(doc_url)
        
        # Check if already scraped
        if doc_id in self.scraped_docs:
            return self.scraped_docs[doc_id]
        
        # Scrape the document
        html_content = self.scrape_public_doc(doc_url)
        main_content, hyperlinks, footer_content = self.parse_html_content(html_content)
        
        # Extract enums and definitions from footer
        enums, definitions = self.extract_enums_and_definitions(footer_content)
        
        # Also check main content for enums/definitions
        main_enums, main_defs = self.extract_enums_and_definitions(main_content)
        enums.update(main_enums)
        definitions.update(main_defs)
        
        # Extract version from content
        version = self._extract_version(main_content)
        
        # Create document object
        doc = DocumentContent(
            doc_id=doc_id,
            title=self._extract_title(html_content),
            main_content=main_content,
            footer_content=footer_content,
            enums=enums,
            definitions=definitions,
            hyperlinks=hyperlinks,
            version=version,
            source_url=doc_url
        )
        
        # Cache the document
        self.scraped_docs[doc_id] = doc
        
        # Scrape linked documents
        if follow_links_depth > 0:
            linked_docs = self.scrape_linked_documents(hyperlinks, follow_links_depth)
            self.scraped_docs.update({doc_id: doc for doc_id, doc in linked_docs.items()})
        
        return doc
    
    def _extract_title(self, html_content: str) -> str:
        """Extract document title from HTML"""
        soup = BeautifulSoup(html_content, 'lxml')
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)
        
        # Try to find first heading
        for tag in ['h1', 'h2', 'h3']:
            heading = soup.find(tag)
            if heading:
                return heading.get_text(strip=True)
        
        return "Untitled Document"
    
    def _extract_version(self, content: str) -> str:
        """Extract version number from content"""
        version_patterns = [
            r'version[:\s]+(\d+\.\d+\.\d+)',
            r'v(\d+\.\d+\.\d+)',
            r'(\d+\.\d+\.\d+)'
        ]
        
        for pattern in version_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "unknown"
    
    def save_scraped_content(self, output_dir: str = "scraped_docs"):
        """Save all scraped content to JSON files"""
        os.makedirs(output_dir, exist_ok=True)
        
        for doc_id, doc in self.scraped_docs.items():
            output_path = os.path.join(output_dir, f"{doc_id}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'doc_id': doc.doc_id,
                    'title': doc.title,
                    'version': doc.version,
                    'source_url': doc.source_url,
                    'main_content': doc.main_content,
                    'footer_content': doc.footer_content,
                    'enums': doc.enums,
                    'definitions': doc.definitions,
                    'hyperlinks': doc.hyperlinks
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved document: {output_path}")


