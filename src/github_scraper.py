import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import requests
from urllib.parse import urlparse, urljoin
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GitHubContent:
    """Represents GitHub markdown content with metadata"""
    file_path: str
    content: str
    title: str
    headers: Dict[str, List[str]]  # Header hierarchy
    code_blocks: List[Dict[str, str]]  # Code snippets with language
    links: List[Dict[str, str]]  # Internal and external links
    tables: List[Dict[str, any]]  # Parsed tables
    commit_sha: str
    last_modified: str
    source_url: str


class GitHubMarkdownScraper:
    """Scrapes and processes GitHub markdown files"""
    
    def __init__(self, github_token: Optional[str] = None):
        self.session = requests.Session()
        if github_token:
            self.session.headers.update({
                'Authorization': f'token {github_token}',
                'Accept': 'application/vnd.github.v3+json'
            })
        self.base_api_url = "https://api.github.com"
        
    def extract_repo_info(self, url: str) -> Tuple[str, str, str]:
        """Extract owner, repo, and path from GitHub URL"""
        # Parse URL patterns:
        # https://github.com/owner/repo/blob/branch/path/file.md
        # https://github.com/owner/repo/tree/branch/path
        parsed = urlparse(url)
        parts = parsed.path.strip('/').split('/')
        
        if len(parts) < 2:
            raise ValueError(f"Invalid GitHub URL: {url}")
            
        owner = parts[0]
        repo = parts[1]
        
        # Extract file path if present
        if len(parts) > 4 and parts[2] in ['blob', 'tree']:
            branch = parts[3]
            file_path = '/'.join(parts[4:])
            return owner, repo, file_path
            
        return owner, repo, ""
    
    def get_file_content(self, owner: str, repo: str, path: str) -> Dict[str, any]:
        """Get file content from GitHub API"""
        url = f"{self.base_api_url}/repos/{owner}/{repo}/contents/{path}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_directory_contents(self, owner: str, repo: str, path: str = "") -> List[Dict[str, any]]:
        """Get directory contents from GitHub API"""
        url = f"{self.base_api_url}/repos/{owner}/{repo}/contents/{path}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def scrape_markdown_file(self, file_url: str) -> GitHubContent:
        """Scrape a single markdown file from GitHub"""
        owner, repo, file_path = self.extract_repo_info(file_url)
        
        # Get file content
        file_data = self.get_file_content(owner, repo, file_path)
        
        if file_data.get('type') != 'file':
            raise ValueError(f"URL does not point to a file: {file_url}")
        
        # Decode content
        content = base64.b64decode(file_data['content']).decode('utf-8')
        
        # Parse markdown content
        return self._parse_markdown_content(
            content=content,
            file_path=file_path,
            commit_sha=file_data.get('sha', ''),
            source_url=file_url
        )
    
    def scrape_repository_docs(self, repo_url: str, doc_paths: Optional[List[str]] = None) -> List[GitHubContent]:
        """Scrape all markdown files from repository documentation"""
        owner, repo, _ = self.extract_repo_info(repo_url)
        
        # Default documentation paths if not specified
        if not doc_paths:
            doc_paths = ['docs', 'documentation', 'README.md', 'wiki']
        
        markdown_files = []
        
        for path in doc_paths:
            try:
                # Check if path exists
                contents = self.get_directory_contents(owner, repo, path)
                
                if isinstance(contents, dict) and contents.get('type') == 'file':
                    # Single file
                    if path.endswith('.md'):
                        file_url = f"https://github.com/{owner}/{repo}/blob/main/{path}"
                        markdown_files.append(self.scrape_markdown_file(file_url))
                elif isinstance(contents, list):
                    # Directory listing
                    for item in contents:
                        if item['type'] == 'file' and item['name'].endswith('.md'):
                            file_url = item['html_url']
                            markdown_files.append(self.scrape_markdown_file(file_url))
                            
            except requests.HTTPError as e:
                if e.response.status_code != 404:
                    logger.error(f"Error accessing {path}: {e}")
                    
        return markdown_files
    
    def _parse_markdown_content(self, content: str, file_path: str, commit_sha: str, source_url: str) -> GitHubContent:
        """Parse markdown content to extract structured information"""
        # Extract title
        title = self._extract_title(content) or os.path.basename(file_path).replace('.md', '')
        
        # Extract headers hierarchy
        headers = self._extract_headers(content)
        
        # Extract code blocks
        code_blocks = self._extract_code_blocks(content)
        
        # Extract links
        links = self._extract_links(content)
        
        # Extract tables
        tables = self._extract_tables(content)
        
        return GitHubContent(
            file_path=file_path,
            content=content,
            title=title,
            headers=headers,
            code_blocks=code_blocks,
            links=links,
            tables=tables,
            commit_sha=commit_sha,
            last_modified=datetime.now().isoformat(),
            source_url=source_url
        )
    
    def _extract_title(self, content: str) -> Optional[str]:
        """Extract title from markdown content"""
        # Look for H1 header
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_headers(self, content: str) -> Dict[str, List[str]]:
        """Extract header hierarchy from markdown"""
        headers = {'h1': [], 'h2': [], 'h3': [], 'h4': [], 'h5': [], 'h6': []}
        
        # Regex for markdown headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        
        for match in re.finditer(header_pattern, content, re.MULTILINE):
            level = len(match.group(1))
            text = match.group(2).strip()
            headers[f'h{level}'].append(text)
            
        return headers
    
    def _extract_code_blocks(self, content: str) -> List[Dict[str, str]]:
        """Extract code blocks with language identifiers"""
        code_blocks = []
        
        # Regex for fenced code blocks
        code_pattern = r'```(\w*)\n(.*?)```'
        
        for match in re.finditer(code_pattern, content, re.DOTALL):
            language = match.group(1) or 'text'
            code = match.group(2).strip()
            code_blocks.append({
                'language': language,
                'code': code,
                'context': self._get_code_context(content, match.start())
            })
            
        return code_blocks
    
    def _extract_links(self, content: str) -> List[Dict[str, str]]:
        """Extract all links from markdown"""
        links = []
        
        # Regex for markdown links
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        
        for match in re.finditer(link_pattern, content):
            text = match.group(1)
            url = match.group(2)
            links.append({
                'text': text,
                'url': url,
                'type': 'external' if url.startswith('http') else 'internal'
            })
            
        return links
    
    def _extract_tables(self, content: str) -> List[Dict[str, any]]:
        """Extract tables from markdown"""
        tables = []
        
        # Simple table extraction (can be enhanced)
        table_pattern = r'(\|.+\|[\r\n]+\|[-:\s|]+\|[\r\n]+(?:\|.+\|[\r\n]+)*)'
        
        for match in re.finditer(table_pattern, content):
            table_text = match.group(0)
            lines = table_text.strip().split('\n')
            
            if len(lines) >= 3:  # Header, separator, at least one row
                headers = [cell.strip() for cell in lines[0].split('|')[1:-1]]
                rows = []
                
                for line in lines[2:]:  # Skip header and separator
                    cells = [cell.strip() for cell in line.split('|')[1:-1]]
                    if cells:
                        rows.append(dict(zip(headers, cells)))
                
                tables.append({
                    'headers': headers,
                    'rows': rows
                })
                
        return tables
    
    def _get_code_context(self, content: str, position: int, context_length: int = 200) -> str:
        """Get surrounding context for a code block"""
        start = max(0, position - context_length)
        end = min(len(content), position + context_length)
        
        # Find nearest header before the code block
        before_content = content[:position]
        headers = re.findall(r'^#{1,6}\s+(.+)$', before_content, re.MULTILINE)
        
        context = {
            'section': headers[-1] if headers else 'Unknown',
            'text': content[start:end].replace('\n', ' ')[:200]
        }
        
        return json.dumps(context)