import os
import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import requests
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SheetContent:
    """Represents Google Sheets content with metadata"""
    sheet_id: str
    sheet_name: str
    headers: List[str]
    rows: List[Dict[str, Any]]
    formulas: Dict[str, str]  # Cell references with formulas
    data_types: Dict[str, str]  # Column data types
    validation_rules: Dict[str, Any]  # Data validation rules
    last_modified: str
    source_url: str


class GoogleSheetsScreener:
    """Scrapes and processes Google Sheets data"""
    
    def __init__(self, credentials_path: Optional[str] = None):
        self.service = self._initialize_service(credentials_path)
        self.use_public_api = self.service is None
        
    def _initialize_service(self, credentials_path: Optional[str] = None):
        """Initialize Google Sheets API service"""
        if credentials_path and os.path.exists(credentials_path):
            try:
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
                )
                return build('sheets', 'v4', credentials=credentials)
            except Exception as e:
                logger.error(f"Failed to initialize Sheets API: {e}")
                return None
        else:
            logger.info("No credentials provided, will use public CSV export method")
            return None
    
    def extract_sheet_id(self, url: str) -> str:
        """Extract Google Sheet ID from URL"""
        patterns = [
            r'/spreadsheets/d/([a-zA-Z0-9-_]+)',
            r'id=([a-zA-Z0-9-_]+)',
            r'/d/([a-zA-Z0-9-_]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError(f"Could not extract sheet ID from URL: {url}")
    
    def scrape_sheet(self, sheet_url: str, sheet_names: Optional[List[str]] = None) -> List[SheetContent]:
        """Scrape Google Sheets data"""
        sheet_id = self.extract_sheet_id(sheet_url)
        
        if self.use_public_api:
            return self._scrape_public_sheets(sheet_id, sheet_url, sheet_names)
        else:
            return self._scrape_with_api(sheet_id, sheet_url, sheet_names)
    
    def _scrape_public_sheets(self, sheet_id: str, sheet_url: str, sheet_names: Optional[List[str]] = None) -> List[SheetContent]:
        """Scrape public sheets using CSV export"""
        sheets_data = []
        
        # If no sheet names specified, try to get the first sheet
        if not sheet_names:
            sheet_names = [None]  # Will use default export
        
        for sheet_name in sheet_names:
            try:
                # Export as CSV
                if sheet_name:
                    export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={sheet_name}"
                else:
                    export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                
                response = requests.get(export_url)
                response.raise_for_status()
                
                # Parse CSV data
                csv_content = response.text
                sheet_content = self._parse_csv_content(
                    csv_content=csv_content,
                    sheet_id=sheet_id,
                    sheet_name=sheet_name or "Sheet1",
                    source_url=sheet_url
                )
                
                sheets_data.append(sheet_content)
                
            except Exception as e:
                logger.error(f"Failed to scrape sheet {sheet_name}: {e}")
                
        return sheets_data
    
    def _scrape_with_api(self, sheet_id: str, sheet_url: str, sheet_names: Optional[List[str]] = None) -> List[SheetContent]:
        """Scrape sheets using Google Sheets API"""
        sheets_data = []
        
        try:
            # Get spreadsheet metadata
            spreadsheet = self.service.spreadsheets().get(spreadsheetId=sheet_id).execute()
            
            # Get all sheet names if not specified
            if not sheet_names:
                sheet_names = [sheet['properties']['title'] for sheet in spreadsheet['sheets']]
            
            for sheet_name in sheet_names:
                try:
                    # Get sheet data
                    result = self.service.spreadsheets().values().get(
                        spreadsheetId=sheet_id,
                        range=f"{sheet_name}!A:Z"  # Adjust range as needed
                    ).execute()
                    
                    values = result.get('values', [])
                    
                    if not values:
                        logger.warning(f"No data found in sheet: {sheet_name}")
                        continue
                    
                    # Get sheet properties for additional metadata
                    sheet_props = next(
                        (s for s in spreadsheet['sheets'] if s['properties']['title'] == sheet_name),
                        None
                    )
                    
                    sheet_content = self._parse_sheet_data(
                        values=values,
                        sheet_id=sheet_id,
                        sheet_name=sheet_name,
                        sheet_props=sheet_props,
                        source_url=sheet_url
                    )
                    
                    sheets_data.append(sheet_content)
                    
                except Exception as e:
                    logger.error(f"Failed to process sheet {sheet_name}: {e}")
                    
        except HttpError as e:
            logger.error(f"API error: {e}")
            
        return sheets_data
    
    def _parse_csv_content(self, csv_content: str, sheet_id: str, sheet_name: str, source_url: str) -> SheetContent:
        """Parse CSV content into structured format"""
        lines = csv_content.strip().split('\n')
        
        if not lines:
            raise ValueError("Empty CSV content")
        
        # Parse headers
        headers = self._parse_csv_line(lines[0])
        
        # Parse rows
        rows = []
        for line in lines[1:]:
            values = self._parse_csv_line(line)
            if values:  # Skip empty rows
                row_dict = {}
                for i, header in enumerate(headers):
                    if i < len(values):
                        row_dict[header] = values[i]
                    else:
                        row_dict[header] = ""
                rows.append(row_dict)
        
        # Infer data types
        data_types = self._infer_data_types(headers, rows)
        
        return SheetContent(
            sheet_id=sheet_id,
            sheet_name=sheet_name,
            headers=headers,
            rows=rows,
            formulas={},  # Not available in CSV export
            data_types=data_types,
            validation_rules={},  # Not available in CSV export
            last_modified=datetime.now().isoformat(),
            source_url=source_url
        )
    
    def _parse_csv_line(self, line: str) -> List[str]:
        """Parse a CSV line handling quoted values"""
        # Simple CSV parser (can be replaced with csv module for robustness)
        values = []
        current = []
        in_quotes = False
        
        for char in line:
            if char == '"':
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                values.append(''.join(current).strip())
                current = []
            else:
                current.append(char)
        
        if current:
            values.append(''.join(current).strip())
            
        return values
    
    def _parse_sheet_data(self, values: List[List[Any]], sheet_id: str, sheet_name: str, 
                         sheet_props: Optional[Dict], source_url: str) -> SheetContent:
        """Parse sheet data from API response"""
        if not values:
            raise ValueError("No data in sheet")
        
        # First row as headers
        headers = [str(h) for h in values[0]]
        
        # Parse rows
        rows = []
        for row_values in values[1:]:
            row_dict = {}
            for i, header in enumerate(headers):
                if i < len(row_values):
                    row_dict[header] = row_values[i]
                else:
                    row_dict[header] = None
            rows.append(row_dict)
        
        # Extract formulas if available
        formulas = {}
        if sheet_props and 'data' in sheet_props:
            for data in sheet_props.get('data', []):
                for row_data in data.get('rowData', []):
                    for cell in row_data.get('values', []):
                        if 'userEnteredValue' in cell and 'formulaValue' in cell['userEnteredValue']:
                            # Store formula with cell reference
                            formulas[f"A1"] = cell['userEnteredValue']['formulaValue']
        
        # Infer data types
        data_types = self._infer_data_types(headers, rows)
        
        # Extract validation rules if available
        validation_rules = {}
        if sheet_props and 'conditionalFormats' in sheet_props:
            validation_rules = sheet_props['conditionalFormats']
        
        return SheetContent(
            sheet_id=sheet_id,
            sheet_name=sheet_name,
            headers=headers,
            rows=rows,
            formulas=formulas,
            data_types=data_types,
            validation_rules=validation_rules,
            last_modified=datetime.now().isoformat(),
            source_url=source_url
        )
    
    def _infer_data_types(self, headers: List[str], rows: List[Dict[str, Any]]) -> Dict[str, str]:
        """Infer data types for each column"""
        data_types = {}
        
        for header in headers:
            types_found = set()
            
            for row in rows[:100]:  # Sample first 100 rows
                value = row.get(header)
                if value is None or value == "":
                    continue
                    
                # Try to infer type
                if isinstance(value, bool):
                    types_found.add('boolean')
                elif isinstance(value, (int, float)):
                    types_found.add('number')
                else:
                    # Try parsing string values
                    str_value = str(value)
                    
                    # Check for boolean
                    if str_value.lower() in ['true', 'false', 'yes', 'no']:
                        types_found.add('boolean')
                    # Check for number
                    elif re.match(r'^-?\d+\.?\d*$', str_value):
                        types_found.add('number')
                    # Check for date
                    elif re.match(r'^\d{4}-\d{2}-\d{2}', str_value):
                        types_found.add('date')
                    # Check for JSON
                    elif str_value.startswith('{') or str_value.startswith('['):
                        try:
                            json.loads(str_value)
                            types_found.add('json')
                        except:
                            types_found.add('string')
                    else:
                        types_found.add('string')
            
            # Determine primary type
            if 'number' in types_found and len(types_found) == 1:
                data_types[header] = 'number'
            elif 'boolean' in types_found and len(types_found) == 1:
                data_types[header] = 'boolean'
            elif 'date' in types_found:
                data_types[header] = 'date'
            elif 'json' in types_found:
                data_types[header] = 'json'
            else:
                data_types[header] = 'string'
                
        return data_types
    
    def extract_schema_info(self, sheet_content: SheetContent) -> Dict[str, Any]:
        """Extract schema information from sheet content"""
        schema = {
            'title': sheet_content.sheet_name,
            'type': 'object',
            'properties': {},
            'required': []
        }
        
        for header in sheet_content.headers:
            data_type = sheet_content.data_types.get(header, 'string')
            
            # Convert to JSON Schema types
            json_type = {
                'number': 'number',
                'boolean': 'boolean',
                'date': 'string',
                'json': 'object',
                'string': 'string'
            }.get(data_type, 'string')
            
            property_schema = {'type': json_type}
            
            # Add format for dates
            if data_type == 'date':
                property_schema['format'] = 'date'
            
            # Check if field appears to be required (non-empty in most rows)
            non_empty_count = sum(1 for row in sheet_content.rows if row.get(header))
            if non_empty_count > len(sheet_content.rows) * 0.9:  # 90% threshold
                schema['required'].append(header)
            
            schema['properties'][header] = property_schema
        
        return schema