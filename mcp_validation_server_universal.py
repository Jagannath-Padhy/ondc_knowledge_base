#!/usr/bin/env python3
"""
Universal MCP Server for ONDC Payload Validation
Validates ANY ONDC payload against appropriate schemas
"""

import json
import os
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniversalONDCValidator:
    """
    Universal ONDC payload validator that works with all domains and actions
    """
    
    def __init__(self):
        self.parsed_schemas_dir = Path("data/parsed_schemas")
        self.knowledge_base_path = Path("scraped_docs")
        
        # Load all parsed schemas
        self.schemas = self._load_all_schemas()
        
        # Initialize Gemini for AI-powered suggestions
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-pro')
        else:
            self.model = None
            
        logger.info(f"Loaded schemas for domains: {list(self.schemas.keys())}")
    
    def _load_all_schemas(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load all parsed schemas from JSON files"""
        schemas = {}
        
        if not self.parsed_schemas_dir.exists():
            logger.warning(f"Parsed schemas directory not found: {self.parsed_schemas_dir}")
            return schemas
        
        # Load each domain's schemas
        for schema_file in self.parsed_schemas_dir.glob("*_schemas.json"):
            if schema_file.name == "schemas_summary.json":
                continue
                
            try:
                with open(schema_file, 'r') as f:
                    domain_schemas = json.load(f)
                    domain = schema_file.stem.replace("_schemas", "")
                    schemas[domain] = domain_schemas
                    logger.info(f"Loaded {len(domain_schemas)} schemas for domain {domain}")
            except Exception as e:
                logger.error(f"Failed to load schemas from {schema_file}: {e}")
        
        return schemas
    
    def get_schema(self, domain: str, action: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get schema for specific domain, action, and version"""
        domain_schemas = self.schemas.get(domain.upper(), [])
        
        for schema_data in domain_schemas:
            if schema_data.get('action') == action:
                if version is None or schema_data.get('version') == version:
                    return schema_data.get('schema')
        
        return None
    
    def validate_payload(self, payload: Dict[str, Any], 
                        domain: Optional[str] = None,
                        action: Optional[str] = None,
                        version: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate ONDC payload against appropriate schema
        
        Args:
            payload: JSON payload to validate
            domain: ONDC domain (if not provided, extracted from payload)
            action: Action type (if not provided, extracted from payload)
            version: API version (if not provided, extracted from payload)
            
        Returns:
            Validation result with errors, warnings, and suggestions
        """
        
        start_time = datetime.now()
        errors = []
        warnings = []
        suggestions = []
        
        try:
            # Extract domain, action, and version from payload if not provided
            if 'context' in payload:
                context = payload['context']
                if not domain and 'domain' in context:
                    # Extract domain from context (e.g., "ONDC:RET10" -> "Retail")
                    domain_str = context['domain']
                    if ':' in domain_str:
                        domain_code = domain_str.split(':')[1]
                        if domain_code.startswith('RET'):
                            domain = 'Retail'
                        elif domain_code.startswith('LOG'):
                            domain = 'Logistics'
                        elif domain_code.startswith('FIS'):
                            domain = 'FIS'
                        elif domain_code.startswith('TRV'):
                            domain = 'TRV'
                    else:
                        domain = domain_str
                
                if not action and 'action' in context:
                    action = context['action']
                
                if not version and 'core_version' in context:
                    version = context['core_version']
            
            # Basic structure validation
            if not isinstance(payload, dict):
                errors.append("Payload must be a JSON object")
                return self._create_result(False, errors, warnings, suggestions, start_time)
            
            # Check required top-level fields
            if 'context' not in payload:
                errors.append("Missing required 'context' field")
            
            if 'message' not in payload:
                errors.append("Missing required 'message' field")
            
            # Find and apply schema validation
            if domain and action:
                schema = self.get_schema(domain, action, version)
                if schema:
                    logger.info(f"Found schema for {domain}/{action} v{version}")
                    self._validate_against_schema(payload, schema, errors, warnings, suggestions)
                else:
                    warnings.append(f"No schema found for {domain}/{action} v{version}")
                    # Try without version
                    schema = self.get_schema(domain, action)
                    if schema:
                        logger.info(f"Using schema for {domain}/{action} (any version)")
                        self._validate_against_schema(payload, schema, errors, warnings, suggestions)
            else:
                warnings.append("Could not determine domain and action from payload")
            
            # Context validation
            if 'context' in payload:
                self._validate_context(payload['context'], action, errors, warnings, suggestions)
            
            # Message validation
            if 'message' in payload:
                self._validate_message(payload['message'], action, errors, warnings, suggestions)
            
            # Domain-specific validations
            if domain:
                self._apply_domain_rules(payload, domain, action, errors, warnings, suggestions)
            
            # Generate AI-powered suggestions if available
            if self.model and (errors or warnings):
                ai_suggestions = self._generate_ai_suggestions(payload, errors, warnings, action)
                suggestions.extend(ai_suggestions)
            
            is_valid = len(errors) == 0
            return self._create_result(is_valid, errors, warnings, suggestions, start_time, 
                                     domain=domain, action=action, version=version)
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return self._create_result(False, errors, warnings, suggestions, start_time)
    
    def _validate_against_schema(self, data: Any, schema: Dict[str, Any], 
                                errors: List[str], warnings: List[str], 
                                suggestions: List[str], path: str = ""):
        """Recursively validate data against schema"""
        if not isinstance(schema, dict):
            return
        
        schema_type = schema.get('type')
        
        if schema_type == 'object':
            if not isinstance(data, dict):
                errors.append(f"{path} must be an object")
                return
            
            # Check required fields
            required = schema.get('required', [])
            for field in required:
                if field not in data:
                    errors.append(f"Missing required field: {path}.{field}" if path else field)
            
            # Validate properties
            properties = schema.get('properties', {})
            for prop, prop_schema in properties.items():
                if prop in data:
                    self._validate_against_schema(
                        data[prop], prop_schema, errors, warnings, suggestions,
                        f"{path}.{prop}" if path else prop
                    )
        
        elif schema_type == 'string':
            if not isinstance(data, str):
                errors.append(f"{path} must be a string")
                return
            
            # Check string constraints
            if 'minLength' in schema and len(data) < schema['minLength']:
                errors.append(f"{path} must be at least {schema['minLength']} characters")
            
            if 'maxLength' in schema and len(data) > schema['maxLength']:
                errors.append(f"{path} must be at most {schema['maxLength']} characters")
            
            if 'pattern' in schema:
                import re
                if not re.match(schema['pattern'], data):
                    errors.append(f"{path} does not match pattern: {schema['pattern']}")
            
            if 'enum' in schema and data not in schema['enum']:
                errors.append(f"{path} must be one of: {', '.join(schema['enum'])}")
            
            if 'const' in schema and data != schema['const']:
                errors.append(f"{path} must be exactly: {schema['const']}")
    
    def _validate_context(self, context: Dict[str, Any], action: str,
                         errors: List[str], warnings: List[str], suggestions: List[str]):
        """Validate context object"""
        required_fields = ['domain', 'action', 'country', 'city', 'core_version',
                          'bap_id', 'bap_uri', 'transaction_id', 'message_id', 'timestamp']
        
        for field in required_fields:
            if field not in context:
                errors.append(f"Missing required context field: {field}")
        
        # Validate specific fields
        if 'action' in context and context['action'] != action:
            errors.append(f"Context action '{context['action']}' does not match expected '{action}'")
        
        if 'country' in context and context['country'] != 'IND':
            warnings.append(f"Country '{context['country']}' is not standard 'IND'")
    
    def _validate_message(self, message: Dict[str, Any], action: str,
                         errors: List[str], warnings: List[str], suggestions: List[str]):
        """Validate message object based on action"""
        # Action-specific message validation
        if action == 'cancel':
            if 'order_id' not in message:
                errors.append("Missing required field: message.order_id")
            if 'cancellation_reason_id' not in message:
                errors.append("Missing required field: message.cancellation_reason_id")
        
        elif action == 'search':
            if 'intent' not in message:
                errors.append("Missing required field: message.intent")
        
        elif action in ['init', 'confirm']:
            if 'order' not in message:
                errors.append(f"Missing required field: message.order for {action}")
    
    def _apply_domain_rules(self, payload: Dict[str, Any], domain: str, action: str,
                           errors: List[str], warnings: List[str], suggestions: List[str]):
        """Apply domain-specific business rules"""
        if domain == 'Retail':
            # Retail-specific validations
            if action == 'cancel' and 'message' in payload:
                reason_id = payload['message'].get('cancellation_reason_id')
                if reason_id:
                    # Validate reason codes
                    valid_codes = {
                        '001': 'Buyer Cancellation',
                        '002': 'Seller Cancellation', 
                        '003': 'Agent Cancellation',
                        '996': 'Order confirmation failure',
                        '997': 'Order confirmation failure'
                    }
                    if reason_id not in valid_codes:
                        warnings.append(f"Unknown cancellation reason code: {reason_id}")
                        suggestions.append(f"Valid codes: {', '.join(f'{k} ({v})' for k, v in valid_codes.items())}")
    
    def _generate_ai_suggestions(self, payload: Dict[str, Any], errors: List[str],
                               warnings: List[str], action: str) -> List[str]:
        """Generate AI-powered suggestions for fixing issues"""
        if not self.model:
            return []
        
        try:
            prompt = f"""
            ONDC {action} payload validation found issues:
            
            Errors: {json.dumps(errors)}
            Warnings: {json.dumps(warnings)}
            
            Payload excerpt: {json.dumps(payload, indent=2)[:1000]}
            
            Provide 2-3 specific suggestions to fix these issues.
            """
            
            response = self.model.generate_content(prompt)
            suggestions = response.text.strip().split('\n')
            return [s.strip() for s in suggestions if s.strip()]
        except Exception as e:
            logger.error(f"AI suggestion generation failed: {e}")
            return []
    
    def _create_result(self, is_valid: bool, errors: List[str], warnings: List[str],
                      suggestions: List[str], start_time: datetime,
                      domain: Optional[str] = None, action: Optional[str] = None,
                      version: Optional[str] = None) -> Dict[str, Any]:
        """Create validation result"""
        return {
            'valid': is_valid,
            'domain': domain,
            'action': action,
            'version': version,
            'errors': errors,
            'warnings': warnings,
            'suggestions': suggestions,
            'validation_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_supported_schemas(self) -> Dict[str, List[str]]:
        """Get list of all supported schemas"""
        supported = {}
        for domain, schemas in self.schemas.items():
            actions = list(set(s['action'] for s in schemas))
            supported[domain] = sorted(actions)
        return supported


def main():
    """Test the universal validator"""
    validator = UniversalONDCValidator()
    
    # Print supported schemas
    print("\n=== Supported Schemas ===")
    supported = validator.get_supported_schemas()
    for domain, actions in supported.items():
        print(f"\n{domain}:")
        for action in actions:
            print(f"  - {action}")
    
    # Test with a sample payload
    test_payload = {
        "context": {
            "domain": "ONDC:RET10",
            "country": "IND",
            "city": "std:080",
            "action": "cancel",
            "core_version": "1.2.0",
            "bap_id": "buyerapp.com",
            "bap_uri": "https://buyerapp.com/ondc",
            "bpp_id": "sellerapp.com",
            "bpp_uri": "https://sellerapp.com/ondc",
            "transaction_id": "T1",
            "message_id": "M1",
            "timestamp": "2023-06-03T08:00:00.000Z",
            "ttl": "PT30M"
        },
        "message": {
            "order_id": "O1",
            "cancellation_reason_id": "001",
            "descriptor": {
                "name": "Order cancelled by buyer",
                "short_desc": "Buyer changed mind"
            }
        }
    }
    
    print("\n=== Testing Validation ===")
    result = validator.validate_payload(test_payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()