#!/usr/bin/env python3
"""
ONDC Payload Validator
Validates ONDC payloads against schemas and business rules from documentation
"""

import json
import os
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ONDCValidator:
    """ONDC payload validator using real schemas and documentation"""
    
    def __init__(self):
        self.schemas_dir = Path("data/parsed_schemas")
        self.docs_dir = Path("scraped_docs")
        
        # Load schemas
        self.retail_schemas = self._load_retail_schemas()
        self.fis_schemas = self._load_fis_schemas()
        
        # Load business rules from documentation
        self.business_rules = self._load_business_rules()
        
        logger.info(f"Loaded {len(self.retail_schemas)} retail schemas")
        logger.info(f"Loaded {len(self.fis_schemas)} FIS schemas")
    
    def _load_retail_schemas(self) -> Dict[str, Any]:
        """Load retail schemas from parsed files"""
        schemas = {}
        
        # Load v1.2.0 schemas
        v120_file = self.schemas_dir / "retail_schemas_v1_2_0.json"
        if v120_file.exists():
            with open(v120_file, 'r') as f:
                data = json.load(f)
                for schema in data:
                    key = f"{schema['action']}_v{schema['version']}"
                    schemas[key] = schema
        
        # Load v1.2.5 schemas
        v125_file = self.schemas_dir / "retail_schemas_v1_2_5.json"
        if v125_file.exists():
            with open(v125_file, 'r') as f:
                data = json.load(f)
                for schema in data:
                    key = f"{schema['action']}_v{schema['version']}"
                    schemas[key] = schema
        
        return schemas
    
    def _load_fis_schemas(self) -> Dict[str, Any]:
        """Load FIS schemas"""
        schemas = {}
        
        fis_file = self.schemas_dir / "FIS_schemas.json"
        if fis_file.exists():
            with open(fis_file, 'r') as f:
                data = json.load(f)
                for schema in data:
                    key = schema['action']
                    schemas[key] = schema
        
        return schemas
    
    def _load_business_rules(self) -> Dict[str, Any]:
        """Load business rules from documentation"""
        rules = {
            "search": {
                "required_context_fields": [
                    "domain", "country", "city", "action", "core_version",
                    "bap_id", "bap_uri", "ttl", "transaction_id", "message_id", "timestamp"
                ],
                "required_message_fields": ["intent"],
                "intent_fields": {
                    "payment": {
                        "@ondc/org/buyer_app_finder_fee_type": {
                            "type": "enum",
                            "values": ["percent", "amount"],
                            "required": True
                        },
                        "@ondc/org/buyer_app_finder_fee_amount": {
                            "type": "string",
                            "pattern": r"^\d+(\.\d{1,2})?$",
                            "required": True,
                            "description": "Must be a numeric value"
                        }
                    }
                },
                "catalog_inc_fields": {
                    "mode": {
                        "type": "enum",
                        "values": ["start", "stop"]
                    },
                    "start_time": {
                        "type": "datetime",
                        "format": "ISO8601"
                    },
                    "end_time": {
                        "type": "datetime",
                        "format": "ISO8601"
                    }
                }
            },
            "cancel": {
                "required_context_fields": [
                    "domain", "country", "city", "action", "core_version",
                    "bap_id", "bap_uri", "bpp_id", "bpp_uri", "transaction_id", 
                    "message_id", "timestamp", "ttl"
                ],
                "required_message_fields": ["order_id", "cancellation_reason_id"],
                "valid_reason_codes": [
                    "001", "002", "003", "004", "005", "006", "009", "010",
                    "011", "013", "014", "016", "017", "018", "020"
                ]
            }
        }
        
        return rules
    
    def validate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate ONDC payload
        
        Returns:
            Dictionary with validation results including errors, warnings, and suggestions
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Extract context info
        context = payload.get('context', {})
        domain = context.get('domain', '')
        action = context.get('action', '')
        version = context.get('core_version', '')
        
        # Basic structure validation
        if not payload.get('context'):
            errors.append("Missing required 'context' field")
        
        if not payload.get('message'):
            errors.append("Missing required 'message' field")
        
        # Validate context
        self._validate_context(context, action, errors, warnings)
        
        # Validate based on action
        if action in self.business_rules:
            self._validate_by_action(payload, action, errors, warnings, suggestions)
        else:
            warnings.append(f"No validation rules found for action: {action}")
        
        # Domain-specific validation
        self._validate_domain(domain, errors, warnings)
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'suggestions': suggestions,
            'metadata': {
                'domain': domain,
                'action': action,
                'version': version,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _validate_context(self, context: Dict[str, Any], action: str, 
                         errors: List[str], warnings: List[str]):
        """Validate context fields"""
        rules = self.business_rules.get(action, {})
        required_fields = rules.get('required_context_fields', [])
        
        for field in required_fields:
            if field not in context:
                errors.append(f"Missing required context field: {field}")
        
        # Validate specific fields
        if context.get('country') != 'IND':
            warnings.append(f"Country should be 'IND', found: {context.get('country')}")
        
        if 'timestamp' in context:
            if not self._is_valid_timestamp(context['timestamp']):
                errors.append(f"Invalid timestamp format: {context['timestamp']}")
        
        if 'ttl' in context:
            if not self._is_valid_duration(context['ttl']):
                errors.append(f"Invalid TTL format: {context['ttl']}")
    
    def _validate_by_action(self, payload: Dict[str, Any], action: str,
                           errors: List[str], warnings: List[str], suggestions: List[str]):
        """Validate based on specific action"""
        rules = self.business_rules.get(action, {})
        message = payload.get('message', {})
        
        # Check required message fields
        required_fields = rules.get('required_message_fields', [])
        for field in required_fields:
            if field not in message:
                errors.append(f"Missing required message field: {field}")
        
        if action == 'search':
            self._validate_search(payload, errors, warnings, suggestions)
        elif action == 'cancel':
            self._validate_cancel(payload, errors, warnings, suggestions)
    
    def _validate_search(self, payload: Dict[str, Any], errors: List[str],
                        warnings: List[str], suggestions: List[str]):
        """Validate search payload"""
        message = payload.get('message', {})
        intent = message.get('intent', {})
        
        # Validate payment fields
        payment = intent.get('payment', {})
        if payment:
            # Validate finder fee type
            fee_type = payment.get('@ondc/org/buyer_app_finder_fee_type')
            if fee_type and fee_type not in ['percent', 'amount']:
                errors.append(f"Invalid buyer_app_finder_fee_type: {fee_type}. Must be 'percent' or 'amount'")
            
            # Validate finder fee amount
            fee_amount = payment.get('@ondc/org/buyer_app_finder_fee_amount')
            if fee_amount:
                if not re.match(r'^\d+(\.\d{1,2})?$', str(fee_amount)):
                    errors.append(f"Invalid buyer_app_finder_fee_amount: {fee_amount}. Must be a numeric value")
                    suggestions.append("Change buyer_app_finder_fee_amount to a numeric value like '3' or '3.5'")
        
        # Validate tags
        tags = intent.get('tags', [])
        for tag in tags:
            if tag.get('code') == 'catalog_inc':
                self._validate_catalog_inc(tag, errors, warnings)
    
    def _validate_catalog_inc(self, tag: Dict[str, Any], errors: List[str], warnings: List[str]):
        """Validate catalog_inc tag"""
        tag_list = tag.get('list', [])
        
        start_time = None
        end_time = None
        
        for item in tag_list:
            code = item.get('code')
            value = item.get('value')
            
            if code == 'start_time':
                if not self._is_valid_timestamp(value):
                    errors.append(f"Invalid start_time format: {value}")
                start_time = value
            
            elif code == 'end_time':
                if not self._is_valid_timestamp(value):
                    errors.append(f"Invalid end_time format: {value}")
                end_time = value
            
            elif code == 'mode':
                if value not in ['start', 'stop']:
                    errors.append(f"Invalid catalog_inc mode: {value}. Must be 'start' or 'stop'")
        
        # Validate time range
        if start_time and end_time:
            try:
                start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                if end <= start:
                    errors.append("end_time must be after start_time")
            except:
                pass
    
    def _validate_cancel(self, payload: Dict[str, Any], errors: List[str],
                        warnings: List[str], suggestions: List[str]):
        """Validate cancel payload"""
        message = payload.get('message', {})
        
        # Validate cancellation reason
        reason_id = message.get('cancellation_reason_id')
        if reason_id:
            valid_codes = self.business_rules['cancel']['valid_reason_codes']
            if reason_id not in valid_codes:
                warnings.append(f"Unknown cancellation_reason_id: {reason_id}")
                suggestions.append(f"Valid cancellation reason codes: {', '.join(valid_codes)}")
    
    def _validate_domain(self, domain: str, errors: List[str], warnings: List[str]):
        """Validate domain format"""
        if domain:
            if not domain.startswith('ONDC:'):
                errors.append(f"Domain must start with 'ONDC:' prefix, found: {domain}")
            
            # Extract domain code
            if ':' in domain:
                domain_code = domain.split(':')[1]
                valid_prefixes = ['RET', 'LOG', 'FIS', 'TRV']
                
                if not any(domain_code.startswith(prefix) for prefix in valid_prefixes):
                    warnings.append(f"Unknown domain code: {domain_code}")
    
    def _is_valid_timestamp(self, timestamp: str) -> bool:
        """Check if timestamp is valid ISO8601 format"""
        try:
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return True
        except:
            return False
    
    def _is_valid_duration(self, duration: str) -> bool:
        """Check if duration is valid ISO8601 duration format"""
        pattern = r'^P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?$'
        return bool(re.match(pattern, duration))


def main():
    """Test the validator with the provided payload"""
    validator = ONDCValidator()
    
    # Test payload from user
    test_payload = {
        "context": {
            "domain": "ONDC:RET15",
            "country": "IND",
            "city": "*",
            "action": "search",
            "core_version": "1.2.0",
            "bap_id": "buyer-app-preprod-v2.ondc.org",
            "bap_uri": "https://buyer-app-preprod-v2.ondc.org/protocol/v1",
            "ttl": "PT30S",
            "transaction_id": "c1732ef7-b89f-4b23-85dc-521f93164be9",
            "message_id": "d1fcbdc7-639c-4add-9811-0d0154a1c241",
            "timestamp": "2024-06-11T08:00:01.639Z"
        },
        "message": {
            "intent": {
                "payment": {
                    "@ondc/org/buyer_app_finder_fee_type": "percent",
                    "@ondc/org/buyer_app_finder_fee_amount": "edw"  # This is invalid - should be numeric
                },
                "tags": [
                    {
                        "code": "catalog_inc",
                        "list": [
                            {
                                "code": "start_time",
                                "value": "2024-06-11T07:00:01.639Z"
                            },
                            {
                                "code": "end_time",
                                "value": "2024-06-11T08:00:01.639Z"
                            }
                        ]
                    }
                ]
            }
        }
    }
    
    print("\n=== ONDC Payload Validation ===")
    print(f"Validating {test_payload['context']['action']} request for {test_payload['context']['domain']}")
    print("-" * 50)
    
    result = validator.validate(test_payload)
    
    print(f"\nValidation Result: {'✅ VALID' if result['valid'] else '❌ INVALID'}")
    
    if result['errors']:
        print("\n🚨 Errors:")
        for error in result['errors']:
            print(f"  - {error}")
    
    if result['warnings']:
        print("\n⚠️  Warnings:")
        for warning in result['warnings']:
            print(f"  - {warning}")
    
    if result['suggestions']:
        print("\n💡 Suggestions:")
        for suggestion in result['suggestions']:
            print(f"  - {suggestion}")
    
    print("\n📊 Metadata:")
    for key, value in result['metadata'].items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()