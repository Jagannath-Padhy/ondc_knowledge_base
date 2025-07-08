#!/usr/bin/env python3
"""
MCP Server for ONDC Payload Validation
Provides validation services through MCP protocol
"""

import json
import asyncio
from typing import Dict, Any
from pathlib import Path
from ondc_validator import ONDCValidator


class MCPValidationServer:
    """MCP Server for ONDC validation"""
    
    def __init__(self):
        self.validator = ONDCValidator()
        self.server_info = {
            "name": "ondc-validator",
            "version": "1.0.0",
            "description": "ONDC payload validation service"
        }
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP requests"""
        method = request.get("method")
        params = request.get("params", {})
        
        if method == "validate":
            return await self.validate_payload(params)
        elif method == "get_schemas":
            return await self.get_supported_schemas()
        elif method == "get_example":
            return await self.get_example_payload(params)
        else:
            return {
                "error": f"Unknown method: {method}",
                "supported_methods": ["validate", "get_schemas", "get_example"]
            }
    
    async def validate_payload(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate ONDC payload"""
        payload = params.get("payload")
        if not payload:
            return {"error": "Missing 'payload' parameter"}
        
        # Run validation
        result = self.validator.validate(payload)
        
        # Format response
        return {
            "valid": result["valid"],
            "validation_result": result,
            "summary": self._create_summary(result)
        }
    
    async def get_supported_schemas(self) -> Dict[str, Any]:
        """Get list of supported schemas"""
        retail_actions = list(set(s.split('_v')[0] for s in self.validator.retail_schemas.keys()))
        fis_actions = list(self.validator.fis_schemas.keys())
        
        return {
            "domains": {
                "Retail": {
                    "codes": ["RET10", "RET11", "RET12", "RET13", "RET14", "RET15", "RET16"],
                    "actions": sorted(retail_actions),
                    "versions": ["1.2.0", "1.2.5"]
                },
                "FIS": {
                    "codes": ["FIS10", "FIS11", "FIS12", "FIS13"],
                    "actions": sorted(fis_actions),
                    "versions": []
                }
            }
        }
    
    async def get_example_payload(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get example payload for specific action"""
        action = params.get("action", "search")
        domain = params.get("domain", "ONDC:RET10")
        
        examples = {
            "search": {
                "context": {
                    "domain": domain,
                    "country": "IND",
                    "city": "std:080",
                    "action": "search",
                    "core_version": "1.2.0",
                    "bap_id": "buyerNP.com",
                    "bap_uri": "https://buyerNP.com/ondc",
                    "transaction_id": "T1",
                    "message_id": "M1",
                    "timestamp": "2023-06-03T08:00:00.000Z",
                    "ttl": "PT30S"
                },
                "message": {
                    "intent": {
                        "category": {
                            "id": "Foodgrains"
                        },
                        "fulfillment": {
                            "type": "Delivery"
                        },
                        "payment": {
                            "@ondc/org/buyer_app_finder_fee_type": "percent",
                            "@ondc/org/buyer_app_finder_fee_amount": "3"
                        }
                    }
                }
            },
            "cancel": {
                "context": {
                    "domain": domain,
                    "country": "IND",
                    "city": "std:080",
                    "action": "cancel",
                    "core_version": "1.2.0",
                    "bap_id": "buyerNP.com",
                    "bap_uri": "https://buyerNP.com/ondc",
                    "bpp_id": "sellerNP.com",
                    "bpp_uri": "https://sellerNP.com/ondc",
                    "transaction_id": "T2",
                    "message_id": "M5",
                    "timestamp": "2023-06-04T10:00:00.000Z",
                    "ttl": "PT30S"
                },
                "message": {
                    "order_id": "O1",
                    "cancellation_reason_id": "003",
                    "descriptor": {
                        "short_desc": "Order cancelled by buyer"
                    }
                }
            }
        }
        
        if action in examples:
            return {
                "action": action,
                "domain": domain,
                "example": examples[action]
            }
        else:
            return {
                "error": f"No example available for action: {action}",
                "available_actions": list(examples.keys())
            }
    
    def _create_summary(self, result: Dict[str, Any]) -> str:
        """Create human-readable summary"""
        if result["valid"]:
            return f"✅ Payload is valid for {result['metadata']['action']} in {result['metadata']['domain']}"
        else:
            error_count = len(result["errors"])
            warning_count = len(result["warnings"])
            return f"❌ Validation failed with {error_count} errors and {warning_count} warnings"


async def main():
    """Run MCP server"""
    server = MCPValidationServer()
    
    print("🚀 ONDC MCP Validation Server Started")
    print("=" * 50)
    
    # Test with the user's payload
    test_request = {
        "method": "validate",
        "params": {
            "payload": {
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
                            "@ondc/org/buyer_app_finder_fee_amount": "edw"
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
        }
    }
    
    print("\n📋 Testing with provided payload...")
    response = await server.handle_request(test_request)
    
    print(f"\nSummary: {response['summary']}")
    
    if not response["valid"]:
        print("\n🚨 Validation Errors:")
        for error in response["validation_result"]["errors"]:
            print(f"  • {error}")
        
        if response["validation_result"]["suggestions"]:
            print("\n💡 Suggestions:")
            for suggestion in response["validation_result"]["suggestions"]:
                print(f"  • {suggestion}")
    
    # Show supported schemas
    print("\n\n📚 Supported Schemas:")
    schemas_response = await server.get_supported_schemas()
    for domain, info in schemas_response["domains"].items():
        print(f"\n{domain}:")
        print(f"  Domain codes: {', '.join(info['codes'])}")
        print(f"  Actions: {', '.join(info['actions'])}")
        if info['versions']:
            print(f"  Versions: {', '.join(info['versions'])}")


if __name__ == "__main__":
    asyncio.run(main())