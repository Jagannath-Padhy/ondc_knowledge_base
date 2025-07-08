#!/usr/bin/env python3
"""
Streamlit App for ONDC Payload Validation
Comprehensive interface for validating ONDC payloads with MCP integration
"""

import streamlit as st
import json
import os
from datetime import datetime
from pathlib import Path
import requests
from typing import Dict, Any

# Import the MCP validation server
from mcp_validation_server_universal import UniversalONDCValidator

# Configure Streamlit page
st.set_page_config(
    page_title="ONDC Payload Validator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "validator" not in st.session_state:
    st.session_state.validator = UniversalONDCValidator()

if "mcp_tools" not in st.session_state:
    # Initialize mock MCP tools until proper implementation
    st.session_state.mcp_tools = {
        "get_cancel_example": lambda params: {
            "context": {
                "domain": "ONDC:RET120",
                "country": "IND",
                "city": "std:080",
                "action": "cancel",
                "core_version": params.get("version", "1.2.0"),
                "bap_id": "buyer-app.ondc.org",
                "bap_uri": "https://buyer-app.ondc.org/ondc",
                "bpp_id": "seller-app.ondc.org",
                "bpp_uri": "https://seller-app.ondc.org/ondc",
                "transaction_id": "T1234567890",
                "message_id": "M1234567890",
                "timestamp": datetime.now().isoformat() + "Z",
                "ttl": "PT30S"
            },
            "message": {
                "order_id": "O1234567890",
                "cancellation_reason_id": "003" if params.get("force", False) else "001",
                "descriptor": {
                    "name": "Buyer Requested",
                    "short_desc": "Force cancellation requested" if params.get("force", False) else "Buyer requested cancellation"
                }
            }
        },
        "validate_payload": lambda params: st.session_state.validator.validate_payload(
            params["payload"],
            params.get("domain"),
            params.get("action"),
            params.get("version")
        ),
        "suggest_fixes": lambda params: {
            "suggestions": [
                "Check that all required fields are present",
                "Ensure timestamp is in ISO format",
                "Verify domain and action match the payload type"
            ]
        }
    }

if "validation_history" not in st.session_state:
    st.session_state.validation_history = []

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🔍 ONDC Payload Validator")
    st.markdown("Validate ONDC payloads with comprehensive error reporting and AI-powered suggestions")
    
    # Schema availability notice
    with st.expander("📌 Currently Supported Schemas", expanded=False):
        st.markdown("""
        **Retail Domain:**
        - Actions: `cancel`, `on_cancel`, `rating`, `status`, `track`
        - Versions: 1.2.0, 1.2.5
        
        **FIS Domain:**
        - Actions: All major flows including `search`, `select`, `init`, `confirm`, `cancel`, `status`, `update` and their callbacks
        - Version: Standard FIS schema
        
        *Note: Additional schemas for other domains and actions are being integrated.*
        """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Domain selection
        domain = st.selectbox(
            "Domain",
            ["retail", "FIS"],
            index=0,
            help="Currently supported domains: Retail, FIS"
        )
        
        # Version selection (only for retail)
        if domain == "retail":
            version = st.selectbox(
                "Version",
                ["1.2.0", "1.2.5"],
                index=1
            )
        else:
            version = None
            st.info("FIS domain uses default versioning")
        
        # Action selection based on domain
        if domain == "retail":
            action = st.selectbox(
                "Action",
                ["cancel", "on_cancel", "rating", "status", "track"],
                index=0,
                help="Available actions for Retail domain"
            )
        else:  # FIS
            action = st.selectbox(
                "Action", 
                ["search", "on_search", "select", "on_select", "init", "on_init", 
                 "confirm", "on_confirm", "cancel", "on_cancel", "status", "on_status",
                 "update", "on_update"],
                index=0,
                help="Available actions for FIS domain"
            )
        
        st.divider()
        
        # Quick examples
        st.header("📝 Quick Examples")
        
        # Domain-specific examples
        if domain == "retail" and action == "cancel":
            if st.button("Load Cancel Payload"):
                example = st.session_state.mcp_tools["get_cancel_example"]({
                    "version": version,
                    "force": False
                })
                st.session_state.payload_input = json.dumps(example, indent=2)
            
            if st.button("Load Force Cancel Payload"):
                example = st.session_state.mcp_tools["get_cancel_example"]({
                    "version": version,
                    "force": True
                })
                st.session_state.payload_input = json.dumps(example, indent=2)
        else:
            st.info("Examples for this action coming soon!")
        
        if st.button("Load Empty Template"):
            # Determine domain code
            if domain == "retail":
                domain_code = f"ONDC:RET{version.replace('.', '')}" if version else "ONDC:RET10"
            elif domain == "FIS":
                domain_code = "ONDC:FIS12"
            else:
                domain_code = domain.upper()
            
            template = {
                "context": {
                    "domain": domain_code,
                    "country": "IND",
                    "city": "std:080",
                    "action": action,
                    "core_version": version if version else "1.2.0",
                    "bap_id": "buyer-app.ondc.org",
                    "bap_uri": "https://buyer-app.ondc.org/ondc",
                    "bpp_id": "seller-app.ondc.org", 
                    "bpp_uri": "https://seller-app.ondc.org/ondc",
                    "transaction_id": "T1234567890",
                    "message_id": "M1234567890",
                    "timestamp": datetime.now().isoformat() + "Z",
                    "ttl": "PT30S"
                },
                "message": {
                    # Action-specific fields will be added here
                }
            }
            st.session_state.payload_input = json.dumps(template, indent=2)
        
        st.divider()
        
        # Validation history
        st.header("📊 Validation History")
        if st.session_state.validation_history:
            for i, result in enumerate(reversed(st.session_state.validation_history[-5:])):
                status = "✅" if result.get("valid", False) else "❌"
                st.caption(f"{status} {result['timestamp'][:19]}")
        else:
            st.caption("No validations yet")
        
        if st.button("Clear History"):
            st.session_state.validation_history = []
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input Payload")
        
        # Payload input area
        payload_input = st.text_area(
            "Enter ONDC payload JSON:",
            height=400,
            value=st.session_state.get("payload_input", ""),
            help="Paste or type your ONDC payload JSON here"
        )
        
        # Validation button
        if st.button("🔍 Validate Payload", type="primary"):
            if payload_input.strip():
                try:
                    # Parse JSON
                    payload = json.loads(payload_input)
                    
                    # Validate using MCP tool
                    result = st.session_state.mcp_tools["validate_payload"]({
                        "payload": payload,
                        "schema_name": action,
                        "domain": domain,
                        "version": version,
                        "action": action
                    })
                    
                    # Store result
                    result["timestamp"] = datetime.now().isoformat()
                    st.session_state.validation_result = result
                    st.session_state.validation_payload = payload
                    st.session_state.validation_history.append(result)
                    
                    st.rerun()
                    
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {str(e)}")
                except Exception as e:
                    st.error(f"Validation error: {str(e)}")
            else:
                st.warning("Please enter a payload to validate")
        
        # Format JSON button
        if st.button("🔧 Format JSON"):
            if payload_input.strip():
                try:
                    formatted = json.dumps(json.loads(payload_input), indent=2)
                    st.session_state.payload_input = formatted
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON - cannot format")
    
    with col2:
        st.header("📋 Validation Results")
        
        if "validation_result" in st.session_state:
            result = st.session_state.validation_result
            
            # Overall status
            if result.get("valid", False):
                st.success("✅ Validation PASSED")
                st.balloons()
            else:
                st.error("❌ Validation FAILED")
            
            # Metrics
            col_metrics1, col_metrics2 = st.columns(2)
            
            with col_metrics1:
                st.metric("Errors", len(result["errors"]))
                st.metric("Warnings", len(result["warnings"]))
            
            with col_metrics2:
                st.metric("Suggestions", len(result["suggestions"]))
                st.metric("Validation Time", f"{result['validation_time']:.2f}s")
            
            # Detailed results
            if result["errors"]:
                st.subheader("🚨 Errors (Must Fix)")
                for i, error in enumerate(result["errors"], 1):
                    st.error(f"{i}. {error}")
            
            if result["warnings"]:
                st.subheader("⚠️ Warnings (Recommended)")
                for i, warning in enumerate(result["warnings"], 1):
                    st.warning(f"{i}. {warning}")
            
            if result["suggestions"]:
                st.subheader("💡 AI Suggestions")
                for i, suggestion in enumerate(result["suggestions"], 1):
                    st.info(f"{i}. {suggestion}")
            
            # Fix suggestions
            if result["errors"]:
                st.subheader("🔧 Quick Fixes")
                
                if st.button("Get Fix Suggestions"):
                    suggestions = st.session_state.mcp_tools["suggest_fixes"]({
                        "payload": st.session_state.validation_payload,
                        "errors": result["errors"],
                        "action": action
                    })
                    
                    for i, suggestion in enumerate(suggestions["suggestions"], 1):
                        st.info(f"{i}. {suggestion}")
            
            # Export results
            st.subheader("📤 Export Results")
            
            export_data = {
                "validation_result": result,
                "payload": st.session_state.validation_payload,
                "config": {
                    "domain": domain,
                    "version": version,
                    "action": action
                }
            }
            
            st.download_button(
                label="Download Validation Report",
                data=json.dumps(export_data, indent=2),
                file_name=f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        else:
            st.info("Enter a payload and click 'Validate Payload' to see results")
    
    # Knowledge Base Section
    st.divider()
    st.header("📚 Knowledge Base")
    
    col_kb1, col_kb2 = st.columns([1, 1])
    
    with col_kb1:
        st.subheader("🔍 Search Documentation")
        
        # Search input
        search_query = st.text_input(
            "Search ONDC documentation:",
            placeholder="e.g., 'what is force cancellation?'"
        )
        
        if st.button("Search"):
            if search_query.strip():
                # Use the old web app search functionality
                try:
                    # This would integrate with the existing knowledge base
                    st.info("Searching ONDC documentation...")
                    
                    # For now, show a placeholder
                    if "force cancellation" in search_query.lower():
                        st.write("""
                        **Force Cancellation** is a feature in ONDC that allows:
                        
                        - Immediate cancellation of orders under specific conditions
                        - Bypass of normal cancellation workflows
                        - Requires special authorization and audit trails
                        
                        **Force Cancellation Payload Example:**
                        """)
                        
                        force_example = st.session_state.mcp_tools["get_cancel_example"]({
                            "version": "1.2.0",
                            "force": True
                        })
                        
                        st.code(json.dumps(force_example, indent=2), language="json")
                    else:
                        st.warning("Search functionality is being optimized. Please use the validation tools for now.")
                        
                except Exception as e:
                    st.error(f"Search error: {str(e)}")
            else:
                st.warning("Please enter a search query")
    
    with col_kb2:
        st.subheader("📖 Common Validations")
        
        # Common validation scenarios
        if domain == "retail":
            scenarios = [
                "Force Cancellation (Retail)",
                "Regular Cancellation (Retail)", 
                "Rating Flow (Retail)",
                "Status Check (Retail)",
                "Order Tracking (Retail)"
            ]
        else:  # FIS
            scenarios = [
                "Search Flow (FIS)",
                "Select & Init (FIS)",
                "Order Confirmation (FIS)",
                "Cancellation (FIS)",
                "Status Update (FIS)"
            ]
        
        selected_scenario = st.selectbox("Select Scenario:", scenarios)
        
        if st.button("Load Scenario"):
            if selected_scenario == "Force Cancellation (Retail)" and domain == "retail":
                example = st.session_state.mcp_tools["get_cancel_example"]({
                    "version": version,
                    "force": True
                })
                st.session_state.payload_input = json.dumps(example, indent=2)
                st.rerun()
            
            elif selected_scenario == "Regular Cancellation (Retail)" and domain == "retail":
                example = st.session_state.mcp_tools["get_cancel_example"]({
                    "version": version,
                    "force": False
                })
                st.session_state.payload_input = json.dumps(example, indent=2)
                st.rerun()
            
            else:
                st.info(f"Example templates for '{selected_scenario}' are being developed. Please use the validation feature with your own payloads.")
    
    # Footer
    st.divider()
    st.caption("🤖 ONDC Payload Validator - Powered by MCP Tools and AI")

if __name__ == "__main__":
    main()