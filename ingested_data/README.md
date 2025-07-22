# Ingested Data

This folder contains all data extracted and processed during ingestion.

## Structure

- `chunks/` - JSON files containing all chunks created from source documents
- `metadata/` - Ingestion summaries and statistics

## How to View

Use the main viewing script:
```bash
python view_data.py
```

Options:
- `--summary` - Show ingestion summary
- `--chunks [type]` - Show chunks of specific type
- `--search [term]` - Search across all chunks
- `--endpoint [name]` - Show data for specific API endpoint

## Chunk Types

1. **api_endpoint_master** - Complete API documentation
2. **json_example** - Individual JSON payload examples
3. **special_topic** - Cross-cutting topics (payment, cancellation, etc.)
4. **cross_reference** - Reference definitions
5. **schema** - YAML schema definitions

## Files Generated

After ingestion, you'll find:
- `chunks/text_contract_chunks.json` - All text-based chunks
- `chunks/yaml_spec_chunks.json` - All YAML-based chunks
- `metadata/ingestion_summary.json` - Overall statistics
- `metadata/text_ingestion.json` - Text ingestion details
- `metadata/yaml_ingestion.json` - YAML ingestion details