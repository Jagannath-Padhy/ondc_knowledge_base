# ONDC Knowledge Base

A semantic search system for ONDC API documentation using RAG (Retrieval Augmented Generation).

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Qdrant database:**
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

3. **Run ingestion:**
   ```bash
   python ingest.py
   ```

4. **Launch the app:**
   ```bash
   streamlit run app.py
   ```

## Usage

### Ingesting Data
```bash
# Use default files from knowledge_sources/
python ingest.py

# Clean database first
python ingest.py --clean

# Specify custom files
python ingest.py --text path/to/contract.txt --yaml path/to/spec.yaml
```

### Viewing Data
```bash
# Show summary
python view_data.py

# Show chunks by type
python view_data.py --chunks json_example

# Search content
python view_data.py --search "on_init"

# View specific endpoint data
python view_data.py --endpoint "/on_init"
```

### Database Management
```bash
# Clean database
python cleanup_db.py
```

## Files Structure

```
ondc-kb-poc/
├── knowledge_sources/        # Source documents
├── ingested_data/           # Processed chunks and metadata
├── src/                     # Core modules
├── ingest.py               # Main ingestion script
├── app.py                  # Streamlit interface
├── view_data.py            # Data viewer
└── cleanup_db.py           # Database cleanup
```

## Data Sources

Place your source files in `knowledge_sources/`:
- `ONDC - API Contract for Retail (v1.2.0).txt` - Text contract (recommended)
- `build.yaml` - OpenAPI specification

## Example Queries

Try these in the Streamlit app:
- "on_init payload"
- "/on_search complete documentation"  
- "payment collection by seller NP"
- "force cancellation process"

## Features

- **Text-based ingestion** with complete JSON payload extraction
- **Cross-reference linking** for comprehensive context
- **Special topic extraction** (payment, cancellation, etc.)
- **Enhanced search** with detailed responses including full payloads
- **Data viewing tools** to inspect ingested content

## Architecture

- **Vector Database**: Qdrant for semantic search
- **Embeddings**: Google Gemini
- **LLM**: Google Gemini for response generation  
- **RAG Pipeline**: Retrieval + generation for comprehensive answers
- **UI**: Streamlit for interactive queries