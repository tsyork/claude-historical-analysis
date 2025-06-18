# Phase 2: RAG Processing System

Document processing pipeline that converts transcripts into searchable vector chunks.

## Files
- `fixed_metadata_first_processor.py` - Latest working processor (recommended)
- `enhanced_rag_processor.py` - Enhanced version with additional features
- `batch_processor.py` - Batch processing utilities
- `setup_laptop2.sh` - Environment setup script

## Features
- Metadata-first processing for reliability
- OpenAI embeddings (text-embedding-3-small)
- Qdrant vector database integration
- Google Cloud Storage metadata tracking
- Comprehensive error handling and resume capability

## Usage
1. Set up Python 3.12 environment
2. Configure credentials and environment variables
3. Run: `python fixed_metadata_first_processor.py`
