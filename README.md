# Claude Historical Analysis Platform

Complete AI-powered historical research platform with automated transcription, vector processing, and sophisticated analysis capabilities.

## 🏗️ System Architecture

### [Phase 1: Transcription](1-transcription/)
AWS Spot Fleet-based podcast transcription using OpenAI Whisper
- GPU-accelerated processing (8-10x faster)
- Cost-effective spot pricing
- Automated deployment and teardown

### [Phase 2: RAG Processing](2-rag-processing/)
Document processing pipeline for vector database preparation
- Metadata-first processing approach
- OpenAI embeddings integration
- Qdrant vector database setup

### [Phase 3: Analysis Interface](3-analysis-interface/)
Claude Sonnet 4-powered web interface for historical analysis
- AI episode detection
- 5 sophisticated analysis types
- Cross-revolution comparative studies

## 🚀 Quick Start

Each phase has its own README with detailed setup instructions. Follow phases in order:

1. **Transcription**: Convert audio to text transcripts
2. **RAG Processing**: Index transcripts into vector database  
3. **Analysis Interface**: Query and analyze historical content

## 📊 System Capabilities

- **Episodes Processed**: 336+ across 11 revolutionary periods
- **Vector Database**: 2880+ searchable chunks
- **Time Coverage**: 275+ years of revolutionary history
- **Query Types**: Basic, comparative, character, thematic, chronological

## 🔧 Requirements

- Python 3.12+
- AWS account (for transcription)
- OpenAI API key
- Anthropic/Claude API key
- Google Cloud credentials (optional)
