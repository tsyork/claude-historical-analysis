# Phase 3: Analysis Interface

Claude Sonnet 4-powered web interface for sophisticated historical analysis.

## Files
- `claude_rag_interface.py` - Main Streamlit application
- `start_production.sh` - Production startup script
- `monitor_services.sh` - System monitoring utilities

## Features
- AI-powered episode detection
- 5 sophisticated analysis types (basic, comparative, character, thematic, chronological)
- Vector search across 2880+ indexed chunks
- External access via Tailscale
- Real-time system monitoring

## Usage
1. Ensure Qdrant is running with indexed data
2. Configure API keys in .env file
3. Run: `./start_production.sh`
