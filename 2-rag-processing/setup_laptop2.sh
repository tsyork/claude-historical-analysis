#!/bin/bash
echo "🔧 Setting up Laptop 2 for processing..."

# Create virtual environment
python3 -m venv processing_env
source processing_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install qdrant-client openai google-cloud-storage google-api-python-client tiktoken python-dotenv

echo "✅ Setup complete!"
echo "📋 To start processing:"
echo "  source processing_env/bin/activate"
echo "  python3 standalone_document_processor.py"
