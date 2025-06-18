#!/bin/bash

# Production Startup Script for Claude Historical Analysis Interface
echo "🏛️ Starting Claude Historical Analysis Interface (Production)"
echo "============================================================="

# Check environment
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Please configure environment variables."
    exit 1
fi

source .env
GCS_CREDS_PATH=${GCS_CREDENTIALS_PATH:-"credentials.json"}
if [ ! -f "$GCS_CREDS_PATH" ]; then
    echo "❌ GCS credentials not found at: $GCS_CREDS_PATH"
    echo "Please check your GCS_CREDENTIALS_PATH in .env file"
    exit 1
fi

# Check API key configuration
if grep -q "your_anthropic_api_key_here" .env; then
    echo "❌ Please configure your Anthropic API key in .env"
    echo "Edit the file and replace 'your_anthropic_api_key_here' with your actual key"
    exit 1
fi

# Load environment
source .env
export $(grep -v '^#' .env | xargs)

# Activate virtual environment
source venv/bin/activate

# Check Qdrant connectivity
echo "🔍 Checking Qdrant connection..."
if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    echo "✅ Qdrant accessible on localhost:6333"
    COLLECTION_COUNT=$(curl -s http://localhost:6333/collections | python -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('result', {}).get('collections', [])))" 2>/dev/null || echo "0")
    echo "📊 Available collections: $COLLECTION_COUNT"
else
    echo "❌ Cannot reach Qdrant. Ensure Docker container is running:"
    echo "   docker run -p 6333:6333 qdrant/qdrant"
    exit 1
fi

# Get current IP for access information
TAILSCALE_IP=$(ifconfig | grep "100\." | awk '{print $2}' | head -1)
echo ""
echo "🌐 Starting web interface accessible from anywhere via Tailscale:"
echo "   Local access: http://localhost:8501"
echo "   Tailscale access: http://$TAILSCALE_IP:8501"
echo "   External access: http://100.72.255.25:8501"
echo ""

# Start Streamlit in production mode
streamlit run claude_rag_interface.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.runOnSave false \
    --server.fileWatcherType none
