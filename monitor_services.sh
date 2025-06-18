#!/bin/bash

# Service Monitoring Script for Laptop 2

echo "🔍 Claude Historical Analysis - Service Status"
echo "=============================================="

# Check Qdrant Docker
echo "📊 Qdrant Docker Container:"
if docker ps | grep -q qdrant; then
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep qdrant
    echo "✅ Qdrant is running"
else
    echo "❌ Qdrant container not running"
fi

echo ""

# Check Qdrant API
echo "🔗 Qdrant API Status:"
if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    COLLECTION_INFO=$(curl -s http://localhost:6333/collections | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    collections = data.get('result', {}).get('collections', [])
    for c in collections:
        print(f'  - {c[\"name\"]}: Active')
except:
    print('  - Unable to parse collection data')
    ")
    echo "✅ API responding"
    echo "$COLLECTION_INFO"
else
    echo "❌ API not responding"
fi

echo ""

# Check Claude interface process
echo "🎛️ Claude Interface Status:"
if pgrep -f "streamlit run claude_rag_interface.py" > /dev/null; then
    PID=$(pgrep -f "streamlit run claude_rag_interface.py")
    echo "✅ Interface running (PID: $PID)"
    echo "🌐 Access URL: http://100.72.255.25:8501"
else
    echo "❌ Interface not running"
    echo "💡 Start with: ./start_production.sh"
fi

echo ""

# Network connectivity
echo "🌐 Network Status:"
TAILSCALE_IP=$(ifconfig | grep "100\." | awk '{print $2}' | head -1)
if [ -n "$TAILSCALE_IP" ]; then
    echo "✅ Tailscale IP: $TAILSCALE_IP"
    echo "🔗 External access: http://$TAILSCALE_IP:8501"
else
    echo "❌ Tailscale not detected"
fi

# Check if port 8501 is open
if netstat -ln | grep -q ":8501 "; then
    echo "✅ Port 8501 is open and listening"
else
    echo "❌ Port 8501 not listening"
fi

