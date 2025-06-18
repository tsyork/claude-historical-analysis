#!/bin/bash

# Cloud Migration Preparation Script
# Helps estimate costs and prepare for cloud deployment

echo "☁️ Cloud Migration Preparation"
echo "=============================="

echo "📊 Current System Analysis:"
echo ""

# Analyze current resource usage
echo "💾 Storage Requirements:"
if [ -d "/var/lib/docker/volumes" ]; then
    QDRANT_SIZE=$(du -sh /var/lib/docker/volumes/*qdrant* 2>/dev/null | head -1 | awk '{print $1}' || echo "Unknown")
    echo "  - Qdrant data: $QDRANT_SIZE"
fi

TOTAL_SIZE=$(du -sh . | awk '{print $1}')
echo "  - Interface code: $TOTAL_SIZE"

echo ""
echo "🔍 Vector Database Stats:"
if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    curl -s http://localhost:6333/collections | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    collections = data.get('result', {}).get('collections', [])
    for c in collections:
        name = c['name']
        # Get detailed collection info
        print(f'Collection: {name}')
        # Additional collection details would require another API call
except Exception as e:
    print(f'Unable to analyze: {e}')
    "
fi

echo ""
echo "💰 Estimated Cloud Costs (Monthly):"
echo "  📋 Option 1: Google Cloud Run + Qdrant Cloud"
echo "     - Cloud Run (1 vCPU, 2GB): ~$15-30"
echo "     - Qdrant Cloud (starter): ~$25-50"
echo "     - Total: ~$40-80/month"
echo ""
echo "  📋 Option 2: Single GCP VM + Docker"
echo "     - e2-standard-2 (2 vCPU, 8GB): ~$50-70"
echo "     - Persistent disk (50GB): ~$8"
echo "     - Total: ~$58-78/month"
echo ""
echo "  📋 Option 3: Cloud Run + Cloud SQL (vector extension)"
echo "     - Cloud Run: ~$15-30"
echo "     - Cloud SQL PostgreSQL: ~$40-60"
echo "     - Total: ~$55-90/month"

echo ""
echo "🚀 Recommended Migration Path:"
echo "  1. Start with Option 2 (single VM) for simplicity"
echo "  2. Use Docker Compose for easy management"
echo "  3. Migrate to Cloud Run + Qdrant Cloud when scaling needed"

echo ""
echo "📋 Migration Checklist:"
echo "  ☐ Export Qdrant data: docker exec <container> tar -czf /backup.tar.gz /qdrant/storage"
echo "  ☐ Create cloud VM with Docker support"
echo "  ☐ Set up Tailscale on cloud VM for seamless migration"
echo "  ☐ Test interface with cloud-hosted Qdrant"
echo "  ☐ Update DNS/access URLs for public access"

