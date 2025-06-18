#!/usr/bin/env python3
"""
Test script for Phase 3 infrastructure connectivity
"""

import os
import sys
from dotenv import load_dotenv

# Load environment
load_dotenv()

def test_qdrant_connection():
    """Test connection to Qdrant on localhost"""
    try:
        from qdrant_client import QdrantClient
        
        host = os.getenv('QDRANT_HOST', 'localhost')
        port = int(os.getenv('QDRANT_PORT', '6333'))
        collection = os.getenv('QDRANT_COLLECTION', 'revolutions_podcast_v1')
        
        print(f"🔍 Testing Qdrant connection to {host}:{port}")
        
        client = QdrantClient(host=host, port=port)
        collections = client.get_collections()
        
        print(f"✅ Qdrant connected successfully")
        print(f"📊 Available collections:")
        for coll in collections.collections:
            print(f"   - {coll.name}")
            if coll.name == collection:
                info = client.get_collection(collection)
                print(f"     └─ {info.vectors_count} vectors")
        
        return True
        
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        return False

def test_anthropic_connection():
    """Test Anthropic API connection"""
    try:
        import anthropic
        
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key or api_key == 'your_anthropic_api_key_here':
            print("❌ Anthropic API key not configured")
            return False
        
        print("🔍 Testing Anthropic API connection")
        
        client = anthropic.Anthropic(api_key=api_key)
        
        # Simple test message
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        print("✅ Anthropic API connected successfully")
        return True
        
    except Exception as e:
        print(f"❌ Anthropic API connection failed: {e}")
        return False

def test_gcs_connection():
    """Test Google Cloud Storage connection"""
    try:
        from google.cloud import storage
        from google.oauth2 import service_account
        
        creds_path = os.getenv('GCS_CREDENTIALS_PATH', 'credentials.json')
        project_id = os.getenv('GCS_PROJECT_ID', 'podcast-transcription-462218')
        bucket_name = os.getenv('GCS_BUCKET', 'ai_knowledgebase')
        
        if not os.path.exists(creds_path):
            print(f"❌ GCS credentials not found: {creds_path}")
            return False
        
        print(f"🔍 Testing GCS connection to {bucket_name}")
        
        credentials = service_account.Credentials.from_service_account_file(creds_path)
        client = storage.Client(credentials=credentials, project=project_id)
        bucket = client.bucket(bucket_name)
        
        # Test listing some blobs
        blobs = list(bucket.list_blobs(prefix="podcasts/revolutions/metadata/", max_results=5))
        
        print(f"✅ GCS connected successfully")
        print(f"📁 Found {len(blobs)} metadata files in bucket")
        
        return True
        
    except Exception as e:
        print(f"❌ GCS connection failed: {e}")
        return False

def main():
    print("🧪 Phase 3 Infrastructure Connectivity Test")
    print("=" * 50)
    
    results = {
        'qdrant': test_qdrant_connection(),
        'anthropic': test_anthropic_connection(),
        'gcs': test_gcs_connection()
    }
    
    print("\n📊 Test Results Summary:")
    print("-" * 30)
    
    all_passed = True
    for service, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{service.upper():>12}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All systems ready for Phase 3!")
        print("Run: ./start_production.sh to start the historical analysis interface")
    else:
        print("\n⚠️  Some systems need configuration. Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
