#!/usr/bin/env python3
"""
Debug script to explore your actual GCS metadata structure
"""

from google.cloud import storage
import json

# Initialize GCS client
client = storage.Client.from_service_account_json("google-cloud-service-account.json")
bucket = client.bucket("ai_knowledgebase")

def explore_metadata_structure():
    print("🔍 Exploring GCS metadata structure...")
    
    # Look for season 8 metadata specifically
    season_8_prefix = "podcasts/revolutions/metadata/season_08/"
    
    print(f"\n📁 Files in {season_8_prefix}:")
    blobs = list(bucket.list_blobs(prefix=season_8_prefix))
    
    for blob in blobs:
        if blob.name.endswith('.json'):
            print(f"  📄 {blob.name}")
            
            # Try to read the first few files to see the structure
            try:
                content = json.loads(blob.download_as_text())
                print(f"      Season: {content.get('season', 'unknown')}")
                print(f"      Episode: {content.get('episode_number', 'unknown')}")
                print(f"      Title: {content.get('title', 'unknown')}")
            except Exception as e:
                print(f"      Error reading: {e}")
            print()
    
    # Also check if there are other metadata folder patterns
    print("\n🔍 Checking other possible metadata locations...")
    
    alternative_prefixes = [
        "podcasts/revolutions/metadata/",
        "metadata/",
        "revolutions/metadata/"
    ]
    
    for prefix in alternative_prefixes:
        print(f"\n📁 Checking {prefix}:")
        try:
            blobs = list(bucket.list_blobs(prefix=prefix, max_results=10))
            for blob in blobs[:5]:  # Show first 5
                print(f"  📄 {blob.name}")
        except:
            print(f"  ❌ No files found")

if __name__ == "__main__":
    explore_metadata_structure()
