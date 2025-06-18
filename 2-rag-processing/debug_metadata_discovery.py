#!/usr/bin/env python3
"""
Debug the metadata discovery logic
"""

from google.cloud import storage
import json

# Initialize GCS client
client = storage.Client.from_service_account_json("google-cloud-service-account.json")
bucket = client.bucket("ai_knowledgebase")

def debug_metadata_discovery():
    print("🔍 Debugging metadata discovery...")
    
    # Test the exact prefix we're using
    metadata_prefix = "podcasts/revolutions/metadata/"
    print(f"📁 Looking for blobs with prefix: {metadata_prefix}")
    
    # List all blobs with this prefix
    all_blobs = list(bucket.list_blobs(prefix=metadata_prefix))
    print(f"📄 Found {len(all_blobs)} total blobs")
    
    # Show first 10 blob names to see the structure
    print(f"\n📋 First 10 blob names:")
    for i, blob in enumerate(all_blobs[:10]):
        print(f"  {i+1}. {blob.name}")
        
        # Split path and analyze
        path_parts = blob.name.split('/')
        print(f"     Path parts: {path_parts}")
        print(f"     Length: {len(path_parts)}")
        
        if len(path_parts) >= 5:
            print(f"     Part 4 (should be season_XX): '{path_parts[4]}'")
            print(f"     Starts with 'season_': {path_parts[4].startswith('season_')}")
        print()
    
    # Let's also try to find season folders manually
    print("\n🔍 Looking for season folders manually...")
    
    # Try alternative approaches
    season_prefixes = [
        "podcasts/revolutions/metadata/season_01/",
        "podcasts/revolutions/metadata/season_06/", 
        "podcasts/revolutions/metadata/season_08/"
    ]
    
    for prefix in season_prefixes:
        blobs = list(bucket.list_blobs(prefix=prefix))
        print(f"📁 {prefix}: {len(blobs)} files")
        if blobs:
            print(f"   First file: {blobs[0].name}")

if __name__ == "__main__":
    debug_metadata_discovery()
