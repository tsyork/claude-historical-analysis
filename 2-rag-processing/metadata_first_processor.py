#!/usr/bin/env python3
"""
Metadata-First RAG Processor
Scans GCS metadata files and processes only unprocessed episodes
"""

import os
import uuid
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import time
import json
import io

from dotenv import load_dotenv
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import tiktoken
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from googleapiclient.http import MediaIoBaseDownload
from google.cloud import storage

load_dotenv()

@dataclass
class DocumentChunk:
    id: str
    content: str
    embedding: Optional[List[float]] = None
    episode_title: str = ""
    season: str = ""
    episode_number: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    word_count: int = 0
    source_file: str = ""
    processed_at: str = ""
    
    def __post_init__(self):
        if self.processed_at == "":
            self.processed_at = datetime.now().isoformat()

class MetadataFirstProcessor:
    def __init__(self):
        self.config = {
            "qdrant_host": "localhost",
            "qdrant_port": 6333,
            "collection_name": "revolutions_podcast_v1",
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "embedding_model": "text-embedding-3-small",
            "embedding_dimensions": 1536,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "gcs_bucket": "ai_knowledgebase"
        }
        
        self.setup_logging()
        self.qdrant_client = self._init_qdrant()
        openai.api_key = self.config["openai_api_key"]
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize Google Drive
        credentials = Credentials.from_service_account_file(
            "google-cloud-service-account.json",
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        self.drive_service = build('drive', 'v3', credentials=credentials)
        
        # Initialize GCS client
        self.gcs_client = storage.Client.from_service_account_json("google-cloud-service-account.json")
        self.gcs_bucket = self.gcs_client.bucket(self.config["gcs_bucket"])
        
        self.stats = {
            "total_episodes_found": 0,
            "already_processed": 0,
            "documents_processed": 0, 
            "chunks_created": 0, 
            "embeddings_generated": 0, 
            "errors": 0
        }
        self.session_start = datetime.now()
        self.progress_log = []
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def _init_qdrant(self):
        client = QdrantClient(host=self.config["qdrant_host"], port=self.config["qdrant_port"], check_compatibility=False)
        self.logger.info(f"Connected to Qdrant at {self.config['qdrant_host']}:{self.config['qdrant_port']}")
        return client
    
    def log_progress(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        self.progress_log.append({"timestamp": timestamp, "level": level, "message": message})
    
    def generate_point_id(self, chunk_id_string: str) -> str:
        namespace = uuid.UUID('12345678-1234-5678-1234-123456789abc')
        return str(uuid.uuid5(namespace, chunk_id_string))
    
    def discover_all_metadata_files(self):
        """Discover all metadata JSON files across all seasons"""
        all_metadata = []
        
        # Look for all season folders in metadata
        metadata_prefix = "podcasts/revolutions/metadata/"
        
        # Get all folders under metadata/
        folders = set()
        for blob in self.gcs_bucket.list_blobs(prefix=metadata_prefix):
            path_parts = blob.name.split('/')
            if len(path_parts) >= 5 and path_parts[4].startswith('season_'):
                folders.add(path_parts[4])
        
        season_folders = sorted(list(folders))
        self.log_progress(f"🔍 Found season folders: {season_folders}")
        
        # Process each season folder
        for season_folder in season_folders:
            season_prefix = f"{metadata_prefix}{season_folder}/"
            season_metadata = []
            
            # Get all JSON files in this season
            for blob in self.gcs_bucket.list_blobs(prefix=season_prefix):
                if blob.name.endswith('.json'):
                    try:
                        metadata_content = json.loads(blob.download_as_text())
                        metadata_info = {
                            "blob_path": blob.name,
                            "season_folder": season_folder,
                            "metadata": metadata_content,
                            "already_processed": "rag_processing" in metadata_content
                        }
                        season_metadata.append(metadata_info)
                        
                    except Exception as e:
                        self.log_progress(f"❌ Error reading {blob.name}: {e}", "ERROR")
                        continue
            
            all_metadata.extend(season_metadata)
            self.log_progress(f"📁 {season_folder}: {len(season_metadata)} episodes found")
        
        return all_metadata
    
    def download_google_doc_from_url(self, google_doc_url, episode_title):
        """Download Google Doc content using the URL from metadata"""
        try:
            # Extract document ID from URL
            # URL format: https://docs.google.com/document/d/DOCUMENT_ID/edit...
            doc_id_match = re.search(r'/document/d/([a-zA-Z0-9-_]+)', google_doc_url)
            if not doc_id_match:
                self.log_progress(f"❌ Could not extract doc ID from URL: {google_doc_url}", "ERROR")
                return None
            
            doc_id = doc_id_match.group(1)
            
            # Download as plain text
            request = self.drive_service.files().export_media(fileId=doc_id, mimeType='text/plain')
            file_io = io.BytesIO()
            downloader = MediaIoBaseDownload(file_io, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            content = file_io.getvalue().decode('utf-8')
            self.log_progress(f"✅ Downloaded content for: {episode_title}")
            return content
            
        except Exception as e:
            self.log_progress(f"❌ Failed to download {episode_title}: {e}", "ERROR")
            return None
    
    def extract_episode_metadata_from_json(self, metadata_json):
        """Extract episode info from the JSON metadata"""
        return {
            "episode_title": metadata_json.get("title", ""),
            "season": metadata_json.get("season", ""),
            "episode_number": metadata_json.get("episode_number", ""),
            "source_file": metadata_json.get("title", "")  # Use title as source file reference
        }
    
    def create_chunks(self, content: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        chunks = []
        content = re.sub(r'\s+', ' ', content.strip())
        
        tokens = self.tokenizer.encode(content)
        chunk_size = self.config["chunk_size"]
        
        # Create chunk ID using season and episode info
        season_str = str(metadata['season']).zfill(2)
        episode_str = str(metadata['episode_number']).zfill(2)
        
        if len(tokens) <= chunk_size:
            chunk = DocumentChunk(
                id=f"{season_str}_{episode_str}_chunk_001",
                content=content,
                chunk_index=0,
                total_chunks=1,
                word_count=len(content.split()),
                **metadata
            )
            chunks.append(chunk)
        else:
            overlap = self.config["chunk_overlap"]
            chunk_starts = list(range(0, len(tokens), chunk_size - overlap))
            
            for i, start in enumerate(chunk_starts):
                end = min(start + chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_content = self.tokenizer.decode(chunk_tokens)
                
                chunk = DocumentChunk(
                    id=f"{season_str}_{episode_str}_chunk_{i+1:03d}",
                    content=chunk_content,
                    chunk_index=i,
                    total_chunks=len(chunk_starts),
                    word_count=len(chunk_content.split()),
                    **metadata
                )
                chunks.append(chunk)
        
        return chunks
    
    def generate_embedding(self, text: str) -> List[float]:
        response = openai.embeddings.create(model=self.config["embedding_model"], input=text)
        self.stats["embeddings_generated"] += 1
        return response.data[0].embedding
    
    def store_chunks_in_qdrant(self, chunks: List[DocumentChunk]) -> tuple[bool, list]:
        try:
            points = []
            point_ids = []
            
            for chunk in chunks:
                if chunk.embedding is None:
                    chunk.embedding = self.generate_embedding(chunk.content)
                
                payload = asdict(chunk)
                payload.pop('embedding', None)
                
                point_id = self.generate_point_id(chunk.id)
                point_ids.append(point_id)
                
                point = PointStruct(id=point_id, vector=chunk.embedding, payload=payload)
                points.append(point)
            
            self.qdrant_client.upsert(collection_name=self.config["collection_name"], points=points)
            self.stats["chunks_created"] += len(chunks)
            
            return True, point_ids
            
        except Exception as e:
            self.logger.error(f"Failed to store chunks: {e}")
            self.stats["errors"] += 1
            return False, []
    
    def update_metadata_with_rag_status(self, blob_path, original_metadata, chunks, point_ids, processing_duration):
        """Update the original metadata file with RAG processing info"""
        try:
            # Add RAG processing section
            original_metadata["rag_processing"] = {
                "status": "completed",
                "processed_at": datetime.now().isoformat(),
                "chunks_created": len(chunks),
                "embedding_model": self.config["embedding_model"],
                "vector_database": "qdrant",
                "collection_name": self.config["collection_name"],
                "point_ids": point_ids,
                "processing_duration_seconds": round(processing_duration, 2),
                "total_word_count": sum(chunk.word_count for chunk in chunks),
                "chunk_details": [
                    {
                        "chunk_id": chunk.id,
                        "word_count": chunk.word_count,
                        "chunk_index": chunk.chunk_index,
                        "point_id": point_ids[i] if i < len(point_ids) else None
                    } for i, chunk in enumerate(chunks)
                ],
                "error_log": None
            }
            
            # Upload updated metadata
            blob = self.gcs_bucket.blob(blob_path)
            blob.upload_from_string(
                json.dumps(original_metadata, indent=2),
                content_type='application/json'
            )
            
            self.log_progress(f"✅ Updated metadata: {blob_path}")
            return True
            
        except Exception as e:
            self.log_progress(f"❌ Failed to update metadata {blob_path}: {e}", "ERROR")
            return False
    
    def process_single_episode(self, metadata_info):
        """Process a single episode from metadata"""
        blob_path = metadata_info["blob_path"]
        metadata_json = metadata_info["metadata"]
        
        # Check if already processed
        if metadata_info["already_processed"]:
            self.log_progress(f"⏭️  Already processed: {metadata_json.get('title', 'Unknown')}")
            self.stats["already_processed"] += 1
            return True
        
        episode_title = metadata_json.get("title", "Unknown")
        google_doc_url = metadata_json.get("google_doc_url", "")
        
        if not google_doc_url:
            self.log_progress(f"❌ No google_doc_url for: {episode_title}", "ERROR")
            self.stats["errors"] += 1
            return False
        
        self.log_progress(f"📄 Processing: {episode_title}")
        processing_start = time.time()
        
        # Download content from Google Doc
        content = self.download_google_doc_from_url(google_doc_url, episode_title)
        if not content:
            self.stats["errors"] += 1
            return False
        
        # Extract metadata for processing
        episode_metadata = self.extract_episode_metadata_from_json(metadata_json)
        
        # Create chunks
        chunks = self.create_chunks(content, episode_metadata)
        
        # Store in Qdrant
        success, point_ids = self.store_chunks_in_qdrant(chunks)
        
        if success:
            processing_duration = time.time() - processing_start
            
            # Update metadata with RAG status
            self.update_metadata_with_rag_status(
                blob_path, metadata_json, chunks, point_ids, processing_duration
            )
            
            self.stats["documents_processed"] += 1
            self.log_progress(f"✅ Complete: {episode_title}")
            return True
        else:
            self.log_progress(f"❌ Failed: {episode_title}", "ERROR")
            return False
    
    def initialize_collection(self):
        collection_name = self.config["collection_name"]
        collections = self.qdrant_client.get_collections()
        existing = [col.name for col in collections.collections]
        
        if collection_name in existing:
            response = input(f"Delete and recreate collection '{collection_name}'? (y/N): ")
            if response.lower() == 'y':
                self.qdrant_client.delete_collection(collection_name)
                self.log_progress(f"Deleted existing collection '{collection_name}'")
            else:
                self.log_progress("Using existing collection")
                return
        
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=self.config["embedding_dimensions"], distance=Distance.COSINE)
        )
        self.log_progress(f"Created collection '{collection_name}'")
    
    def process_all_episodes(self):
        """Main processing function - discovers and processes all unprocessed episodes"""
        self.log_progress("🚀 Starting Metadata-First RAG Processing")
        
        # Initialize collection
        self.initialize_collection()
        
        # Discover all metadata files
        self.log_progress("🔍 Discovering all metadata files...")
        all_metadata = self.discover_all_metadata_files()
        
        self.stats["total_episodes_found"] = len(all_metadata)
        unprocessed = [m for m in all_metadata if not m["already_processed"]]
        
        self.log_progress(f"📊 Discovery Summary:")
        self.log_progress(f"  📁 Total episodes found: {len(all_metadata)}")
        self.log_progress(f"  ✅ Already processed: {len(all_metadata) - len(unprocessed)}")
        self.log_progress(f"  📄 Need processing: {len(unprocessed)}")
        
        if not unprocessed:
            self.log_progress("🎉 All episodes already processed!")
            return
        
        # Process unprocessed episodes
        self.log_progress(f"\n🎬 Processing {len(unprocessed)} episodes...")
        
        for i, metadata_info in enumerate(unprocessed, 1):
            self.log_progress(f"\n📄 [{i}/{len(unprocessed)}] Processing episode...")
            
            success = self.process_single_episode(metadata_info)
            
            if not success:
                self.log_progress(f"❌ Episode failed, continuing...")
            
            # Brief pause between episodes
            time.sleep(1)
        
        # Final report
        total_duration = datetime.now() - self.session_start
        self.log_progress(f"\n🏁 Metadata-First Processing Complete!")
        self.log_progress(f"⏱️  Total Duration: {total_duration}")
        self.log_progress(f"📊 Final Stats:")
        self.log_progress(f"  📁 Total episodes found: {self.stats['total_episodes_found']}")
        self.log_progress(f"  ⏭️  Already processed: {self.stats['already_processed']}")
        self.log_progress(f"  ✅ Successfully processed: {self.stats['documents_processed']}")
        self.log_progress(f"  🧩 Total chunks created: {self.stats['chunks_created']}")
        self.log_progress(f"  🔢 Total embeddings: {self.stats['embeddings_generated']}")
        self.log_progress(f"  ❌ Errors: {self.stats['errors']}")

if __name__ == "__main__":
    processor = MetadataFirstProcessor()
    
    print("🚀 Metadata-First RAG Processor")
    print("=" * 50)
    print("This will:")
    print("- Scan all GCS metadata files")
    print("- Process only unprocessed episodes")
    print("- Use exact filename matching")
    print("- Perfect resumability")
    print("=" * 50)
    
    confirm = input("Start metadata-first processing? (y/n): ").lower()
    if confirm == 'y':
        processor.process_all_episodes()
    else:
        print("👋 Processing cancelled.")
