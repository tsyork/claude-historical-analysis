#!/usr/bin/env python3
"""
Enhanced RAG Processor with GCS metadata updates
Updates your existing season-based metadata structure
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

class EnhancedRAGProcessor:
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
        
        self.stats = {"documents_processed": 0, "chunks_created": 0, "embeddings_generated": 0, "errors": 0}
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
    
    def update_gcs_metadata(self, episode_metadata, chunks, point_ids, processing_duration):
        """Update existing GCS metadata with RAG processing information"""
        try:
            season = episode_metadata['season'].zfill(2)
            episode = episode_metadata['episode_number'].zfill(2)
            
            # Use your existing metadata path structure
            metadata_path = f"podcasts/revolutions/metadata/season_{season}/{episode_metadata['source_file']}.json"
            
            # Try to find existing metadata file using episode info
            # Your files seem to be named by the actual episode title
            episode_title_clean = re.sub(r'[^\w\s-]', '', episode_metadata['episode_title'])
            episode_title_clean = re.sub(r'\s+', ' ', episode_title_clean).strip()
            
            # Try multiple possible metadata file paths based on your structure
            possible_paths = [
                f"podcasts/revolutions/metadata/season_{season}/{season}.{episode}- {episode_title_clean}.json",
                f"podcasts/revolutions/metadata/season_{season}/{season}.{episode} {episode_title_clean}.json",
                f"podcasts/revolutions/metadata/season_{season}/{episode_metadata['episode_title']}.json"
            ]
            
            # Alternative: search for files in the season metadata folder
            metadata_folder_path = f"podcasts/revolutions/metadata/season_{season}/"
            
            existing_metadata = None
            metadata_blob_path = None
            
            try:
                # List all files in the season metadata folder
                blobs = list(self.gcs_bucket.list_blobs(prefix=metadata_folder_path))
                
                # Find the file that matches this episode
                for blob in blobs:
                    if blob.name.endswith('.json'):
                        # Download and check if it matches our episode
                        try:
                            blob_content = json.loads(blob.download_as_text())
                            if (blob_content.get('season') == season and 
                                blob_content.get('episode_number') == episode):
                                existing_metadata = blob_content
                                metadata_blob_path = blob.name
                                break
                        except:
                            continue
                            
            except Exception as e:
                self.log_progress(f"Could not search metadata folder: {e}", "WARNING")
            
            if existing_metadata is None:
                self.log_progress(f"Could not find existing metadata for S{season}E{episode}", "WARNING")
                return False
            
            # Add RAG processing section to existing metadata
            existing_metadata["rag_processing"] = {
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
            
            # Upload updated metadata back to GCS
            blob = self.gcs_bucket.blob(metadata_blob_path)
            blob.upload_from_string(
                json.dumps(existing_metadata, indent=2),
                content_type='application/json'
            )
            
            self.log_progress(f"✅ Updated GCS metadata: {metadata_blob_path}")
            return True
            
        except Exception as e:
            self.log_progress(f"⚠️  Failed to update GCS metadata: {e}", "WARNING")
            # Don't fail the whole process if metadata update fails
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
    
    def extract_episode_metadata(self, filename: str, content: str):
        """Extract metadata from Drive filename: 'Revolutions – S9E01 – 9.01- New Spain'"""
        metadata = {"episode_title": "", "season": "", "episode_number": "", "source_file": filename}
        
        # Extract from filename: "Revolutions – S9E01 – 9.01- New Spain"
        pattern = r"S(\d+)E(\d+)\s*–\s*[\d.]+\s*-\s*(.+)"
        match = re.search(pattern, filename)
        if match:
            metadata["season"] = match.group(1).zfill(2)
            metadata["episode_number"] = match.group(2).zfill(2)
            metadata["episode_title"] = match.group(3).strip()
        
        return metadata
    
    def create_chunks(self, content: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        chunks = []
        content = re.sub(r'\s+', ' ', content.strip())
        
        tokens = self.tokenizer.encode(content)
        chunk_size = self.config["chunk_size"]
        
        if len(tokens) <= chunk_size:
            chunk = DocumentChunk(
                id=f"{metadata['season']}_{metadata['episode_number']}_chunk_001",
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
                    id=f"{metadata['season']}_{metadata['episode_number']}_chunk_{i+1:03d}",
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
    
    def download_google_doc_content(self, file_id, file_name):
        try:
            request = self.drive_service.files().export_media(fileId=file_id, mimeType='text/plain')
            file_io = io.BytesIO()
            downloader = MediaIoBaseDownload(file_io, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            content = file_io.getvalue().decode('utf-8')
            return content
        except Exception as e:
            self.logger.error(f"Failed to download {file_name}: {e}")
            return None
    
    def process_season_documents(self, season_name, max_episodes=None):
        self.log_progress(f"🎬 Processing {season_name}")
        
        # Find season folder (same logic as before)
        ai_transcripts_id = "136Nmn3gJe0DPVh8p4vUl3oD4-qDNRySh"
        
        # Get Revolutions Podcast folder
        revolutions_query = f"'{ai_transcripts_id}' in parents and name='Revolutions Podcast'"
        revolutions_results = self.drive_service.files().list(q=revolutions_query, fields="files(id, name)").execute()
        
        if not revolutions_results.get('files'):
            self.log_progress("❌ Could not find Revolutions Podcast folder", "ERROR")
            return False
        
        revolutions_folder_id = revolutions_results['files'][0]['id']
        
        # Find specific season folder
        season_query = f"'{revolutions_folder_id}' in parents and name contains '{season_name}'"
        season_results = self.drive_service.files().list(q=season_query, fields="files(id, name)").execute()
        
        if not season_results.get('files'):
            self.log_progress(f"❌ Could not find {season_name} folder", "ERROR")
            return False
        
        season_folder_id = season_results['files'][0]['id']
        
        # Get all documents in season
        docs_query = f"'{season_folder_id}' in parents and mimeType='application/vnd.google-apps.document'"
        all_docs = []
        page_token = None
        
        while True:
            docs_results = self.drive_service.files().list(
                q=docs_query, pageSize=100, pageToken=page_token,
                fields="nextPageToken, files(id, name, size)", orderBy="name"
            ).execute()
            
            docs = docs_results.get('files', [])
            all_docs.extend(docs)
            
            page_token = docs_results.get('nextPageToken')
            if not page_token:
                break
        
        self.log_progress(f"📄 Found {len(all_docs)} documents to process")
        
        # Process documents
        docs_to_process = all_docs[:max_episodes] if max_episodes else all_docs
        processed_count = 0
        
        for i, doc in enumerate(docs_to_process):
            self.log_progress(f"📄 Processing {i+1}/{len(docs_to_process)}: {doc['name']}")
            processing_start = time.time()
            
            content = self.download_google_doc_content(doc['id'], doc['name'])
            if content:
                metadata = self.extract_episode_metadata(doc['name'], content)
                chunks = self.create_chunks(content, metadata)
                success, point_ids = self.store_chunks_in_qdrant(chunks)
                
                if success:
                    processing_duration = time.time() - processing_start
                    
                    # Update GCS metadata
                    self.update_gcs_metadata(metadata, chunks, point_ids, processing_duration)
                    
                    processed_count += 1
                    self.stats["documents_processed"] += 1
                    self.log_progress(f"✅ Complete: RAG + GCS metadata updated")
                else:
                    self.log_progress(f"❌ Processing failed", "ERROR")
            else:
                self.log_progress(f"❌ Download failed", "ERROR")
            
            time.sleep(1)  # Be nice to APIs
        
        self.log_progress(f"📊 {season_name} Complete: {processed_count} processed")
        return processed_count > 0
    
    def process_all_seasons(self):
        self.log_progress("🚀 Starting FULL processing with GCS metadata updates")
        
        self.initialize_collection()
        
        seasons = [
            "Season 8", "Season 6", "Season 2", "Season 4", "Season 1",
            "Season 9", "Season 5", "Season 7", "Season 3", "Season 10"
        ]
        
        for season in seasons:
            success = self.process_season_documents(season)
            if not success:
                self.log_progress(f"❌ Failed to process {season}, stopping", "ERROR")
                break
            time.sleep(5)
        
        # Final report
        total_duration = datetime.now() - self.session_start
        self.log_progress(f"🏁 Processing Complete! Duration: {total_duration}")
        self.log_progress(f"📊 Final Stats: {self.stats}")
        self.log_progress("🗄️  GCS metadata updated with RAG processing status")

if __name__ == "__main__":
    processor = EnhancedRAGProcessor()
    
    print("🚀 Enhanced RAG Processor with GCS Metadata Updates")
    print("This will process ALL 336 episodes AND update your GCS metadata.")
    print("Estimated time: 4-6 hours")
    
    confirm = input("Start enhanced processing? (y/n): ").lower()
    if confirm == 'y':
        processor.process_all_seasons()
    else:
        print("👋 Processing cancelled.")
