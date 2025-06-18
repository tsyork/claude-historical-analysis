#!/usr/bin/env python3
"""
Standalone Document Processing Pipeline for Laptop 2
All dependencies included, no external imports needed
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

class Laptop2Processor:
    def __init__(self):
        self.config = {
            "qdrant_host": "localhost",  # Local Qdrant on Laptop 2
            "qdrant_port": 6333,
            "collection_name": "revolutions_podcast_v1",
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "embedding_model": "text-embedding-3-small",
            "embedding_dimensions": 1536,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "development_mode": True
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
    
    def initialize_collection(self):
        collection_name = self.config["collection_name"]
        collections = self.qdrant_client.get_collections()
        existing = [col.name for col in collections.collections]
        
        if collection_name in existing:
            if self.config["development_mode"]:
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
            # Multiple chunks
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
    
    def store_chunks_in_qdrant(self, chunks: List[DocumentChunk]) -> bool:
        try:
            points = []
            for chunk in chunks:
                if chunk.embedding is None:
                    chunk.embedding = self.generate_embedding(chunk.content)
                
                payload = asdict(chunk)
                payload.pop('embedding', None)
                
                point_id = self.generate_point_id(chunk.id)
                point = PointStruct(id=point_id, vector=chunk.embedding, payload=payload)
                points.append(point)
            
            self.qdrant_client.upsert(collection_name=self.config["collection_name"], points=points)
            self.stats["chunks_created"] += len(chunks)
            return True
        except Exception as e:
            self.logger.error(f"Failed to store chunks: {e}")
            self.stats["errors"] += 1
            return False
    
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
        
        # Find season folder
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
            
            content = self.download_google_doc_content(doc['id'], doc['name'])
            if content:
                metadata = self.extract_episode_metadata(doc['name'], content)
                chunks = self.create_chunks(content, metadata)
                success = self.store_chunks_in_qdrant(chunks)
                
                if success:
                    processed_count += 1
                    self.stats["documents_processed"] += 1
                    self.log_progress(f"✅ Processed successfully")
                else:
                    self.log_progress(f"❌ Processing failed", "ERROR")
            else:
                self.log_progress(f"❌ Download failed", "ERROR")
            
            time.sleep(1)  # Be nice to APIs
        
        self.log_progress(f"📊 {season_name} Complete: {processed_count} processed")
        return processed_count > 0
    
    def process_all_seasons(self):
        self.log_progress("🚀 Starting FULL processing of 336 Revolutions Podcast episodes")
        
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

if __name__ == "__main__":
    processor = Laptop2Processor()
    
    print("🚀 Laptop 2 Standalone Processor Ready!")
    print("This will process ALL 336 episodes.")
    print("Estimated time: 4-6 hours")
    
    confirm = input("Start full processing? (y/n): ").lower()
    if confirm == 'y':
        processor.process_all_seasons()
    else:
        print("👋 Processing cancelled.")
