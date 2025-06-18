#!/usr/bin/env python3
"""
Enhanced Document Processing Pipeline for Revolutions Podcast RAG System
Processes Google Drive transcripts into searchable vector embeddings
"""

import uuid
import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime

# Core dependencies
from dotenv import load_dotenv
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# Google Cloud dependencies
from google.cloud import storage
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials

# Text processing
import tiktoken

# Load environment
load_dotenv()

@dataclass
class DocumentChunk:
    """Represents a processed document chunk with metadata"""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    
    # Episode metadata
    episode_title: str = ""
    season: str = ""
    episode_number: str = ""
    
    # Chunk metadata
    chunk_index: int = 0
    total_chunks: int = 0
    start_char: int = 0
    end_char: int = 0
    
    # Content metadata
    topics: List[str] = None
    characters: List[str] = None
    time_period: str = ""
    revolution_type: str = ""
    
    # Processing metadata
    source_file: str = ""
    processed_at: str = ""
    word_count: int = 0
    
    def __post_init__(self):
        if self.topics is None:
            self.topics = []
        if self.characters is None:
            self.characters = []
        if self.processed_at == "":
            self.processed_at = datetime.now().isoformat()

class DocumentProcessor:
    """
    Enhanced document processing pipeline for Revolutions Podcast transcripts
    """
    
    def __init__(self):
        self.config = self._load_config()
        self.setup_logging()
        
        # Initialize clients
        self.qdrant_client = self._init_qdrant()
        self.gcs_client = self._init_gcs()
        self.drive_service = self._init_drive()
        
        # Initialize OpenAI
        openai.api_key = self.config["openai_api_key"]
        
        # Text processing
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Processing stats
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "errors": 0
        }

    def generate_point_id(self, chunk_id_string: str) -> str:
        """Generate a valid Qdrant point ID from string ID"""
        namespace = uuid.UUID('12345678-1234-5678-1234-123456789abc')
        return str(uuid.uuid5(namespace, chunk_id_string))
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        return {
            # Qdrant
            "qdrant_host": os.getenv("QDRANT_HOST", "100.72.255.25"),
            "qdrant_port": int(os.getenv("QDRANT_PORT", "6333")),
            "collection_name": os.getenv("QDRANT_COLLECTION_NAME", "revolutions_podcast_v1"),
            
            # OpenAI
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            "embedding_dimensions": int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1536")),
            
            # Google Cloud
            "gcs_bucket": os.getenv("GCS_BUCKET_NAME", "ai_knowledgebase"),
            "google_credentials": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            
            # Processing
            "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
            "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
            "max_chunks_per_episode": int(os.getenv("MAX_CHUNKS_PER_EPISODE", "100")),
            "batch_size": int(os.getenv("BATCH_SIZE", "10")),
            
            # Development
            "development_mode": os.getenv("DEVELOPMENT_MODE", "true").lower() == "true"
        }
    
    def setup_logging(self):
        """Configure logging"""
        log_level = logging.DEBUG if self.config["development_mode"] else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('document_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Document processor initialized")
    
    def _init_qdrant(self) -> QdrantClient:
        """Initialize Qdrant client"""
        try:
            client = QdrantClient(
                host=self.config["qdrant_host"],
                port=self.config["qdrant_port"],
                check_compatibility=False
            )
            self.logger.info(f"Connected to Qdrant at {self.config['qdrant_host']}:{self.config['qdrant_port']}")
            return client
        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def _init_gcs(self) -> storage.Client:
        """Initialize Google Cloud Storage client"""
        try:
            credentials = Credentials.from_service_account_file(self.config["google_credentials"])
            client = storage.Client(credentials=credentials)
            self.logger.info(f"Connected to GCS bucket: {self.config['gcs_bucket']}")
            return client
        except Exception as e:
            self.logger.error(f"Failed to connect to GCS: {e}")
            raise
    
    def _init_drive(self):
        """Initialize Google Drive API client"""
        try:
            credentials = Credentials.from_service_account_file(
                self.config["google_credentials"],
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
            service = build('drive', 'v3', credentials=credentials)
            self.logger.info("Connected to Google Drive API")
            return service
        except Exception as e:
            self.logger.error(f"Failed to connect to Google Drive: {e}")
            raise
    
    def initialize_collection(self):
        """Initialize or recreate the Qdrant collection"""
        collection_name = self.config["collection_name"]
        
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            existing_collections = [col.name for col in collections.collections]
            
            if collection_name in existing_collections:
                self.logger.info(f"Collection '{collection_name}' already exists")
                
                if self.config["development_mode"]:
                    response = input(f"Delete and recreate collection '{collection_name}'? (y/N): ")
                    if response.lower() == 'y':
                        self.qdrant_client.delete_collection(collection_name)
                        self.logger.info(f"Deleted existing collection '{collection_name}'")
                    else:
                        self.logger.info("Using existing collection")
                        return
            
            # Create new collection
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.config["embedding_dimensions"],
                    distance=Distance.COSINE
                )
            )
            self.logger.info(f"Created collection '{collection_name}'")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize collection: {e}")
            raise
    
    def extract_episode_metadata(self, filename: str, content: str) -> Dict[str, Any]:
        """Extract metadata from filename and content"""
        metadata = {
            "episode_title": "",
            "season": "",
            "episode_number": "",
            "topics": [],
            "characters": [],
            "time_period": "",
            "revolution_type": "",
            "source_file": filename
        }
        
        # Extract from filename patterns
        filename_patterns = [
            r"(\d+)\.(\d+)\s*-\s*(.+?)(?:\s*-\s*(.+?))?\.(?:txt|docx?)$",
            r"Season\s*(\d+),?\s*Episode\s*(\d+)\s*-\s*(.+?)(?:\s*-\s*(.+?))?\.(?:txt|docx?)$"
        ]
        
        for pattern in filename_patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                metadata["season"] = match.group(1).zfill(2)
                metadata["episode_number"] = match.group(2).zfill(2)
                metadata["episode_title"] = match.group(3).strip() if match.group(3) else ""
                if match.group(4):
                    metadata["episode_title"] += f" - {match.group(4).strip()}"
                break
        
        # Extract revolution type from content/title
        revolution_keywords = {
            "french": ["french revolution", "france", "robespierre", "louis xvi", "marie antoinette"],
            "russian": ["russian revolution", "russia", "lenin", "tsar", "bolshevik"],
            "american": ["american revolution", "washington", "adams", "jefferson"],
            "haitian": ["haitian revolution", "haiti", "toussaint", "l'ouverture"],
            "1848": ["1848", "spring of nations", "paris commune"],
            "english": ["english civil war", "cromwell", "charles i"]
        }
        
        content_lower = content.lower()
        title_lower = metadata["episode_title"].lower()
        
        for revolution, keywords in revolution_keywords.items():
            if any(keyword in content_lower or keyword in title_lower for keyword in keywords):
                metadata["revolution_type"] = revolution
                break
        
        # Extract time periods
        year_matches = re.findall(r'\b(1[6-9]\d{2}|20\d{2})\b', content)
        if year_matches:
            years = [int(year) for year in year_matches]
            metadata["time_period"] = f"{min(years)}-{max(years)}"
        
        return metadata
    
    def create_chunks(self, content: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Split document into overlapping chunks with metadata"""
        chunks = []
        
        # Clean and normalize content
        content = re.sub(r'\n+', '\n', content.strip())
        content = re.sub(r'\s+', ' ', content)
        
        # Calculate chunk parameters
        chunk_size = self.config["chunk_size"]
        overlap = self.config["chunk_overlap"]
        
        # Tokenize for accurate chunking
        tokens = self.tokenizer.encode(content)
        
        if len(tokens) <= chunk_size:
            # Single chunk
            chunk = DocumentChunk(
                id=f"{metadata['season']}_{metadata['episode_number']}_chunk_001",
                content=content,
                chunk_index=0,
                total_chunks=1,
                start_char=0,
                end_char=len(content),
                word_count=len(content.split()),
                **metadata
            )
            chunks.append(chunk)
            return chunks
        
        # Multiple chunks
        chunk_starts = list(range(0, len(tokens), chunk_size - overlap))
        total_chunks = len(chunk_starts)
        
        for i, start in enumerate(chunk_starts):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_content = self.tokenizer.decode(chunk_tokens)
            
            # Calculate character positions (approximate)
            char_start = int((start / len(tokens)) * len(content))
            char_end = int((end / len(tokens)) * len(content))
            
            chunk_id = f"{metadata['season']}_{metadata['episode_number']}_chunk_{i+1:03d}"
            
            chunk = DocumentChunk(
                id=chunk_id,
                content=chunk_content,
                chunk_index=i,
                total_chunks=total_chunks,
                start_char=char_start,
                end_char=char_end,
                word_count=len(chunk_content.split()),
                **metadata
            )
            chunks.append(chunk)
            
            # Limit chunks per episode
            if len(chunks) >= self.config["max_chunks_per_episode"]:
                break
        
        self.logger.info(f"Created {len(chunks)} chunks for {metadata['episode_title']}")
        return chunks
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI"""
        try:
            response = openai.embeddings.create(
                model=self.config["embedding_model"],
                input=text,
                encoding_format="float"
            )
            self.stats["embeddings_generated"] += 1
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            self.stats["errors"] += 1
            raise
    
    def store_chunks_in_qdrant(self, chunks: List[DocumentChunk]) -> bool:
        """Store processed chunks in Qdrant"""
        try:
            points = []
            
            for chunk in chunks:
                # Generate embedding if not already done
                if chunk.embedding is None:
                    chunk.embedding = self.generate_embedding(chunk.content)
                
                # Create a clean payload
                payload = asdict(chunk)
                payload.pop('embedding', None)
                
                # Generate valid point ID
                point_id = self.generate_point_id(chunk.id)
                
                # Convert to Qdrant point
                point = PointStruct(
                    id=point_id,  # Use UUID instead of string
                    vector=chunk.embedding,
                    payload=payload
                )
                points.append(point)
            
            # Batch upload to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.config["collection_name"],
                points=points
            )
            
            self.stats["chunks_created"] += len(chunks)
            self.logger.info(f"Stored {len(chunks)} chunks in Qdrant")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store chunks in Qdrant: {e}")
            self.stats["errors"] += 1
            return False
    
    def process_single_document(self, file_path: str, content: str) -> bool:
        """Process a single document through the entire pipeline"""
        try:
            self.logger.info(f"Processing document: {file_path}")
            
            # Extract metadata
            metadata = self.extract_episode_metadata(file_path, content)
            
            # Create chunks
            chunks = self.create_chunks(content, metadata)
            
            # Store in Qdrant
            success = self.store_chunks_in_qdrant(chunks)
            
            if success:
                self.stats["documents_processed"] += 1
                self.logger.info(f"Successfully processed: {metadata['episode_title']}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to process document {file_path}: {e}")
            self.stats["errors"] += 1
            return False
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return {
            **self.stats,
            "collection_info": self._get_collection_info()
        }
    
    def _get_collection_info(self) -> Dict[str, Any]:
        """Get information about the Qdrant collection"""
        try:
            collection_info = self.qdrant_client.get_collection(self.config["collection_name"])
            return {
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}

# Example usage and testing
if __name__ == "__main__":
    # Initialize processor
    processor = DocumentProcessor()
    
    # Initialize collection
    processor.initialize_collection()
    
    # Test with a sample document
    sample_content = """
    Episode Title: The French Revolution - The Tennis Court Oath
    
    In the spring of 1789, the financial crisis in France reached a breaking point. 
    King Louis XVI was forced to convene the Estates-General for the first time since 1614. 
    The Third Estate, representing the common people, found themselves locked out of their 
    usual meeting hall on June 20, 1789.
    
    Led by figures like Abbé Emmanuel Joseph Sieyès and Jean-Paul Marat, the representatives 
    of the Third Estate gathered in a nearby tennis court. There, they swore the famous 
    Tennis Court Oath, vowing not to separate until they had given France a new constitution.
    
    This moment marked a crucial turning point in the French Revolution, as it represented 
    the first time the common people formally challenged the absolute authority of the monarchy.
    """
    
    sample_filename = "03.01 - The French Revolution - Tennis Court Oath.txt"
    
    print("🧪 Testing document processing pipeline...")
    success = processor.process_single_document(sample_filename, sample_content)
    
    if success:
        print("✅ Sample document processed successfully!")
        stats = processor.get_processing_stats()
        print(f"📊 Processing stats: {stats}")
    else:
        print("❌ Sample document processing failed")
