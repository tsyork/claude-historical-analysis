#!/usr/bin/env python3
"""
Batch processor for all 336 Revolutions Podcast transcripts
Processes Google Docs from Google Drive into Qdrant vector database
"""

from src.document_processor import DocumentProcessor
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
import os
import time
from dotenv import load_dotenv
import io
from googleapiclient.http import MediaIoBaseDownload

load_dotenv()

class RevolutionsBatchProcessor:
    def __init__(self):
        self.processor = DocumentProcessor()
        
        # Initialize Google Drive API
        credentials = Credentials.from_service_account_file(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        self.drive_service = build('drive', 'v3', credentials=credentials)
        
        # Season folder IDs (we'll discover these)
        self.season_folders = {}
        
        # Processing stats
        self.total_processed = 0
        self.total_errors = 0
        self.start_time = None
    
    def discover_season_folders(self):
        """Discover all season folder IDs"""
        print("🔍 Discovering season folders...")
        
        # Start from Revolutions Podcast folder
        revolutions_folder_id = "1ZVgEcqGIq4X7YvEOZWDT8QlBKZoA0xB1"  # We'll need to find this
        
        # For now, let's manually map what we know
        # We can get these IDs from the previous exploration
        self.season_folders = {
            "Season 1": {"name": "Season 1 – English Revolution", "episodes": 24},
            "Season 2": {"name": "Season 2 – American Revolution", "episodes": 18},
            "Season 3": {"name": "Season 3 – French Revolution", "episodes": 63},
            "Season 4": {"name": "Season 4 – Haitian Revolution", "episodes": 20},
            "Season 5": {"name": "Season 5 – Spanish American Wars of Independence", "episodes": 28},
            "Season 6": {"name": "Season 6 – July Revolution", "episodes": 12},
            "Season 7": {"name": "Season 7 – 1848 Revolutions", "episodes": 33},
            "Season 8": {"name": "Season 8 – Paris Commune", "episodes": 8},
            "Season 9": {"name": "Season 9 – Mexican Revolution", "episodes": 27},
            "Season 10": {"name": "Season 10 – Russian Revolution", "episodes": 103}
        }
        
        print(f"✅ Found {len(self.season_folders)} seasons with 336 total episodes")
    
    def download_google_doc_content(self, file_id, file_name):
        """Download content from a Google Doc"""
        try:
            # Export as plain text
            request = self.drive_service.files().export_media(
                fileId=file_id,
                mimeType='text/plain'
            )
            
            file_io = io.BytesIO()
            downloader = MediaIoBaseDownload(file_io, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            content = file_io.getvalue().decode('utf-8')
            return content
            
        except Exception as e:
            print(f"❌ Failed to download {file_name}: {e}")
            return None
    
    def process_season_documents(self, season_name, max_episodes=None):
        """Process all documents in a specific season"""
        print(f"\n🎬 Processing {season_name}")
        print("=" * 50)
        
        # Find season folder by name
        ai_transcripts_id = "136Nmn3gJe0DPVh8p4vUl3oD4-qDNRySh"
        
        # Get Revolutions Podcast folder
        revolutions_query = f"'{ai_transcripts_id}' in parents and name='Revolutions Podcast'"
        revolutions_results = self.drive_service.files().list(
            q=revolutions_query,
            fields="files(id, name)"
        ).execute()
        
        if not revolutions_results.get('files'):
            print("❌ Could not find Revolutions Podcast folder")
            return False
        
        revolutions_folder_id = revolutions_results['files'][0]['id']
        
        # Find specific season folder
        season_query = f"'{revolutions_folder_id}' in parents and name contains '{season_name}'"
        season_results = self.drive_service.files().list(
            q=season_query,
            fields="files(id, name)"
        ).execute()
        
        if not season_results.get('files'):
            print(f"❌ Could not find {season_name} folder")
            return False
        
        season_folder_id = season_results['files'][0]['id']
        print(f"✅ Found season folder: {season_folder_id}")
        
        # Get all documents in season folder
        docs_query = f"'{season_folder_id}' in parents and mimeType='application/vnd.google-apps.document'"
        
        all_docs = []
        page_token = None
        
        while True:
            docs_results = self.drive_service.files().list(
                q=docs_query,
                pageSize=100,
                pageToken=page_token,
                fields="nextPageToken, files(id, name, size)",
                orderBy="name"
            ).execute()
            
            docs = docs_results.get('files', [])
            all_docs.extend(docs)
            
            page_token = docs_results.get('nextPageToken')
            if not page_token:
                break
        
        print(f"📄 Found {len(all_docs)} documents to process")
        
        # Process documents
        processed_count = 0
        error_count = 0
        
        # Limit episodes if specified (for testing)
        docs_to_process = all_docs[:max_episodes] if max_episodes else all_docs
        
        for i, doc in enumerate(docs_to_process):
            print(f"\n📄 Processing {i+1}/{len(docs_to_process)}: {doc['name']}")
            
            # Download content
            content = self.download_google_doc_content(doc['id'], doc['name'])
            
            if content:
                # Process through pipeline
                success = self.processor.process_single_document(doc['name'], content)
                
                if success:
                    processed_count += 1
                    self.total_processed += 1
                    print(f"✅ Processed successfully")
                else:
                    error_count += 1
                    self.total_errors += 1
                    print(f"❌ Processing failed")
            else:
                error_count += 1
                self.total_errors += 1
                print(f"❌ Download failed")
            
            # Small delay to be nice to APIs
            time.sleep(1)
        
        print(f"\n📊 {season_name} Complete:")
        print(f"  ✅ Processed: {processed_count}")
        print(f"  ❌ Errors: {error_count}")
        
        return processed_count > 0
    
    def process_all_seasons(self, test_mode=True):
        """Process all seasons (or test with small batches)"""
        print("🚀 Starting Revolutions Podcast Batch Processing")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Initialize collection
        self.processor.initialize_collection()
        
        if test_mode:
            print("🧪 TEST MODE: Processing 3 episodes from Paris Commune (smallest season)")
            self.process_season_documents("Season 8", max_episodes=3)
        else:
            # Process all seasons
            seasons_to_process = [
                "Season 8",  # Start small (8 episodes)
                "Season 6",  # Medium (12 episodes)
                "Season 2",  # Medium (18 episodes)
                "Season 4",  # Medium (20 episodes)
                "Season 1",  # Medium (24 episodes)
                "Season 9",  # Large (27 episodes)
                "Season 5",  # Large (28 episodes)
                "Season 7",  # Large (33 episodes)
                "Season 3",  # Very large (63 episodes)
                "Season 10"  # Massive (103 episodes)
            ]
            
            for season in seasons_to_process:
                success = self.process_season_documents(season)
                if not success:
                    print(f"❌ Failed to process {season}, stopping")
                    break
                
                # Brief pause between seasons
                time.sleep(5)
        
        # Final stats
        elapsed = time.time() - self.start_time
        print(f"\n🏁 Batch Processing Complete!")
        print(f"⏱️  Total time: {elapsed/60:.1f} minutes")
        print(f"✅ Documents processed: {self.total_processed}")
        print(f"❌ Errors: {self.total_errors}")
        print(f"📊 Success rate: {(self.total_processed/(self.total_processed+self.total_errors)*100):.1f}%")
        
        # Show final collection stats
        stats = self.processor.get_processing_stats()
        print(f"🗄️  Vector database stats: {stats}")

if __name__ == "__main__":
    processor = RevolutionsBatchProcessor()
    
    # Start with test mode (just 3 episodes)
    processor.process_all_seasons(test_mode=True)
    
    print("\n🎯 Test complete! Ready for full processing when you're ready.")
    print("To process all 336 episodes, run:")
    print("  processor.process_all_seasons(test_mode=False)")
