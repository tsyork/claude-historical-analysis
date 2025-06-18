import os
import json
import time
import re
from pathlib import Path
from tqdm import tqdm
import whisper
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.cloud import storage
import certifi
import subprocess

os.environ['SSL_CERT_FILE'] = certifi.where()

# ===== CONFIGURATION =====
CREDENTIALS_FILE = "credentials.json"
ROOT_FOLDER_ID = "136Nmn3gJe0DPVh8p4vUl3oD4-qDNRySh"
TEMP_AUDIO_DIR = Path("./temp_audio")

# Google Cloud Storage Configuration
GCS_PROJECT_ID = "podcast-transcription-462218"
GCS_BUCKET_NAME = "ai_knowledgebase"
GCS_METADATA_PREFIX = "podcasts/revolutions/metadata/"

SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/cloud-platform"
]

# Global variables that will be set by user input
SELECTED_SEASON = None
WHISPER_MODEL = None
SPECIFIC_FILENAME = None
FORCE_RETRANSCRIBE = None
SKIP_EXISTING = None
AUTO_SHUTDOWN = None

# ===== INTERACTIVE CONFIGURATION =====
def get_user_settings():
    """Interactive configuration prompts"""
    print("🎙️ Interactive Podcast Transcription Setup")
    print("=" * 50)
    
    # Season selection
    print("\n📂 Season Selection:")
    print("Available seasons:")
    print("  1 - Season 1 (English Revolution)")
    print("  2 - Season 2 (American Revolution)") 
    print("  3 - Season 3 (French Revolution)")
    print("  4 - Season 4 (Haitian Revolution)")
    print("  5 - Season 5 (Spanish American Wars of Independence)")
    print("  6 - Season 6 (July Revolution)")
    print("  7 - Season 7 (1848 Revolutions)")
    print("  8 - Season 8 (Second French Empire)")
    print("  9 - Season 9 (Mexican Revolution)")
    print("  10 - Season 10 (Russian Revolution)")
    print("  all - All seasons")
    
    while True:
        season_choice = input("\nEnter season number (1-10) or 'all': ").strip().lower()
        if season_choice == "all":
            selected_season = None
            break
        elif season_choice in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
            selected_season = f"season_{int(season_choice):02d}"
            break
        else:
            print("❌ Invalid choice. Please enter 1-6 or 'all'")
    
    # Whisper model selection
    print("\n🤖 Whisper Model Selection:")
    print("Available models:")
    print("  1 - large-v3 (Best quality, slower ~15-30 min per hour)")
    print("  2 - medium (Good quality, faster ~10-20 min per hour)")
    print("  3 - small (Fast, decent quality ~5-15 min per hour)")
    
    while True:
        model_choice = input("\nEnter model choice (1-3): ").strip()
        if model_choice == "1":
            whisper_model = "large-v3"
            break
        elif model_choice == "2":
            whisper_model = "medium"
            break
        elif model_choice == "3":
            whisper_model = "small"
            break
        else:
            print("❌ Invalid choice. Please enter 1-3")
    
    # Specific episode (optional)
    print("\n📄 Specific Episode (optional):")
    print("Enter a specific episode filename to process only that episode")
    print("Example: '6.01 - Introduction.mp3'")
    specific_filename = input("Enter specific episode filename (or press Enter for all): ").strip()
    if not specific_filename:
        specific_filename = None
    
    # Force retranscribe
    print("\n🔄 Force Retranscribe:")
    print("Force retranscribe will re-process episodes that already have transcripts")
    force_retranscribe = input("Force retranscribe existing episodes? (y/N): ").strip().lower()
    force_retranscribe = force_retranscribe in ['y', 'yes']
    
    # Skip existing
    print("\n⏭️ Skip Existing:")
    print("Skip existing will ignore episodes that already have transcripts")
    skip_existing = input("Skip episodes that already have transcripts? (Y/n): ").strip().lower()
    skip_existing = skip_existing not in ['n', 'no']
    
    # Auto shutdown
    print("\n💤 Auto Shutdown:")
    print("Auto shutdown will turn off the instance when transcription is complete")
    auto_shutdown = input("Automatically shutdown instance when complete? (Y/n): ").strip().lower()
    auto_shutdown = auto_shutdown not in ['n', 'no']
    
    # Summary
    print("\n📋 Configuration Summary:")
    print("=" * 30)
    print(f"Season: {selected_season or 'All seasons'}")
    print(f"Model: {whisper_model}")
    print(f"Specific episode: {specific_filename or 'All episodes'}")
    print(f"Force retranscribe: {force_retranscribe}")
    print(f"Skip existing: {skip_existing}")
    print(f"Auto shutdown: {auto_shutdown}")
    
    confirm = input("\nProceed with these settings? (Y/n): ").strip().lower()
    if confirm in ['n', 'no']:
        print("❌ Cancelled by user")
        exit(0)
    
    return {
        'SELECTED_SEASON': selected_season,
        'WHISPER_MODEL': whisper_model,
        'SPECIFIC_FILENAME': specific_filename,
        'FORCE_RETRANSCRIBE': force_retranscribe,
        'SKIP_EXISTING': skip_existing,
        'AUTO_SHUTDOWN': auto_shutdown
    }

# ===== AUTHENTICATION =====
def authenticate():
    """Authenticate with Google APIs and Cloud Storage"""
    creds = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE, scopes=SCOPES)
    drive_service = build('drive', 'v3', credentials=creds)
    docs_service = build('docs', 'v1', credentials=creds)

    # Initialize GCS client
    gcs_client = storage.Client(credentials=creds, project=GCS_PROJECT_ID)

    return drive_service, docs_service, gcs_client

# ===== HELPER FUNCTIONS =====
def parse_episode_number(filename_stem):
    """Extract season and episode numbers for proper sorting"""
    # Extract pattern like "1.05" or "2.10" from filenames like "1.05 - Title"
    match = re.match(r"(\d+)\.(\d+)", filename_stem)
    if match:
        season = int(match.group(1))
        episode = int(match.group(2))
        return season * 1000 + episode  # Creates proper sort order: 1001, 1002, ..., 1010, 2001, etc.
    return float('inf')  # Put unrecognized formats at the end

def normalize_season_format(season_input):
    """Convert season input to proper folder format (season_XX)"""
    if season_input is None:
        return None

    # If already in correct format (season_XX), return as-is
    if season_input.startswith("season_"):
        return season_input

    # If just a number like "4", convert to "season_04"
    try:
        season_num = int(season_input)
        return f"season_{season_num:02d}"
    except ValueError:
        # If it's something like "season_4", convert to "season_04"
        if "season_" in season_input:
            try:
                season_num = int(season_input.replace("season_", ""))
                return f"season_{season_num:02d}"
            except ValueError:
                pass

    return season_input  # Return as-is if we can't parse it

def list_metadata_files_in_gcs(gcs_client):
    """List all JSON metadata files from Google Cloud Storage"""
    try:
        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        blobs = bucket.list_blobs(prefix=GCS_METADATA_PREFIX)

        json_files = []
        for blob in blobs:
            if blob.name.endswith('.json'):
                json_files.append(blob.name)

        print(f"📂 Found {len(json_files)} metadata files in GCS")
        return json_files

    except Exception as e:
        print(f"❌ Failed to list metadata files from GCS: {e}")
        return []

def download_metadata_from_gcs(blob_path, gcs_client):
    """Download and parse JSON metadata from GCS"""
    try:
        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(blob_path)

        # Download as text and parse JSON
        json_content = blob.download_as_text()
        metadata = json.loads(json_content)

        # Add blob path for later updates
        metadata['_gcs_blob_path'] = blob_path

        return metadata

    except Exception as e:
        print(f"❌ Failed to download metadata from {blob_path}: {e}")
        return None

def upload_metadata_to_gcs(metadata, blob_path, gcs_client):
    """Upload updated metadata back to GCS"""
    try:
        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(blob_path)

        # Remove internal fields before saving
        save_metadata = metadata.copy()
        save_metadata.pop('_gcs_blob_path', None)

        # Upload updated JSON
        blob.upload_from_string(json.dumps(save_metadata, indent=2))
        print(f"💾 Updated metadata in GCS: {blob_path}")
        return True

    except Exception as e:
        print(f"❌ Failed to upload metadata to {blob_path}: {e}")
        return False

def download_from_gcs(gcs_uri, local_path, gcs_client):
    """Download file from Google Cloud Storage to local temporary path"""
    try:
        # Parse GCS URI (gs://bucket/path)
        if not gcs_uri.startswith("gs://"):
            raise ValueError(f"Invalid GCS URI: {gcs_uri}")

        uri_parts = gcs_uri[5:].split("/", 1)  # Remove 'gs://' and split
        bucket_name = uri_parts[0]
        blob_path = uri_parts[1]

        print(f"📥 Downloading from GCS: {blob_path}")

        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        # Ensure local directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download file
        blob.download_to_filename(str(local_path))
        print(f"✅ Downloaded to: {local_path}")
        return True

    except Exception as e:
        print(f"❌ Failed to download from GCS: {e}")
        return False

def get_or_create_folder(name, parent_id, drive_service):
    """Get existing folder or create new one in Google Drive"""
    # Search for existing folder
    query = f"name='{name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    folders = results.get('files', [])

    if folders:
        return folders[0]['id']

    # Create new folder
    folder_metadata = {
        'name': name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent_id]
    }
    folder = drive_service.files().create(body=folder_metadata, fields='id').execute()
    return folder.get('id')

def update_google_doc(doc_id, content, docs_service):
    """Update existing Google Doc with new content"""
    try:
        # Get current document to find end index
        doc = docs_service.documents().get(documentId=doc_id).execute()
        end_index = doc["body"]["content"][-1]["endIndex"] - 1

        # Replace all content
        requests = [
            {"deleteContentRange": {"range": {"startIndex": 1, "endIndex": end_index}}},
            {"insertText": {"location": {"index": 1}, "text": content}}
        ]

        docs_service.documents().batchUpdate(
            documentId=doc_id,
            body={"requests": requests}
        ).execute()

        return True
    except Exception as e:
        print(f"❌ Failed to update Google Doc: {e}")
        return False

def create_google_doc(title, content, folder_id, drive_service, docs_service):
    """Create new Google Doc and return ID and URL"""
    try:
        # Create document
        doc = docs_service.documents().create(body={"title": title}).execute()
        doc_id = doc["documentId"]

        # Move to correct folder
        drive_service.files().update(
            fileId=doc_id,
            addParents=folder_id,
            removeParents='root',
            fields='id, parents'
        ).execute()

        # Add content
        requests = [{"insertText": {"location": {"index": 1}, "text": content}}]
        docs_service.documents().batchUpdate(
            documentId=doc_id,
            body={"requests": requests}
        ).execute()

        doc_url = f"https://docs.google.com/document/d/{doc_id}"
        return doc_id, doc_url

    except HttpError as error:
        print(f"❌ Failed to create Google Doc: {error}")
        return None, None

def format_transcript_segments(segments):
    """Format transcript segments into readable paragraphs"""
    if not segments:
        return ""

    paragraphs = []
    current_para = ""
    sentence_count = 0
    max_sentences = 4
    pause_threshold = 2.0  # seconds

    for i, seg in enumerate(segments):
        text = seg["text"].strip()
        if not text:
            continue

        current_para += " " + text if current_para else text
        sentence_count += len(re.findall(r'[.!?]+', text))

        is_last_segment = (i + 1 == len(segments))
        next_pause = segments[i + 1]["start"] - seg["end"] if not is_last_segment else 0

        if sentence_count >= max_sentences or next_pause >= pause_threshold or is_last_segment:
            paragraphs.append(current_para.strip())
            current_para = ""
            sentence_count = 0

    return "\n\n".join(paragraphs)

def format_metadata_block(meta, segments):
    """Create formatted document with metadata header"""
    formatted_transcript = format_transcript_segments(segments)
    return f"""---
title: {meta['title']}
season: {meta['season']}
episode: {meta['episode_number']}
revolution: {meta['revolution_name']}
published: {meta['published']}
source: Revolutions Podcast
---

{formatted_transcript}
"""

def filter_metadata_files(metadata_files):
    """Filter metadata files based on user options"""
    print("🔍 Filtering episodes...")

    # Apply season filter
    if SELECTED_SEASON:
        normalized_season = normalize_season_format(SELECTED_SEASON)
        print(f"📂 Filtering for {normalized_season} (input: {SELECTED_SEASON})")

        # Filter files that contain the normalized season in their path
        metadata_files = [f for f in metadata_files if normalized_season in f]
        print(f"📂 Found {len(metadata_files)} episodes in {normalized_season}")

    # Apply filename filter
    if SPECIFIC_FILENAME:
        print(f"📄 Filtering for specific file: {SPECIFIC_FILENAME}")
        filename_stem = Path(SPECIFIC_FILENAME).stem
        metadata_files = [f for f in metadata_files if filename_stem in f]
        print(f"📄 Found {len(metadata_files)} matching episodes")

    print(f"📋 Total episodes to process: {len(metadata_files)}")
    return metadata_files

# ===== MAIN PROCESSING LOOP =====
def main():
    global SELECTED_SEASON, WHISPER_MODEL, SPECIFIC_FILENAME, FORCE_RETRANSCRIBE, SKIP_EXISTING, AUTO_SHUTDOWN

    # Get user settings interactively
    settings = get_user_settings()
    
    # Apply settings
    SELECTED_SEASON = settings['SELECTED_SEASON']
    WHISPER_MODEL = settings['WHISPER_MODEL']
    SPECIFIC_FILENAME = settings['SPECIFIC_FILENAME']
    FORCE_RETRANSCRIBE = settings['FORCE_RETRANSCRIBE']
    SKIP_EXISTING = settings['SKIP_EXISTING']
    AUTO_SHUTDOWN = settings['AUTO_SHUTDOWN']

    print(f"\n🚀 Starting GCS-based podcast transcription process...")
    print(f"🎯 Target: {SELECTED_SEASON or 'All seasons'}")
    print(f"🤖 Model: {WHISPER_MODEL}")

    # Authenticate
    drive_service, docs_service, gcs_client = authenticate()

    # Load Whisper model
    print(f"🤖 Loading Whisper model: {WHISPER_MODEL}")
    model = whisper.load_model(WHISPER_MODEL)

    # Create temporary audio directory
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Get metadata files from GCS
    all_metadata_files = list_metadata_files_in_gcs(gcs_client)
    if not all_metadata_files:
        print("❌ No metadata files found in GCS")
        return

    # Filter based on user options
    filtered_files = filter_metadata_files(all_metadata_files)
    if not filtered_files:
        print("❌ No episodes found matching the specified criteria.")
        return

    # Sort by episode number (extract from filename)
    filtered_files = sorted(filtered_files, key=lambda f: parse_episode_number(Path(f).stem))

    # Process each episode
    processed_count = 0
    skipped_count = 0
    
    for blob_path in filtered_files:
        print(f"\n" + "=" * 60)

        # Download and load metadata from GCS
        metadata = download_metadata_from_gcs(blob_path, gcs_client)
        if not metadata:
            print(f"⚠️  Failed to load metadata from {blob_path}")
            continue

        # Check if we have a GCS URI for the audio file
        gcs_uri = metadata.get("gcs_uri")
        if not gcs_uri:
            print(f"⚠️  No GCS URI found for {metadata['title']} - run cloud download first")
            continue

        print(f"📻 Processing: {metadata['title']}")

        # Check transcription status
        segments = metadata.get("segments")
        transcript = metadata.get("transcript")
        existing_doc_id = metadata.get("google_doc_id")

        # Determine if we need to transcribe
        needs_transcription = (
                FORCE_RETRANSCRIBE or
                not transcript or
                not segments
        )

        # Check if we should skip
        if not needs_transcription and SKIP_EXISTING and existing_doc_id:
            print(f"⏭️  Skipping (already transcribed): {metadata['title']}")
            skipped_count += 1
            continue

        # Download audio from cloud storage if needed
        temp_audio_path = None
        if needs_transcription:
            temp_audio_path = TEMP_AUDIO_DIR / f"{metadata['title']}.mp3"

            if not download_from_gcs(gcs_uri, temp_audio_path, gcs_client):
                print(f"❌ Failed to download audio from cloud storage")
                continue

        # Transcribe if needed
        if needs_transcription:
            print(f"🎙️  Transcribing audio...")
            try:
                # Get audio duration for progress estimation
                audio = whisper.load_audio(str(temp_audio_path))
                duration_seconds = len(audio) / whisper.audio.SAMPLE_RATE
                duration_minutes = duration_seconds / 60

                print(f"📏 Audio duration: {duration_minutes:.1f} minutes")
                
                # GPU vs CPU estimation
                if hasattr(model, 'device') and 'cuda' in str(model.device):
                    print(f"⚡ GPU acceleration detected - estimated time: {duration_minutes * 0.2:.1f} minutes")
                else:
                    print(f"⏱️  CPU processing - estimated time: {duration_minutes * 0.55:.1f} minutes")

                # Create progress bar with time-based updates
                with tqdm(total=100, desc=f"🎙️  {metadata['title'][:25]}...",
                          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:

                    # Start transcription
                    start_time = time.time()
                    result = model.transcribe(str(temp_audio_path), verbose=False)
                    transcript = result["text"].strip()
                    segments = result["segments"]

                    # Update progress bar to completion
                    pbar.update(100)

                    # Show final timing
                    elapsed_time = time.time() - start_time
                    print(f"✅ Transcription completed in {elapsed_time / 60:.1f} minutes")

                # Update metadata
                metadata["transcript"] = transcript
                metadata["segments"] = segments
                metadata["whisper_model"] = WHISPER_MODEL
                metadata["transcribed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

                print(f"✅ Transcription complete")
            except Exception as e:
                print(f"❌ Transcription failed: {e}")
                # Clean up temp file
                if temp_audio_path and temp_audio_path.exists():
                    temp_audio_path.unlink()
                continue
            finally:
                # Clean up temporary audio file
                if temp_audio_path and temp_audio_path.exists():
                    temp_audio_path.unlink()
                    print(f"🗑️  Cleaned up temporary audio file")
        else:
            print(f"📝 Using existing transcript")

        # Prepare Google Drive structure
        podcast_folder = get_or_create_folder("Revolutions Podcast", ROOT_FOLDER_ID, drive_service)
        season_name = f"Season {metadata['season']} – {metadata['revolution_name']}"
        season_folder = get_or_create_folder(season_name, podcast_folder, drive_service)

        # Create or update Google Doc
        doc_title = f"Revolutions – S{metadata['season']}E{metadata['episode_number']} – {metadata['title']}"
        full_text = format_metadata_block(metadata, segments)

        if existing_doc_id and not FORCE_RETRANSCRIBE:
            # Update existing document
            print(f"📝 Updating existing Google Doc...")
            if update_google_doc(existing_doc_id, full_text, docs_service):
                print(f"✅ Updated: https://docs.google.com/document/d/{existing_doc_id}")
            else:
                print(f"❌ Failed to update existing document")
                continue
        else:
            # Create new document
            print(f"📄 Creating new Google Doc...")
            doc_id, doc_url = create_google_doc(doc_title, full_text, season_folder, drive_service, docs_service)

            if doc_id:
                metadata["google_doc_id"] = doc_id
                metadata["google_doc_url"] = doc_url
                print(f"✅ Created: {doc_url}")
            else:
                print(f"❌ Failed to create Google Doc")
                continue

        # Update metadata fields
        metadata["transcribed"] = True
        metadata["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        # Upload updated metadata back to GCS
        if upload_metadata_to_gcs(metadata, blob_path, gcs_client):
            print(f"💾 Metadata updated in cloud storage")
        else:
            print(f"⚠️  Failed to update metadata in cloud storage")

        processed_count += 1

    # Final summary
    print(f"\n🎉 Processing complete!")
    print(f"📊 Summary:")
    print(f"   Processed: {processed_count} episodes")
    print(f"   Skipped: {skipped_count} episodes")
    print(f"   Total: {processed_count + skipped_count} episodes")
    
    if AUTO_SHUTDOWN:
        print("💤 Auto-shutdown in 60 seconds...")
        print("💡 All transcripts have been saved to Google Drive")
        time.sleep(60)
        subprocess.run(["sudo", "shutdown", "-h", "now"])
    else:
        print("🎯 Transcription complete. Instance will remain running.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Transcription cancelled by user")
    except Exception as e:
        print(f"\n❌ Transcription failed: {e}")
        if AUTO_SHUTDOWN:
            print("💤 Auto-shutdown in 60 seconds due to error...")
            time.sleep(60)
            subprocess.run(["sudo", "shutdown", "-h", "now"])
