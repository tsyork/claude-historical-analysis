# Phase 1: Transcription System

AWS Spot Fleet-based podcast transcription using OpenAI Whisper.

## Files
- `interactive_transcribe.py` - Main transcription script with interactive configuration
- `deploy-transcription.sh` - Automated AWS Spot Fleet deployment 
- `spot-fleet-config.json` - AWS Spot Fleet configuration

## Features
- GPU-accelerated transcription (8-10x faster than local)
- Cost-effective spot pricing (~.50-5.00 for 150 hours)
- Automatic instance setup and teardown
- Multiple Whisper model support (tiny to large-v3)

## Usage
1. Configure AWS credentials
2. Update spot-fleet-config.json with your AWS resource IDs
3. Run: `./deploy-transcription.sh`
