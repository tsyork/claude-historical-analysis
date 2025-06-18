#!/usr/bin/env python3
"""
Claude Sonnet 4-Powered Historical Analysis Interface - Production Version
Optimized for Laptop 2 deployment with external access

Architecture:
- Production Machine (Laptop 2): This interface + Qdrant Docker
- Network: Tailscale VPN for external access
- Docker: Qdrant running in container on localhost:6333
- Python: 3.12 virtual environment for optimal performance
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import anthropic
from google.cloud import storage
from google.oauth2 import service_account
import logging
import sys

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('claude_interface.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class HistoricalQuery:
    """Represents a sophisticated historical analysis query"""
    question: str
    query_type: str  # "basic", "comparative", "thematic", "chronological", "character"
    revolutions: List[str] = None
    time_period: str = None
    characters: List[str] = None
    themes: List[str] = None

@dataclass
class AnalysisResult:
    """Results from historical analysis with citations"""
    answer: str
    sources: List[Dict[str, Any]]
    query_metadata: Dict[str, Any]
    reasoning: str = None

class ClaudeHistoricalAnalyzer:
    """
    Production Claude Sonnet 4-powered historical analysis interface
    
    Optimized for:
    - Laptop 2 production deployment
    - External access via Tailscale
    - Local Qdrant Docker container
    - Cost-effective cloud migration path
    """
    
    def __init__(self, 
                 qdrant_host: str = "localhost",  # Local Docker container
                 qdrant_port: int = 6333,
                 qdrant_collection: str = None,
                 anthropic_api_key: str = None,
                 gcs_credentials_path: str = None,
                 gcs_project: str = "podcast-transcription-462218",
                 gcs_bucket: str = "ai_knowledgebase"):
        """
        Initialize the production Claude-powered historical analysis interface
        
        Args:
            qdrant_host: Docker container host (localhost for production)
            qdrant_port: Qdrant service port
            qdrant_collection: Name of the vector collection
            anthropic_api_key: Claude API key
            gcs_credentials_path: Path to Google Cloud service account credentials
            gcs_project: Google Cloud project ID
            gcs_bucket: GCS bucket containing episode metadata
        """
        
        logger.info("🚀 Initializing Claude Historical Analyzer (Production)")
        
        # Initialize Qdrant connection to local Docker container
        try:
            self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
            self.collection_name = qdrant_collection or os.getenv('QDRANT_COLLECTION', 'revolutions_podcast_v1')
            
            # Test connection and log collection info
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            logger.info(f"✅ Qdrant connected: {len(collection_names)} collections available")
            
            if self.collection_name in collection_names:
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                logger.info(f"📊 Collection '{self.collection_name}': {collection_info.vectors_count} vectors")
                logger.info(f"📊 Collection points_count: {collection_info.points_count}")
                logger.info(f"📊 Collection status: {collection_info.status}")
                logger.info(f"📊 Full collection info: {collection_info}")
            else:
                logger.warning(f"⚠️ Collection '{self.collection_name}' not found")

        except Exception as e:
            logger.error(f"❌ Qdrant connection failed: {e}")
            raise ConnectionError(f"Cannot connect to Qdrant at {qdrant_host}:{qdrant_port}")
        
        # Initialize Claude client
        try:
            self.claude_client = anthropic.Anthropic(
                api_key=anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
            )
            logger.info("✅ Claude API client initialized")
        except Exception as e:
            logger.error(f"❌ Claude initialization failed: {e}")
            raise ValueError("Invalid Anthropic API key")
        
        # Initialize Google Cloud Storage for metadata enrichment
        try:
            actual_creds_path = gcs_credentials_path or os.getenv('GCS_CREDENTIALS_PATH')
            if not actual_creds_path:
                logger.warning("No GCS credentials path specified")
                self.gcs_client = None
                self.bucket = None
                return
            if os.path.exists(actual_creds_path):
                credentials = service_account.Credentials.from_service_account_file(actual_creds_path)
                self.gcs_client = storage.Client(credentials=credentials, project=gcs_project)
                self.bucket = self.gcs_client.bucket(gcs_bucket)
                logger.info("✅ Google Cloud Storage connected")
            else:
                logger.warning(f"⚠️ GCS credentials not found at {gcs_credentials_path}")
                self.gcs_client = None
                self.bucket = None
        except Exception as e:
            logger.warning(f"⚠️ GCS initialization failed: {e}")
            self.gcs_client = None
            self.bucket = None
        
        # Revolution mapping for 10 seasons
        self.revolution_mapping = {
            "1": "English Revolution",
            "2": "American Revolution", 
            "3": "French Revolution",
            "4": "Haitian Revolution",
            "5": "Spanish American Wars of Independence",
            "6": "July Revolution",
            "7": "1848 Revolutions",
            "8": "Paris Commune",
            "9": "Mexican Revolution",
            "10": "Russian Revolution"
        }
        
        # Initialize validated prompt templates
        self._initialize_prompt_templates()
        
        logger.info("🎉 Claude Historical Analyzer ready for production queries")

    def _initialize_prompt_templates(self):
        """Initialize sophisticated prompt templates validated through Dust.tt testing"""
        
        self.prompts = {
            "basic_analysis": """
You are a sophisticated historical analyst with deep expertise in revolutionary periods. 
Analyze the following question using the provided transcript excerpts from the Revolutions podcast.

Question: {question}

Relevant transcript excerpts:
{context}

Instructions:
- Provide a comprehensive analysis drawing from the transcript content
- Include specific episode citations when making claims
- Highlight key patterns, causes, and consequences
- Use clear, academic but accessible language
- Structure your response with clear sections and conclusions

Analysis:
""",

            "comparative_analysis": """
You are conducting a sophisticated comparative analysis of multiple revolutionary periods.
Use the provided transcript excerpts to analyze similarities, differences, and patterns.

Comparative Question: {question}
Revolutions being compared: {revolutions}

Relevant transcript excerpts:
{context}

Instructions:
- Compare and contrast the specified revolutionary periods
- Identify common patterns and unique characteristics  
- Analyze causal factors, social dynamics, and outcomes
- Highlight cross-revolution influences and connections
- Provide specific citations from relevant episodes
- Structure comparison with clear categories and conclusions

Comparative Analysis:
""",

            "character_analysis": """
You are analyzing historical figures across multiple revolutionary periods.
Use the transcript excerpts to trace character development, relationships, and influence.

Character Analysis Question: {question}
Focus Characters: {characters}

Relevant transcript excerpts:
{context}

Instructions:
- Trace the character's involvement across different revolutionary periods
- Analyze their evolving political views, relationships, and influence
- Examine how different revolutionary contexts shaped their actions
- Identify key relationships with other historical figures
- Provide timeline of major activities with episode citations
- Assess their lasting impact on revolutionary movements

Character Analysis:
""",

            "thematic_evolution": """
You are tracking the evolution of ideas, concepts, and themes across revolutionary periods.
Use the transcript excerpts to analyze how themes developed and transformed over time.

Thematic Question: {question}
Theme Focus: {themes}

Relevant transcript excerpts:
{context}

Instructions:
- Trace how the specified themes evolved across different revolutions
- Identify key transformations, adaptations, and innovations
- Analyze the influence of earlier revolutions on later ones
- Examine how social, economic, and political contexts shaped theme development
- Provide specific examples with episode citations
- Assess the lasting impact and modern relevance

Thematic Evolution Analysis:
""",

            "chronological_analysis": """
You are conducting a timeline-based analysis of revolutionary events and developments.
Use the transcript excerpts to analyze patterns, sequences, and causal relationships over time.

Chronological Question: {question}
Time Period: {time_period}

Relevant transcript excerpts:
{context}

Instructions:
- Create a clear chronological framework for analysis
- Identify key sequences, patterns, and turning points
- Analyze causal relationships between events across time
- Examine how earlier developments influenced later outcomes
- Highlight recurring patterns and unique historical moments
- Provide precise dating with episode citations where available

Chronological Analysis:
"""
        }

    def detect_episode_with_ai(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Use Claude to intelligently detect if query is asking for a specific episode
        """
        detection_prompt = f"""
        Analyze this user query to determine if they're asking about a specific episode of a podcast:

        Query: "{query}"

        If this query is asking for a specific episode, extract:
        1. Season number (if mentioned)
        2. Episode number (if mentioned) 
        3. Episode title or keywords (if mentioned)
        4. Whether this is clearly an episode-specific request

        Respond in JSON format:
        {{
            "is_episode_request": true/false,
            "season": "number or null",
            "episode": "number or null", 
            "title_keywords": ["list", "of", "keywords"] or null,
            "confidence": "high/medium/low"
        }}

        Examples:
        - "summarize episode 7.21" → {{"is_episode_request": true, "season": "7", "episode": "21", "confidence": "high"}}
        - "what happened in cracking down and backing down" → {{"is_episode_request": true, "title_keywords": ["cracking", "down", "backing"], "confidence": "medium"}}
        - "causes of French Revolution" → {{"is_episode_request": false, "confidence": "high"}}
        """
        
        try:
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                temperature=0.1,
                messages=[{"role": "user", "content": detection_prompt}]
            )
            
            response_text = response.content[0].text.strip()
            logger.info(f"🤖 Raw AI response: {response_text}")
            
            # Try to extract JSON from response
            import json
            import re
            
            # Look for JSON block in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                
                if result.get("is_episode_request") and result.get("confidence") in ["high", "medium"]:
                    logger.info(f"🤖 AI detected episode request: {result}")
                    return result
            else:
                logger.warning(f"🤖 No JSON found in AI response: {response_text}")
                
        except Exception as e:
            logger.error(f"Episode detection error: {e}")

        return None

    def retrieve_relevant_chunks(self, 
                                query: str, 
                                limit: int = 10,
                                filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant transcript chunks from Qdrant vector database
        
        Args:
            query: The search query
            limit: Maximum number of chunks to retrieve
            filters: Optional filters for seasons, revolutions, etc.
            
        Returns:
            List of relevant chunks with metadata and scores
        """
        try:
            # Build Qdrant filter if specified
            qdrant_filter = None
            conditions = []
            if filters:
                if "seasons" in filters and filters["seasons"]:
                    season_conditions = [
                        FieldCondition(key="season", match=MatchValue(value=season))
                        for season in filters["seasons"]
                    ]
                    conditions.extend(season_conditions)
                
                if "revolutions" in filters and filters["revolutions"]:
                    # Map revolution names to season numbers
                    season_map = {
                        "French Revolution": "3",  
                        "Russian Revolution": "10"  # Adjust if Russian Revolution is season 10
                    }
                    
                    # Filter by season numbers
                    seasons_to_filter = []
                    for rev in filters["revolutions"]:
                        if rev in season_map:
                            seasons_to_filter.append(season_map[rev])

                    # Add debug logging here
                    logger.info(f"🔍 Debug - Revolution filters: {filters['revolutions']}")
                    logger.info(f"🔍 Debug - Seasons to filter: {seasons_to_filter}")

                    
                    if seasons_to_filter:
                        season_conditions = [
                            FieldCondition(key="season", match=MatchValue(value=season))
                            for season in seasons_to_filter
                        ]
                        conditions.extend(season_conditions)
                
                if conditions:
                    qdrant_filter = Filter(should=conditions)

            # AI-powered episode detection
            episode_info = self.detect_episode_with_ai(query)

            if episode_info:
                # Build smart filters based on AI detection using structured metadata
                if episode_info.get("season") and episode_info.get("episode"):
                    # Use structured fields for precise filtering
                    season_condition = FieldCondition(key="season", match=MatchValue(value=episode_info["season"]))
                    episode_condition = FieldCondition(key="episode_number", match=MatchValue(value=episode_info["episode"]))
                    
                    conditions.extend([season_condition, episode_condition])
                    logger.info(f"🤖 Filtering for Season {episode_info['season']} Episode {episode_info['episode']} using structured metadata")
                    
                elif episode_info.get("season"):
                    # Fall back to season filtering if no specific episode
                    season_condition = FieldCondition(key="season", match=MatchValue(value=episode_info["season"]))
                    conditions.append(season_condition)
                    logger.info(f"🤖 Filtering for Season {episode_info['season']} using structured metadata")
                
                logger.info(f"🤖 AI-detected episode request, applying smart filtering")
            
            # Embed the query using OpenAI (same model as your data)
            try:
                from openai import OpenAI
                import os
                
                client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                response = client.embeddings.create(
                    model="text-embedding-3-small",  # Match your original embedding model
                    input=query
                )
                query_vector = response.data[0].embedding
                
                # Debug: Log the final filter being used
                if conditions:
                    qdrant_filter = Filter(must=conditions)
                    logger.info(f"🔍 Debug: Final filter with {len(conditions)} conditions")
                    for i, condition in enumerate(conditions):
                        logger.info(f"🔍 Debug: Condition {i+1}: {condition}")
                else:
                    logger.info(f"🔍 Debug: No filters applied")

                logger.info(f"🔍 Debug: About to search with filter: {qdrant_filter}")
                
                # Now search with the embedded query
                search_results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=qdrant_filter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False
                )
                
            except Exception as e:
                logger.error(f"Search embedding error: {e}")
                return []            
            # Format results
            chunks = []
            for result in search_results:  # scroll returns (points, next_page_offset)
                chunk = {
                    "content": result.payload.get("content", ""),
                    "metadata": result.payload,
                    "score": result.score,  # Default score since scroll doesn't provide relevance scores
                    "episode_title": result.payload.get("episode_title", "Unknown"),
                    "season": result.payload.get("season", "Unknown"),
                    "revolution_name": result.payload.get("revolution_name", "Unknown"),
                    "chunk_index": result.payload.get("chunk_index", 0)
                }
                chunks.append(chunk)
            
            logger.info(f"📊 Retrieved {len(chunks)} relevant chunks for query: {query[:100]}...")
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []

    def enrich_with_metadata(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich chunks with additional metadata from GCS
        
        Args:
            chunks: List of chunks from vector search
            
        Returns:
            Enriched chunks with additional metadata
        """
        if not self.gcs_client:
            return chunks
        
        enriched_chunks = []
        for chunk in chunks:
            try:
                season = chunk["metadata"].get("season")
                episode_title = chunk["metadata"].get("episode_title", "")
                
                if season and episode_title:
                    # Construct metadata path
                    metadata_path = f"podcasts/revolutions/metadata/season_{int(season):02d}/{episode_title}.json"
                    
                    try:
                        blob = self.bucket.blob(metadata_path)
                        if blob.exists():
                            metadata_content = json.loads(blob.download_as_text())
                            chunk["metadata"].update(metadata_content)
                    except Exception as e:
                        logger.debug(f"Could not load metadata for {metadata_path}: {e}")
                
                enriched_chunks.append(chunk)
                
            except Exception as e:
                logger.warning(f"Error enriching chunk metadata: {e}")
                enriched_chunks.append(chunk)
        
        return enriched_chunks

    def format_context_for_claude(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context for Claude analysis
        
        Args:
            chunks: List of relevant chunks
            
        Returns:
            Formatted context string for Claude
        """
        if not chunks:
            return "No relevant transcript content found."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            episode_title = chunk.get("episode_title", "Unknown Episode")
            season = chunk.get("season", "Unknown")
            revolution = chunk.get("revolution_name", "Unknown Revolution")
            content = chunk.get("content", "")
            
            context_part = f"""
[Source {i}] Season {season} - {revolution}
Episode: {episode_title}
Content: {content.strip()}
---
"""
            context_parts.append(context_part.strip())
        
        return "\n\n".join(context_parts)

    def analyze_with_claude(self, 
                           query: HistoricalQuery, 
                           context: str) -> str:
        """
        Perform sophisticated analysis using Claude Sonnet 4
        
        Args:
            query: The historical query object
            context: Formatted context from vector search
            
        Returns:
            Claude's analysis response
        """
        try:
            # Select appropriate prompt template
            if query.query_type == "comparative":
                prompt_template = self.prompts["comparative_analysis"]
                prompt = prompt_template.format(
                    question=query.question,
                    revolutions=", ".join(query.revolutions) if query.revolutions else "Multiple",
                    context=context
                )
            elif query.query_type == "character":
                prompt_template = self.prompts["character_analysis"]
                prompt = prompt_template.format(
                    question=query.question,
                    characters=", ".join(query.characters) if query.characters else "Historical figures",
                    context=context
                )
            elif query.query_type == "thematic":
                prompt_template = self.prompts["thematic_evolution"]
                prompt = prompt_template.format(
                    question=query.question,
                    themes=", ".join(query.themes) if query.themes else "Revolutionary themes",
                    context=context
                )
            elif query.query_type == "chronological":
                prompt_template = self.prompts["chronological_analysis"]
                prompt = prompt_template.format(
                    question=query.question,
                    time_period=query.time_period or "Revolutionary periods",
                    context=context
                )
            else:
                # Default to basic analysis
                prompt_template = self.prompts["basic_analysis"]
                prompt = prompt_template.format(
                    question=query.question,
                    context=context
                )
            
            # Call Claude Sonnet 4 with production settings
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",  # Latest Claude model
                max_tokens=4000,
                temperature=0.1,  # Low temperature for historical accuracy
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Claude analysis error: {e}")
            return f"Error performing analysis: {str(e)}"

    def query(self, 
              question: str,
              query_type: str = "basic",
              revolutions: List[str] = None,
              time_period: str = None,
              characters: List[str] = None,
              themes: List[str] = None,
              max_chunks: int = 10) -> AnalysisResult:
        """
        Perform a complete historical analysis query
        
        Args:
            question: The historical question to analyze
            query_type: Type of analysis ("basic", "comparative", "thematic", "chronological", "character")
            revolutions: List of specific revolutions to focus on
            time_period: Specific time period for chronological analysis
            characters: List of historical figures for character analysis
            themes: List of themes for thematic analysis
            max_chunks: Maximum number of chunks to retrieve and analyze
            
        Returns:
            AnalysisResult with Claude's analysis and source citations
        """
        start_time = datetime.now()
        
        logger.info(f"🔍 Processing {query_type} query: {question[:100]}...")
        
        # Create query object
        query_obj = HistoricalQuery(
            question=question,
            query_type=query_type,
            revolutions=revolutions,
            time_period=time_period,
            characters=characters,
            themes=themes
        )
        
        # Build search filters
        search_filters = {}
        if revolutions:
            search_filters["revolutions"] = revolutions
        
        # Retrieve relevant chunks
        chunks = self.retrieve_relevant_chunks(
            query=question,
            limit=max_chunks,
            filters=search_filters
        )
        
        if not chunks:
            logger.warning("No relevant chunks found for query")
            return AnalysisResult(
                answer="No relevant information found in the transcripts for this query.",
                sources=[],
                query_metadata={"query_type": query_type, "processing_time": 0, "error": "no_chunks"}
            )
        
        # Enrich with metadata
        enriched_chunks = self.enrich_with_metadata(chunks)
        
        # Format context for Claude
        context = self.format_context_for_claude(enriched_chunks)
        
        # Perform Claude analysis
        logger.info(f"🧠 Analyzing with Claude Sonnet 4...")
        analysis = self.analyze_with_claude(query_obj, context)
        
        # Prepare source citations
        sources = []
        for chunk in enriched_chunks:  # Limit to top 5 sources for citations
            source = {
                "episode_title": chunk.get("episode_title", "Unknown"),
                "season": chunk.get("season", "Unknown"),
                "revolution_name": chunk.get("revolution_name", "Unknown"),
                "relevance_score": chunk.get("score", 0),
                "chunk_preview": chunk.get("content", "")[:200] + "..."
            }
            sources.append(source)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        query_metadata = {
            "query_type": query_type,
            "processing_time": processing_time,
            "chunks_analyzed": len(chunks),
            "revolutions_searched": revolutions or "All",
            "timestamp": datetime.now().isoformat(),
            "success": True
        }
        
        logger.info(f"✅ Analysis complete in {processing_time:.2f}s")
        
        return AnalysisResult(
            answer=analysis,
            sources=sources,
            query_metadata=query_metadata
        )


def create_production_interface():
    """
    Create the production Streamlit web interface for external access
    """
    
    st.set_page_config(
        page_title="Historical Analysis - Revolutions Podcast",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Production header with system status
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.title("🏛️ Historical Analysis Interface")
        st.subheader("Sophisticated Analysis of Revolutionary Periods")
    
    with col2:
        st.metric("System Status", "🟢 Online")
        
    with col3:
        # Get Tailscale IP for access info
        try:
            import subprocess
            tailscale_ip = subprocess.check_output(['hostname', '-I']).decode().strip().split()
            tailscale_ip = [ip for ip in tailscale_ip if ip.startswith('100.')][0]
            st.metric("External Access", f"🌐 {tailscale_ip}:8501")
        except:
            st.metric("External Access", "🌐 Available")
    
    # Initialize analyzer with error handling for production
    @st.cache_resource
    def get_analyzer():
        try:
            return ClaudeHistoricalAnalyzer()
        except ConnectionError as e:
            st.error(f"❌ Database Connection Error: {e}")
            st.info("Please ensure Qdrant Docker container is running: `docker ps | grep qdrant`")
            return None
        except ValueError as e:
            st.error(f"❌ API Configuration Error: {e}")
            st.info("Please check your Anthropic API key in the .env file")
            return None
        except Exception as e:
            st.error(f"❌ Initialization Error: {e}")
            st.info("Please check system configuration and try again")
            return None
    
    analyzer = get_analyzer()
    
    if not analyzer:
        st.stop()
    
    # Production sidebar with system info
    with st.sidebar:
        st.header("🔧 Query Configuration")
        
        # System status section
        with st.expander("📊 System Status", expanded=False):
            try:
                # Test Qdrant connection
                collections = analyzer.qdrant_client.get_collections()
                collection_names = [c.name for c in collections.collections]
                st.success(f"✅ Qdrant: {len(collection_names)} collections")
                
                # Show collection stats
                if analyzer.collection_name in collection_names:
                    info = analyzer.qdrant_client.get_collection(analyzer.collection_name)
                    vector_count = info.vectors_count if info.vectors_count is not None else info.points_count
                    st.info(f"📊 Vectors: {vector_count:,}")
                
                # GCS status
                if analyzer.gcs_client:
                    st.success("✅ Google Cloud Storage")
                else:
                    st.warning("⚠️ GCS metadata disabled")
                    
            except Exception as e:
                st.error(f"❌ System check failed: {e}")
        
        # Query type selection
        query_type = st.selectbox(
            "Analysis Type",
            ["basic", "comparative", "thematic", "chronological", "character"],
            help="Select the type of historical analysis to perform"
        )
        
        # Revolution selection for filtering
        available_revolutions = list(analyzer.revolution_mapping.values())
        selected_revolutions = st.multiselect(
            "Focus Revolutions (optional)",
            available_revolutions,
            help="Leave empty to search all revolutions"
        )
        
        # Additional parameters based on query type
        characters = []
        themes = []
        time_period = ""
        
        if query_type == "character":
            characters_input = st.text_input(
                "Historical Figures",
                placeholder="e.g., Lafayette, Robespierre, Toussaint",
                help="Comma-separated list of historical figures to analyze"
            )
            characters = [c.strip() for c in characters_input.split(",") if c.strip()]
        
        elif query_type == "thematic":
            themes_input = st.text_input(
                "Themes to Track",
                placeholder="e.g., social class, economic causes, republicanism",
                help="Comma-separated list of themes to analyze"
            )
            themes = [t.strip() for t in themes_input.split(",") if t.strip()]
        
        elif query_type == "chronological":
            time_period = st.text_input(
                "Time Period",
                placeholder="e.g., 1789-1799, Late 18th century",
                help="Specific time period for chronological analysis"
            )
        
        # Advanced options
        max_chunks = st.slider(
            "Max Sources",
            min_value=5,
            max_value=20,
            value=10,
            help="Maximum number of transcript chunks to analyze"
        )
    
    # Main query interface
    st.header("💭 Ask Your Historical Question")
    
    # Example queries with production-focused examples
    example_queries = {
        "Basic Analysis": "What were the main causes of the French Revolution?",
        "Comparative": "Compare the economic factors leading to the French Revolution versus the Russian Revolution.",
        "Character": "How did Lafayette's role evolve across the American and French Revolutions?",
        "Thematic": "How did concepts of republicanism evolve from the English Revolution through the French Revolution?",
        "Chronological": "What were the key turning points in the Haitian Revolution between 1791-1804?"
    }
    
    # Example query selection
    example_category = st.selectbox(
        "📝 Example Queries (optional)",
        [""] + list(example_queries.keys()),
        help="Select an example query to get started"
    )
    
    default_question = example_queries.get(example_category, "")
    
    # Main question input
    question = st.text_area(
        "Your Historical Question",
        value=default_question,
        height=100,
        placeholder="Ask a sophisticated historical question about revolutionary periods..."
    )
    
    # Analysis button with production monitoring
    if st.button("🔍 Analyze", type="primary", disabled=not question.strip()):
        if question.strip():
            # Create progress tracking for production
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🔍 Searching transcript database...")
                progress_bar.progress(25)
                
                # Perform the analysis
                result = analyzer.query(
                    question=question,
                    query_type=query_type,
                    revolutions=selected_revolutions if selected_revolutions else None,
                    time_period=time_period if time_period else None,
                    characters=characters if characters else None,
                    themes=themes if themes else None,
                    max_chunks=max_chunks
                )
                
                progress_bar.progress(75)
                status_text.text("🧠 Generating analysis with Claude...")
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Display results in production format
                st.header("📊 Analysis Results")
                
                # Main analysis with copy functionality
                st.subheader("🎯 Historical Analysis")
                
                # Show metadata first for transparency
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Processing Time", f"{result.query_metadata['processing_time']:.2f}s")
                with col2:
                    st.metric("Sources Analyzed", result.query_metadata['chunks_analyzed'])
                with col3:
                    st.metric("Query Type", result.query_metadata['query_type'].title())
                
                # Main analysis content
                st.markdown(result.answer)
                
                # Sources and additional info
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📚 Source Citations")
                    for i, source in enumerate(result.sources, 1):
                        with st.expander(f"Source {i}: {source['episode_title']} (Season {source['season']})"):
                            st.write(f"**Revolution:** {source['revolution_name']}")
                            st.write(f"**Relevance Score:** {source['relevance_score']:.3f}")
                            st.write(f"**Content Preview:** {source['chunk_preview']}")
                
                with col2:
                    st.subheader("⚙️ System Metrics")
                    metrics_data = {
                        "Query Type": result.query_metadata['query_type'],
                        "Processing Time": f"{result.query_metadata['processing_time']:.2f}s",
                        "Chunks Retrieved": result.query_metadata['chunks_analyzed'],
                        "Search Scope": result.query_metadata['revolutions_searched'],
                        "Timestamp": result.query_metadata['timestamp'][:19]
                    }
                    
                    for key, value in metrics_data.items():
                        st.text(f"{key}: {value}")
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Analysis failed: {str(e)}")
                logger.error(f"Query processing error: {e}")
                
        else:
            st.warning("Please enter a question to analyze.")
    
    # Production footer with system info
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 System Coverage**")
        st.text("• 336+ episodes processed")
        st.text("• 11 seasons of history")
        st.text("• 275+ years covered")
    
    with col2:
        st.markdown("**🔧 Technical Stack**")
        st.text("• Claude Sonnet 4")
        st.text("• Qdrant Vector DB")
        st.text("• Python 3.12")
    
    with col3:
        st.markdown("**🌐 Access Information**")
        st.text("• External access via Tailscale")
        st.text("• Production deployment")
        st.text("• Cloud migration ready")


def main():
    """Main entry point for production deployment"""
    try:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Set production environment variables
        os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
        os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
        
        # Create and run the production interface
        create_production_interface()
        
    except Exception as e:
        st.error(f"❌ Production startup failed: {e}")
        logger.error(f"Production startup error: {e}")


if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        import streamlit as st
        # If we can access streamlit's session state, we're in streamlit
        _ = st.session_state
        main()
    except:
        # Command-line interface for testing
        print("🏛️ Claude Historical Analysis Interface - Production")
        print("Run with: streamlit run claude_rag_interface.py")
        
        # Quick production test
        try:
            analyzer = ClaudeHistoricalAnalyzer()
            test_query = "What were the main causes of the French Revolution?"
            print(f"\n🧪 Testing production setup with: {test_query}")
            
            result = analyzer.query(
                question=test_query,
                query_type="basic",
                max_chunks=5
            )
            
            print(f"\n✅ Production test successful!")
            print(f"Answer length: {len(result.answer)} characters")
            print(f"Sources found: {len(result.sources)}")
            print(f"Processing time: {result.query_metadata['processing_time']:.2f}s")
            print(f"\n🌐 Ready for external access via Tailscale")
            
        except Exception as e:
            print(f"❌ Production test failed: {e}")
            print("Please check configuration and Docker services.")
