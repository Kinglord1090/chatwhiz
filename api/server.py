"""
FastAPI Backend Server for ChatWhiz - Improved Version
Provides independent API endpoints with multi-stage progress tracking
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import os
import sys
import tempfile
import yaml
import uuid
import json
from datetime import datetime
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

from modules.embedder import create_embedder_from_config
from modules.vector_store import create_vector_store_from_config
from modules.loader import ChatLoader, ChatMessage
from modules.retriever import ChatRetriever
from modules.llm import create_rag_system
from modules.encryptor import is_file_encrypted

app = FastAPI(title="ChatWhiz API", description="Advanced Chat Search System API", version="2.1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system components
system_components = {
    'embedder': None,
    'vector_store': None,
    'retriever': None,
    'rag_system': None,
    'config': None,
    'initialized': False
}

# Background task storage - now supports multiple concurrent tasks
background_tasks = {}
# Separate storage for file processing tasks
file_processing_tasks = {}

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

# Persistent state file
STATE_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'indexing_state.json')

# Shutdown flag for clean termination
shutdown_flag = threading.Event()

# Models
class SearchQuery(BaseModel):
    query: str
    mode: str = "semantic"
    top_k: int = 5
    threshold: float = 0.7

class RAGQuery(BaseModel):
    query: str
    mode: str = "semantic"
    top_k: int = 5
    threshold: float = 0.7
    llm_provider: Optional[str] = None
    instruction: Optional[str] = None  # Allow custom instruction

class IndexingRequest(BaseModel):
    rebuild: bool = False

class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: float
    message: str
    result: Optional[Dict[str, Any]] = None
    stages: Optional[Dict[str, Dict[str, Any]]] = None  # Multi-stage progress

def load_config():
    """Load configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    if not os.path.exists(config_path):
        return {
            'embedding_model': 'hkunlp/instructor-large',
            'instruction': 'Represent the chat message for semantic search:',
            'device': 'auto',
            'llm_provider': 'none',
            'retrieval_mode': 'semantic',
            'top_k': 5,
            'similarity_threshold': 0.3,
            'data_dir': 'data',
            'vectorstore_dir': 'data/vectorstore',
            'bm25_dir': 'data/bm25',
            'cache_dir': 'data/cache',
            'processed_dir': 'data/processed'
        }
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

async def initialize_system():
    """Initialize the ChatWhiz system asynchronously."""
    if system_components['initialized']:
        return
    
    try:
        print("Initializing ChatWhiz system...")
        config = load_config()
        system_components['config'] = config
        
        # Create embedder
        embedder = create_embedder_from_config(config)
        
        # Get embedding dimension
        dimension = embedder.get_embedding_dimension()
        
        # Create vector store
        vector_store = create_vector_store_from_config(config, dimension)
        
        # Create retriever
        retriever = ChatRetriever(embedder, vector_store, config=config)
        
        # Create RAG system
        rag_system = create_rag_system(config)
        
        system_components.update({
            'embedder': embedder,
            'vector_store': vector_store,
            'retriever': retriever,
            'rag_system': rag_system,
            'initialized': True
        })
        
        print("ChatWhiz system initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing system: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize system: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    # Initialize system first
    await initialize_system()
    # Then check for any unfinished indexing tasks and resume them
    await resume_unfinished_tasks()

async def resume_unfinished_tasks():
    """Resume any unfinished indexing tasks from previous session."""
    try:
        state_file = STATE_FILE
        print(f"Checking for resumable tasks at: {state_file}")
        if os.path.exists(state_file):
            print(f"✓ Found indexing state file")
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    print(f"  State: status={state.get('status')}, messages={len(state.get('messages', []))}, batch={state.get('current_batch', 0)}")
                    
                if state.get('status') == 'indexing' and state.get('messages'):
                    filename = state.get('filename', 'Unknown file')
                    message_count = len(state.get('messages', []))
                    current_batch = state.get('current_batch', 0)
                    print(f"\n{'='*60}")
                    print(f"RESUMING INDEXING TASK:")
                    print(f"  File: {filename}")
                    print(f"  Progress: {current_batch}/{message_count} messages processed")
                    print(f"  Rebuild mode: {state.get('rebuild', False)}")
                    print(f"{'='*60}\n")
                    
                    # Use the saved task_id if available, otherwise generate new one
                    task_id = state.get('task_id', str(uuid.uuid4()))
                    
                    # Extract progress info from state
                    current_batch = state.get('current_batch', 0)
                    total_messages = len(state.get('messages', []))
                    # Calculate actual progress percentage
                    # current_batch is the number of messages already processed
                    embedding_progress = (current_batch / total_messages * 100) if total_messages > 0 else 0
                    bm25_progress = (current_batch / total_messages * 100) if total_messages > 0 else 0
                    overall_progress = (embedding_progress + bm25_progress) / 200  # Average normalized to 0-1
                    
                    background_tasks[task_id] = {
                        "status": "processing",
                        "progress": overall_progress,
                        "message": f"Resuming: {current_batch}/{total_messages} messages processed",
                        "result": None,
                        "stages": state.get('stages', {
                            "embedding": {"progress": embedding_progress, "status": "processing"},
                            "bm25": {"progress": bm25_progress, "status": "processing"},
                            "saving": {"progress": 0, "status": "pending"}
                        })
                    }
                    
                    # Update state with task_id and save it
                    state['task_id'] = task_id
                    with open(state_file, 'w') as f:
                        json.dump(state, f)
                    
                    print(f"Task {task_id} created with status: processing, progress: {overall_progress:.2f}")
                    
                    # Resume indexing in background
                    threading.Thread(target=resume_indexing_task, args=(task_id, state)).start()
                    return task_id
                else:
                    print(f"  State file exists but not resumable (status={state.get('status')})")
            except json.JSONDecodeError as e:
                print(f"✗ ERROR: Corrupted state file - {e}")
                # Remove corrupted file
                os.remove(state_file)
                print("  Removed corrupted state file")
        else:
            print("✓ No indexing state file found - starting fresh")
    except Exception as e:
        print(f"✗ ERROR checking for resumable tasks: {e}")
        import traceback
        traceback.print_exc()
    return None

@app.get("/")
async def root():
    """Serve the main HTML page."""
    # Serve the default UI (modern) as index.html
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    return {"message": "ChatWhiz API is running"}

@app.get("/api/status")
async def get_status():
    """Get system status."""
    if not system_components['initialized']:
        await initialize_system()
    
    retriever = system_components['retriever']
    embedder = system_components['embedder']
    config = system_components['config']
    rag_system = system_components['rag_system']
    
    stats = retriever.get_stats()
    
    # Check available LLM providers
    available_llms = []
    if config.get('llm_provider') and config.get('llm_provider') != 'none':
        if rag_system and rag_system.llm_provider.is_available():
            available_llms.append(config.get('llm_provider'))
    
    # Add instructor-large as default option for RAG (built-in Q&A)
    if 'instructor-qa' not in available_llms:
        available_llms.append('instructor-qa')  # Better name for Q&A mode
    
    # Derive model/device info
    try:
        device = embedder._get_device() if embedder else 'cpu'
    except Exception:
        device = 'cpu'
    
    # Check for active tasks (both indexing and file processing)
    active_tasks = []
    for task_id, task_info in {**background_tasks, **file_processing_tasks}.items():
        if task_info.get('status') in ['started', 'processing']:
            active_tasks.append({
                'task_id': task_id,
                'type': task_info.get('type', 'indexing'),
                'filename': task_info.get('filename'),
                'progress': task_info.get('progress', 0),
                'message': task_info.get('message', '')
            })
    
    return {
        "initialized": system_components['initialized'],
        "stats": stats,
        "config": {
            "embedding_model": config.get('embedding_model'),
            "device": device,
            "llm_provider": config.get('llm_provider') if config.get('llm_provider') != 'none' else (available_llms[0] if available_llms else 'none'),
            "retrieval_mode": config.get('retrieval_mode'),
            "available_llms": available_llms
        },
        "active_tasks": active_tasks
    }

@app.post("/api/search")
async def search_messages(query: SearchQuery):
    """Search messages."""
    if not system_components['initialized']:
        await initialize_system()
    
    retriever = system_components['retriever']
    
    try:
        # Allow frontend to request all results by using a high top_k value
        # Cap at a reasonable maximum to prevent memory issues
        max_results = min(query.top_k, 10000)  # Cap at 10000 results max
        
        results = retriever.search(
            query=query.query,
            mode=query.mode,
            k=max_results,
            threshold=query.threshold
        )
        
        # Convert results to serializable format
        search_results = []
        for result in results:
            search_results.append({
                "text": result.text,
                "score": float(result.score),
                "metadata": result.metadata,
                "search_type": result.search_type
            })
        
        return {
            "results": search_results,
            "query": query.query,
            "mode": query.mode,
            "total_results": len(search_results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag")
async def generate_rag_response(query: RAGQuery):
    """Generate RAG response with improved Q&A capabilities."""
    if not system_components['initialized']:
        await initialize_system()
    
    retriever = system_components['retriever']
    embedder = system_components['embedder']
    config = system_components['config']
    
    try:
        # First get search results
        results = retriever.search(
            query=query.query,
            mode=query.mode,
            k=query.top_k,
            threshold=query.threshold
        )
        
        if not results:
            return {
                "answer": "No relevant messages found to generate a response.",
                "results": [],
                "llm_used": "none"
            }
        
        # Determine which LLM to use
        llm_provider = query.llm_provider or "instructor-qa"
        
        if llm_provider == "instructor-qa":
            # Improved Q&A mode - actually answer questions instead of just listing messages
            context_texts = [result.text for result in results[:5]]  # Use top 5 results
            
            # Build a context-aware response based on the question type
            user_question = query.query.lower()
            
            # Analyze the question to provide a better answer
            if any(word in user_question for word in ['who', 'which', 'whose']):
                # Person-related question
                # Extract unique senders from results
                senders = {}
                for result in results:
                    sender = result.metadata.get('sender', 'Unknown')
                    if sender not in senders:
                        senders[sender] = 0
                    senders[sender] += 1
                
                # Sort by frequency
                sorted_senders = sorted(senders.items(), key=lambda x: x[1], reverse=True)
                
                if 'most' in user_question or 'talked' in user_question:
                    answer = f"Based on the search results, **{sorted_senders[0][0]}** talked most about '{query.query.replace('?', '').strip()}' "
                    answer += f"with {sorted_senders[0][1]} relevant messages.\n\n"
                    if len(sorted_senders) > 1:
                        answer += "Other participants who discussed this topic:\n"
                        for sender, count in sorted_senders[1:3]:
                            answer += f"- {sender}: {count} messages\n"
                else:
                    answer = f"The following people discussed '{query.query.replace('?', '').strip()}':\n"
                    for sender, count in sorted_senders[:5]:
                        answer += f"- {sender}: {count} messages\n"
                
                answer += f"\n**Sample relevant messages:**\n"
                for i, result in enumerate(results[:3], 1):
                    answer += f"{i}. {result.text[:150]}...\n"
                    
            elif any(word in user_question for word in ['what', 'how', 'why', 'when', 'where']):
                # Information-seeking question
                answer = f"Based on the chat history about '{query.query.replace('?', '').strip()}':\n\n"
                
                # Group messages by topic/sender
                seen_content = set()
                unique_points = []
                for result in results[:7]:
                    # Avoid duplicate content
                    content_key = result.text[:50].lower()
                    if content_key not in seen_content:
                        seen_content.add(content_key)
                        unique_points.append(result)
                
                answer += "**Key points from the conversation:**\n"
                for i, result in enumerate(unique_points[:5], 1):
                    sender = result.metadata.get('sender', 'Someone')
                    answer += f"{i}. {sender} said: {result.text[:200]}\n"
                    if result.metadata.get('timestamp'):
                        answer += f"   (at {result.metadata['timestamp']})\n"
                    answer += "\n"
                    
            elif any(word in user_question for word in ['count', 'many', 'number']):
                # Counting question
                answer = f"Found **{len(results)}** messages related to '{query.query.replace('?', '').strip()}'.\n\n"
                
                # Count by sender
                senders = {}
                for result in results:
                    sender = result.metadata.get('sender', 'Unknown')
                    senders[sender] = senders.get(sender, 0) + 1
                
                answer += "**Breakdown by participant:**\n"
                for sender, count in sorted(senders.items(), key=lambda x: x[1], reverse=True):
                    answer += f"- {sender}: {count} messages\n"
                    
            else:
                # Default case - provide a summary
                answer = f"Here's a summary of the conversation about '{query.query}':\n\n"
                for i, result in enumerate(results[:5], 1):
                    sender = result.metadata.get('sender', 'Someone')
                    answer += f"{i}. {sender}: {result.text[:150]}...\n"
                
                if len(results) > 1:
                    answer += f"\n(Found {len(results)} relevant messages with similarity scores from {results[0].score:.3f} to {results[-1].score:.3f})"
                else:
                    answer += f"\n(Found 1 relevant message with similarity score {results[0].score:.3f})"
            
            return {
                "answer": answer,
                "results": [{
                    "text": r.text, 
                    "score": float(r.score), 
                    "search_type": r.search_type,
                    "metadata": r.metadata
                } for r in results],
                "llm_used": "instructor-qa",
                "context_used": len(results)
            }
        
        else:
            # Use configured RAG system (external LLM)
            rag_system = system_components['rag_system']
            if not rag_system or not rag_system.llm_provider.is_available():
                raise HTTPException(status_code=400, detail=f"LLM provider {llm_provider} is not available")
            
            # Add custom instruction if provided
            if query.instruction:
                rag_result = rag_system.generate_answer(
                    query.query, 
                    results,
                    custom_instruction=query.instruction
                )
            else:
                rag_result = rag_system.generate_answer(query.query, results)
            
            return {
                "answer": rag_result.get('answer', 'Failed to generate response'),
                "results": [{
                    "text": r.text, 
                    "score": float(r.score), 
                    "search_type": r.search_type,
                    "metadata": r.metadata
                } for r in results],
                "llm_used": llm_provider,
                "context_used": rag_result.get('context_used', 0),
                "error": rag_result.get('error')
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...), rebuild: bool = False):
    """Upload and process chat files - supports multiple concurrent uploads."""
    # Create separate task for each file
    task_ids = []
    
    for file in files:
        task_id = str(uuid.uuid4())
        file_processing_tasks[task_id] = {
            "status": "started",
            "type": "file_processing",
            "filename": file.filename,
            "progress": 0.0,
            "message": f"Starting to process {file.filename}...",
            "result": None,
            "rebuild": rebuild,  # Store rebuild flag
            "stages": {
                "upload": {"progress": 0, "status": "pending"},
                "processing": {"progress": 0, "status": "pending"},
                "deduplication": {"progress": 0, "status": "pending"}
            }
        }
        
        # Process each file in a separate thread
        threading.Thread(target=process_single_file_background, args=(task_id, file, rebuild)).start()
        task_ids.append(task_id)
    
    return {"task_ids": task_ids}

def process_single_file_background(task_id: str, file: UploadFile, rebuild: bool = False):
    """Process a single uploaded file in background."""
    try:
        loader = ChatLoader()
        
        # Update upload stage
        file_processing_tasks[task_id]["stages"]["upload"]["status"] = "processing"
        file_processing_tasks[task_id]["message"] = f"Reading {file.filename}..."
        
        # Read file content
        file.file.seek(0)
        content = file.file.read()
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        file_processing_tasks[task_id]["stages"]["upload"]["progress"] = 100
        file_processing_tasks[task_id]["stages"]["upload"]["status"] = "completed"
        file_processing_tasks[task_id]["progress"] = 0.33
        
        try:
            # Check if encrypted
            if is_file_encrypted(tmp_path):
                file_processing_tasks[task_id]["status"] = "error"
                file_processing_tasks[task_id]["message"] = f"File {file.filename} is encrypted"
                return
            
            # Process file
            file_processing_tasks[task_id]["stages"]["processing"]["status"] = "processing"
            file_processing_tasks[task_id]["message"] = f"Extracting messages from {file.filename}..."
            
            messages = loader.auto_detect_and_load(tmp_path)
            filtered_messages = loader.filter_messages(messages)
            
            file_processing_tasks[task_id]["stages"]["processing"]["progress"] = 100
            file_processing_tasks[task_id]["stages"]["processing"]["status"] = "completed"
            file_processing_tasks[task_id]["progress"] = 0.66
            
            # Deduplicate
            file_processing_tasks[task_id]["stages"]["deduplication"]["status"] = "processing"
            file_processing_tasks[task_id]["message"] = f"Deduplicating messages from {file.filename}..."
            
            unique_messages = loader.deduplicate_messages(filtered_messages)
            
            file_processing_tasks[task_id]["stages"]["deduplication"]["progress"] = 100
            file_processing_tasks[task_id]["stages"]["deduplication"]["status"] = "completed"
            file_processing_tasks[task_id]["progress"] = 1.0
            
            # Store messages for indexing
            messages_data = [msg.to_dict() for msg in unique_messages]
            
            # Create indexing task
            index_task_id = str(uuid.uuid4())
            background_tasks[index_task_id] = {
                "status": "queued",
                "type": "indexing",
                "filename": file.filename,
                "progress": 0.0,
                "message": f"Queued for indexing: {file.filename}",
                "result": None,
                "rebuild": rebuild,  # Pass rebuild flag
                "stages": {
                    "embedding": {"progress": 0, "status": "pending"},
                    "bm25": {"progress": 0, "status": "pending"},
                    "saving": {"progress": 0, "status": "pending"}
                }
            }
            
            # Save initial indexing state IMMEDIATELY and SYNCHRONOUSLY
            # This ensures state is persisted even if server stops right after upload
            try:
                save_indexing_state({
                    "status": "indexing",
                    "task_id": index_task_id,
                    "filename": file.filename,  # Add filename for better logging
                    "messages": [msg.to_dict() for msg in unique_messages],
                    "current_batch": 0,
                    "rebuild": rebuild,
                    "progress": 0,
                    "stages": background_tasks[index_task_id]["stages"]
                })
                print(f"Saved initial indexing state for {file.filename} (task {index_task_id})")
            except Exception as e:
                print(f"Warning: Failed to save initial indexing state: {e}")
                # Continue anyway - indexing can still proceed
            
            # Start indexing in background
            threading.Thread(
                target=index_messages_multithreaded,
                args=(index_task_id, unique_messages, 0, rebuild)
            ).start()
            
            file_processing_tasks[task_id]["status"] = "completed"
            file_processing_tasks[task_id]["message"] = f"Successfully processed {file.filename}"
            file_processing_tasks[task_id]["result"] = {
                "total_messages": len(filtered_messages),
                "unique_messages": len(unique_messages),
                "index_task_id": index_task_id
            }
            
        finally:
            os.unlink(tmp_path)
            
    except Exception as e:
        file_processing_tasks[task_id]["status"] = "error"
        file_processing_tasks[task_id]["message"] = f"Error processing {file.filename}: {str(e)}"
        file_processing_tasks[task_id]["progress"] = 0

def index_messages_multithreaded(task_id: str, messages: List[ChatMessage], start_batch: int = 0, rebuild: bool = False):
    """Index messages using multithreading for better performance with parallel BM25 indexing.
    
    Args:
        task_id: The task ID
        messages: List of messages to index
        start_batch: Starting batch index for resuming (default 0)
        rebuild: If True, clear cache and rebuild indexes from scratch
    """
    try:
        if not messages:
            background_tasks[task_id]["status"] = "completed"
            background_tasks[task_id]["message"] = "No messages to index"
            return
        
        retriever = system_components['retriever']
        embedder = system_components['embedder']
        
        total_messages = len(messages)
        background_tasks[task_id]["status"] = "processing"
        
        # Prepare texts and metadata
        texts = [f"{msg.sender}: {msg.text}" for msg in messages]
        metadata = [retriever._get_message_metadata(msg) for msg in messages]
        
        # Check if we need to rebuild or clear stores
        if start_batch == 0:
            if rebuild:
                # User requested rebuild - clear everything
                print(f"Rebuild requested - clearing all caches and indexes")
                retriever.vector_store.clear()
                retriever.bm25_store.clear()
                # Clear embedding cache too
                cache_dir = system_components['config'].get('cache_dir', 'data/cache')
                if os.path.exists(cache_dir):
                    import shutil
                    shutil.rmtree(cache_dir)
                    os.makedirs(cache_dir, exist_ok=True)
                    print(f"Cleared embedding cache at {cache_dir}")
            else:
                # Normal processing - check existing data
                vector_count = retriever.vector_store.get_stats()['total_vectors']
                bm25_count = retriever.bm25_store.get_stats()['total_documents']
                
                # Only clear if we have mismatched or no data
                if vector_count == 0 and bm25_count == 0:
                    print(f"Starting fresh indexing for {total_messages} messages")
                elif vector_count != total_messages or bm25_count != total_messages:
                    # Data mismatch, clear and restart
                    retriever.vector_store.clear()
                    retriever.bm25_store.clear()
                    print(f"Cleared mismatched indexes (had {vector_count} vectors, {bm25_count} BM25 docs, need {total_messages})")
                else:
                    # Data matches, don't clear - just mark as complete
                    print(f"Indexes already complete with {vector_count} items, skipping indexing")
                    background_tasks[task_id]["status"] = "completed"
                    background_tasks[task_id]["progress"] = 1.0
                    background_tasks[task_id]["message"] = f"Already indexed {total_messages} messages"
                    background_tasks[task_id]["stages"]["embedding"]["progress"] = 100
                    background_tasks[task_id]["stages"]["embedding"]["status"] = "completed"
                    background_tasks[task_id]["stages"]["bm25"]["progress"] = 100
                    background_tasks[task_id]["stages"]["bm25"]["status"] = "completed"
                    background_tasks[task_id]["stages"]["saving"]["progress"] = 100
                    background_tasks[task_id]["stages"]["saving"]["status"] = "completed"
                    clear_indexing_state()
                    return
        
        # Set initial status
        background_tasks[task_id]["stages"]["embedding"]["status"] = "processing"
        background_tasks[task_id]["stages"]["bm25"]["status"] = "processing"
        
        # Use batch processing for both embeddings and BM25
        batch_size = 50  # Keep smaller batches for more granular progress
        save_interval = 500  # Save every 500 messages for better searchability
        
        # Track progress
        embeddings_processed = start_batch * batch_size if start_batch > 0 else 0
        bm25_processed = start_batch * batch_size if start_batch > 0 else 0
        
        # Collect all embeddings and process in batches
        all_embeddings = []
        all_texts_processed = []
        all_metadata_processed = []
        
        # Process messages in batches starting from where we left off
        num_batches = (total_messages + batch_size - 1) // batch_size
        
        for batch_idx in range(start_batch, num_batches):
            # Check shutdown flag before processing each batch
            if shutdown_flag.is_set():
                print(f"Shutdown requested, stopping indexing at batch {batch_idx}")
                background_tasks[task_id]["status"] = "interrupted"
                background_tasks[task_id]["message"] = f"Indexing interrupted at batch {batch_idx}"
                # Save current state for resumption
                save_indexing_state({
                    "status": "indexing",
                    "task_id": task_id,
                    "messages": [msg.to_dict() for msg in messages],
                    "current_batch": batch_idx * batch_size,
                    "progress": background_tasks[task_id]["progress"],
                    "stages": background_tasks[task_id]["stages"]
                })
                return
            
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_messages)
            batch_texts = texts[batch_start:batch_end]
            batch_metadata = metadata[batch_start:batch_end]
            
            # Process embeddings for this batch
            # Don't use cache if rebuild is requested
            batch_embeddings = embedder.encode(batch_texts, use_cache=(not rebuild), batch_size=min(32, len(batch_texts)))
            all_embeddings.extend(batch_embeddings)
            all_texts_processed.extend(batch_texts)
            all_metadata_processed.extend(batch_metadata)
            
            # Add small delay when using cache to allow UI to see progress
            # (cached embeddings are returned instantly, UI can't keep up)
            import time
            if batch_idx < 10:  # Only for first few batches to show initial progress
                time.sleep(0.05)  # 50ms delay
            
            # Add to vector store immediately
            retriever.vector_store.add_embeddings(batch_embeddings, batch_texts, batch_metadata)
            
            # Add to BM25 store immediately (parallel with embeddings)
            retriever.bm25_store.add_documents(batch_texts, batch_metadata)
            
            # Update progress
            embeddings_processed = batch_end
            bm25_processed = batch_end
            
            # Update task progress
            embedding_progress = (embeddings_processed / total_messages) * 100
            bm25_progress = (bm25_processed / total_messages) * 100
            
            background_tasks[task_id]["stages"]["embedding"]["progress"] = embedding_progress
            background_tasks[task_id]["stages"]["bm25"]["progress"] = bm25_progress
            
            # Overall progress (average of both)
            overall_progress = (embedding_progress + bm25_progress) / 200  # Normalize to 0-1
            background_tasks[task_id]["progress"] = overall_progress
            
            # Update message
            background_tasks[task_id]["message"] = f"Processing: {batch_end}/{total_messages} messages (Embeddings: {embedding_progress:.1f}%, BM25: {bm25_progress:.1f}%)"
            
            # Save indexes periodically for searchability
            if batch_end % save_interval == 0 or batch_end == total_messages:
                # Save both stores in parallel
                def save_stores():
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        futures = [
                            executor.submit(retriever.vector_store.save, "default"),
                            executor.submit(retriever.bm25_store.save, "default")
                        ]
                        for future in as_completed(futures):
                            future.result()
                    print(f"Saved indexes at {batch_end} messages")
                
                save_stores()
                
                # Save state for resumption (save actual message count, not batch index)
                save_indexing_state({
                    "status": "indexing",
                    "task_id": task_id,
                    "messages": [msg.to_dict() for msg in messages],
                    "current_batch": batch_end,  # Save message count for clearer resumption
                    "progress": background_tasks[task_id]["progress"],
                    "stages": background_tasks[task_id]["stages"]
                })
        
        # Mark stages as completed
        background_tasks[task_id]["stages"]["embedding"]["status"] = "completed"
        background_tasks[task_id]["stages"]["embedding"]["progress"] = 100
        background_tasks[task_id]["stages"]["bm25"]["status"] = "completed" 
        background_tasks[task_id]["stages"]["bm25"]["progress"] = 100
        
        # Final save
        background_tasks[task_id]["stages"]["saving"]["status"] = "processing"
        background_tasks[task_id]["message"] = "Finalizing indexes..."
        
        # Save both stores one final time
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(retriever.vector_store.save, "default"),
                executor.submit(retriever.bm25_store.save, "default")
            ]
            for future in as_completed(futures):
                future.result()
        
        # Get final stats
        stats = retriever.get_stats()
        
        background_tasks[task_id]["stages"]["saving"]["progress"] = 100
        background_tasks[task_id]["stages"]["saving"]["status"] = "completed"
        
        # Mark task as completed
        background_tasks[task_id]["status"] = "completed"
        background_tasks[task_id]["progress"] = 1.0
        background_tasks[task_id]["message"] = f"Successfully indexed {total_messages} messages"
        background_tasks[task_id]["result"] = {
            "messages_indexed": total_messages,
            "total_vectors": stats.get('semantic_vectors', 0),
            "total_bm25_docs": stats.get('bm25_documents', 0)
        }
        
        # Clear indexing state on successful completion
        clear_indexing_state()
        
    except Exception as e:
        background_tasks[task_id]["status"] = "error"
        background_tasks[task_id]["message"] = f"Indexing error: {str(e)}"
        background_tasks[task_id]["progress"] = 0
        print(f"Error in indexing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Check if we need to save state before shutdown
        if shutdown_flag.is_set() and task_id in background_tasks:
            if background_tasks[task_id]["status"] == "processing":
                print("Saving indexing state before shutdown...")
                # State should already be saved periodically, but ensure it's saved
                pass

@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    """Get background task status."""
    # Check both task stores
    if task_id in background_tasks:
        return background_tasks[task_id]
    elif task_id in file_processing_tasks:
        return file_processing_tasks[task_id]
    else:
        raise HTTPException(status_code=404, detail="Task not found")

@app.get("/api/tasks")
async def get_all_tasks():
    """Get all active tasks."""
    all_tasks = []
    
    # Add file processing tasks
    for task_id, task_info in file_processing_tasks.items():
        if task_info.get('status') in ['started', 'processing', 'queued']:
            all_tasks.append({
                'task_id': task_id,
                'type': 'file_processing',
                'filename': task_info.get('filename'),
                'status': task_info.get('status'),
                'progress': task_info.get('progress', 0),
                'message': task_info.get('message', ''),
                'stages': task_info.get('stages', {})
            })
    
    # Add indexing tasks
    for task_id, task_info in background_tasks.items():
        if task_info.get('status') in ['started', 'processing', 'queued']:
            all_tasks.append({
                'task_id': task_id,
                'type': 'indexing',
                'filename': task_info.get('filename'),
                'status': task_info.get('status'),
                'progress': task_info.get('progress', 0),
                'message': task_info.get('message', ''),
                'stages': task_info.get('stages', {})
            })
    
    return {"tasks": all_tasks}

def save_indexing_state(state: dict):
    """Save the current indexing state to disk."""
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        # Write to a temp file first, then rename (atomic operation)
        temp_file = STATE_FILE + '.tmp'
        with open(temp_file, 'w') as f:
            json.dump(state, f, indent=2)
            f.flush()  # Ensure data is written
            os.fsync(f.fileno())  # Force write to disk
        
        # Windows-compatible file replacement
        # First try os.replace (works on most systems)
        try:
            os.replace(temp_file, STATE_FILE)
        except (OSError, PermissionError):
            # On Windows, if file is locked, try alternative approach
            import shutil
            max_retries = 3
            for retry in range(max_retries):
                try:
                    # Try to remove the old file first
                    if os.path.exists(STATE_FILE):
                        try:
                            os.remove(STATE_FILE)
                        except:
                            pass  # File might be locked, we'll overwrite it
                    # Use shutil.move which is more robust on Windows
                    shutil.move(temp_file, STATE_FILE)
                    break
                except Exception as e:
                    if retry == max_retries - 1:
                        # Last retry failed, just copy the content
                        with open(temp_file, 'r') as src:
                            content = src.read()
                        with open(STATE_FILE, 'w') as dst:
                            dst.write(content)
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                    else:
                        time.sleep(0.1)  # Brief pause before retry
        
        print(f"✓ Saved indexing state: {state.get('filename', 'unknown')} - {len(state.get('messages', []))} messages, batch {state.get('current_batch', 0)}")
    except Exception as e:
        print(f"✗ ERROR saving indexing state: {e}")
        import traceback
        traceback.print_exc()

def clear_indexing_state():
    """Clear the saved indexing state."""
    if os.path.exists(STATE_FILE):
        try:
            os.remove(STATE_FILE)
            print("✓ Cleared indexing state file")
        except Exception as e:
            print(f"✗ Warning: Could not clear indexing state: {e}")

def resume_indexing_task(task_id: str, state: dict):
    """Resume an interrupted indexing task."""
    try:
        # Wait for system initialization
        max_wait = 30
        wait_time = 0
        while not system_components['initialized'] and wait_time < max_wait:
            time.sleep(1)
            wait_time += 1
        
        if not system_components['initialized']:
            background_tasks[task_id]["status"] = "error"
            background_tasks[task_id]["message"] = "System initialization failed"
            return
        
        # Reconstruct messages from saved state
        messages = []
        for msg_data in state.get('messages', []):
            messages.append(ChatMessage(
                text=msg_data['text'],
                sender=msg_data['sender'],
                timestamp=datetime.fromisoformat(msg_data['timestamp']),
                message_id=msg_data.get('message_id'),
                metadata=msg_data.get('metadata', {})
            ))
        
        if not messages:
            background_tasks[task_id]["status"] = "error"
            background_tasks[task_id]["message"] = "No messages found to resume"
            clear_indexing_state()
            return
        
        # Calculate the starting batch from saved state
        current_messages_done = state.get('current_batch', 0)  # This is the message count
        batch_size = 50  # Must match the batch_size in index_messages_multithreaded
        start_batch_index = current_messages_done // batch_size  # Calculate which batch to start from
        rebuild = state.get('rebuild', False)  # Get rebuild flag from saved state
        
        print(f"Resuming indexing from batch {start_batch_index} (message {current_messages_done}/{len(messages)})")
        print(f"  Rebuild mode: {rebuild}")
        
        # Resume indexing from where we left off, passing rebuild flag
        index_messages_multithreaded(task_id, messages, start_batch=start_batch_index, rebuild=rebuild)
        # Note: clear_indexing_state() is called by index_messages_multithreaded on completion
        
    except Exception as e:
        background_tasks[task_id]["status"] = "error"
        background_tasks[task_id]["message"] = f"Error resuming: {str(e)}"
        clear_indexing_state()

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Shutdown handler
@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown of background tasks."""
    print("\nShutting down ChatWhiz server...")
    shutdown_flag.set()
    
    # Give threads a moment to save state
    await asyncio.sleep(1)
    
    # Shutdown the thread pool executor
    executor.shutdown(wait=False)
    
    print("ChatWhiz server shutdown complete.")

def handle_signal(signum, frame):
    """Handle termination signals."""
    import signal
    print("\nReceived termination signal, initiating shutdown...")
    shutdown_flag.set()
    
    # Give threads time to save state
    time.sleep(2)
    
    # Force exit if needed
    sys.exit(0)

if __name__ == "__main__":
    import uvicorn
    import signal
    
    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, shutting down...")
        shutdown_flag.set()
        executor.shutdown(wait=False)
