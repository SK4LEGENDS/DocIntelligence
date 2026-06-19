import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
import shutil
import logging
import json
import sqlite3
import ollama
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.database.sqlite_db import (
    init_db, add_pdf, update_pdf_status, update_pdf_page_count, get_pdfs, get_pdf,
    add_pdf_table, create_chat, get_chats, get_chat, add_message, get_chat_messages
)
from backend.ingestion.parser import parse_pdf, parse_pdf_fast
from backend.ingestion.chunker import chunk_document
from backend.ingestion.indexing import index_pdf_content, delete_pdf_vectors
from backend.agent.graph import run_agent_stream

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Enable uvloop for high performance asyncio on supported OS (Linux/macOS)
import asyncio
if os.name != "nt":
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        logger.info("Successfully applied uvloop EventLoopPolicy.")
    except ImportError:
        logger.info("uvloop not installed, using default event loop.")

# Initialize FastAPI
app = FastAPI(title="Hierarchical Agentic PDF Chatbot API")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Initialize Relational DB
init_db()

# Background Ingestion Pipeline Task
def process_pdf_background(pdf_id: int, file_path: str, mode: str = "auto"):
    """
    Background worker task running natively in FastAPI thread pool:
    1. Read PDF page count using PyMuPDF (fitz).
    2. Route to Fast Mode (PyMuPDF) or Advanced Mode (Docling).
    3. Update SQLite database with dynamic statuses.
    4. Save tables (Advanced mode only) and index sections/chunks in Qdrant.
    5. Fall back to Fast Mode automatically if Advanced Mode fails (e.g. out of memory).
    """
    import fitz
    
    logger.info(f"Background task starting for PDF ID {pdf_id} using mode '{mode}'...")
    
    try:
        # Determine page count
        doc_fitz = fitz.open(file_path)
        page_count = len(doc_fitz)
        doc_fitz.close()
        
        logger.info(f"PDF ID {pdf_id} has {page_count} pages.")
        update_pdf_page_count(pdf_id, page_count)
        
        # Decide mode automatically if mode is "auto"
        selected_mode = mode
        if mode == "auto":
            selected_mode = "fast" if page_count < 100 else "advanced"
            logger.info(f"Auto-selected mode '{selected_mode}' based on page count {page_count}.")
            
        run_fast = (selected_mode == "fast")
        
        if not run_fast:
            try:
                logger.info(f"Running Advanced Ingestion (Docling) for PDF ID {pdf_id}...")
                
                # Step 1: Conversion using Docling
                update_pdf_status(pdf_id, "parsing: converting document")
                doc = parse_pdf(file_path)
                
                # Step 2: Semantic chunking
                update_pdf_status(pdf_id, "parsing: chunking sections")
                structured_sections = chunk_document(doc)
                
                # Step 3: Table extraction & saving to Relational DB
                update_pdf_status(pdf_id, "parsing: extracting tables")
                tables = []
                for table in doc.tables:
                    page_no = table.prov[0].page_no if table.prov else 1
                    try:
                        table_md = table.export_to_markdown(doc=doc)
                        tables.append({"page_number": page_no, "markdown": table_md})
                    except Exception as table_err:
                        logger.warning(f"Could not export table on page {page_no}: {str(table_err)}")
                        
                logger.info(f"Saving {len(tables)} extracted tables to SQLite database...")
                for table in tables:
                    add_pdf_table(pdf_id, table["page_number"], table["markdown"])
                    
                # Step 4: Hybrid Indexing (Dense + Sparse BM25)
                update_pdf_status(pdf_id, "parsing: indexing vectors")
                logger.info(f"Indexing PDF {pdf_id} in Qdrant & BM25...")
                index_pdf_content(pdf_id, structured_sections, fit_bm25=True)
                
            except Exception as adv_err:
                logger.error(
                    f"Advanced Ingestion failed for PDF {pdf_id}: {str(adv_err)}. "
                    f"Falling back to Fast Ingestion (PyMuPDF) for safety...",
                    exc_info=True
                )
                run_fast = True
                
        if run_fast:
            logger.info(f"Running Fast Ingestion (PyMuPDF) for PDF ID {pdf_id}...")
            
            # Step 1: Text extraction & chunking
            update_pdf_status(pdf_id, "parsing: extracting text")
            structured_sections = parse_pdf_fast(file_path)
            
            # Step 2: Dense + Sparse Indexing
            update_pdf_status(pdf_id, "parsing: indexing vectors")
            logger.info(f"Indexing PDF {pdf_id} in Qdrant (fast)...")
            index_pdf_content(pdf_id, structured_sections, fit_bm25=False)
            
        # Complete
        update_pdf_status(pdf_id, "ready")
        logger.info(f"Background ingestion completed successfully for PDF ID {pdf_id}!")
        
    except Exception as e:
        logger.error(f"Error in background processing for PDF ID {pdf_id}: {str(e)}", exc_info=True)
        update_pdf_status(pdf_id, "error", str(e))

# --- API ROUTES ---

@app.post("/api/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...), mode: str = Form("auto")):
    """
    Uploads a PDF file and starts background parsing/indexing.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    try:
        # Save file to disk
        filename = file.filename
        filepath = os.path.join(UPLOADS_DIR, f"{uuid_name(filename)}")
        
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Add entry in SQLite
        pdf_id = add_pdf(filename, filepath)
        
        # Queue background processing
        background_tasks.add_task(process_pdf_background, pdf_id, filepath, mode)
        
        return {"id": pdf_id, "filename": filename, "status": "parsing"}
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

def uuid_name(filename: str) -> str:
    import uuid
    ext = os.path.splitext(filename)[1]
    return f"{uuid.uuid4().hex}{ext}"

@app.get("/api/pdfs")
async def list_pdfs():
    """
    Lists all uploaded PDFs and their ingestion status.
    """
    return get_pdfs()

@app.get("/api/pdfs/{pdf_id}/file")
async def get_pdf_file(pdf_id: int):
    """
    Serves the raw PDF binary file.
    """
    pdf = get_pdf(pdf_id)
    if not pdf:
        raise HTTPException(status_code=404, detail="PDF not found")
    if not os.path.exists(pdf["filepath"]):
        raise HTTPException(status_code=404, detail="PDF file not found on disk")
        
    # Return file with inline headers so it renders inside the browser iframe instead of downloading
    headers = {"Content-Disposition": f'inline; filename="{pdf["filename"]}"'}
    return FileResponse(
        pdf["filepath"],
        media_type="application/pdf",
        headers=headers
    )

@app.get("/api/pdfs/{pdf_id}/tables")
async def get_tables_endpoint(pdf_id: int):
    """
    Retrieves all tables extracted from the PDF by Docling.
    """
    from backend.database.sqlite_db import get_pdf_tables
    pdf = get_pdf(pdf_id)
    if not pdf:
        raise HTTPException(status_code=404, detail="PDF not found")
    return get_pdf_tables(pdf_id)

@app.delete("/api/pdfs/{pdf_id}")
async def delete_pdf(pdf_id: int):
    """
    Deletes a PDF, its vector entries, database records, and files.
    """
    pdf = get_pdf(pdf_id)
    if not pdf:
        raise HTTPException(status_code=404, detail="PDF not found")
        
    # Remove from disk
    if os.path.exists(pdf["filepath"]):
        os.remove(pdf["filepath"])
        
    # Delete vectors & BM25
    delete_pdf_vectors(pdf_id)
    
    # Delete SQLite records (Cascaded)
    conn = sqlite3.connect(os.path.join(BASE_DIR, "chatbot.db"))
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute("DELETE FROM pdfs WHERE id = ?", (pdf_id,))
    conn.commit()
    conn.close()
    
    return {"status": "success", "message": f"Successfully deleted PDF {pdf_id}."}

# CHAT ENDPOINTS
@app.post("/api/chats")
async def start_chat(pdf_id: int = Form(...), title: str = Form(...)):
    """
    Creates a new chat session linked to a PDF.
    """
    pdf = get_pdf(pdf_id)
    if not pdf:
        raise HTTPException(status_code=404, detail="Linked PDF not found")
    if pdf["status"] != "ready":
        raise HTTPException(status_code=400, detail="PDF must be fully processed and 'ready' before chatting")
        
    chat_id = create_chat(pdf_id, title)
    return {"id": chat_id, "pdf_id": pdf_id, "title": title}

@app.get("/api/chats")
async def list_chats():
    """
    Lists all chats.
    """
    return get_chats()

@app.get("/api/models")
async def list_models():
    """
    Retrieves all available chat models pulled in local Ollama.
    """
    try:
        response = ollama.list()
        # Filter for models that support chat (exclude embedding models)
        chat_models = []
        for model in response.models:
            name = model.model
            if "embed" not in name:
                chat_models.append(name)
        # Sort so that qwen2.5/qwen3 models are preferred as defaults if they exist
        chat_models.sort(key=lambda x: ("qwen" not in x.lower(), x))
        return chat_models
    except Exception as e:
        logger.error(f"Error fetching Ollama models: {str(e)}")
        return ["qwen2.5:7b"] # Fallback

@app.get("/api/chats/{chat_id}/messages")
async def get_messages(chat_id: int):
    """
    Returns message history for a chat.
    """
    chat = get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return get_chat_messages(chat_id)

@app.post("/api/chats/{chat_id}/message")
async def send_chat_message(chat_id: int, content: str = Form(...), model: str = Form("qwen2.5:7b")):
    """
    Sends a message to the chat and returns the streaming response from the agent.
    """
    chat = get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat session not found")
        
    pdf_id = chat["pdf_id"]
    
    # 1. Save User Message
    add_message(chat_id, "user", content)
    
    # 2. Get chat history
    history = get_chat_messages(chat_id)
    
    # 3. Create stream wrapper to save assistant response when complete
    def sse_generator():
        full_answer = []
        try:
            for sse_msg in run_agent_stream(pdf_id, content, history, model_name=model):
                # If it's a token, accumulate it
                if "type\": \"token" in sse_msg:
                    # Extract token from json structure
                    try:
                        # Event is e.g. data: {"type": "token", "content": "hello"}
                        data_part = sse_msg.replace("data: ", "").strip()
                        obj = json.loads(data_part)
                        full_answer.append(obj["content"])
                    except Exception:
                        pass
                yield sse_msg
                
            # Stream complete, save final message in DB
            assistant_response = "".join(full_answer)
            if assistant_response.strip():
                add_message(chat_id, "assistant", assistant_response)
                logger.info(f"Saved assistant response to DB for chat {chat_id}.")
        except Exception as err:
            logger.error(f"Error in SSE streamer for chat {chat_id}: {str(err)}", exc_info=True)
            yield f"data: {json.dumps({'type': 'token', 'content': f'Stream crashed: {str(err)}'})}\n\n"
            
    return StreamingResponse(sse_generator(), media_type="text/event-stream")

# Mount Static Files (Production Frontend)
frontend_dist = os.path.join(os.path.dirname(BASE_DIR), "frontend", "dist")
if os.path.exists(frontend_dist):
    logger.info(f"Mounting static files from {frontend_dist}...")
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
else:
    logger.warning(
        f"Frontend distribution folder not found at {frontend_dist}. "
        f"FastAPI will only serve the API routes. Start the Vite React dev server independently."
    )
