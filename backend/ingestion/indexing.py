import os
import uuid
import pickle
import logging
import re
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi

from backend.ingestion.embeddings import get_embeddings_batch

logger = logging.getLogger(__name__)

from backend.database.qdrant_db import qdrant_client, save_qdrant_snapshot

BM25_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database", "bm25_indexes")
os.makedirs(BM25_DIR, exist_ok=True)

def init_qdrant_collections():
    """
    Creates Qdrant collections if they do not exist.
    """
    # nomic-embed-text size is 768
    vector_size = 768
    
    # parent_sections
    if not qdrant_client.collection_exists("parent_sections"):
        logger.info("Creating parent_sections collection in Qdrant...")
        qdrant_client.create_collection(
            collection_name="parent_sections",
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        
    # child_chunks
    if not qdrant_client.collection_exists("child_chunks"):
        logger.info("Creating child_chunks collection in Qdrant...")
        qdrant_client.create_collection(
            collection_name="child_chunks",
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

def bm25_tokenize(text: str) -> List[str]:
    """
    Tokenizes text for BM25 keyword matching (lowercase, alphanumeric words).
    """
    return re.findall(r'\w+', text.lower())

def index_pdf_content(pdf_id: int, structured_sections: List[Dict[str, Any]], model_name: str = "nomic-embed-text:latest", fit_bm25: bool = True):
    """
    Indexes parent sections and child chunks in Qdrant, and optionally builds BM25 keyword indexes on disk.
    """
    init_qdrant_collections()
    
    parent_points = []
    child_points = []
    
    parent_summaries = []
    parent_metadata = []
    
    child_contents = []
    child_metadata = []
    
    # 1. First Pass: Prepare texts for batch embedding
    all_parent_texts = [sec["summary"] for sec in structured_sections]
    
    all_child_texts = []
    all_child_info = [] # Keep mapping: (parent_uuid, page_no)
    
    for sec in structured_sections:
        parent_uuid = uuid.uuid4().hex
        sec["uuid"] = parent_uuid
        
        for child in sec["children"]:
            all_child_texts.append(child["content"])
            all_child_info.append((parent_uuid, child["page_number"]))
            
    logger.info(f"PDF {pdf_id}: Embedding {len(all_parent_texts)} parents and {len(all_child_texts)} children...")
    
    # 2. Get batch embeddings
    parent_embeddings = get_embeddings_batch(all_parent_texts, model_name=model_name)
    child_embeddings = get_embeddings_batch(all_child_texts, model_name=model_name)
    
    # 3. Create Qdrant Point objects
    # Index Parent Sections
    for idx, sec in enumerate(structured_sections):
        parent_uuid = sec["uuid"]
        vector = parent_embeddings[idx]
        
        payload = {
            "pdf_id": pdf_id,
            "heading": sec["heading"],
            "summary": sec["summary"],
            "page_numbers": sec["page_numbers"]
        }
        
        parent_points.append(
            PointStruct(id=parent_uuid, vector=vector, payload=payload)
        )
        
        parent_summaries.append(sec["summary"])
        parent_metadata.append({
            "id": parent_uuid,
            "heading": sec["heading"],
            "summary": sec["summary"],
            "page_numbers": sec["page_numbers"]
        })
        
    # Index Child Chunks
    child_idx = 0
    for sec in structured_sections:
        parent_uuid = sec["uuid"]
        for child in sec["children"]:
            child_uuid = uuid.uuid4().hex
            vector = child_embeddings[child_idx]
            
            payload = {
                "pdf_id": pdf_id,
                "parent_id": parent_uuid,
                "content": child["content"],
                "page_number": child["page_number"]
            }
            
            child_points.append(
                PointStruct(id=child_uuid, vector=vector, payload=payload)
            )
            
            child_contents.append(child["content"])
            child_metadata.append({
                "id": child_uuid,
                "parent_id": parent_uuid,
                "content": child["content"],
                "page_number": child["page_number"]
            })
            child_idx += 1
            
    # 4. Upsert vectors to Qdrant
    logger.info(f"PDF {pdf_id}: Upserting {len(parent_points)} parent sections to Qdrant...")
    qdrant_client.upsert(collection_name="parent_sections", points=parent_points)
    
    logger.info(f"PDF {pdf_id}: Upserting {len(child_points)} child chunks to Qdrant...")
    qdrant_client.upsert(collection_name="child_chunks", points=child_points)
    
    # 5. Index in SQLite FTS5 Virtual Tables
    logger.info(f"PDF {pdf_id}: Indexing text in SQLite FTS5 virtual tables...")
    from backend.database.sqlite_db import add_fts_parent_sections, add_fts_child_chunks
    import json
    
    fts_parents = []
    for sec in parent_metadata:
        fts_parents.append((
            pdf_id,
            sec["id"],
            sec["heading"],
            sec["summary"],
            json.dumps(sec["page_numbers"])
        ))
        
    fts_children = []
    for chunk in child_metadata:
        fts_children.append((
            pdf_id,
            chunk["id"],
            chunk["parent_id"],
            chunk["content"],
            chunk["page_number"]
        ))
        
    if fts_parents:
        add_fts_parent_sections(fts_parents)
    if fts_children:
        add_fts_child_chunks(fts_children)
        
    # Persist to disk
    save_qdrant_snapshot()
    logger.info(f"PDF {pdf_id}: Ingestion indexing completed successfully.")
    
def delete_pdf_vectors(pdf_id: int):
    """
    Deletes all vector records and FTS5 indexes associated with a PDF.
    """
    logger.info(f"Deleting vector indexes for PDF: {pdf_id}")
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # Delete from parent_sections
        qdrant_client.delete(
            collection_name="parent_sections",
            points_selector=Filter(
                must=[FieldCondition(key="pdf_id", match=MatchValue(value=pdf_id))]
            )
        )
        
        # Delete from child_chunks
        qdrant_client.delete(
            collection_name="child_chunks",
            points_selector=Filter(
                must=[FieldCondition(key="pdf_id", match=MatchValue(value=pdf_id))]
            )
        )
        
        # Delete FTS5 database entries
        from backend.database.sqlite_db import delete_fts_data
        delete_fts_data(pdf_id)
            
        # Persist to disk
        save_qdrant_snapshot()
        logger.info(f"Deleted vector indexes and FTS5 entries for PDF: {pdf_id}")
    except Exception as e:
        logger.error(f"Error deleting vector data for PDF {pdf_id}: {str(e)}")
