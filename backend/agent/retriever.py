import os
import pickle
import logging
import re
import ollama
from typing import List, Dict, Any
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from backend.ingestion.embeddings import get_embedding
from backend.ingestion.indexing import BM25_DIR, bm25_tokenize
from backend.database.qdrant_db import qdrant_client
from backend.agent.reranker import get_reranker
from backend.database.cache import fts_cache, rerank_cache

logger = logging.getLogger(__name__)

# Module-level caches
BM25_CACHE = {}      # Key: pdf_id, Value: {"parent": parent_bm25_data, "child": child_bm25_data}
RETRIEVAL_CACHE = {} # Key: (pdf_id, query, top_sections, top_chunks, complexity), Value: List[Dict[str, Any]]

def generate_multi_queries(query: str, model_name: str = "qwen2.5:7b") -> List[str]:
    """
    Generates 2 alternative search queries using local Ollama.
    """
    prompt = (
        "You are a search query optimizer. We are retrieving text from a PDF document.\n"
        "Generate exactly 2 alternative search queries using different keywords or synonyms for the question below.\n"
        f"Original Question: {query}\n\n"
        "Return ONLY the 2 alternative queries, one per line. Do not include numbers, bullet points, explanations, or quotes."
    )
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.4}
        )
        content = response["message"]["content"].strip()
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        
        cleaned = []
        for line in lines:
            line = re.sub(r'^[\d\-\*\•\>]+\.?\s*', '', line).strip()
            line = line.strip('"\'')
            if line:
                cleaned.append(line)
        logger.info(f"Generated alternative queries: {cleaned[:2]}")
        return cleaned[:2]
    except Exception as e:
        logger.error(f"Error generating multi-queries: {str(e)}")
        return []

def run_rrf(dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]], key_field: str = "id", c: int = 60) -> List[Dict[str, Any]]:
    """
    Combines dense vector search results and sparse keyword search results using Reciprocal Rank Fusion (RRF).
    """
    rrf_scores = {}
    items_map = {}
    
    # Process dense results
    for rank, item in enumerate(dense_results):
        item_id = item[key_field]
        rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + (1.0 / (c + rank + 1))
        items_map[item_id] = item
        
    # Process sparse results
    for rank, item in enumerate(sparse_results):
        item_id = item[key_field]
        rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + (1.0 / (c + rank + 1))
        if item_id not in items_map:
            items_map[item_id] = item
            
    # Sort by score descending
    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    merged_results = []
    for item_id, score in sorted_ids:
        merged_item = dict(items_map[item_id])
        merged_item["rrf_score"] = score
        merged_results.append(merged_item)
        
    return merged_results

def retrieve_hierarchical_context(
    pdf_id: int, 
    query: str, 
    top_sections: int = 3, 
    top_chunks: int = 5,
    complexity: str = "simple",
    model_name: str = "qwen2.5:7b"
) -> List[Dict[str, Any]]:
    """
    Performs Hierarchical Parent-Child retrieval using Hybrid Dense+Sparse Search:
    1. Detects exact-match/page/section lookup queries to run FTS5 search directly and skip Qdrant.
    2. Runs query complexity routing. For COMPLEX queries, runs multi-query RRF expansion.
    3. Searches parent_sections in Qdrant + cached BM25/FTS5, merges via RRF.
    4. Searches child chunks matching top parent section IDs, merges via RRF.
    5. Reranks child chunks using CrossEncoder reranker (COMPLEX queries only, sliced to top 20 candidates).
    """
    # 0. Check Retrieval Cache
    cache_key = (pdf_id, query, top_sections, top_chunks, complexity)
    if cache_key in RETRIEVAL_CACHE:
        logger.info(f"Retrieval Cache HIT for PDF {pdf_id} with query '{query}'")
        return RETRIEVAL_CACHE[cache_key]

    logger.info(f"RAG Retrieval: Starting query pipeline for PDF {pdf_id} (Complexity: {complexity})...")
    
    # Detect exact FTS-preferred query patterns (e.g., page 5, section 4, clause 8)
    is_exact_lookup = False
    lower_q = query.lower()
    if re.search(r'\b(page|section|clause|chapter)\s+\d+', lower_q) or (lower_q.startswith('"') and lower_q.endswith('"')):
        is_exact_lookup = True
        logger.info(f"FTS5 Direct Shortcut: Detected exact match query pattern: '{query}'. Bypassing vector search.")
        
    global BM25_CACHE
    
    # 1. Multi-Query Formulation for Complex Queries (skip for FTS5 exact lookups)
    all_queries = [query]
    if complexity == "complex" and not is_exact_lookup:
        alt_queries = generate_multi_queries(query, model_name=model_name)
        all_queries.extend(alt_queries)
        logger.info(f"Multi-query expansion enabled. Query pool: {all_queries}")

    # --- LEVEL 1: SEARCH PARENT SECTIONS (Accumulate RRF scores across all queries) ---
    parent_rrf_sums = {}
    parent_items = {}

    for q_idx, q in enumerate(all_queries):
        dense_parents = []
        
        # A. Qdrant Dense Search (Skip if exact lookup)
        if not is_exact_lookup:
            query_vector = get_embedding(q, model_name="nomic-embed-text:latest")
            response_parents = qdrant_client.query_points(
                collection_name="parent_sections",
                query=query_vector,
                query_filter=Filter(
                    must=[FieldCondition(key="pdf_id", match=MatchValue(value=pdf_id))]
                ),
                limit=15
            )
            dense_parents = [
                {
                    "id": hit.id,
                    "heading": hit.payload["heading"],
                    "summary": hit.payload["summary"],
                    "page_numbers": hit.payload["page_numbers"],
                    "score": hit.score
                }
                for hit in response_parents.points
            ]
        
        # B. FTS5 Sparse Search (with caching)
        cache_key_fts_parent = (pdf_id, q, "parents")
        sparse_parents = fts_cache.get(cache_key_fts_parent)
        if sparse_parents is None:
            from backend.database.sqlite_db import query_fts_parent_sections
            fts_res = query_fts_parent_sections(pdf_id, q, limit=15)
            sparse_parents = [
                {
                    "id": r["section_id"],
                    "heading": r["heading"],
                    "summary": r["summary"],
                    "page_numbers": r["page_numbers"],
                    "score": r["score"]
                }
                for r in fts_res
            ]
            fts_cache.set(cache_key_fts_parent, sparse_parents)
            
        # C. Run RRF for this query and accumulate
        q_merged_parents = run_rrf(dense_parents, sparse_parents, key_field="id")
        for item in q_merged_parents:
            item_id = item["id"]
            # Weight original query higher if multi-query is running
            weight = 1.0 if q_idx == 0 else 0.5
            parent_rrf_sums[item_id] = parent_rrf_sums.get(item_id, 0.0) + (item["rrf_score"] * weight)
            parent_items[item_id] = item

    # Sort merged parent list and slice top K
    sorted_parents = sorted(parent_rrf_sums.items(), key=lambda x: x[1], reverse=True)
    top_parent_sections = [parent_items[p_id] for p_id, _ in sorted_parents[:top_sections]]
    
    if not top_parent_sections:
        logger.warning("No parent sections retrieved. Query returned empty results.")
        return []
        
    parent_ids = [p["id"] for p in top_parent_sections]
    parent_headings = [p["heading"] for p in top_parent_sections]
    logger.info(f"Level 1 RAG Complete. Top parent sections: {parent_headings}")
    
    # --- LEVEL 2: SEARCH CHILD CHUNKS WITHIN SECTIONS ---
    child_rrf_sums = {}
    child_items = {}

    for q_idx, q in enumerate(all_queries):
        dense_children = []
        
        # A. Qdrant Dense Search (Skip if exact lookup)
        if not is_exact_lookup:
            query_vector = get_embedding(q, model_name="nomic-embed-text:latest")
            response_children = qdrant_client.query_points(
                collection_name="child_chunks",
                query=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(key="pdf_id", match=MatchValue(value=pdf_id)),
                        FieldCondition(key="parent_id", match=MatchAny(any=parent_ids))
                    ]
                ),
                limit=30
            )
            dense_children = [
                {
                    "id": hit.id,
                    "parent_id": hit.payload["parent_id"],
                    "content": hit.payload["content"],
                    "page_number": hit.payload["page_number"],
                    "score": hit.score
                }
                for hit in response_children.points
            ]
        
        # B. FTS5 Sparse Search (with caching)
        parent_ids_tuple = tuple(parent_ids)
        cache_key_fts_child = (pdf_id, parent_ids_tuple, q, "children")
        sparse_children = fts_cache.get(cache_key_fts_child)
        if sparse_children is None:
            from backend.database.sqlite_db import query_fts_child_chunks
            fts_res = query_fts_child_chunks(pdf_id, parent_ids, q, limit=30)
            sparse_children = [
                {
                    "id": r["chunk_id"],
                    "parent_id": r["parent_id"],
                    "content": r["content"],
                    "page_number": r["page_number"],
                    "score": r["score"]
                }
                for r in fts_res
            ]
            fts_cache.set(cache_key_fts_child, sparse_children)
            
        # C. Run RRF for this query and accumulate
        q_merged_children = run_rrf(dense_children, sparse_children, key_field="id")
        for item in q_merged_children:
            item_id = item["id"]
            weight = 1.0 if q_idx == 0 else 0.5
            child_rrf_sums[item_id] = child_rrf_sums.get(item_id, 0.0) + (item["rrf_score"] * weight)
            child_items[item_id] = item

    # Sort merged children
    sorted_children = sorted(child_rrf_sums.items(), key=lambda x: x[1], reverse=True)
    merged_children = [child_items[c_id] for c_id, _ in sorted_children]
    logger.info(f"Level 2 RAG Complete. Merged {len(merged_children)} child candidates.")
    
    # --- RERANKING STAGE (Only run for COMPLEX queries) ---
    parent_heading_map = {p["id"]: p["heading"] for p in top_parent_sections}
    
    if complexity == "simple" or not merged_children:
        logger.info("Adaptive Retrieval: Skipping CrossEncoder reranking for Simple query.")
        final_grounded_chunks = merged_children[:top_chunks]
    else:
        # Limit reranking to top 20 candidates to optimize CrossEncoder performance
        candidates_to_rerank = merged_children[:20]
        candidate_ids = tuple(c["id"] for c in candidates_to_rerank)
        cache_key_rerank = (query, candidate_ids)
        
        final_grounded_chunks = rerank_cache.get(cache_key_rerank)
        if final_grounded_chunks is None:
            logger.info("Adaptive Retrieval: Scoring child chunks using CrossEncoder reranker...")
            reranker = get_reranker()
            final_grounded_chunks = reranker.rerank(query, candidates_to_rerank, top_n=top_chunks)
            rerank_cache.set(cache_key_rerank, final_grounded_chunks)
    
    # Append parent heading context for final injection
    for chunk in final_grounded_chunks:
        chunk["heading"] = parent_heading_map.get(chunk["parent_id"], "Unknown Section")
        
    logger.info(f"RAG Retrieval Complete. Yielding top {len(final_grounded_chunks)} evidence chunks.")
    
    # Save to Cache
    RETRIEVAL_CACHE[cache_key] = final_grounded_chunks
    return final_grounded_chunks
