import logging
from typing import List, Dict, Any
from qdrant_client.models import Filter, FieldCondition, MatchValue

from backend.agent.retriever import retrieve_hierarchical_context
from backend.database.qdrant_db import qdrant_client
from backend.database.sqlite_db import get_pdf, get_pdf_tables

logger = logging.getLogger(__name__)

def search_pdf_content(pdf_id: int, query: str) -> str:
    """
    Searches the PDF content using hierarchical parent-child hybrid search.
    Returns the top matching chunks as a formatted string with page numbers.
    """
    logger.info(f"Tool Executing: search_pdf_content for PDF {pdf_id} with query '{query}'")
    try:
        chunks = retrieve_hierarchical_context(pdf_id, query, top_sections=3, top_chunks=5)
        if not chunks:
            return "No matching content found in the PDF for this query."
            
        formatted_results = []
        for idx, chunk in enumerate(chunks):
            formatted_results.append(
                f"--- Result {idx+1} (Page {chunk['page_number']}, Section: {chunk['heading']}) ---\n"
                f"{chunk['content']}\n"
            )
        return "\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Error in search_pdf_content tool: {str(e)}")
        return f"Error searching PDF content: {str(e)}"

def read_page(pdf_id: int, page_number: int) -> str:
    """
    Retrieves the complete parsed text content of a specific page from the PDF.
    """
    logger.info(f"Tool Executing: read_page for PDF {pdf_id}, page {page_number}")
    try:
        # Scroll child chunks to fetch all chunks on this page
        # Note: scroll returns a tuple (points, offset)
        points, _ = qdrant_client.scroll(
            collection_name="child_chunks",
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="pdf_id", match=MatchValue(value=pdf_id)),
                    FieldCondition(key="page_number", match=MatchValue(value=page_number))
                ]
            ),
            limit=100
        )
        
        if not points:
            return f"No content found on Page {page_number}."
            
        # Sort chunks by order if available, otherwise just concatenate them
        # Let's concatenate text
        chunk_texts = [p.payload["content"] for p in points]
        page_text = "\n\n".join(chunk_texts)
        return f"--- Content of Page {page_number} ---\n{page_text}"
    except Exception as e:
        logger.error(f"Error in read_page tool: {str(e)}")
        return f"Error reading page content: {str(e)}"

def get_pdf_metadata(pdf_id: int) -> str:
    """
    Returns metadata about the PDF (filename, total pages, number of tables extracted).
    """
    logger.info(f"Tool Executing: get_pdf_metadata for PDF {pdf_id}")
    try:
        pdf = get_pdf(pdf_id)
        if not pdf:
            return "PDF not found."
            
        # Get page count by scrolling or checking max page number in chunks
        points, _ = qdrant_client.scroll(
            collection_name="child_chunks",
            scroll_filter=Filter(
                must=[FieldCondition(key="pdf_id", match=MatchValue(value=pdf_id))]
            ),
            limit=1000
        )
        page_numbers = {p.payload["page_number"] for p in points} if points else set()
        page_count = max(page_numbers) if page_numbers else 1
        
        # Get table count from SQLite
        tables = get_pdf_tables(pdf_id)
        table_count = len(tables)
        
        return (
            f"PDF Metadata:\n"
            f"- ID: {pdf_id}\n"
            f"- Filename: {pdf['filename']}\n"
            f"- Estimated Pages: {page_count}\n"
            f"- Extracted Tables: {table_count}\n"
            f"- Uploaded At: {pdf['created_at']}\n"
        )
    except Exception as e:
        logger.error(f"Error in get_pdf_metadata tool: {str(e)}")
        return f"Error fetching PDF metadata: {str(e)}"

def get_tables(pdf_id: int) -> str:
    """
    Retrieves all tables extracted from the PDF formatted as markdown.
    """
    logger.info(f"Tool Executing: get_tables for PDF {pdf_id}")
    try:
        tables = get_pdf_tables(pdf_id)
        if not tables:
            return "No tables were found or extracted from this PDF."
            
        formatted_tables = []
        for idx, table in enumerate(tables):
            formatted_tables.append(
                f"--- Table {idx+1} (Page {table['page_number']}) ---\n"
                f"{table['markdown']}\n"
            )
        return "\n".join(formatted_tables)
    except Exception as e:
        logger.error(f"Error in get_tables tool: {str(e)}")
        return f"Error retrieving tables: {str(e)}"
