import logging
from typing import List, Dict, Any, Tuple
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker

logger = logging.getLogger(__name__)

def generate_section_summary(section_text: str, heading_path: str) -> str:
    """
    Generates a fast heuristic summary of a document section.
    """
    if not section_text.strip():
        return "Empty section."
    
    snippet = section_text[:800].strip()
    if len(section_text) > 800:
        snippet += "..."
    return f"Section Heading: {heading_path}\nContent: {snippet}"

def chunk_document(doc, model_name: str = "qwen3:8b") -> List[Dict[str, Any]]:
    """
    Performs hierarchical, semantic section chunking on a DoclingDocument.
    Groups chunks into parent sections (by heading paths) and generates section summaries.
    Returns:
        List of dicts: [
            {
                "heading": "Section Heading Path",
                "summary": "LLM section summary",
                "page_numbers": [1, 2],
                "children": [
                    {"content": "chunk text", "page_number": 1},
                    ...
                ]
            }
        ]
    """
    logger.info("Initializing Docling HierarchicalChunker...")
    chunker = HierarchicalChunker(
        merge_list_items=True,
        always_emit_headings=True
    )
    
    doc_chunks = list(chunker.chunk(doc))
    logger.info(f"Hierarchical chunker generated {len(doc_chunks)} base chunks.")
    
    # Group chunks by heading hierarchy
    # Key: Tuple of headings, Value: list of chunks
    sections_map: Dict[Tuple[str, ...], List[Any]] = {}
    
    for chunk in doc_chunks:
        # Extract headings path
        headings = tuple(chunk.meta.headings) if chunk.meta.headings else ("General",)
        
        if headings not in sections_map:
            sections_map[headings] = []
        sections_map[headings].append(chunk)
    
    structured_sections = []
    
    for headings_tuple, chunks_list in sections_map.items():
        heading_path = " > ".join(headings_tuple)
        
        # Assemble children and page numbers
        children = []
        page_numbers_set = set()
        section_text_parts = []
        
        for chunk in chunks_list:
            # Extract page number from chunk provenance
            chunk_pages = sorted(list(set(
                prov.page_no 
                for item in chunk.meta.doc_items 
                for prov in getattr(item, "prov", []) 
                if hasattr(prov, "page_no")
            )))
            page_no = chunk_pages[0] if chunk_pages else 1
            page_numbers_set.update(chunk_pages if chunk_pages else [1])
            
            children.append({
                "content": chunk.text,
                "page_number": page_no
            })
            section_text_parts.append(chunk.text)
            
        full_section_text = "\n".join(section_text_parts)
        page_list = sorted(list(page_numbers_set))
        
        logger.info(f"Summarizing section: {heading_path} (Pages: {page_list}, Chunks: {len(children)})")
        # Generate summary
        summary = generate_section_summary(full_section_text, heading_path)
        
        structured_sections.append({
            "heading": heading_path,
            "summary": summary,
            "page_numbers": page_list,
            "children": children
        })
        
    return structured_sections
