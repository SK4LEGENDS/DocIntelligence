import logging
from typing import List, Dict, Any
from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

def parse_pdf(file_path: str):
    """
    Parses a PDF file using IBM's Docling tool and returns the structured DoclingDocument.
    """
    logger.info(f"Starting Docling conversion for: {file_path}")
    try:
        converter = DocumentConverter()
        result = converter.convert(file_path)
        logger.info(f"Successfully converted PDF: {file_path}")
        return result.document
    except Exception as e:
        logger.error(f"Failed to convert PDF {file_path}: {str(e)}", exc_info=True)
        raise e

def parse_pdf_fast(file_path: str) -> List[Dict[str, Any]]:
    """
    Fast parsing using PyMuPDF (fitz) for PDFs < 50 pages.
    Groups paragraphs/lines into chunks and returns structured sections compatible with indexing.
    Uses horizontal span consolidation to handle absolute-positioned template overlays correctly.
    """
    import fitz
    logger.info(f"Starting PyMuPDF fast extraction for: {file_path}")
    
    def extract_clean_text(page) -> str:
        blocks = page.get_text("dict")["blocks"]
        spans = []
        
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        bbox = span["bbox"]
                        y_center = (bbox[1] + bbox[3]) / 2.0
                        spans.append({
                            "x0": bbox[0],
                            "y_center": y_center,
                            "text": span["text"]
                        })
                        
        if not spans:
            return ""
            
        # Group spans into lines by y_center tolerance (3.0 points)
        lines = []
        spans.sort(key=lambda s: (s["y_center"], s["x0"]))
        
        current_line = []
        current_y = None
        
        for span in spans:
            if current_y is None:
                current_y = span["y_center"]
                current_line.append(span)
            elif abs(span["y_center"] - current_y) <= 3.0:
                current_line.append(span)
            else:
                current_line.sort(key=lambda s: s["x0"])
                lines.append(current_line)
                current_line = [span]
                current_y = span["y_center"]
                
        if current_line:
            current_line.sort(key=lambda s: s["x0"])
            lines.append(current_line)
            
        # Reconstruct text line by line
        final_lines = []
        for line in lines:
            line_text = ""
            for span in line:
                span_text = span["text"]
                if line_text and not line_text.endswith(" ") and not span_text.startswith(" "):
                    line_text += " "
                line_text += span_text
            final_lines.append(line_text)
            
        return "\n".join(final_lines)

    try:
        doc = fitz.open(file_path)
        structured_sections = []
        
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            text = extract_clean_text(page)
            
            if not text.strip():
                continue
                
            # Split page text into lines
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
            
            chunks = []
            current_chunk = []
            current_len = 0
            
            for p in paragraphs:
                current_chunk.append(p)
                current_len += len(p)
                
                if current_len >= 800: # Target chunk size ~800 characters
                    chunks.append(" \n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                    
            if current_chunk:
                chunks.append(" \n".join(current_chunk))
                
            # Format as a parent section per page
            children = [{"content": chunk, "page_number": page_num} for chunk in chunks]
            
            if children:
                # Fast summary using first 200 characters of the page text
                summary_snippet = text[:200].strip().replace("\n", " ")
                if len(text) > 200:
                    summary_snippet += "..."
                    
                structured_sections.append({
                    "heading": f"Page {page_num}",
                    "summary": f"Content of Page {page_num}: {summary_snippet}",
                    "page_numbers": [page_num],
                    "children": children
                })
                
        doc.close()
        logger.info(f"Successfully parsed PDF {file_path} with {len(structured_sections)} page sections.")
        return structured_sections
    except Exception as e:
        logger.error(f"Failed to parse PDF {file_path} with PyMuPDF: {str(e)}", exc_info=True)
        raise e
