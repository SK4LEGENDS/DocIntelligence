import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
import sys
import argparse
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("parser_worker")

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.ingestion.parser import parse_pdf
from backend.ingestion.chunker import chunk_document

def main():
    parser = argparse.ArgumentParser(description="Parse PDF using Docling in a separate process.")
    parser.add_argument("--file_path", required=True, help="Path to the PDF file.")
    parser.add_argument("--output_path", required=True, help="Path to write the JSON result.")
    args = parser.parse_args()

    try:
        logger.info(f"Worker starting for file: {args.file_path}")
        doc = parse_pdf(args.file_path)
        
        logger.info("Chunking document...")
        structured_sections = chunk_document(doc)
        
        # Extract tables
        tables = []
        for table in doc.tables:
            page_no = table.prov[0].page_no if table.prov else 1
            try:
                table_md = table.export_to_markdown(doc=doc)
                tables.append({"page_number": page_no, "markdown": table_md})
            except Exception as table_err:
                logger.warning(f"Could not export table on page {page_no}: {str(table_err)}")
        
        result = {
            "sections": structured_sections,
            "tables": tables
        }
        
        # Write to JSON
        logger.info(f"Writing parsed results to {args.output_path}...")
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        logger.info("Worker completed successfully!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Worker failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
