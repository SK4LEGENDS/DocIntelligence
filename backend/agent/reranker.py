import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class LocalReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = None
        self.model_name = model_name
        logger.info(f"Initializing local CrossEncoder reranker with model {model_name}...")
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            logger.info("Successfully loaded CrossEncoder reranker model.")
        except Exception as e:
            logger.warning(
                f"Failed to load sentence-transformers CrossEncoder ({str(e)}). "
                f"RAG workflow will fall back to using hybrid RRF scoring directly."
            )

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Reranks a list of chunks based on cross-encoder similarity with the query.
        """
        if not chunks:
            return []
            
        if self.model is None:
            # Fallback: return the top_n chunks directly as sorted by the retriever
            logger.info("Reranker offline; returning top chunks based on hybrid RRF score.")
            return chunks[:top_n]
            
        try:
            # Prepare pairs: [query, chunk_content]
            pairs = [[query, chunk["content"]] for chunk in chunks]
            scores = self.model.predict(pairs)
            
            # Update chunks with cross-encoder scores
            for chunk, score in zip(chunks, scores):
                chunk["rerank_score"] = float(score)
                
            # Sort descending
            reranked_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
            logger.info(f"Successfully reranked {len(chunks)} chunks using CrossEncoder.")
            return reranked_chunks[:top_n]
        except Exception as e:
            logger.error(f"Error during reranking process: {str(e)}. Falling back to direct slice.")
            return chunks[:top_n]

# Singleton instance
_reranker_instance = None

def get_reranker() -> LocalReranker:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = LocalReranker()
    return _reranker_instance
