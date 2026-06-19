import logging
import time
import functools
from typing import List
from ollama import Client

logger = logging.getLogger(__name__)

# Use custom client with high timeout to prevent HTTP timeout errors
ollama_client = Client(timeout=300.0)

@functools.lru_cache(maxsize=1024)
def get_embedding(text: str, model_name: str = "nomic-embed-text:latest") -> List[float]:
    """
    Generates a single dense vector embedding for the input text using local Ollama.
    """
    try:
        truncated_text = text[:6000] if text.strip() else "Empty chunk."
        response = ollama_client.embed(model=model_name, input=truncated_text)
        return response["embeddings"][0]
    except Exception as e:
        logger.error(f"Error generating embedding for text: {str(e)}")
        raise e

def get_embeddings_batch(texts: List[str], model_name: str = "nomic-embed-text:latest", batch_size: int = 32) -> List[List[float]]:
    """
    Generates dense vector embeddings for a list of texts in batches to prevent HTTP timeouts.
    """
    embeddings = []
    total_texts = len(texts)
    logger.info(f"Generating embeddings for {total_texts} texts in batches of {batch_size} using model: {model_name}...")
    
    for i in range(0, total_texts, batch_size):
        batch = texts[i : i + batch_size]
        retries = 3
        while retries > 0:
            try:
                # Clean texts to remove empty content and truncate to 6000 chars to avoid context length limits
                cleaned_batch = [t[:6000] if t.strip() else "Empty chunk." for t in batch]
                response = ollama_client.embed(model=model_name, input=cleaned_batch)
                embeddings.extend(response["embeddings"])
                break
            except Exception as e:
                retries -= 1
                logger.warning(f"Embedding batch {i//batch_size + 1} failed. Retries left: {retries}. Error: {str(e)}")
                if retries == 0:
                    logger.error(f"Failed embedding batch after retries.")
                    raise e
                time.sleep(2) # Backoff
                
    logger.info(f"Successfully generated {len(embeddings)} embeddings.")
    return embeddings
