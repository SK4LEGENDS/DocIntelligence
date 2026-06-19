import logging
import ollama
from typing import List

logger = logging.getLogger(__name__)

def evaluate_evidence_sufficiency(query: str, context: str, model_name: str = "qwen2.5:7b") -> bool:
    """
    Evaluates if the retrieved context contains enough specific evidence to answer the query.
    Returns True if sufficient, False otherwise.
    """
    if not context or "No matching content" in context:
        return False
        
    prompt = (
        "You are a strict factual validation engine. Analyze the User Query and the Retrieved Context below.\n"
        "Evaluate whether the context contains enough specific, verified details to fully and accurately answer the user's question. "
        "Do not make assumptions, extrapolate, or bring in outside knowledge.\n\n"
        f"User Query: {query}\n\n"
        f"Retrieved Context:\n{context}\n\n"
        "Format your output:\n"
        "Line 1: Either 'YES' or 'NO'\n"
        "Line 2: A one-sentence explanation of why the context is sufficient or what specific information is missing."
    )
    
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0} # Deterministic
        )
        content = response["message"]["content"].strip()
        first_line = content.split("\n")[0].upper()
        
        logger.info(f"Sufficiency Evaluation: {content}")
        return "YES" in first_line
    except Exception as e:
        logger.error(f"Error in evaluate_evidence_sufficiency: {str(e)}")
        # Default fallback to True to prevent infinite loops if LLM errors out
        return True

def generate_expanded_query(query: str, previous_queries: List[str], model_name: str = "qwen2.5:7b") -> str:
    """
    Uses the selected model to rewrite the user query into a broader or alternative search query
    to retrieve missing details in the next RAG iteration.
    """
    prompt = (
        "You are a search query optimizer. We are searching a PDF document. "
        "The previous search queries failed to find enough evidence to answer the user's question.\n\n"
        f"Original User Question: {query}\n"
        f"Failed Search Queries: {previous_queries}\n\n"
        "Generate a new, alternative search query using different keywords, synonyms, or a broader scope "
        "that is likely to locate the relevant section of the document. "
        "Do not include any greeting or explanation. Return ONLY the search query text."
    )
    
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.5}
        )
        expanded_query = response["message"]["content"].strip()
        logger.info(f"Query expanded from '{query}' to '{expanded_query}'")
        return expanded_query
    except Exception as e:
        logger.error(f"Error generating expanded query: {str(e)}")
        return query # Fallback to original
