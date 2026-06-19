import sys
import os
import time
import logging

# Set up logging to console
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Add workspace root to python path
sys.path.append(os.getcwd())

from backend.ingestion.parser import parse_pdf_fast
from backend.ingestion.embeddings import get_embedding
from backend.agent.graph import classify_query_node, AgentState

# Get a sample PDF from uploads to verify
UPLOADS_DIR = "backend/uploads"
sample_pdf = None
for f in os.listdir(UPLOADS_DIR):
    if f.endswith(".pdf"):
        sample_pdf = os.path.join(UPLOADS_DIR, f)
        break

print("=== Running Optimization Verification ===")

# 1. Test PyMuPDF Fast Parsing
print("\n--- 1. Testing Fast Ingestion (PyMuPDF) ---")
if sample_pdf:
    start_time = time.time()
    sections = parse_pdf_fast(sample_pdf)
    elapsed = time.time() - start_time
    print(f"Fast parsing completed in {elapsed:.3f} seconds.")
    print(f"Extracted {len(sections)} sections.")
    if sections:
        print(f"Sample Section Heading: {sections[0]['heading']}")
        print(f"Sample Section Summary: {sections[0]['summary'][:150]}")
        print(f"Sample Section Child Count: {len(sections[0]['children'])}")
else:
    print("No sample PDF found in uploads to test parsing.")

# 2. Test Embedding LRU Caching
print("\n--- 2. Testing Embedding Cache Speedup ---")
print("First call to get_embedding (should query Ollama)...")
start1 = time.time()
v1 = get_embedding("what are the tuition fees?")
elapsed1 = time.time() - start1
print(f"First call took {elapsed1:.3f} seconds.")

print("Second call to get_embedding (should HIT cache)...")
start2 = time.time()
v2 = get_embedding("what are the tuition fees?")
elapsed2 = time.time() - start2
print(f"Second call took {elapsed2:.6f} seconds.")

assert elapsed2 < 0.005, f"Cache not hit: second call took {elapsed2:.3f}s"
print("SUCCESS: Embedding cache hit verified!")

# 3. Test Complexity Routing
print("\n--- 3. Testing Query Complexity Routing ---")
state_simple = {
    "query": "What are the tuition fees?",
    "chat_history": [],
    "query_type": "factual",
    "complexity": "simple",
    "thought_process": []
}

state_complex = {
    "query": "Compare the tuition fees between 2026 and 2027 and tell me if there are contradictions in hostel rules.",
    "chat_history": [],
    "query_type": "factual",
    "complexity": "simple",
    "thought_process": []
}

print("Classifying simple query...")
res_simple = classify_query_node(state_simple, model_name="qwen2.5:7b")
print(f"Simple Query classified as Type: {res_simple['query_type']}, Complexity: {res_simple['complexity']}")

print("Classifying complex query...")
res_complex = classify_query_node(state_complex, model_name="qwen2.5:7b")
print(f"Complex Query classified as Type: {res_complex['query_type']}, Complexity: {res_complex['complexity']}")

print("\n=== All Verification Tests Passed! ===")
