import os
import json
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QDRANT_SNAPSHOT_DIR = os.path.join(BASE_DIR, "database", "qdrant_snapshots")
os.makedirs(QDRANT_SNAPSHOT_DIR, exist_ok=True)

# Use in-memory Qdrant client (no file locks, no portalocker issues)
logger.info("Initializing in-memory Qdrant client...")
qdrant_client = QdrantClient(location=":memory:")

# Vector size for nomic-embed-text
VECTOR_SIZE = 768

def _collections_file():
    return os.path.join(QDRANT_SNAPSHOT_DIR, "collections.json")

def save_qdrant_snapshot():
    """Save all Qdrant collections data to disk as JSON for persistence."""
    try:
        snapshot_data = {}
        for collection in qdrant_client.get_collections().collections:
            name = collection.name
            points, _ = qdrant_client.scroll(collection_name=name, limit=100000, with_vectors=True)
            snapshot_data[name] = [
                {
                    "id": p.id,
                    "vector": p.vector,
                    "payload": p.payload
                }
                for p in points
            ]
        
        with open(_collections_file(), "w", encoding="utf-8") as f:
            json.dump(snapshot_data, f)
        logger.info(f"Qdrant snapshot saved: {len(snapshot_data)} collections.")
    except Exception as e:
        logger.error(f"Failed to save Qdrant snapshot: {e}")

def load_qdrant_snapshot():
    """Load Qdrant data from disk snapshot into memory."""
    cfile = _collections_file()
    if not os.path.exists(cfile):
        logger.info("No Qdrant snapshot found on disk. Starting fresh.")
        return
    
    try:
        with open(cfile, "r", encoding="utf-8") as f:
            snapshot_data = json.load(f)
        
        from qdrant_client.models import PointStruct
        
        for name, points_data in snapshot_data.items():
            # Create collection if it doesn't exist
            if not qdrant_client.collection_exists(name):
                qdrant_client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
                )
            
            if points_data:
                points = [
                    PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
                    for p in points_data
                ]
                qdrant_client.upsert(collection_name=name, points=points)
        
        total_points = sum(len(v) for v in snapshot_data.values())
        logger.info(f"Qdrant snapshot loaded: {len(snapshot_data)} collections, {total_points} total points.")
    except Exception as e:
        logger.error(f"Failed to load Qdrant snapshot: {e}. Starting fresh.")

# Load existing data on startup
load_qdrant_snapshot()
logger.info("Qdrant in-memory client ready.")
