import time
import threading
from typing import Any, Optional

class InMemoryCache:
    def __init__(self, default_ttl: float = 300.0, max_size: int = 1000):
        """
        Thread-safe in-memory cache with TTL support (seconds) and size bounds.
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache = {}
        self._lock = threading.Lock()

    def get(self, key: Any) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                return None
            
            val, expire_time = self._cache[key]
            if time.time() > expire_time:
                # Evict expired entry
                del self._cache[key]
                return None
                
            return val

    def set(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        with self._lock:
            # Enforce max size limit (evict first arbitrary entry if exceeded)
            if len(self._cache) >= self.max_size and key not in self._cache:
                # Evict oldest or first expired
                expired_keys = [k for k, (_, exp) in self._cache.items() if time.time() > exp]
                if expired_keys:
                    for k in expired_keys:
                        del self._cache[k]
                else:
                    # Just remove an arbitrary key
                    arbitrary_key = next(iter(self._cache))
                    del self._cache[arbitrary_key]

            set_ttl = ttl if ttl is not None else self.default_ttl
            expire_time = time.time() + set_ttl
            self._cache[key] = (value, expire_time)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

# Global Singleton caches for different RAG pipeline stages
fts_cache = InMemoryCache(default_ttl=300.0, max_size=500)
rerank_cache = InMemoryCache(default_ttl=300.0, max_size=500)
reflection_cache = InMemoryCache(default_ttl=600.0, max_size=200)
metadata_cache = InMemoryCache(default_ttl=120.0, max_size=200)
