"""
Redis-based caching system for improved performance.
"""
import redis
import json
import hashlib
from typing import Optional, Dict, Any
from app.core.config import get_settings

settings = get_settings()

class CacheManager:
    """Advanced caching system with Redis backend."""
    
    def __init__(self):
        self.redis_client = redis.from_url(settings.REDIS_URL)
        self.default_ttl = 3600  # 1 hour
    
    def generate_audio_hash(self, audio_data: bytes) -> str:
        """Generate hash for audio data."""
        return hashlib.sha256(audio_data).hexdigest()
    
    def generate_text_hash(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
        """Generate hash for text translation."""
        key_data = f"{text}:{source_lang}:{target_lang}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    async def get_transcription(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached transcription result."""
        try:
            cached_data = self.redis_client.get(f"transcription:{cache_key}")
            if cached_data:
                return json.loads(cached_data)
        except Exception:
            pass
        return None
    
    async def cache_transcription(self, cache_key: str, result: Dict[str, Any]):
        """Cache transcription result."""
        try:
            self.redis_client.setex(
                f"transcription:{cache_key}",
                self.default_ttl,
                json.dumps(result)
            )
        except Exception:
            pass  # Continue without caching if Redis is unavailable
    
    async def get_translation(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached translation result."""
        try:
            cached_data = self.redis_client.get(f"translation:{cache_key}")
            if cached_data:
                return json.loads(cached_data)
        except Exception:
            pass
        return None
    
    async def cache_translation(self, cache_key: str, result: Dict[str, Any]):
        """Cache translation result."""
        try:
            self.redis_client.setex(
                f"translation:{cache_key}",
                self.default_ttl,
                json.dumps(result)
            )
        except Exception:
            pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Redis connection health."""
        try:
            self.redis_client.ping()
            return {"status": "healthy", "backend": "redis"}
        except Exception as e:
            return {"status": "error", "backend": "redis", "error": str(e)}
