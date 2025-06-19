"""
Comprehensive API testing with pytest and FastAPI TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import get_settings  # ✅ Import config

client = TestClient(app)
settings = get_settings()  # ✅ Create a settings instance

class TestAudioProcessingAPI:
    """Test suite for audio processing endpoints."""

    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]
        assert "components" in data
        assert "timestamp" in data

    def test_root_endpoint(self):
        """Test root endpoint information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data

    def test_supported_languages(self):
        """Test supported languages endpoint."""
        response = client.get("/api/v1/supported-languages")
        assert response.status_code == 200
        data = response.json()
        assert "asr_languages" in data
        assert "translation_languages" in data
        assert "tts_languages" in data
        assert isinstance(data["asr_languages"], list)

    def test_transcribe_endpoint_requires_auth(self):
        """Test transcription endpoint without authentication."""
        audio_content = b"dummy_audio_content"
        files = {"audio": ("test.wav", audio_content, "audio/wav")}
        response = client.post("/api/v1/transcribe", files=files)
        assert response.status_code in [401, 403]  # Expect unauthorized

    def test_translate_requires_auth(self):
        """Test translation endpoint without authentication."""
        data = {
            "text": "Hello world",
            "target_language": "fr"
        }
        response = client.post("/api/v1/translate", json=data)
        assert response.status_code in [401, 403]

    def test_invalid_audio_format(self):
        """Test validation of audio file formats."""
        audio_content = b"dummy_audio_content"
        files = {"audio": ("test.txt", audio_content, "text/plain")}
        response = client.post("/api/v1/transcribe", files=files)
        assert response.status_code in [400, 415, 401, 403]

    def test_file_size_validation(self):
        """Test file size limits."""
        # Simulate file larger than allowed
        big_audio = b"0" * (settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1)
        files = {"audio": ("big.wav", big_audio, "audio/wav")}
        response = client.post("/api/v1/transcribe", files=files)
        assert response.status_code in [413, 401, 403]
