import os
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.security import SecurityService

client = TestClient(app)

def get_auth_headers():
    token = SecurityService().create_token({"sub": "test_user"})
    return {"Authorization": f"Bearer {token}"}

class TestEndpointIntegration:

    def test_transcribe_success(self):
        """Test successful transcription with a valid WAV file."""
        # ✅ Make sure this path and file exist: tests/sample1.wav
        test_audio_path = "tests/sample1.wav"

        # Ensure the file exists before the test
        assert os.path.exists(test_audio_path), f"Audio file not found at {test_audio_path}"

        with open(test_audio_path, "rb") as f:
            files = {"audio": ("sample1.wav", f, "audio/wav")}
            headers = get_auth_headers()
            response = client.post("/api/v1/transcribe", files=files, headers=headers)

        print("Transcribe response:", response.json())
        assert response.status_code == 200
        assert "text" in response.json()

    def test_translate_success(self):
        """Test successful translation."""
        payload = {
            "text": "Hello world",
            "source_lang": "en",
            "target_lang": "es"
        }
        headers = get_auth_headers()
        response = client.post("/api/v1/translate", json=payload, headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert "translated_text" in data
