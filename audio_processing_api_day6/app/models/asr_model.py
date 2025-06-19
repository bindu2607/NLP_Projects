"""
ASRModel for production-grade speech-to-text using OpenAI Whisper.
Accepts np.ndarray (audio array) for robust API integration.
"""
from typing import Optional, Dict, Any
import numpy as np

try:
    import whisper
except ImportError:
    whisper = None

class ASRModel:
    def __init__(self, model_name: str = "base"):
        if whisper is None:
            raise ImportError("Please install openai-whisper: pip install openai-whisper")
        self.model_name = model_name
        self._model = whisper.load_model(model_name)
        self.supported_languages = ["en", "fr", "de", "es", "hi", "auto"]

    async def transcribe(
        self,
        audio_array: np.ndarray,
        language: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        # Whisper expects np.ndarray for in-memory audio
        result = self._model.transcribe(
            audio_array,
            language=language if language and language != "auto" else None,
            word_timestamps=True,
            **kwargs
        )

        response = {
            "text": result.get("text", ""),
            "language": result.get("language", language or "auto"),
            "language_probability": result.get("language_probability", 1.0),
            "duration": result.get("duration", 0),
            "segments": result.get("segments", []),
            "words": [],
            "model_info": {
                "model_name": self.model_name,
                "device": str(self._model.device)
            }
        }
        if "segments" in result:
            for seg in result["segments"]:
                if "words" in seg:
                    response["words"].extend(seg["words"])
        return response
