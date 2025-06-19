import asyncio
import base64
from typing import List, Dict, Any, Optional
from TTS.api import TTS
import numpy as np
import io
import soundfile as sf
from app.core.config import get_settings

settings = get_settings()

class TTSModel:
    """Advanced TTS model with voice cloning and multi-engine support."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.TTS_MODEL_NAME
        self._model = None
        self.supported_languages = [
            "en", "es", "fr", "de", "it", "pt", "pl", "tr",
            "ru", "nl", "cs", "ar", "zh", "ja", "hi"
        ]
        self._load_model()

    def _load_model(self):
        """Initialize the TTS model."""
        try:
            self._model = TTS(self.model_name, progress_bar=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load TTS model: {str(e)}")

    async def synthesize(
        self,
        text: str,
        language: str = "en",
        speaker_wav: Optional[bytes] = None,
        speed: float = 1.0,
        output_format: str = "wav"
    ) -> Dict[str, Any]:
        """
        Synthesize speech from text with optional voice cloning.
        """
        if not self._model:
            raise RuntimeError("TTS model not loaded")

        if language not in self.supported_languages:
            raise ValueError(
                f"Language {language} not supported. Supported: {self.supported_languages}"
            )

        try:
            loop = asyncio.get_running_loop()

            # Generate speech
            if speaker_wav is not None and len(speaker_wav) > 0:
                # Voice cloning mode
                audio_array = await loop.run_in_executor(
                    None,
                    self._synthesize_with_voice_cloning,
                    text, speaker_wav, language
                )
            else:
                # Standard synthesis
                audio_array = await loop.run_in_executor(
                    None,
                    self._synthesize_standard,
                    text, language, speed
                )

            # Convert to requested format
            audio_bytes = self._convert_audio_format(audio_array, output_format)

            return {
                "audio_data": base64.b64encode(audio_bytes).decode('utf-8'),
                "format": output_format,
                "sample_rate": 22050,
                "duration": len(audio_array) / 22050,
                "language": language,
                "text": text,
                "model_used": self.model_name,
                "voice_cloned": speaker_wav is not None and len(speaker_wav) > 0
            }

        except Exception as e:
            raise RuntimeError(f"Speech synthesis failed: {str(e)}")

    def _synthesize_with_voice_cloning(
        self, text: str, speaker_wav: bytes, language: str
    ) -> np.ndarray:
        """Perform voice cloning synthesis."""
        return self._model.tts_with_vc(
            text=text,
            speaker_wav=speaker_wav,
            language=language
        )

    def _synthesize_standard(
        self, text: str, language: str, speed: float
    ) -> np.ndarray:
        """Perform standard TTS synthesis."""
        return self._model.tts(
            text=text,
            language=language,
            speed=speed
        )

    def _convert_audio_format(self, audio_array: np.ndarray, format: str) -> bytes:
        """Convert audio array to specified format with proper WAV header."""
        if format == "wav":
            with io.BytesIO() as buf:
                sf.write(buf, audio_array, 22050, format="WAV")
                return buf.getvalue()
        else:
            raise ValueError(f"Format {format} not supported")

    async def get_available_speakers(self) -> List[str]:
        """Get list of available speaker voices."""
        if hasattr(self._model, 'speakers'):
            return self._model.speakers
        return []
