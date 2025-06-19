import io
import wave
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from fastapi import HTTPException
from app.core.config import get_settings

try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False

try:
    import librosa
    import soundfile as sf
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

settings = get_settings()

class AudioProcessor:
    def __init__(self):
        self.max_file_size = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        self.allowed_formats = settings.ALLOWED_AUDIO_FORMATS
        self.target_sr = 16000

    def validate_audio_file(self, audio_data: bytes, filename: str):
        if len(audio_data) > self.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit"
            )
        file_ext = filename.lower().split('.')[-1]
        if file_ext not in self.allowed_formats:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported format. Allowed: {', '.join(self.allowed_formats)}"
            )

    async def enhance_audio(self, audio_data: bytes, filename: str = None) -> bytes:
        try:
            audio_array, sr = self._decode_audio(audio_data, filename)
            if HAS_NOISEREDUCE:
                reduced_array = nr.reduce_noise(y=audio_array, sr=sr)
            else:
                reduced_array = audio_array
            normalized_array = self._normalize_audio(reduced_array)
            return self._encode_wav(normalized_array, sr=self.target_sr)
        except Exception:
            return audio_data

    def _decode_audio(self, audio_bytes: bytes, filename: Optional[str] = None) -> Tuple[np.ndarray, int]:
        if HAS_LIBROSA:
            audio_io = io.BytesIO(audio_bytes)
            y, sr = librosa.load(audio_io, sr=self.target_sr, mono=True)
            return y, sr
        else:
            audio_io = io.BytesIO(audio_bytes)
            with wave.open(audio_io, 'rb') as wav_file:
                sr = wav_file.getframerate()
                frames = wav_file.readframes(-1)
                audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                return audio_array, sr

    def _encode_wav(self, audio_array: np.ndarray, sr: int = 16000) -> bytes:
        audio_int16 = (audio_array * 32767).astype(np.int16)
        audio_io = io.BytesIO()
        with wave.open(audio_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sr)
            wav_file.writeframes(audio_int16.tobytes())
        return audio_io.getvalue()

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val * 0.8
        return audio
