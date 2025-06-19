import webrtcvad
from langdetect import detect

class NoiseSuppressor:
    def __init__(self, aggressiveness=3):
        """Initialize WebRTC VAD with the given aggressiveness (0-3)."""
        self.vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        """Check if the given audio frame contains speech."""
        return self.vad.is_speech(frame, sample_rate)

    def filter_speech(self, audio_frames: list, sample_rate: int) -> list:
        """Filter and return only speech frames from the list of audio frames."""
        return [frame for frame in audio_frames if self.is_speech(frame, sample_rate)]

class SpeakerDiarizer:
    def diarize(self, audio_path: str):
        """
        Placeholder for speaker diarization logic.
        In production, integrate with pyannote.audio or similar library.
        """
        # Example static output for demonstration
        return [
            {"speaker": "Speaker 1", "start": 0.0, "end": 5.0},
            {"speaker": "Speaker 2", "start": 5.0, "end": 10.0}
        ]

class LanguageRouter:
    def __init__(self, translation_models: dict):
        """
        Initialize with a dict mapping source languages to translation models.
        Example: {'en': MarianMTModel, 'fr': SomeOtherModel, ...}
        """
        self.translation_models = translation_models

    def route(self, text: str):
        """
        Detect source language and return the appropriate translation model.
        Defaults to English model if language is unsupported.
        """
        src_lang = detect(text)
        if src_lang not in self.translation_models:
            src_lang = "en"  # Default to English if unsupported
        return self.translation_models[src_lang]
