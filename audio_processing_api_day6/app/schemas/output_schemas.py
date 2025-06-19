from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class WordTimestamp(BaseModel):
    """Word-level timing information."""
    word: str = Field(..., description="The spoken word")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    probability: float = Field(..., description="Confidence score for this word")

class SegmentInfo(BaseModel):
    """Segment-level transcription information."""
    start: float = Field(..., description="Segment start time")
    end: float = Field(..., description="Segment end time")
    text: str = Field(..., description="Segment text")
    avg_logprob: float = Field(..., description="Average log probability")
    no_speech_prob: float = Field(..., description="No speech probability")

class TranscriptionResponse(BaseModel):
    """Response schema for transcription results."""
    text: str = Field(..., description="Complete transcribed text")
    language: str = Field(..., description="Detected language")
    language_probability: float = Field(..., description="Language detection confidence")
    duration: float = Field(..., description="Audio duration in seconds")
    segments: List[SegmentInfo] = Field(..., description="Detailed segment information")
    words: List[WordTimestamp] = Field(..., description="Word-level timestamps")
    model_info: Dict[str, str] = Field(..., description="Model metadata")
    processing_time: float = Field(..., description="Processing time in seconds")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Welcome to our advanced audio processing system",
                "language": "en",
                "language_probability": 0.99,
                "duration": 3.5,
                "segments": [
                    {
                        "start": 0.0,
                        "end": 3.5,
                        "text": "Welcome to our advanced audio processing system",
                        "avg_logprob": -0.2,
                        "no_speech_prob": 0.01
                    }
                ],
                "words": [
                    {"word": "Welcome", "start": 0.0, "end": 0.5, "probability": 0.99},
                    {"word": "to", "start": 0.5, "end": 0.7, "probability": 0.98}
                ],
                "model_info": {"model_name": "large-v2", "device": "cpu"},
                "processing_time": 1.2
            }
        }
    }

class TranslationResponse(BaseModel):
    """Response schema for translation results."""
    translated_text: str = Field(..., description="Translated text")
    source_language: str = Field(..., description="Source language code")
    target_language: str = Field(..., description="Target language code")
    confidence_score: float = Field(..., description="Translation confidence")
    model_used: str = Field(..., description="Translation model identifier")
    original_text: str = Field(..., description="Original input text")
    processing_time: float = Field(..., description="Processing time in seconds")

    model_config = {
        "json_schema_extra": {
            "example": {
                "translated_text": "Bienvenue dans notre système de traitement audio avancé",
                "source_language": "en",
                "target_language": "fr",
                "confidence_score": 0.95,
                "model_used": "Helsinki-NLP/opus-mt-en-fr",
                "original_text": "Welcome to our advanced audio processing system",
                "processing_time": 0.8
            }
        }
    }

class VoiceCloningResponse(BaseModel):
    """Response schema for voice cloning results."""
    audio_data: str = Field(..., description="Base64 encoded audio data")
    format: str = Field(..., description="Audio format")
    sample_rate: int = Field(..., description="Audio sample rate")
    duration: float = Field(..., description="Audio duration in seconds")
    similarity_score: float = Field(..., description="Voice similarity score")
    quality_rating: str = Field(..., description="Quality assessment")
    cloning_successful: bool = Field(..., description="Whether cloning met threshold")
    processing_time: float = Field(..., description="Processing time in seconds")

    model_config = {
        "json_schema_extra": {
            "example": {
                "audio_data": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
                "format": "wav",
                "sample_rate": 22050,
                "duration": 2.5,
                "similarity_score": 0.89,
                "quality_rating": "good",
                "cloning_successful": True,
                "processing_time": 3.2
            }
        }
    }

class PipelineResponse(BaseModel):
    """Response schema for complete pipeline processing."""
    final_audio: str = Field(..., description="Final processed audio (base64)")
    transcription: Optional[str] = Field(None, description="Original transcription")
    translation: Optional[str] = Field(None, description="Translated text")
    similarity_score: Optional[float] = Field(None, description="Voice similarity if cloning used")
    total_processing_time: float = Field(..., description="Total pipeline processing time")
    stages_completed: List[str] = Field(..., description="List of completed processing stages")
    metadata: Dict[str, Any] = Field(..., description="Additional processing metadata")

    model_config = {
        "json_schema_extra": {
            "example": {
                "final_audio": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
                "transcription": "Hello, welcome to our service",
                "translation": "Hola, bienvenido a nuestro servicio",
                "similarity_score": 0.92,
                "total_processing_time": 5.7,
                "stages_completed": ["transcription", "translation", "voice_cloning"],
                "metadata": {
                    "source_language": "en",
                    "target_language": "es",
                    "quality_score": 0.95
                }
            }
        }
    }

class ErrorResponse(BaseModel):
    """Standard error response schema."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "HTTP_403",
                "message": "Not authenticated",
                "details": None,
                "timestamp": "2025-06-10T20:17:51.906249",
                "request_id": None
            }
        }
    }
from pydantic import BaseModel
from typing import Optional

class TTSSpeakResponse(BaseModel):
    audio_data: str
    format: str
    sample_rate: int
    duration: float
    language: str
    text: str
    model_used: str
    processing_time: Optional[float] = None
    quality_rating: Optional[str] = None
