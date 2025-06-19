"""
Comprehensive Pydantic schemas for request validation with examples.
Follows OpenAPI 3.0 standards for excellent API documentation.
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from enum import Enum

class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"
    M4A = "m4a"

class LanguageCode(str, Enum):
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    ARABIC = "ar"

class TranscriptionRequest(BaseModel):
    """Request schema for audio transcription."""
    language: Optional[LanguageCode] = Field(
        None, 
        description="Source language hint for better accuracy"
    )
    include_word_timestamps: bool = Field(
        True, 
        description="Include word-level timing information"
    )

    class Config:
        schema_extra = {
            "example": {
                "language": "en",
                "include_word_timestamps": True
            }
        }

class TranslationRequest(BaseModel):
    """Request schema for text translation."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to translate")
    target_language: LanguageCode = Field(..., description="Target language code")
    source_language: Optional[LanguageCode] = Field(
        None, 
        description="Source language (auto-detected if not provided)"
    )

    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "text": "Hello, how are you today?",
                "target_language": "fr",
                "source_language": "en"
            }
        }

class TTSRequest(BaseModel):
    """Request schema for text-to-speech synthesis."""
    text: str = Field(..., min_length=1, max_length=2000, description="Text to synthesize")
    language: LanguageCode = Field(default=LanguageCode.ENGLISH, description="Synthesis language")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier")
    output_format: AudioFormat = Field(default=AudioFormat.WAV, description="Output audio format")

    class Config:
        schema_extra = {
            "example": {
                "text": "Welcome to our advanced audio processing API",
                "language": "en",
                "speed": 1.0,
                "output_format": "wav"
            }
        }

class VoiceCloningRequest(BaseModel):
    """Request schema for voice cloning."""
    text: str = Field(..., min_length=1, max_length=2000, description="Text to synthesize")
    target_language: LanguageCode = Field(..., description="Target language for synthesis")
    similarity_threshold: float = Field(
        default=0.75, 
        ge=0.0, 
        le=1.0, 
        description="Minimum similarity threshold for successful cloning"
    )

    class Config:
        schema_extra = {
            "example": {
                "text": "This is a demonstration of voice cloning technology",
                "target_language": "fr",
                "similarity_threshold": 0.8
            }
        }

class PipelineRequest(BaseModel):
    """Request schema for complete audio processing pipeline."""
    target_language: LanguageCode = Field(..., description="Final output language")
    include_intermediate_results: bool = Field(
        default=False, 
        description="Include transcription and translation in response"
    )
    processing_options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional processing parameters"
    )

    class Config:
        schema_extra = {
            "example": {
                "target_language": "es",
                "include_intermediate_results": True,
                "processing_options": {
                    "enhance_audio": True,
                    "remove_noise": True
                }
            }
        }
