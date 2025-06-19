from app.schemas.output_schemas import TTSSpeakResponse
import time
from datetime import datetime
from io import BytesIO
from typing import Dict, Any

import librosa
from fastapi import (
    APIRouter, UploadFile, File, Form, HTTPException,
    Depends, BackgroundTasks, status
)
from fastapi.responses import JSONResponse

# --- Internal imports ---
from app.models.asr_model import ASRModel
from app.models.translation_model import TranslationModel
from app.models.tts_model import TTSModel
from app.schemas.input_schemas import *
from app.schemas.output_schemas import *
from app.services.pdf_logger import PDFLogger
from app.utils.audio_processing import AudioProcessor
from app.utils.cache import CacheManager
from app.core.security import get_current_user, require_role, SecurityService
from app.core.config import get_settings

settings = get_settings()
router = APIRouter()

# --- Demo users (replace with DB in production) ---
users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": SecurityService.hash_password("admin"),
        "role": "admin"
    },
    "test": {
        "username": "test",
        "hashed_password": SecurityService.hash_password("test"),
        "role": "user"
    }
}

# --- Login Endpoint ---
@router.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    user = users_db.get(username)
    if not user or not SecurityService.verify_password(password, user["hashed_password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    token = SecurityService.create_access_token({"sub": user["username"], "role": user["role"]})
    return {"access_token": token, "token_type": "bearer"}

# --- Instantiate Services ---
asr_model = ASRModel()
translation_model = TranslationModel()
tts_model = TTSModel()
pdf_logger = PDFLogger()
audio_processor = AudioProcessor()
cache_manager = CacheManager()

# --- Transcription ---
@router.post("/transcribe", response_model=TranscriptionResponse, tags=["Audio Processing"])
async def transcribe_audio(
    audio: UploadFile = File(...),
    request_data: TranscriptionRequest = Depends(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    start_time = time.time()
    try:
        audio_data = await audio.read()
        audio_processor.validate_audio_file(audio_data, audio.filename)
        cache_key = cache_manager.generate_audio_hash(audio_data)
        cached = await cache_manager.get_transcription(cache_key)
        if cached:
            cached["processing_time"] = time.time() - start_time
            cached["cache_hit"] = True
            return TranscriptionResponse(**cached)
        enhanced = await audio_processor.enhance_audio(audio_data)
        audio_io = BytesIO(enhanced)
        audio_array, sr = librosa.load(audio_io, sr=16000, mono=True)
        result = await asr_model.transcribe(audio_array, language=request_data.language)
        result.update({
            "processing_time": time.time() - start_time,
            "cache_hit": False,
            "user_id": current_user.get("sub")
        })
        background_tasks.add_task(cache_manager.cache_transcription, cache_key, result)
        background_tasks.add_task(pdf_logger.log_transcription, audio.filename, result, current_user.get("sub"))
        return TranscriptionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

# --- Translation ---
@router.post("/translate", response_model=TranslationResponse, tags=["Text Processing"])
async def translate_text(
    request: TranslationRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    start_time = time.time()
    try:
        source = request.source_language or "en"
        target = request.target_language or "fr"
        lang_pair = (source, target)
        if lang_pair not in translation_model.language_pairs:
            supported = [f"{src}-{tgt}" for (src, tgt) in translation_model.language_pairs.keys()]
            raise HTTPException(status_code=400, detail=f"Unsupported language pair {source}-{target}. Supported: {supported}")
        cache_key = cache_manager.generate_text_hash(request.text, target, source)
        cached = await cache_manager.get_translation(cache_key)
        if cached:
            cached["processing_time"] = time.time() - start_time
            cached["cache_hit"] = True
            return TranslationResponse(**cached)
        result = await translation_model.translate(
            text=request.text,
            source_language=source,
            target_language=target
        )
        result.update({
            "processing_time": time.time() - start_time,
            "cache_hit": False
        })
        background_tasks.add_task(cache_manager.cache_translation, cache_key, result)
        return TranslationResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

# --- Speech Synthesis (TTS Only) ---
@router.post("/speak", response_model=TTSSpeakResponse, tags=["Voice Processing"])
async def speak(
    text: str = Form(..., description="Text to synthesize"),
    target_lang: str = Form(..., description="Target language code (e.g., 'en', 'fr')"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    start_time = time.time()
    try:
        # Synthesize speech using TTS model (no reference audio, no cloning)
        result = await tts_model.synthesize(
            text=text,
            language=target_lang,
            speaker_wav=None  # No reference audio for TTS-only
        )
        result["processing_time"] = time.time() - start_time
        background_tasks.add_task(
            pdf_logger.log_tts_speak, text, result, current_user.get("sub")
        )
        return TTSSpeakResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")

# --- Language Info ---
@router.get("/supported-languages", tags=["Information"])
async def get_supported_languages():
    try:
        translation_languages = await translation_model.get_supported_languages()
        return {
            "asr_languages": ["auto"] + asr_model.supported_languages,
            "translation_languages": translation_languages,
            "tts_languages": tts_model.supported_languages,
            "language_pairs": [f"{src}-{tgt}" for (src, tgt) in translation_model.language_pairs.keys()],
            "total_supported": len(set(translation_languages))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve language info: {str(e)}")

# --- Health Check ---
@router.get("/health", tags=["System"])
async def health_check():
    health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.VERSION,
        "components": {}
    }
    try:
        health["components"]["asr"] = {
            "status": "healthy" if asr_model._model else "error",
            "model": asr_model.model_name
        }
        health["components"]["translation"] = {
            "status": "healthy",
            "supported_pairs": len(translation_model.language_pairs)
        }
        health["components"]["tts"] = {
            "status": "healthy" if tts_model._model else "error",
            "model": tts_model.model_name
        }
        health["components"]["cache"] = await cache_manager.health_check()
        return health
    except Exception as e:
        health["status"] = "unhealthy"
        health["error"] = str(e)
        return JSONResponse(status_code=503, content=health)
