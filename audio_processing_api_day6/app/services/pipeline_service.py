"""
End-to-end speech pipeline: Whisper/Vosk ASR → MarianMT/OpenNMT Translation → TTS Synthesis (XTTS/ElevenLabs).
Supports word-level diff, color highlighting, and speaker similarity.
"""

import asyncio
from typing import Dict, Any, Optional
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS
from resemblyzer import VoiceEncoder, preprocess_wav
from difflib import SequenceMatcher
import numpy as np

# Optional: Vosk integration for Windows/offline ASR
try:
    from vosk import Model as VoskModel, KaldiRecognizer
    HAS_VOSK = True
except ImportError:
    HAS_VOSK = False

# Optional: ElevenLabs for cloud TTS/voice cloning
try:
    from elevenlabs import generate, set_api_key
    HAS_ELEVENLABS = True
except ImportError:
    HAS_ELEVENLABS = False

# Optional: OpenNMT REST API integration
import requests

class OpenNMTTranslation:
    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")

    def translate(self, text: str, src_lang: str = "en", tgt_lang: str = "fr") -> str:
        payload = [{"src": text}]
        url = f"{self.api_url}/translate"
        response = requests.post(url, json=payload)
        if response.ok:
            result = response.json()
            if isinstance(result, list) and "tgt" in result[0]:
                return result[0]["tgt"]
            elif "result" in result:
                return result["result"]
            else:
                return str(result)
        else:
            raise RuntimeError(f"OpenNMT API error: {response.text}")

class ElevenLabsTTS:
    def __init__(self, api_key: str):
        set_api_key(api_key)

    def synthesize(self, text: str, voice: str = "Rachel", model: str = "eleven_monolingual_v1") -> bytes:
        return generate(text=text, voice=voice, model=model)

class PipelineService:
    def __init__(
        self,
        whisper_model_name: str = "large-v2",
        whisper_device: str = "cpu",
        translation_model_name: str = "Helsinki-NLP/opus-mt-en-fr",
        tts_model_name: str = "tts_models/multilingual/multi-dataset/your_tts",
        vosk_model_path: Optional[str] = None,
        elevenlabs_api_key: Optional[str] = None,
        opennmt_url: Optional[str] = None
    ):
        # ASR (Whisper)
        self.whisper = WhisperModel(whisper_model_name, device=whisper_device, compute_type="int8")
        # MarianMT
        self.trans_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
        self.trans_model = MarianMTModel.from_pretrained(translation_model_name)
        # TTS (XTTS/YourTTS)
        self.tts = TTS(tts_model_name, progress_bar=False)
        # Resemblyzer for speaker similarity
        self.voice_encoder = VoiceEncoder()
        # Vosk for optional offline ASR
        self.vosk = VoskModel(vosk_model_path) if HAS_VOSK and vosk_model_path else None
        # ElevenLabs for cloud TTS
        self.elevenlabs = ElevenLabsTTS(elevenlabs_api_key) if HAS_ELEVENLABS and elevenlabs_api_key else None
        # OpenNMT for custom NMT
        self.opennmt = OpenNMTTranslation(opennmt_url) if opennmt_url else None

    async def asr_whisper(self, audio_bytes: bytes, language: Optional[str] = None) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        segments, info = await loop.run_in_executor(
            None,
            lambda: self.whisper.transcribe(
                audio_bytes,
                language=language,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
        )
        full_text = " ".join([seg.text.strip() for seg in segments])
        return {
            "text": full_text,
            "segments": [seg.text.strip() for seg in segments],
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "words": [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "probability": getattr(w, "probability", 0.0)
                }
                for seg in segments if hasattr(seg, "words") and seg.words
                for w in seg.words
            ]
        }

    def asr_vosk(self, audio_bytes: bytes, sample_rate: int = 16000) -> Dict[str, Any]:
        if not self.vosk:
            raise RuntimeError("Vosk model not initialized or not available.")
        import wave, io, json
        wf = wave.open(io.BytesIO(audio_bytes), "rb")
        rec = KaldiRecognizer(self.vosk, wf.getframerate())
        text = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                part = rec.Result()
                text += json.loads(part)["text"] + " "
        final = rec.FinalResult()
        text += json.loads(final)["text"]
        return {"text": text.strip()}

    async def translate(self, text: str, target_language: str = "fr", source_language: Optional[str] = None, backend: str = "marianmt") -> Dict[str, Any]:
        if backend == "opennmt" and self.opennmt:
            translated = self.opennmt.translate(text, src_lang=source_language or "en", tgt_lang=target_language)
            return {
                "translated_text": translated,
                "target_language": target_language,
                "original_text": text
            }
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._translate_sync,
                text,
                target_language
            )
            return result

    def _translate_sync(self, text: str, target_language: str) -> Dict[str, Any]:
        inputs = self.trans_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        output = self.trans_model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
        translated = self.trans_tokenizer.decode(output[0], skip_special_tokens=True)
        return {
            "translated_text": translated,
            "target_language": target_language,
            "original_text": text
        }

    async def synthesize(self, text: str, language: str = "fr", speaker_wav: Optional[bytes] = None, backend: str = "xtts") -> bytes:
        loop = asyncio.get_event_loop()
        if backend == "elevenlabs" and self.elevenlabs:
            return self.elevenlabs.synthesize(text)
        elif speaker_wav:
            audio = await loop.run_in_executor(
                None,
                self.tts.tts_with_vc,
                text,
                speaker_wav,
                language
            )
        else:
            audio = await loop.run_in_executor(
                None,
                self.tts.tts,
                text,
                language
            )
        return audio

    def speaker_similarity(self, audio1: bytes, audio2: bytes) -> float:
        wav1 = preprocess_wav(audio1)
        wav2 = preprocess_wav(audio2)
        emb1 = self.voice_encoder.embed_utterance(wav1)
        emb2 = self.voice_encoder.embed_utterance(wav2)
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        return float(np.dot(emb1_norm, emb2_norm))

    def word_level_diff(self, ref_text: str, hyp_text: str) -> list:
        ref_words = ref_text.strip().split()
        hyp_words = hyp_text.strip().split()
        matcher = SequenceMatcher(None, ref_words, hyp_words)
        diff_result = []
        for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
            if opcode == 'equal':
                for w in ref_words[i1:i2]:
                    diff_result.append({"word": w, "status": "correct"})
            elif opcode == 'replace':
                for w in ref_words[i1:i2]:
                    diff_result.append({"word": w, "status": "mismatch"})
                for w in hyp_words[j1:j2]:
                    diff_result.append({"word": w, "status": "mismatch"})
            elif opcode == 'delete':
                for w in ref_words[i1:i2]:
                    diff_result.append({"word": w, "status": "missing"})
            elif opcode == 'insert':
                for w in hyp_words[j1:j2]:
                    diff_result.append({"word": w, "status": "extra"})
        return diff_result

    def colorize_diff(self, diff: list) -> str:
        html = []
        for item in diff:
            word = item["word"]
            status = item["status"]
            if status == "correct":
                html.append(f'<span style="color:green">{word}</span>')
            elif status == "missing":
                html.append(f'<span style="color:red;text-decoration:line-through">{word}</span>')
            elif status == "extra":
                html.append(f'<span style="color:orange;text-decoration:underline">{word}</span>')
            elif status == "mismatch":
                html.append(f'<span style="color:blue;font-style:italic">{word}</span>')
        return " ".join(html)


# Example usage (in an async FastAPI endpoint)
# pipeline = PipelineService()
# transcription = await pipeline.asr_whisper(audio_bytes)
# translation = await pipeline.translate(transcription["text"], target_language="fr")
# tts_audio = await pipeline.synthesize(translation["translated_text"], language="fr", speaker_wav=reference_audio)
# similarity = pipeline.speaker_similarity(reference_audio, tts_audio)
# diff = pipeline.word_level_diff(transcription["text"], translation["translated_text"])
# html_diff = pipeline.colorize_diff(diff)
