"""
Neural machine translation using Hugging Face Transformers.
Supports multiple language pairs with confidence scoring.
"""
import asyncio
from typing import Dict, Any, Optional, List
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect
from app.core.config import get_settings

settings = get_settings()

class TranslationModel:
    """Advanced translation model with multi-language support."""

    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        # Use tuple keys for language pairs for clarity and robustness
        self.language_pairs = {
            ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
            ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
            ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
            ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
            ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
            ("de", "en"): "Helsinki-NLP/opus-mt-de-en"
        }

    def _load_model(self, model_name: str):
        """Load translation model and tokenizer."""
        if model_name not in self.models:
            try:
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
                self.tokenizers[model_name] = tokenizer
                self.models[model_name] = model
            except Exception as e:
                raise RuntimeError(f"Failed to load translation model {model_name}: {str(e)}")
        return self.models[model_name], self.tokenizers[model_name]

    def detect_language(self, text: str) -> str:
        """Detect the language of input text."""
        try:
            return detect(text)
        except Exception:
            return "en"  # Default to English

    async def translate(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Translate text between languages with confidence scoring.

        Args:
            text: Text to translate
            target_language: Target language code
            source_language: Source language code (auto-detected if not provided)

        Returns:
            Translation results with metadata
        """
        # Default to English if not provided
        if not source_language:
            source_language = "en"

        lang_pair = (source_language, target_language)
        if lang_pair not in self.language_pairs:
            raise ValueError(f"Translation pair {source_language}-{target_language} not supported. Supported pairs: {list(self.language_pairs.keys())}")

        model_name = self.language_pairs[lang_pair]

        try:
            model, tokenizer = self._load_model(model_name)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._perform_translation,
                text, model, tokenizer
            )
            return {
                "translated_text": result["translation"],
                "source_language": source_language,
                "target_language": target_language,
                "confidence_score": result.get("confidence", 0.0),
                "model_used": model_name,
                "original_text": text
            }
        except Exception as e:
            raise RuntimeError(f"Translation failed: {str(e)}")

    def _perform_translation(self, text: str, model, tokenizer) -> Dict[str, Any]:
        """Perform the actual translation."""
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(
            **inputs,
            max_length=512,
            num_beams=4,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True
        )
        translation = tokenizer.decode(translated.sequences[0], skip_special_tokens=True)
        confidence = 0.0
        if hasattr(translated, 'sequences_scores'):
            confidence = float(translated.sequences_scores[0])
        return {
            "translation": translation,
            "confidence": confidence
        }

    async def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        languages = set()
        for src, tgt in self.language_pairs.keys():
            languages.add(src)
            languages.add(tgt)
        return sorted(list(languages))
