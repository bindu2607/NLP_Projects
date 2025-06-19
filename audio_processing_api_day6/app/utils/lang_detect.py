"""
Batch language detection and translation model routing utility.
Designed for multimodal AI, speech recognition, and translation pipelines.
"""

from langdetect import detect, DetectorFactory

# Ensure deterministic language detection
DetectorFactory.seed = 0

def detect_text_lang(text: str) -> str:
    """
    Detect the language of a single text string.
    Returns a two-letter language code (e.g., 'en', 'fr').
    Falls back to 'en' if detection fails.
    """
    try:
        return detect(text)
    except Exception:
        return "en"

def batch_detect_lang(texts):
    """
    Detect languages for a list of texts.
    Returns a list of detected language codes.
    """
    return [detect_text_lang(text) for text in texts]

class TranslationRouter:
    """
    Routes text to the appropriate translation model based on detected language.
    Example usage:
        router = TranslationRouter({'en': english_model, 'fr': french_model})
        model = router.route("Bonjour le monde")
    """
    def __init__(self, translation_models: dict):
        self.translation_models = translation_models

    def route(self, text: str):
        lang = detect_text_lang(text)
        return self.translation_models.get(lang, self.translation_models.get("en"))

    def batch_route(self, texts):
        """
        For a list of texts, returns a list of translation models (one per text).
        """
        langs = batch_detect_lang(texts)
        return [
            self.translation_models.get(lang, self.translation_models.get("en"))
            for lang in langs
        ]
