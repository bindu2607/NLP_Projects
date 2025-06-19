from utils.whisper_integration import transcribe_audio
from utils.context_optimizer import optimize_context
from models.huggingface_model import Translator

def translate_pipeline(audio_file, src_lang, tgt_lang):
    """
    Full pipeline:
    1) Transcribe audio file to text.
    2) Optimize context for better translation.
    3) Translate optimized text.
    """
    transcript = transcribe_audio(audio_file)
    optimized_text = optimize_context(transcript, src_lang, tgt_lang)
    translator = Translator()
    translation = translator.translate(optimized_text, src_lang, tgt_lang)
    
    return {
        "transcript": transcript,
        "optimized_text": optimized_text,
        "translation": translation
    }

if __name__ == "__main__":
    class DummyFile:
        def read(self):
            with open("tests/assets/sample.wav", "rb") as f:
                return f.read()

    audio_file = DummyFile()
    result = translate_pipeline(audio_file, "en", "zh")

    print("Transcript:")
    print(result["transcript"])
    print("\nOptimized Text:")
    print(result["optimized_text"])
    print("\nTranslation:")
    print(result["translation"])
