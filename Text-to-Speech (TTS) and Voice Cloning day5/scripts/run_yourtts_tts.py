import logging
import os
from TTS.api import TTS

os.makedirs("outputs", exist_ok=True)
logging.basicConfig(filename='outputs/app.log', level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    speaker_wav = "input/original_speaker.wav"
    with open("input/translated_text.txt", "r", encoding="utf-8") as f:
        text = f.read().strip()

    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False)
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language="fr-fr",  # Use "fr-fr" for French, "en" for English, "pt-br" for Portuguese
        file_path="outputs/tts_output.wav"
    )
    logger.info("TTS output saved to outputs/tts_output.wav")
    print("TTS synthesis complete. Output saved to outputs/tts_output.wav")
except Exception as e:
    logger.error(f"TTS synthesis failed: {e}")
    print("Error during TTS synthesis:", e)
