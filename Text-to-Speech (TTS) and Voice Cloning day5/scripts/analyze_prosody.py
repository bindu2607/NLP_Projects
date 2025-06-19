import librosa
import numpy as np
import logging
import os

os.makedirs("outputs", exist_ok=True)
logging.basicConfig(filename='outputs/app.log', level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_prosody(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=500)
    energy = np.sum(librosa.feature.rms(y=y))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return {
        "mean_f0": float(np.nanmean(f0)),
        "std_f0": float(np.nanstd(f0)),
        "energy": float(energy),
        "tempo": float(tempo)
    }

try:
    orig = extract_prosody("input/original_speaker.wav")
    gen = extract_prosody("outputs/tts_output.wav")

    with open("outputs/prosody_analysis.txt", "w") as f:
        f.write("Prosody Analysis\n")
        f.write(f"Original: {orig}\n")
        f.write(f"Generated: {gen}\n")
        f.write(f"F0 diff: {abs(orig['mean_f0']-gen['mean_f0']):.2f}\n")
        f.write(f"Energy diff: {abs(orig['energy']-gen['energy']):.2f}\n")
        f.write(f"Tempo diff: {abs(orig['tempo']-gen['tempo']):.2f}\n")
    logger.info("Prosody analysis completed.")
    print("Prosody analysis complete. Results saved to outputs/prosody_analysis.txt")
except Exception as e:
    logger.error(f"Prosody analysis failed: {e}")
    print("Error during prosody analysis:", e)
