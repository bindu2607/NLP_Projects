import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import logging
import os

os.makedirs("outputs", exist_ok=True)
logging.basicConfig(filename='outputs/app.log', level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_pitch(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=None)
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=500)
    plt.figure(figsize=(10, 4))
    plt.plot(f0, label="Pitch (F0)")
    plt.title("Pitch Contour")
    plt.xlabel("Frame")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Pitch contour saved to {output_path}")

def plot_spectrogram(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=None)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Spectrogram saved to {output_path}")

try:
    plot_pitch("outputs/tts_output.wav", "outputs/pitch_contour.png")
    plot_spectrogram("outputs/tts_output.wav", "outputs/spectrogram.png")
    print("Visualization complete. Check outputs/pitch_contour.png and outputs/spectrogram.png")
except Exception as e:
    logger.error(f"Visualization failed: {e}")
    print("Error during visualization:", e)
