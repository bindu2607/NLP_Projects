import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import noisereduce as nr
import webrtcvad
import crepe

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw'))
PROC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))

def apply_noise_suppression(y, sr):
    # Noise suppression using noisereduce (Python alternative to RNNoise/SpeexDSP for Windows)
    return nr.reduce_noise(y=y, sr=sr)

def apply_vad(y, sr, aggressiveness=2):
    vad = webrtcvad.Vad(aggressiveness)
    frame_duration = 30  # ms
    frame_length = int(sr * frame_duration / 1000)
    voiced = []
    for i in range(0, len(y), frame_length):
        frame = y[i:i+frame_length]
        if len(frame) < frame_length:
            break
        is_speech = vad.is_speech((frame * 32768).astype(np.int16).tobytes(), sr)
        if is_speech:
            voiced.extend(frame)
    return np.array(voiced)

def extract_mfcc(y, sr):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

def extract_pitch_crepe(y, sr):
    audio = librosa.util.normalize(y)
    audio = audio.astype(np.float32)
    time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)
    return frequency

def plot_waveforms(y_orig, y_proc, sr, fname):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y_orig, sr=sr)
    plt.title("Original Audio")
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(y_proc, sr=sr)
    plt.title("Processed Audio (Denoised + VAD)")
    plt.tight_layout()
    plt.savefig(os.path.join(PROC_DIR, fname + "_waveforms.png"))
    plt.close()

def main():
    os.makedirs(PROC_DIR, exist_ok=True)
    for file in os.listdir(RAW_DIR):
        if file.endswith(".wav"):
            path = os.path.join(RAW_DIR, file)
            y, sr = librosa.load(path, sr=16000)
            y_denoised = apply_noise_suppression(y, sr)
            y_vad = apply_vad(y_denoised, sr)
            mfcc = extract_mfcc(y_vad, sr)
            pitch = extract_pitch_crepe(y_vad, sr)
            # Save processed audio
            sf.write(os.path.join(PROC_DIR, file.replace(".wav", "_processed.wav")), y_vad, sr)
            # Save features
            np.save(os.path.join(PROC_DIR, file.replace(".wav", "_mfcc.npy")), mfcc)
            np.save(os.path.join(PROC_DIR, file.replace(".wav", "_pitch.npy")), pitch)
            # Plot comparison
            plot_waveforms(y, y_vad, sr, file.replace(".wav", ""))
            print(f"Processed {file}")

if __name__ == "__main__":
    main()
