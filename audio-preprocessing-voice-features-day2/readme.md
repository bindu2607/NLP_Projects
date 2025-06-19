# 🎧 Audio Preprocessing & Voice Feature Extraction

## 📌 Overview

This project demonstrates how to preprocess raw audio files by applying noise suppression, voice activity detection (VAD), and feature extraction (MFCC and pitch). It also includes audio format conversion and waveform visualization to compare raw vs processed signals.

---

## 🎯 Objectives

- Convert raw `.mp3` files to `.wav`
- Apply noise reduction and VAD for clean speech segments
- Extract MFCC and pitch features using Librosa and CREPE
- Save processed audio and feature arrays
- Plot original vs processed waveforms for comparison

---

## 🛠️ Technologies & Libraries Used

| Tool/Library        | Purpose                                                       |
|---------------------|---------------------------------------------------------------|
| **Python 3.8+**      | Core programming language                                     |
| **MoviePy**          | Convert `.mp3` to `.wav` audio format                        |
| **Librosa**          | Audio loading, MFCC extraction, and waveform visualization   |
| **CREPE**            | Pitch extraction using deep learning                         |
| **Noisereduce**      | Noise suppression (alternative to RNNoise / SpeexDSP)        |
| **WebRTC VAD**       | Voice Activity Detection using fixed 10/20/30 ms frames      |
| **SoundFile**        | Saving processed audio to disk                               |
| **Matplotlib**       | Plotting waveforms for visual comparison                     |
| **NumPy**            | Numerical processing and feature storage                     |

---

---

## 🚀 Features Implemented

- 🎧 **MP3 to WAV Conversion** — with `moviepy.editor.AudioFileClip`
- 🧹 **Noise Suppression** — using `noisereduce`
- 🗣️ **Voice Activity Detection (VAD)** — via `webrtcvad`
- 🎼 **MFCC Extraction** — using `librosa.feature.mfcc`
- 📈 **Pitch Detection** — with `crepe.predict(...)`
- 🖼️ **Waveform Plotting** — before and after processing
- 💾 **Feature Saving** — to `.npy` and cleaned `.wav` formats

---

## 📦 Setup Instructions

1. Create a virtual environment
On Windows:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Prepare audio files
- Place .mp3 or .wav files inside data/raw/.

- Convert MP3 to WAV
```
python src/utils/convert_mp3_to_wav.py
```
4. Run the preprocessing pipeline
```
python src/preprocessing/audio_features.py
```

## 🧪 Output Files

After running the script, the following will be saved to data/processed/:

File Type	Description
*_processed.wav	Cleaned and speech-only audio
*_mfcc.npy	MFCC feature array (13 coefficients)
*_pitch.npy	Pitch frequency array (CREPE)
*_waveforms.png	Comparison plot (original vs clean)

## 🤝 Contributions
Feel free to fork the repository, submit issues, or open pull requests. Contributions are welcome!

