# 🎙️ StreamLingo VoiceSync
### Real-Time Voice-Preserving Translation System

**StreamLingo VoiceSync** is an AI project developed at **Oscowl AI**.  
I designed and implemented modular pipelines for **speech recognition (ASR)**, **audio preprocessing**, **machine translation (NMT)**, and **text-to-speech with voice cloning (TTS)** — with strong focus on **speaker identity preservation, reproducibility, and scalability**.

---

## 📌 Overview
- 🌐 Real-time multilingual **speech-to-speech translation** (40+ languages)  
- 🗣️ **Voice-preserving TTS** with improved speaker similarity  
- 🔐 **FastAPI backend** with authentication, monitoring & automated reporting  
- 📊 Comprehensive **evaluation toolkit** with WER, BLEU, cosine similarity, MOS  
- 📈 **Streamlit dashboards** for real-time analytics and visualization  

---

## 🔹 My Contributions (Section-Wise)

### 1. Automatic Speech Recognition (ASR)
- Implemented **Whisper** and **Vosk** pipelines for transcription and translation.  
- Supported **batch + live audio** transcription.  
- Automated **WER computation** and **word-level diff highlighting**.  
- Generated structured outputs (JSON, transcripts).  

### 2. Audio Preprocessing
- Designed preprocessing workflow:  
  - MP3 → WAV conversion  
  - **Noise suppression** (noisereduce, WebRTC VAD)  
  - **Voice Activity Detection (VAD)** for silence trimming  
  - **MFCC & pitch extraction** (Librosa, CREPE)  
  - Waveform & spectrogram visualization  
- Produced **clean audio segments + extracted features** for downstream tasks.  

### 3. Machine Translation (NMT)
- Built **English ↔ French** pipeline with **MarianMT**.  
- Integrated **Facebook NLLB-200** for **English ↔ Chinese bidirectional translation**.  
- Automated evaluation with **BLEU scores** and detailed reports.  
- Added **confidence scoring** for translations.  

### 4. TTS & Voice Cloning
- Integrated **XTTS** and **YourTTS** for natural speech synthesis.  
- Built evaluation toolkit:  
  - Speaker similarity (cosine similarity, Resemblyzer embeddings)  
  - Prosody analysis (pitch, tempo, energy)  
  - Visualization: **spectrograms, pitch contours**  
- Achieved **20% improvement in speaker identity retention**  
  - Cosine similarity **0.70 → 0.80** using **ECAPA-TDNN** on AISHELL-3.  

### 5. Backend Development
- Engineered **FastAPI backend** with modular APIs (ASR, NMT, TTS).  
- Added **JWT authentication** and role-based access.  
- Implemented **audit logging & Prometheus monitoring**.  
- Automated **PDF report generation** (ReportLab) + JSON outputs.  

### 6. Evaluation Framework
- **ASR**: Word Error Rate (WER), diff highlighting.  
- **Translation**: BLEU metrics, confidence scores.  
- **Speaker Similarity**: Cosine similarity, Equal Error Rate (EER).  
- **Audio Quality**: PESQ, Mel-Cepstral Distortion.  
- **Subjective**: Mean Opinion Score (MOS), AB testing.  

### 7. UI & Scalability
- Built **Streamlit dashboards** for analytics and reporting.  
- Extended pipelines with **batch processing** for large-scale evaluation.  
- Scaled platform to handle **10,000+ waitlisted users**.  

---

## 🛠️ Tech Stack
- **ASR**: Whisper, Vosk  
- **NMT**: MarianMT, Facebook NLLB-200  
- **TTS/Voice Cloning**: YourTTS, XTTS, Resemblyzer  
- **Speaker Embeddings**: ECAPA-TDNN (SpeechBrain)  
- **Backend**: FastAPI, JWT, Prometheus, ReportLab  
- **Audio Processing**: Librosa, CREPE, noisereduce, WebRTC VAD  
- **Visualization/UI**: Streamlit, Matplotlib  

---

## 📊 Results
- ✅ **10% boost** in speaker identity retention (0.70 → 0.75 cosine similarity)  
- ✅ **BLEU-scored translations** validated for English ↔ French & Chinese  
- ✅ **WER benchmarking** between Whisper and Vosk  
- ✅ Production-ready **secure backend** with monitoring and reporting  

---

## 📂 Repository Links
- [Whisper ASR](https://github.com/bindu2607/NLP_Projects/tree/main/whisper-asr-transcriber-day1)  
- [Audio Preprocessing](https://github.com/bindu2607/NLP_Projects/tree/main/audio-preprocessing-voice-features-day2)  
- [ASR Comparison](https://github.com/bindu2607/NLP_Projects/tree/main/Automatic%20Speech%20Recognition%20day3)  
- [Machine Translation (MarianMT)](https://github.com/bindu2607/NLP_Projects/tree/main/Machine%20Translation%20day4)  
- [TTS & Voice Cloning](https://github.com/bindu2607/NLP_Projects/tree/main/Text-to-Speech%20(TTS)%20and%20Voice%20Cloning%20day5)  
- [Audio Processing API](https://github.com/bindu2607/NLP_Projects/tree/main/audio_processing_api_day6)  
- [Whisper + NLLB Translation](https://github.com/bindu2607/NLP_Projects/tree/main/whisper-nllb-translation-day7)  

---

## 🚀 Next Steps
- Add **speaker diarization** for multi-speaker handling.  
- Expand **language coverage** across ASR, NMT, and TTS.  
- Optimize for **low-latency real-time deployment**.  
- Conduct **large-scale subjective listening studies**.  

---

## 📜 License
MIT License © 2025 [Marpini Himabindu](https://github.com/bindu2607)
