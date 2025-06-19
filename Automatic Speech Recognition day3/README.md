# 🎙️ Whisper & Vosk Speech Recognition

## 📌 Overview

This project demonstrates how to use OpenAI’s Whisper and Vosk for automatic speech recognition (ASR). It includes both batch audio transcription and live speech recognition via microphone input.
---

## 🚀 Features

- ✅ Transcribe audio files to text using **Whisper (batch mode)**
- ✅ Real-time speech-to-text using **Whisper + Microphone**
- ✅ (Optional) Real-time ASR using **Vosk** as an alternative
- 💾 Saves all transcriptions to the `outputs/` directory

---

## ⚙️ Setup Instructions

1. Clone the Project

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd Day-3
```
2. Create a Virtual Environment (Optional but Recommended)
```
python -m venv .venv
source .venv/bin/activate     
# On Windows: .venv\Scripts\activate
```
3. Install Dependencies
```
pip install -r requirements.txt
```
Make sure ffmpeg is also installed and added to your system PATH.

## 🧪 Usage

**🔁 Transcribe an Audio File (Batch Mode)**
```
python whisper_batch.py
```
Input: sample1.mp3

Output: outputs/whisper_transcription.txt

**🎤 Live Speech-to-Text with Whisper**
```
python whisper_live.py
```
**🎤 Live ASR with Vosk**
```
python vosk_live.py
```

## 📂 Output Examples

```
outputs/
├── whisper_transcription.txt   # Example: "Hello, welcome to our demo..."
├── vosk_transcription.txt      # Example: "This is live speech recognition..."
├──live_transcription.txt
```

## 🤝 Contributions

Feel free to fork this repository, improve the code, and open a pull request. Suggestions and enhancements are welcome!

