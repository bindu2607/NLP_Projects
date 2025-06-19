# MT_specialist

**Speech-to-Text & Translation | NLP | Whisper ASR | NLLB-200 | Streamlit | GPU Accelerated**

🌐 English–Chinese Speech & Text Translation System

## 📖 Overview

This project implements a high-performance speech translation pipeline that transcribes spoken input and translates it bidirectionally between English and Chinese, using OpenAI Whisper for transcription and Meta’s NLLB-200 1.3B model for translation.

## Key components include:

- 🎧 Speech-to-text transcription (OpenAI Whisper)

- 🧠 Context optimization for idiomatic phrase handling

- 🌍 Multilingual translation (Meta’s NLLB-200 1.3B)

- 📊 BLEU score evaluation for translation quality benchmarking

## ✨ Key Features

- 🎤 Accurate Audio Transcription
Utilizes OpenAI’s Whisper model (large) for multilingual, high-accuracy speech recognition.

- 💡 Idiom-Aware Context Optimization
Automatically replaces or rephrases idiomatic expressions to improve translation clarity and fluency.

- 🔁 Bidirectional Translation Support
Enables seamless English ↔ Chinese translation using Meta’s NLLB-200 multilingual transformer.

- ⚡ GPU Acceleration
Automatically detects and utilizes GPU hardware (CUDA) for efficient inference.

- 📈 Translation Quality Evaluation
Supports BLEU score computation to measure the accuracy and fluency of translations.

- 🛠️ Modular Python Codebase
Well-organized structure for easy customization, testing, and scaling to additional languages or models.


---

## ⚙️ Technologies Used

| **Component**          | **Purpose / Description**                                                                    |
| ---------------------- | -------------------------------------------------------------------------------------------- |
| 🎧 **Whisper (Large)** | Advanced multilingual Automatic Speech Recognition (ASR) model developed by OpenAI.          |
| 🌐 **NLLB-200 (1.3B)** | Meta AI’s state-of-the-art multilingual machine translation model covering 600+ languages.   |
| 🖥️ **Streamlit**      | Lightweight Python framework for building interactive, real-time web apps with minimal code. |
| 🔥 **PyTorch**         | High-performance deep learning framework used for model training and inference.              |
| 🤗 **Transformers**    | Hugging Face’s powerful library enabling seamless integration with the NLLB model.           |

---

## 🌐 Supported Languages

| Source Language | Target Language |
|-----------------|-----------------|
| English (`en`)  | Chinese (`zh`)  |
| Chinese (`zh`)  | English (`en`)  |

---

## Prerequisites
Before you start, please make sure you have:

Python 3.8 or higher installed on your system

- ffmpeg installed for audio processing
You can install ffmpeg using your system’s package manager:
```
On Ubuntu/Debian: sudo apt-get install ffmpeg
```
```
On macOS (with Homebrew): brew install ffmpeg
```

## ⚙️ Setup and Usage

1.  **Clone the repository:**
 ```bash
 git clone https://github.com/StreamLingo-VoiceSync/MT_specialist.git
 cd MT_specialist
 ```

 2. Install Requirements
  ```
 pip install -r requirements.txt
 ```

✅ Make sure ffmpeg is installed and accessible in your system PATH.

3. Launch the Streamlit App
 ```
 cd streamlit_app
 streamlit run app.py
 ```

4. Run Inference from CLI
 ```
 python main.py
 ```

5. Test the model
   Run all project tests using:
```
   pytest tests/
 ```

## 🧠 Models & Inference

**🔊 Whisper (Transcription)**

- Model: openai/whisper-large

- Input: .wav, .mp3, .m4a audio formats

- Output: Full transcript with segment-wise and word-level confidence scores

- Purpose: High-accuracy multilingual speech-to-text transcription

**🌐 NLLB-200 (Translation)**

- Model: facebook/nllb-200-1.3B

- Optimized For: High-quality English ⇄ Chinese translation

- Features: Idiomatic phrase handling and strong contextual understanding

- Coverage: Supports over 200 languages (used here for English–Chinese)

📁 Sample Outputs

**1.Audio Translation JSON**

```
{
  "filename": "sample.wav",
  "source_language": "en",
  "target_language": "zh",
  "full_transcript": "Good afternoon, everyone...",
  "full_translation": "大家下午好...",
  "segments": [...],
  "word_details": [...]
}
```

**2.Text Translation JSON**

```
{
  "timestamp": "20250617_162938",
  "source_language": "en",
  "target_language": "zh",
  "original_text": "Good Morning\n",
  "translation": "您好,早上",
  "detected_language": "en"
}
```

## 🤝 Contributions

Contributions are welcome!
Feel free to fork the repo, make changes, and open a pull request.
