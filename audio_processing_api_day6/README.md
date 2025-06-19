# 🎙️ Audio Processing API – Speech Recognition, Translation, TTS & Voice Cloning

## 📖 Overview

This project delivers a modular, secure, and scalable **backend API** for advanced audio and language processing. It integrates state-of-the-art models for:

- ✅ **Automatic Speech Recognition (ASR)**
- 🌐 **Neural Machine Translation (NMT)**
- 🗣️ **Text-to-Speech (TTS) & Voice Cloning**
- 🔁 **Speaker Similarity Evaluation**

Accessible via **REST and WebSocket endpoints**, this architecture is built for extensibility, observability, and production-grade security.



## ⚙️ Technologies & Core Components

| Technology        | Purpose                                          |
|-------------------|--------------------------------------------------|
| **Python**        | Primary backend language                         |
| **FastAPI**       | High-performance framework for RESTful APIs      |
| **Whisper / Vosk**| ASR models for robust speech-to-text             |
| **MarianMT / OpenNMT** | Translation engines for multilingual NMT |
| **XTTS / YourTTS / ElevenLabs** | TTS & voice cloning models     |
| **Resemblyzer**   | Speaker embedding & similarity scoring           |
| **Docker**        | Containerized deployment                         |


---

## ✨ Key Features

- 🔐 **Secure Login**: Token-based authentication required for advanced features
- 🧠 **Speech-to-Text (ASR)**: Fast and accurate transcription via Whisper or Vosk
- 🌍 **Text Translation**: English ⇄ Other language translation via MarianMT / OpenNMT
- 🔊 **Text-to-Speech & Voice Cloning**: Generate natural speech or clone voices using reference audio
- 🧬 **Speaker Similarity**: Cosine similarity scoring using speaker embeddings
- ⚙️ **Pipeline Service**: End-to-end flow (Transcribe → Translate → Synthesize → Evaluate)
- 📑 **PDF Report Generation**: Generate logs, scores, and outputs in printable format
- 📈 **Centralized Logging**: Unified `app.log` for debugging, monitoring, and tracing

---

## 🧪 Example Workflow

1. 🔑 **Authenticate** – Login securely to obtain a token  
2. 🎧 **Upload Audio** – Send an audio file for transcription or cloning  
3. 📝 **Transcribe** – Convert speech to text  
4. 🌐 **Translate** – Convert the transcript to a target language  
5. 🗣️ **Synthesize** – Generate TTS or clone a voice  
6. 🔁 **Compare** – Evaluate speaker similarity  
7. 📄 **Report** – Get results in JSON or downloadable PDF

---

## 🚀 How to Run

### 🔧 Install Dependencies

```bash
pip install -r requirements.txt
```
1. Install Dependencies
```
pip install -r requirements.txt
```
2. Start the FastAPI Backend
From the root directory (where main.py is located):
```
uvicorn main:app --reload
```
The API will be available at: http://localhost:8000

Interactive docs at: http://localhost:8000/docs.

3. (Optional) Deploy with Docker
If you have a docker/ directory with Dockerfile and docker-compose:
```
cd docker
docker-compose up --build
```
4. Streamlit Frontend
```
streamlit run streamlit_app.py
```

## 🔒 Security & Best Practices

- All endpoints are protected with JWT-based authentication

- Input/output validated using Pydantic schemas

- Centralized logging with unique request tracking

- Modular architecture for testing and CI/CD integration

- Supports both REST and WebSocket streaming interfaces

## 📦 Example Endpoints

Endpoint	Method	Description
/api/v1/asr/	POST	Upload audio and receive transcription
/api/v1/translate/	POST	Translate text between languages
/api/v1/tts/	POST	Generate speech from text
/api/v1/clone/	POST	Clone voice using reference audio
/api/v1/similarity/	POST	Compare voices using embeddings
/api/v1/health/	GET	Check API health and uptime

## 👨‍💻 Contributions

Contributions are welcome!

Fork this repo and create your feature branch:
git checkout -b feature/YourFeature

Commit your changes:
git commit -m "Add your message"

Push to the branch:
git push origin feature/YourFeature

Open a Pull Request 🚀



