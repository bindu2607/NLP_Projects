# Whisper ASR Demo

A simple project to transcribe and translate audio files using OpenAI's Whisper model.

## Requirements

- Python 3.8–3.11
- ffmpeg installed and added to your system PATH
- Python packages: `openai-whisper`, `torch`, `ffmpeg-python`

## Setup Instructions

1. **Clone or download this project folder.**

2. **Install Python dependencies:**
 ```
  pip install openai-whisper torch ffmpeg-python
  ```

3. **Install ffmpeg:**
- **Windows (recommended):**
  - Open Command Prompt as administrator and run:
    ```
    choco install ffmpeg
    ```
  - Or download from [ffmpeg.org](https://ffmpeg.org/download.html), extract, and add the `bin` folder to your PATH.
- **macOS:**  
  ```
  brew install ffmpeg
  ```
- **Linux:**  
  ```
  sudo apt install ffmpeg
  ```

4. **Verify ffmpeg installation:**
   ffmpeg -version

You should see version information if installed correctly.

## Usage

1. Place your audio file (e.g., `sample1.mp3`) in the project directory.
2. Run the demo script:
```
python whisper_demo.py
```
3. The script will:
- Detect the language of the audio
- Transcribe the speech
- If not in English, provide an English translation

## Example Output

Detected language: es
Transcription in original language:
¿Dónde está la parada del autobús? ¿Dónde está la parada del autobús?

Translation to English:
Where is the bus stop? Where is the bus stop?

##
- Supports common audio formats: `.mp3`, `.wav`, `.flac`, etc.
- For best results, use clear recordings.
- For advanced audio processing (noise suppression, feature extraction), consider tools like Librosa, webrtcvad, or RNNoise[1].
