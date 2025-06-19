# Audio Processing API Documentation

## Endpoints

### POST `/api/v1/transcribe`
- **Description:** Transcribe an audio file.
Audio file to transcribe. Supported formats: WAV, MP3, OGG, FLAC, etc.
- **Input:** `audio` (file, required)
- **Response:**
{
  "text": "Transcribed text from the audio file.",
  "language": "en",
  "language_probability": 0.98,
  "duration": 12.34,
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "text": "First segment of speech.",
      "avg_logprob": -0.15,
      "no_speech_prob": 0.01
    }
    // ...more segments
  ],
  "words": [
    {
      "word": "Hello",
      "start": 0.0,
      "end": 0.5,
      "probability": 0.97
    }
    // ...more words
  ],
  "model_info": {
    "model_name": "large-v2",
    "device": "cpu",
    "backend": "whisper"
  }
}
