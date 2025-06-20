import whisper
import tempfile
import os

# Load the Whisper model only once
model = whisper.load_model("large")

def transcribe_audio(audio_file):
    """
    Transcribes an uploaded audio file using OpenAI's Whisper model.

    Parameters:
    - audio_file: A file-like object (e.g., from Streamlit uploader).

    Returns:
    - dict: A dictionary containing 'text', and optionally 'segments' and 'words'.
    """
    # Create a temporary file from uploaded audio
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        tmp.write(audio_file.read())
        tmp.close()

        # Run transcription (remove `word_timestamps=True` if you’re not using whisperx)
        result = model.transcribe(tmp.name)

        # Ensure structure matches Streamlit app expectations
        output = {
            "text": result.get("text", ""),
            "segments": result.get("segments", []),
            "words": []  # Whisper (non-X) doesn't return word timestamps
        }

    finally:
        os.remove(tmp.name)

    return output
