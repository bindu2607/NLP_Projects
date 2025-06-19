import whisper
import tempfile
import os

# Load the Whisper model globally once
model = whisper.load_model("large")

def transcribe_audio(audio_file):
    """
    Transcribe an uploaded audio file (BytesIO or similar) to text using Whisper.
    audio_file: a file-like object (e.g., Streamlit uploaded file).
    Returns transcript string.
    """
    # Create temp file with delete=False
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        tmp.write(audio_file.read())
        tmp.close()  # Close the file so ffmpeg can access it
        
        # Now pass the temp file path to whisper
        result = model.transcribe(tmp.name)
    finally:
        # Remove the temp file explicitly
        os.remove(tmp.name)
    
    return result["text"]
