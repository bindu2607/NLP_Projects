# tests/test_whisper_integration.py

from utils.whisper_integration import transcribe_audio

def test_transcribe_audio():
    class DummyFile:
        def read(self):
            with open("tests/assets/sample.wav", "rb") as f:
                return f.read()

    dummy_audio = DummyFile()
    transcript = transcribe_audio(dummy_audio)
    assert isinstance(transcript, str)
    assert len(transcript) > 0
