import os
from elevenlabs import generate, set_api_key

class ElevenLabsTTS:
    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("ElevenLabs API key not set")
        set_api_key(api_key)

    def synthesize(self, text: str, voice: str = "Rachel", model: str = "eleven_monolingual_v1") -> bytes:
        # See https://elevenlabs.io/docs for available voices/models
        audio = generate(text=text, voice=voice, model=model)
        return audio
