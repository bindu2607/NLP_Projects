import vosk
import pyaudio
import json
import os

# Path to your downloaded Vosk model folder
model_path = "vosk-model-small-en-us-0.15"  # Change if your folder name is different
output_file = "outputs/vosk_transcription.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Load the Vosk model and set up the recognizer
model = vosk.Model(model_path)
rec = vosk.KaldiRecognizer(model, 16000)

# Set up microphone input
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
stream.start_stream()

print("Speak into the microphone. Press Ctrl+C to stop.")
try:
    with open(output_file, "a", encoding="utf-8") as f:
        while True:
            data = stream.read(4000, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                result = rec.Result()
                text = json.loads(result)["text"]
                if text:
                    print("Live transcript:", text)
                    f.write(text + "\n")
except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
