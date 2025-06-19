import whisper
import pyaudio
import numpy as np
import queue
import threading
import os

# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_SECONDS = 8

audio_queue = queue.Queue()
model = whisper.load_model("base")  # You can use "small" or "medium" for better accuracy

# Save to a different file
output_file = "outputs/live_transcript.txt"

# Make sure the outputs directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

def record_audio():
    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Speak into the microphone. Press Ctrl+C to stop.")
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_queue.put(data)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

def transcribe_live():
    buffer = b""
    chunk_window = int(RATE * CHUNK_SECONDS)
    print("Transcribing live audio...")
    while True:
        try:
            buffer += audio_queue.get(timeout=1)
            if len(buffer) >= chunk_window * 2:
                audio_np = np.frombuffer(buffer[:chunk_window*2], np.int16).astype("float32") / 32768.0
                if np.abs(audio_np).mean() < 0.01:
                    buffer = buffer[chunk_window*2:]
                    continue
                result = model.transcribe(audio_np, language="en", fp16=False)
                text = result["text"].strip()
                if text:
                    print("Live transcript:", text)
                    # Save to the new file
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(text + "\n")
                buffer = buffer[chunk_window*2:]
        except queue.Empty:
            continue

if __name__ == "__main__":
    threading.Thread(target=record_audio, daemon=True).start()
    transcribe_live()
