import whisper
import os

audio_path = "sample1.mp3"  # Your audio file
output_dir = "outputs"      # Folder to save results
os.makedirs(output_dir, exist_ok=True)  # Create outputs/ if it doesn't exist
output_file = os.path.join(output_dir, "whisper_transcription.txt")  # Output text file

# Load the Whisper model
model = whisper.load_model("base")  # You can change to "small", "medium", or "large" for better accuracy

# Transcribe the audio file
result = model.transcribe(audio_path)
detected_language = result['language']
original_text = result['text']

# Prepare the output lines
lines = [
    f"Detected language: {detected_language}",
    "Transcription in original language:",
    original_text
]

# Translate to English if needed
if detected_language != "en":
    result_en = model.transcribe(audio_path, task="translate")
    lines.append("\nTranslation to English:")
    lines.append(result_en["text"])
else:
    lines.append("\nAudio is already in English.")

# Save the output to a text file
with open(output_file, "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n")

print(f"Whisper transcription saved to: {output_file}")
