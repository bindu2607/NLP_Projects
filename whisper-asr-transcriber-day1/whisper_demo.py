import whisper
import os

# Define paths
audio_path = "sample1.mp3"  # Input audio file
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "transcription.txt")

# Load Whisper model
model = whisper.load_model("base")  # Options: "tiny", "small", "medium", "large"

# Transcribe and detect language
result = model.transcribe(audio_path)
detected_language = result['language']
original_text = result['text']

# Prepare output content
output_lines = [
    f"Detected language: {detected_language}",
    "Transcription in original language:",
    original_text
]

# If not English, translate
if detected_language != "en":
    result_en = model.transcribe(audio_path, task="translate")
    output_lines.append("\nTranslation to English:")
    output_lines.append(result_en["text"])
else:
    output_lines.append("\nAudio is already in English.")

# Print output to console
for line in output_lines:
    print(line)

# Save output to file
with open(output_file, "w", encoding="utf-8") as f:
    for line in output_lines:
        f.write(line + "\n")

print(f"\nTranscription and translation saved to: {output_file}")
