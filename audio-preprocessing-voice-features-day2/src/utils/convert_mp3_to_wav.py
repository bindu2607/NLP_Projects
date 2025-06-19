from moviepy.editor import AudioFileClip
import os

input_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw/sample1.mp3'))
output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw/sample1.wav'))

if not os.path.exists(input_file):
    print(f"File not found: {input_file}")
    exit(1)

audio_clip = AudioFileClip(input_file)
audio_clip.write_audiofile(output_file)
print("Conversion complete! WAV file saved at:", output_file)
