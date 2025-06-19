import os
import subprocess

def convert_to_wav(input_path, output_path, sample_rate=16000):
    """Convert audio file to WAV format with specified sample rate."""
    command = [
        'ffmpeg',
        '-y',  # overwrite output
        '-i', input_path,
        '-ar', str(sample_rate),
        '-ac', '1',  # mono
        output_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
