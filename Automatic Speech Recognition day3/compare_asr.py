import re
from termcolor import colored
from jiwer import wer

# Helper: Normalize text (lowercase, remove punctuation)
def normalize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Read lines from file, skipping blank lines
def read_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

# File paths
whisper_file = 'outputs/live_transcript.txt'
vosk_file = 'outputs/vosk_transcription.txt'

# Read and normalize lines
whisper_lines = [normalize(line) for line in read_lines(whisper_file)]
vosk_lines = [normalize(line) for line in read_lines(vosk_file)]

# Pad shorter transcript so both have the same number of lines
max_len = max(len(whisper_lines), len(vosk_lines))
whisper_lines += [''] * (max_len - len(whisper_lines))
vosk_lines += [''] * (max_len - len(vosk_lines))

print(f"\n{'Whisper':<55} | {'Vosk'}")
print('-'*55 + '+' + '-'*55)

for w_line, v_line in zip(whisper_lines, vosk_lines):
    w_words = w_line.split()
    v_words = v_line.split()
    max_words = max(len(w_words), len(v_words))
    w_words += [''] * (max_words - len(w_words))
    v_words += [''] * (max_words - len(v_words))
    w_colored = []
    v_colored = []
    for w, v in zip(w_words, v_words):
        if w == v:
            w_colored.append(w)
            v_colored.append(v)
        else:
            # Highlight differences in red
            w_colored.append(colored(w, 'red') if w else '')
            v_colored.append(colored(v, 'red') if v else '')
    print(f"{' '.join(w_colored):<55} | {' '.join(v_colored)}")

# Calculate WER
whisper_text = ' '.join(whisper_lines)
vosk_text = ' '.join(vosk_lines)
word_error_rate = wer(whisper_text, vosk_text)

print("\n" + "="*50)
print(f"Word Error Rate (WER) between Whisper and Vosk: {word_error_rate:.2%}")
print("="*50)
