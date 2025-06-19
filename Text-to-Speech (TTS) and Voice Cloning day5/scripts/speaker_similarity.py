from resemblyzer import preprocess_wav, VoiceEncoder
import numpy as np
import logging
import os

os.makedirs("outputs", exist_ok=True)
logging.basicConfig(filename='outputs/app.log', level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    encoder = VoiceEncoder()
    orig = preprocess_wav("input/original_speaker.wav")
    gen = preprocess_wav("outputs/tts_output.wav")
    orig_embed = encoder.embed_utterance(orig)
    gen_embed = encoder.embed_utterance(gen)
    similarity = np.dot(orig_embed, gen_embed)
    with open("outputs/similarity_score.txt", "w") as f:
        f.write(f"Cosine similarity: {similarity:.4f}\n")
    logger.info(f"Speaker similarity computed: {similarity:.4f}")
    print(f"Speaker similarity (cosine): {similarity:.4f} (saved to outputs/similarity_score.txt)")
except Exception as e:
    logger.error(f"Speaker similarity computation failed: {e}")
    print("Error during speaker similarity computation:", e)
