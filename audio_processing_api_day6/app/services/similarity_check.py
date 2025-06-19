"""
Speaker similarity and word-level difference highlighting service.
Compares speaker embeddings and highlights transcription mismatches.
"""

from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
from typing import List, Dict, Any
import difflib

class SimilarityCheckService:
    def __init__(self):
        self.encoder = VoiceEncoder()

    async def compare(self, audio1: bytes, audio2: bytes) -> float:
        """Compute cosine similarity between two audio samples."""
        wav1 = preprocess_wav(audio1)
        wav2 = preprocess_wav(audio2)
        emb1 = self.encoder.embed_utterance(wav1)
        emb2 = self.encoder.embed_utterance(wav2)
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        similarity = float(np.dot(emb1_norm, emb2_norm))
        return similarity

    async def batch_compare(self, reference_audio: bytes, audio_list: List[bytes]) -> List[Dict[str, Any]]:
        """Batch compare reference audio to a list of audios."""
        ref_wav = preprocess_wav(reference_audio)
        ref_emb = self.encoder.embed_utterance(ref_wav)
        ref_emb_norm = ref_emb / np.linalg.norm(ref_emb)
        results = []
        for idx, audio in enumerate(audio_list):
            try:
                wav = preprocess_wav(audio)
                emb = self.encoder.embed_utterance(wav)
                emb_norm = emb / np.linalg.norm(emb)
                score = float(np.dot(ref_emb_norm, emb_norm))
                results.append({
                    "index": idx,
                    "similarity": score,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "index": idx,
                    "similarity": None,
                    "success": False,
                    "error": str(e)
                })
        return results

    def word_level_diff(self, ref_text: str, hyp_text: str) -> List[Dict[str, Any]]:
        """
        Highlight word-level differences between two texts.
        Returns a list of dicts: {word, status}
        status: 'correct', 'missing', 'extra', 'mismatch'
        """
        ref_words = ref_text.strip().split()
        hyp_words = hyp_text.strip().split()
        matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
        diff_result = []
        for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
            if opcode == 'equal':
                for w in ref_words[i1:i2]:
                    diff_result.append({"word": w, "status": "correct"})
            elif opcode == 'replace':
                for w in ref_words[i1:i2]:
                    diff_result.append({"word": w, "status": "mismatch"})
                for w in hyp_words[j1:j2]:
                    diff_result.append({"word": w, "status": "mismatch"})
            elif opcode == 'delete':
                for w in ref_words[i1:i2]:
                    diff_result.append({"word": w, "status": "missing"})
            elif opcode == 'insert':
                for w in hyp_words[j1:j2]:
                    diff_result.append({"word": w, "status": "extra"})
        return diff_result

    def colorize_diff(self, diff: List[Dict[str, Any]]) -> str:
        """
        Returns an HTML string with color highlighting for mismatches:
        - correct: green
        - missing: red (strikethrough)
        - extra: orange (underline)
        - mismatch: blue (italic)
        """
        html = []
        for item in diff:
            word = item["word"]
            status = item["status"]
            if status == "correct":
                html.append(f'<span style="color:green">{word}</span>')
            elif status == "missing":
                html.append(f'<span style="color:red;text-decoration:line-through">{word}</span>')
            elif status == "extra":
                html.append(f'<span style="color:orange;text-decoration:underline">{word}</span>')
            elif status == "mismatch":
                html.append(f'<span style="color:blue;font-style:italic">{word}</span>')
        return " ".join(html)
