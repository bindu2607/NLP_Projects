# 🗣️ Text-to-Speech (TTS) and Voice Cloning Evaluation Toolkit

## 📖 Overview

This project offers a complete, modular pipeline for evaluating **Text-to-Speech (TTS)** and **Voice Cloning** systems. It enables users to:

- Synthesize speech using a reference speaker 
- Evaluate similarity using speaker embeddings
- Analyze prosody (pitch, tempo, energy)
- Visualize pitch contours and spectrograms
- Log and save reproducible results

## 🚀 Usage Instructions

### 1. Prepare Inputs
Place your reference speaker audio and the translated text in the `input/` folder:
- `original_speaker.wav`
- `translated_text.txt` (e.g., `"Il s'agit d'une séquence de sons purs à différentes fréquences."`)

### 2. Run TTS Synthesis

```bash
python scripts/run_yourtts_tts.py
```
✅ Output: outputs/tts_output.wav

3. Compute Speaker Similarity
```
python scripts/speaker_similarity.py
```
✅ Output: outputs/similarity_score.txt

4. Analyze Prosody
```
python scripts/analyze_prosody.py
```
✅ Output: outputs/prosody_analysis.txt

5. Visualize Audio Features
```
python scripts/plot_visualizations.py
```
✅ Outputs:

outputs/pitch_contour.png

outputs/spectrogram.png

## 🛠️ Technologies Used

Component	Description
Coqui TTS (YourTTS)	Multilingual, zero-shot TTS synthesis engine
Resemblyzer	Computes speaker embeddings for similarity scoring
librosa	Audio processing and feature extraction
matplotlib	Visualizes pitch contour and spectrogram
logging	Unified logging across scripts

## 📦 Output Summary

File	Description
tts_output.wav	Synthesized speech output
similarity_score.txt	Cosine similarity score between reference and generated voice
prosody_analysis.txt	Mean pitch, energy, and tempo comparisons
pitch_contour.png	Visualization of F0 pitch curve
spectrogram.png	Audio spectrogram of the generated output
app.log	Unified log for all script operations

## 🤝 Contributions

Contributions are welcome! If you'd like to improve this toolkit, fix bugs, or add new features:

1. ⭐ Star the repository to show support.
2. Fork the repo and create a new branch:  
   `git checkout -b feature-name`
3. Make your changes and commit:  
   `git commit -m "Add feature/fix"`
4. Push to your fork:  
   `git push origin feature-name`
5. Open a pull request with a clear explanation of your changes.

For major changes or design proposals, feel free to open an issue first to discuss your ideas.

---

> 📧 For collaboration inquiries, please contact via email or GitHub Discussions.




