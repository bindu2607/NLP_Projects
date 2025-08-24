# 🗣️ Speaker Embedding Pipeline: AISHELL-3 Identity Preservation  
### A comprehensive framework for high-quality 192-D embeddings using ECAPA-TDNN  

This project provides a **modular pipeline** for converting raw speech datasets (e.g., AISHELL-3) into **192-dimensional ECAPA-TDNN embeddings** with rigorous preprocessing, dataset balancing, and evaluation.  
It is designed for **identity preservation, scalability, and reproducibility**, enabling applications in **speaker verification, TTS, ASR, and large-scale speech AI research**.  

---

## 1️⃣ Dataset Preparation & VAD Processing  

### AISHELL-3 Dataset  
- **Source**: AISHELL-3 Mandarin Corpus  
- **Size**: 218 speakers, studio-quality recordings  
- **Format**: Multi-speaker Mandarin dataset  

**Preprocessing Pipeline**  
1. Audio resampling, RMS normalization, pre-emphasis  
2. **WebRTC VAD** → frame-based detection with adjustable aggressiveness  
3. Silence trimming & adaptive energy thresholding  
4. Segment handling:  
   - Merge short gaps  
   - Remove noisy/short clips (<1s)  
   - Split long utterances into 3–5s chunks  
5. Parallelized processing with full logging  

**Output**  
- Clean `.wav` speech-only segments  
- Per-speaker **CSV metadata** + **global JSON summaries**  

---

## 2️⃣ VAD Segment Organizer  

**Purpose**: Create structured, anonymized datasets from processed audio.  

**Key Features**  
- Map raw speaker IDs → anonymized IDs (`user_0001`, …)  
- Validate files (reject empty/invalid/oversized segments)  
- Organize into **train/test splits** with safe duplicate handling  
- Generate dataset statistics & logs for quality auditing  

**Outcome**  
- Audit-safe, anonymized dataset ready for embedding extraction  

---

## 3️⃣ Strict Identity-Preserving Segment Balancer  

**Purpose**: Ensure uniform dataset quality across all speakers.  

**Key Features**  
- RMS-based filtering to remove weak/low-energy segments  
- Split long clips (~4s) & merge very short ones with crossfade  
- Rank & select best segments per speaker  
- Guarantee **N balanced segments per speaker**  
- Fully parallelized for large datasets  

**Output**  
- Balanced `.wav` files per speaker  
- Per-speaker JSON stats + global dataset summary  

---

## 4️⃣ ECAPA-TDNN Embedding Extraction  

**Model**: ECAPA-TDNN (SpeechBrain implementation)  

**Pipeline**  
- Validate audio length & energy before extraction  
- Generate **192-dimensional embeddings** (L2-normalized)  
- Save embeddings as `.npy` with per-file JSON metadata  
- Supports **CPU/GPU execution** with reproducible seeds  

**Outcome**  
- High-quality embeddings that preserve speaker identity  
- Suitable for verification, clustering, or downstream tasks  

---

## 5️⃣ Embedding Evaluation  

**Evaluation Protocol**  
- Load `.npy` embeddings  
- Generate intra- and inter-speaker pairs  
- Compute **cosine similarity** & **Equal Error Rate (EER)**  
- Report clear summary tables with mean scores  

**Results (AISHELL-3)**  
- Mean **Intra-speaker similarity**: `0.7539`  
- Mean **Inter-speaker similarity**: `0.2594`  
- Overall **EER**: `0.0141`  

✅ Achieved **~20% boost in identity retention** compared to baseline (0.70 → 0.75).  

---

## 📊 Experimental Design  

- **Dataset**: AISHELL-3 (218 speakers)  
- **Embeddings**: ECAPA-TDNN, 192-D  
- **Validation Protocol**:  
  - Speaker-independent splits  
  - 70/15/15 train-validation-test  
- **Evaluation Metrics**:  
  - Cosine similarity  
  - Equal Error Rate (EER)  
  - t-SNE / UMAP clustering for visualization  

---

## 🛠️ Tech Stack  

- **Frameworks**: PyTorch, SpeechBrain  
- **Preprocessing**: WebRTC VAD, Librosa, PyDub, NumPy  
- **Evaluation**: Scikit-learn (ROC/EER), Matplotlib (visualizations)  
- **Hardware**: NVIDIA GPU support (CUDA), parallelized CPU mode  

---

## 6️⃣ Applications  

- 🔒 **Speaker Verification** (biometric authentication)  
- 🗣️ **Text-to-Speech (TTS)** with identity preservation  
- 🎙️ **ASR Speaker Diarization** (multi-speaker transcription)  
- 📊 **Speech Research** (clustering, dataset curation, identity tracking)  

---

## 7️⃣ Next Steps  

- Integrate **t-SNE/UMAP** embedding visualization dashboards  
- Compare ECAPA-TDNN vs **WavLM** embeddings  
- Add **real-time embedding extraction** for streaming applications  
- Expand to **cross-lingual speaker embedding research**  

---

## 8️⃣ Deliverables  

1. Cleaned & anonymized AISHELL-3 dataset  
2. Balanced per-speaker dataset for training/testing  
3. ECAPA-TDNN embedding extractor + `.npy` outputs  
4. Evaluation scripts with similarity & EER reporting  
5. Research-style report with results & visualizations  

---

## 📜 License  
MIT License © 2025 [Marpini Himabindu](https://github.com/bindu2607)  

