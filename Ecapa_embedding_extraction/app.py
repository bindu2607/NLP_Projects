#!/usr/bin/env python3
import streamlit as st
import numpy as np
import torch
import torchaudio
from pathlib import Path
from sklearn.preprocessing import normalize

from speechbrain.inference.speaker import EncoderClassifier

# ------------------ CONFIG ------------------
TARGET_SR = 16000
EMB_DIM = 192
MIN_DUR = 1.2
MAX_DUR = 600.0
MIN_ENERGY = 1e-5

# Default AISHELL3 processed path
DEFAULT_DATA_ROOT = Path(
    r"D:\Oscowl ai\AISHELL3 Speaker Embedding Extractor day8\data\processed\balanced_segments"
)

# ------------------ HELPERS ------------------
@st.cache_resource
def load_model(device="cpu"):
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )

def load_audio(path: Path):
    wav, sr = torchaudio.load(str(path))
    if sr != TARGET_SR:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)(wav)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)
    dur = wav.shape[-1] / TARGET_SR
    if dur < MIN_DUR or dur > MAX_DUR:
        return None
    energy = torch.mean(wav**2).item()
    if energy < MIN_ENERGY:
        return None
    wav = wav - torch.mean(wav)
    peak = torch.max(torch.abs(wav))
    if float(peak) > 1e-6:
        wav = wav / peak * 0.95
    return wav

def extract_embedding(model, wav: torch.Tensor):
    with torch.no_grad():
        emb = model.encode_batch(wav).squeeze()
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()
        emb = emb.reshape(1, -1)
        emb = normalize(emb, axis=1).astype(np.float32).reshape(-1)
        return emb

# ------------------ STREAMLIT UI ------------------
st.set_page_config(page_title="ECAPA-TDNN Embedding Demo", layout="wide")
st.title("🔊 ECAPA-TDNN Speaker Embedding Extraction (AISHELL-3)")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Using device: **{device}**")
model = load_model(device)

mode = st.radio("Select input mode", ["Upload .wav", "Choose from AISHELL3 balanced_segments"])

wav_path = None
if mode == "Upload .wav":
    uploaded = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded is not None:
        tmp_path = Path("temp.wav")
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())
        wav_path = tmp_path
elif mode == "Choose from AISHELL3 balanced_segments":
    split_choice = st.selectbox("Select split", ["train", "test"])
    split_root = DEFAULT_DATA_ROOT / split_choice
    speakers = sorted([d for d in split_root.glob("*") if d.is_dir()])
    spk_choice = st.selectbox("Select speaker", speakers, format_func=lambda x: x.name)
    if spk_choice:
        wav_files = sorted(spk_choice.glob("*.wav"))
        wav_choice = st.selectbox("Select wav file", wav_files, format_func=lambda x: x.name)
        if wav_choice:
            wav_path = wav_choice

if wav_path is not None:
    st.audio(str(wav_path))
    audio = load_audio(wav_path)
    if audio is None:
        st.error("Audio is too short, too long, or low energy.")
    else:
        emb = extract_embedding(model, audio)
        st.success(f"Embedding extracted! Shape: {emb.shape}")
        st.write(emb[:20])  # preview first 20 dims

        npy_name = wav_path.stem + "_embedding.npy"
        np.save(npy_name, emb)
        with open(npy_name, "rb") as f:
            st.download_button("Download Embedding (.npy)", f, file_name=npy_name)
