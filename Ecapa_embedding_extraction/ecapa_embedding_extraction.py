
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from sklearn.preprocessing import normalize

try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    print("ERROR: Install dependencies: pip install speechbrain torch torchaudio scikit-learn tqdm")
    sys.exit(1)

TARGET_SR = 16000
EMB_DIM = 192
MIN_DUR = 1.2   # seconds
MAX_DUR = 600.0 # seconds
MIN_ENERGY = 1e-5 
DEFAULT_DATA_ROOT = Path(r"D:\Oscowl ai\AISHELL3 Speaker Embedding Extractor day8\processed\balanced_segments")

DEFAULT_OUT_ROOT  = Path(r"D:\Oscowl ai\AISHELL3 Speaker Embedding Extractor day8\data\processed\embedding_extraction")
DEFAULT_LOG_DIR   = Path(r"D:\Oscowl ai\AISHELL3 Speaker Embedding Extractor day8\logs")


# ------------------ LOGGER ------------------
def setup_logger(log_file: Path, level="INFO", to_console=True):
    logger = logging.getLogger("ECAPA_EXTRACT")
    logger.setLevel(getattr(logging, level))
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(getattr(logging, level))
    logger.addHandler(fh)
    if to_console:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        ch.setLevel(getattr(logging, level))
        logger.addHandler(ch)
    return logger


# ------------------ AUDIO LOADER ------------------
def load_audio_16k_mono(path: Path, logger: logging.Logger):
    try:
        wav, sr = torchaudio.load(str(path))
        if sr != TARGET_SR:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)(wav)
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0)

        dur = wav.shape[-1] / TARGET_SR
        if dur < MIN_DUR or dur > MAX_DUR:
            logger.debug(f"Skip (duration {dur:.2f}s): {path}")
            return None
        energy = torch.mean(wav**2).item()
        if energy < MIN_ENERGY:
            logger.debug(f"Skip (low energy {energy:.6f}): {path}")
            return None

        wav = wav - torch.mean(wav)
        peak = torch.max(torch.abs(wav))
        if float(peak) > 1e-6:
            wav = wav / peak * 0.95
        return wav
    except Exception as e:
        logger.debug(f"Load failed {path}: {e}")
        return None


# ------------------ EMBEDDING EXTRACTION ------------------
def extract_emb(model: EncoderClassifier, wav: torch.Tensor, logger: logging.Logger):
    try:
        with torch.no_grad():
            emb = model.encode_batch(wav).squeeze()
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            emb = emb.reshape(1, -1)
            emb = normalize(emb, axis=1).astype(np.float32).reshape(-1)
            return emb
    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        return None


# ------------------ PROCESS SPLIT ------------------
def process_split(split: str, data_root: Path, out_root: Path, model, logger, save_emb=True):
    in_root  = data_root / split
    out_root = out_root / split

    if not in_root.exists():
        logger.warning(f"Split '{split}' not found: {in_root}")
        return

    speakers = [d for d in in_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not speakers:
        logger.warning(f"No speakers found in {in_root}")
        return

    logger.info(f"[{split}] Found {len(speakers)} speakers")

    files_total, files_ok, files_failed = 0, 0, 0

    for spk in speakers:
        wavs = sorted(spk.glob("*.wav"))
        if not wavs:
            continue
        logger.info(f"[{split}] Speaker {spk.name}: {len(wavs)} files")

        spk_out = out_root / "embeddings" / spk.name
        if save_emb:
            spk_out.mkdir(parents=True, exist_ok=True)

        for wav in tqdm(wavs, desc=f"Extract {split}/{spk.name}", leave=False):
            files_total += 1
            audio = load_audio_16k_mono(wav, logger)
            if audio is None:
                files_failed += 1
                continue

            emb = extract_emb(model, audio, logger)
            if emb is None:
                files_failed += 1
                continue

            if emb.shape != (EMB_DIM,):
                logger.error(f"Unexpected embedding shape {emb.shape} for file {wav}")
                files_failed += 1
                continue

            files_ok += 1

            if save_emb:
                try:
                    np_path = spk_out / (wav.stem + ".npy")
                    np.save(str(np_path), emb)
                except Exception as e:
                    logger.warning(f"Failed to save embedding npy for {wav}: {e}")

                meta = {
                    "audio_path": str(wav),
                    "embedding_path": str(np_path),
                    "embedding_dim": EMB_DIM,
                    "dtype": "float32",
                    "sr": TARGET_SR,
                    "timestamp": datetime.now().isoformat()
                }
                try:
                    json_path = spk_out / (wav.stem + ".json")
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)
                except Exception as e:
                    logger.debug(f"Failed to save metadata for {wav}: {e}")

    logger.info(f"[{split}] Processed {files_ok}/{files_total} ok, {files_failed} failed.")


# ------------------ MAIN ------------------
def main():
    ap = argparse.ArgumentParser(description="ECAPA-TDNN speaker embedding extraction (train + val + test)")
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Input data root containing train/val/test folders")
    ap.add_argument("--out-root",  default=str(DEFAULT_OUT_ROOT), help="Output folder root")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Which splits to process")
    ap.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="Log folder")
    ap.add_argument("--save-emb", action="store_true", help="Save embeddings as .npy and metadata JSON")
    ap.add_argument("--device", default="auto", help="Device: auto | cpu | cuda | cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    ap.add_argument("--no-console", action="store_true", help="Disable console logging")
    args = ap.parse_args()

    # Seeds and device setup
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir / "ecapa_extract.log", level=args.log_level, to_console=not args.no_console)

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model once
    logger.info("Loading ECAPA-TDNN model (SpeechBrain, 192-dim)...")
    t0 = time.time()
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    logger.info(f"Model loaded in {time.time() - t0:.2f}s")

    data_root = Path(args.data_root)
    out_root  = Path(args.out_root)

    for split in args.splits:
        process_split(split, data_root, out_root, model, logger, save_emb=args.save_emb)

    logger.info("All splits processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
