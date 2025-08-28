#!/usr/bin/env python3
import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import math
import sys

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import scipy.signal as sg

SAMPLE_RATE = 16000
TARGET_LUFS = -22.0
SEED = 42

# Default dataset roots (adjust)
DEFAULT_MUSAN_DIR = Path(r"D:\Oscowl ai\AISHELL3 Speaker Embedding Extractor day8\data\musan\musan\musan")
DEFAULT_RIRS_DIR = Path(r"D:\Oscowl ai\AISHELL3 Speaker Embedding Extractor day8\data\RIRS_NOISES\RIRS_NOISES\RIRS_NOISES")
DEFAULT_IN_DIR = Path(r"D:\Oscowl ai\AISHELL3 Speaker Embedding Extractor day8\data\processed\balanced_segments")
DEFAULT_OUT_DIR = Path(r"D:\Oscowl ai\AISHELL3 Speaker Embedding Extractor day8\data\processed\augmented_segments_identity")

# Stricter, identity-preserving AUG_CONFIG
AUG_CONFIG = {
    "noise": {"prob": 0.06, "snr_db": (29.0, 30.0)},
    "reverb": {"prob": 0.02, "rt60": (0.12, 0.20), "wet": (0.04, 0.045)},
    "speed": {"prob": 0.02, "range": (0.999, 1.001)},
    "pitch": {"prob": 0.0, "steps": (-0.05, 0.05)},
    "volume": {"prob": 0.04, "gain_db": (-0.6, 0.6)},
    "eq": {"prob": 0.02, "bands": [(200, 400), (1800, 3200)], "gain_db": (-0.6, 0.6)},
}

def setup_logger(log_file: Path, level: str = "INFO", to_console: bool = True) -> logging.Logger:
    logger = logging.getLogger("AUG_IDENTITY")
    logger.setLevel(getattr(logging, level))
    if logger.hasHandlers():
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

def set_seed(s: int = SEED):
    random.seed(s)
    np.random.seed(s)

def normalize_lufs_simple(x: np.ndarray, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    target_lin = 10 ** (target_lufs / 20.0)
    y = x * (target_lin / rms)
    peak = np.max(np.abs(y))
    if peak > 0.999:
        y = y / peak * 0.98
    return y.astype(np.float32)

def soft_limiter(x: np.ndarray, clip: float = 0.995) -> np.ndarray:
    peak = np.max(np.abs(x)) + 1e-12
    if peak <= clip:
        return x.astype(np.float32)
    return (x / peak * clip).astype(np.float32)

def fade_edges(x: np.ndarray, fade_dur: float = 0.008, sr: int = SAMPLE_RATE) -> np.ndarray:
    n = int(sr * fade_dur)
    if len(x) <= 2 * n or n <= 0:
        return x
    f = np.linspace(0.0, 1.0, n, dtype=x.dtype)
    y = x.copy()
    y[:n] *= f
    y[-n:] *= f[::-1]
    return y

def match_length(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) == target_len:
        return x
    if len(x) > target_len:
        start = (len(x) - target_len) // 2
        return x[start:start + target_len]
    pad = target_len - len(x)
    left = pad // 2
    right = pad - left
    return np.pad(x, (left, right), mode="reflect")

def load_random_noise_file(musan_dir: Optional[Path]) -> Optional[np.ndarray]:
    if musan_dir is None or not musan_dir.exists():
        return None
    folder = musan_dir / "noise"
    if not folder.exists():
        folder = musan_dir
    files = list(folder.glob("**/*.wav"))
    if not files:
        return None
    choice = random.choice(files)
    a, _ = librosa.load(str(choice), sr=SAMPLE_RATE, mono=True)
    return a.astype(np.float32)

def load_random_rir(rirs_dir: Optional[Path]) -> Optional[np.ndarray]:
    if rirs_dir is None or not rirs_dir.exists():
        return None
    files = list(rirs_dir.glob("**/*.wav"))
    if not files:
        return None
    choice = random.choice(files)
    a, _ = librosa.load(str(choice), sr=SAMPLE_RATE, mono=True)
    peak = np.max(np.abs(a)) + 1e-12
    a = (a / peak).astype(np.float32)
    return a

def colored_noise(length: int, sr: int = SAMPLE_RATE, lowpass: int = 6000) -> np.ndarray:
    w = np.random.randn(length).astype(np.float32)
    if lowpass and lowpass < sr / 2:
        b, a = sg.butter(4, lowpass / (sr / 2), btype="low")
        w = sg.lfilter(b, a, w).astype(np.float32)
    w /= (np.sqrt(np.mean(w ** 2)) + 1e-12)
    return w

def add_noise(x: np.ndarray, snr_db_range: Tuple[float, float], musan_dir: Optional[Path]) -> np.ndarray:
    noise = load_random_noise_file(musan_dir)
    if noise is None:
        noise = colored_noise(len(x))
    noise = match_length(noise, len(x))
    snr_db = random.uniform(*snr_db_range)
    rms_x = np.sqrt(np.mean(x ** 2)) + 1e-12
    rms_n = np.sqrt(np.mean(noise ** 2)) + 1e-12
    noise_scaled = noise * (rms_x / (10 ** (snr_db / 20) * rms_n))
    y = x + noise_scaled
    y = fade_edges(y, 0.008)
    return soft_limiter(y)

def synthetic_ir(rt60: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    length = int(min(max(rt60 * sr, 64), sr * 2))
    t = np.arange(length, dtype=np.float32) / sr
    tau = rt60 / math.log(1000.0)
    ir = np.exp(-t / tau).astype(np.float32)
    ir *= np.hanning(length).astype(np.float32)
    ir /= (np.sqrt(np.mean(ir ** 2)) + 1e-12)
    return ir

def add_reverb(x: np.ndarray, rirs_dir: Optional[Path], rt60_range: Tuple[float, float], wet_range: Tuple[float, float]) -> np.ndarray:
    rir = load_random_rir(rirs_dir)
    if rir is None:
        rir = synthetic_ir(rt60=random.uniform(*rt60_range))
    max_len = int(min(len(rir), max(64, int(random.uniform(*rt60_range) * SAMPLE_RATE))))
    if len(rir) > max_len:
        rir = rir[:max_len]
    conv = sg.fftconvolve(x, rir, mode="full")[:len(x)].astype(np.float32)
    wet = random.uniform(*wet_range)
    mixed = (1.0 - wet) * x + wet * conv
    rms_x = np.sqrt(np.mean(x ** 2)) + 1e-12
    rms_m = np.sqrt(np.mean(mixed ** 2)) + 1e-12
    mixed = (mixed * (rms_x / rms_m)).astype(np.float32)
    mixed = fade_edges(mixed, 0.008)
    return soft_limiter(mixed)

def change_speed(x: np.ndarray, speed_range: Tuple[float, float]) -> np.ndarray:
    rate = random.uniform(*speed_range)
    try:
        y = librosa.effects.time_stretch(x, rate)
    except Exception:
        y = librosa.resample(x, orig_sr=SAMPLE_RATE, target_sr=int(SAMPLE_RATE * rate))
    y = match_length(y, len(x))
    y = fade_edges(y, 0.006)
    return soft_limiter(y)

def change_pitch(x: np.ndarray, steps_range: Tuple[float, float]) -> np.ndarray:
    steps = random.uniform(*steps_range)
    try:
        y = librosa.effects.pitch_shift(x, sr=SAMPLE_RATE, n_steps=steps, bins_per_octave=24)
    except Exception:
        return x
    y = match_length(y, len(x))
    y = fade_edges(y, 0.006)
    return soft_limiter(y)

def change_volume(x: np.ndarray, gain_db_range: Tuple[float, float]) -> np.ndarray:
    g = random.uniform(*gain_db_range)
    y = np.clip(x * (10 ** (g / 20.0)), -0.999, 0.999).astype(np.float32)
    y = fade_edges(y, 0.006)
    return soft_limiter(y)

def apply_eq(x: np.ndarray, bands: List[Tuple[int, int]], gain_db_range: Tuple[float, float]) -> np.ndarray:
    y = x.copy().astype(np.float32)
    for low, high in bands:
        if low <= 0 or high >= SAMPLE_RATE / 2 or low >= high:
            continue
        g = random.uniform(*gain_db_range)
        b, a = sg.butter(2, [low / (SAMPLE_RATE / 2), high / (SAMPLE_RATE / 2)], btype='band')
        try:
            band = sg.lfilter(b, a, x).astype(np.float32)
        except Exception:
            continue
        y = y + band * (10 ** (g / 20.0) - 1.0)
    y = fade_edges(y, 0.006)
    return soft_limiter(y.astype(np.float32))

def choose_one_augmentation(cfg: Dict[str, Any]) -> Optional[str]:
    ops = list(cfg.keys())
    random.shuffle(ops)
    for k in ops:
        prob = float(cfg.get(k, {}).get("prob", 0.0))
        if random.random() < prob:
            return k
    return None

def process_file(
    path: Path, out_path: Path, musan_dir: Optional[Path], rirs_dir: Optional[Path],
    overwrite: bool, dry_run: bool, logger: logging.Logger, write_original: bool
) -> Optional[str]:
    try:
        x, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        if x.size == 0:
            logger.warning(f"Empty audio: {path}")
            return None
        x = normalize_lufs_simple(x)

        aug_choice = choose_one_augmentation(AUG_CONFIG)
        y = x.copy()
        if aug_choice == "noise":
            y = add_noise(y, AUG_CONFIG["noise"]["snr_db"], musan_dir)
        elif aug_choice == "reverb":
            y = add_reverb(y, rirs_dir, AUG_CONFIG["reverb"]["rt60"], AUG_CONFIG["reverb"]["wet"])
        elif aug_choice == "speed":
            y = change_speed(y, AUG_CONFIG["speed"]["range"])
        elif aug_choice == "pitch":
            y = change_pitch(y, AUG_CONFIG["pitch"]["steps"])
        elif aug_choice == "volume":
            y = change_volume(y, AUG_CONFIG["volume"]["gain_db"])
        elif aug_choice == "eq":
            y = apply_eq(y, AUG_CONFIG["eq"]["bands"], AUG_CONFIG["eq"]["gain_db"])
        else:
            # keep original (and optionally write a clean copy if requested)
            aug_choice = "none"

        y = fade_edges(y, 0.006)
        y = soft_limiter(y)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if write_original and aug_choice != "none":
            # Also store a clean twin for evaluation/ablation
            clean_out = out_path.with_name(out_path.stem.replace("_aug", "_clean") + out_path.suffix)
            if (not clean_out.exists()) or overwrite:
                sf.write(str(clean_out), x, SAMPLE_RATE, subtype="PCM_16")

        if (not out_path.exists()) or overwrite:
            sf.write(str(out_path), y, SAMPLE_RATE, subtype="PCM_16")

        meta = {
            "input": str(path),
            "output": str(out_path),
            "sr": SAMPLE_RATE,
            "aug": aug_choice,
            "targets": {"same_speaker_similarity_min": 0.80, "eer_max_percent": 1.0},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        try:
            with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not write metadata for {out_path}: {e}")
        return aug_choice
    except Exception as e:
        logger.error(f"Failed processing {path}: {e}")
        return None

def process_dataset(
    in_root: Path, out_root: Path, musan_dir: Optional[Path], rirs_dir: Optional[Path],
    num_aug_per_file: int, overwrite: bool, dry_run: bool, logger: logging.Logger,
    write_original: bool
) -> Dict[str, Any]:
    t0 = time.time()
    stats = {
        "speakers": 0,
        "inputs": 0,
        "outputs": 0,
        "files_skipped_existing": 0,
        "augmented_counts": {},
        "aug_none": 0,
        "duration_sec": 0.0,
        "config": {"sample_rate": SAMPLE_RATE, "target_lufs": TARGET_LUFS, "seed": SEED, "aug": AUG_CONFIG}
    }

    all_files = sorted(in_root.glob("**/*.wav"))
    if not all_files:
        logger.warning("No audio files found.")
        return stats

    speaker_ids = set()
    for path in tqdm(all_files, desc="Augmenting"):
        speaker_ids.add(path.parent.name)
        stats["inputs"] += 1

        for i in range(num_aug_per_file):
            rel = path.relative_to(in_root)
            out_path = out_root / rel.parent / f"{rel.stem}_aug{i+1}.wav"

            if out_path.exists() and not overwrite:
                stats["files_skipped_existing"] += 1
                continue
            if dry_run:
                stats["outputs"] += 1
                continue

            aug_choice = process_file(path, out_path, musan_dir, rirs_dir, overwrite, dry_run, logger, write_original)
            if aug_choice is not None:
                stats["outputs"] += 1
                if aug_choice == "none":
                    stats["aug_none"] += 1
                else:
                    stats["augmented_counts"][aug_choice] = stats["augmented_counts"].get(aug_choice, 0) + 1

    stats["speakers"] = len(speaker_ids)
    stats["duration_sec"] = round(time.time() - t0, 2)
    return stats

def convert_paths_to_str(obj):
    if isinstance(obj, dict):
        return {k: convert_paths_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths_to_str(i) for i in obj]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj

def main():
    parser = argparse.ArgumentParser(description="Identity-preserving augmentation (verification-friendly)")
    parser.add_argument("--in", dest="in_dir", default=DEFAULT_IN_DIR, help="Input root")
    parser.add_argument("--out", dest="out_dir", default=DEFAULT_OUT_DIR, help="Output root")
    parser.add_argument("--musan", dest="musan_dir", default=str(DEFAULT_MUSAN_DIR), help="MUSAN root")
    parser.add_argument("--rirs", dest="rirs_dir", default=str(DEFAULT_RIRS_DIR), help="RIRS_NOISES root")
    parser.add_argument("--num_aug", type=int, default=1, help="Augmentations per file (use 1)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--dry_run", action="store_true", help="Simulate without writing files")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--no-console", action="store_true", help="Disable console logging")
    parser.add_argument("--summary", dest="summary_file", default="augment_summary.json", help="Summary JSON output")
    parser.add_argument("--write-original", action="store_true", help="Also save a clean copy for analysis")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    logs_dir = Path("logs")
    logger = setup_logger(logs_dir / "augment_identity.log", level=args.log_level, to_console=not args.no_console)

    in_root = Path(args.in_dir)
    out_root = Path(args.out_dir)
    musan = Path(args.musan_dir) if args.musan_dir else None
    rirs = Path(args.rirs_dir) if args.rirs_dir else None

    if not in_root.exists():
        logger.error(f"Input directory does not exist: {in_root}")
        sys.exit(2)

    logger.info(f"Starting augmentation: input={in_root} out={out_root} musan={musan} rirs={rirs}")
    try:
        stats = process_dataset(in_root, out_root, musan, rirs, args.num_aug, args.overwrite, args.dry_run, logger, args.write_original)
        stats["run_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        stats["cmd_args"] = vars(args)
        safe_stats = convert_paths_to_str(stats)
        summary_path = Path(args.summary_file)
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(safe_stats, f, indent=2)
            logger.info(f"Wrote summary to {summary_path}")
        except Exception as e:
            logger.warning(f"Could not write summary file {summary_path}: {e}")

        # brief stats to console
        logger.info(json.dumps({
            "inputs": stats.get("inputs"),
            "outputs": stats.get("outputs"),
            "augmented_counts": stats.get("augmented_counts"),
            "aug_none": stats.get("aug_none"),
            "speakers": stats.get("speakers"),
            "duration_sec": stats.get("duration_sec")
        }, indent=2))
        logger.info("Augmentation finished.")
    except KeyboardInterrupt:
        logger.warning("User interrupted the run.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception during processing: {e}", exc_info=True)
        sys.exit(3)

if __name__ == "__main__":
    main()
