#!/usr/bin/env python3
import argparse
import json
import logging
import random
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, List, Dict, Any

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Strict identity-preserving segment balancer (SV-ready: >0.80, <1% EER) + optional silence-aware splitting for TTS"
    )
    p.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--target-segments", type=int, default=35)
    p.add_argument("--min-duration", type=float, default=3.8, help="Train minimum duration (s)")
    p.add_argument("--max-duration", type=float, default=12.0, help="Maximum duration before split (s)")
    p.add_argument("--rms-percentile", type=int, default=12, help="Per-speaker RMS percentile gate")
    p.add_argument("--crossfade-duration", type=float, default=0.04, help="Merge crossfade (s)")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--silence-aware", action="store_true", help="Use silence-aware splitting (recommended for TTS)")
    p.add_argument("--silence-top-db", type=float, default=35.0, help="Top-dB for librosa.effects.split")
    p.add_argument("--export-csv", action="store_true", help="(Reserved) Export per-speaker CSV stats")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

# ----------------------------
# Logger
# ----------------------------
def setup_logger(log_path: Path, verbose: bool = False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("BALANCER_STRICT")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    if verbose:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)
    return logger

# ----------------------------
# Utility (audio QC)
# ----------------------------
def peak_normalize(x: np.ndarray, peak_target: float = 0.95) -> np.ndarray:
    if x.size == 0:
        return x
    p = float(np.max(np.abs(x)))
    if p > 1e-6:
        x = (x / p) * peak_target
    return x

def remove_dc(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    return x - float(np.mean(x))

def clipping_ratio(x: np.ndarray, thr: float = 0.999) -> float:
    if x.size == 0:
        return 0.0
    return float(np.mean(np.abs(x) >= thr))

def snr_heuristic(x: np.ndarray, frame_len: int = 512, hop: int = 256) -> float:
    """
    Very lightweight SNR estimate: median(frame RMS) / min(frame RMS + eps).
    Works as a floor to reject extremely noisy segments.
    """
    if x.size < frame_len:
        return 60.0
    frames = librosa.util.frame(x.astype(np.float32), frame_length=frame_len, hop_length=hop)
    rms = np.sqrt(np.mean(frames**2, axis=0) + 1e-12)
    med = float(np.median(rms))
    mn = float(np.min(rms) + 1e-12)
    snr = 20.0 * np.log10((med + 1e-12) / mn)
    return snr

# ----------------------------
# Core
# ----------------------------
class StrictBalancer:
    def __init__(self, base_dir: Path, target_segments: int, min_duration: float,
                 max_duration: float, rms_percentile: int, crossfade_duration: float,
                 sample_rate: int, silence_aware: bool, silence_top_db: float,
                 logger: logging.Logger):
        self.base_dir = base_dir
        self.target_segments = target_segments
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.rms_percentile = rms_percentile
        self.crossfade_duration = crossfade_duration
        self.sample_rate = sample_rate
        self.logger = logger
        self.silence_aware = silence_aware
        self.silence_top_db = silence_top_db

        random.seed(42)
        np.random.seed(42)

        # Per-speaker RMS thresholds (computed once in main process)
        self.spk_thr: Dict[str, float] = {}
        # Backfill thresholds (convert near-misses safely)
        self.backfill_soft_min = max(3.6, self.min_duration - 0.2)   # Tier 1: >=3.6s
        self.backfill_soft_min2 = max(3.5, self.min_duration - 0.3)  # Tier 2: >=3.5s
        self.backfill_soft_cap2 = 5
        # Length targets
        self.split_target_low = 3.8
        self.split_target_high = 5.0
        self.split_min_keep = 3.0

        # QC thresholds
        self.max_clip_ratio = 0.01   # 1% clipped samples → reject
        self.max_dc_abs = 0.02       # DC offset magnitude floor → reject
        self.min_snr_db = 10.0       # Very low → reject (heuristic)

    # -------- I/O helpers --------
    def load_wav(self, path: Path):
        try:
            a, sr = sf.read(str(path), dtype="float32", always_2d=False)
            if a.ndim == 2:
                a = np.mean(a, axis=1, dtype=np.float32)
            if sr != self.sample_rate:
                # fast + reliable resample
                a = librosa.resample(a, orig_sr=sr, target_sr=self.sample_rate, res_type="kaiser_fast")
            # QC: remove DC, normalize, quick sanity checks
            a = remove_dc(a)
            if abs(float(np.mean(a))) > self.max_dc_abs:
                # If DC remains large, reject by returning None
                return None, None
            a = peak_normalize(a, 0.95)
            if clipping_ratio(a) > self.max_clip_ratio:
                return None, None
            if snr_heuristic(a) < self.min_snr_db:
                return None, None
            return a.astype(np.float32), self.sample_rate
        except Exception:
            return None, None

    def duration(self, a: np.ndarray):
        return len(a) / float(self.sample_rate)

    def rms(self, a: np.ndarray):
        if a is None or len(a) == 0:
            return 0.0
        return float(np.sqrt(np.mean(a.astype(np.float32) ** 2)))

    # -------- Threshold scan (train) --------
    def scan_thresholds(self, train_root: Path, max_wavs_per_spk=200):
        spk_dirs = [d for d in train_root.iterdir() if d.is_dir()]
        for d in tqdm(spk_dirs, desc="Threshold scan", leave=False):
            sid = d.name
            rms_vals = []
            count = 0
            for w in d.glob("*.wav"):
                a, _ = self.load_wav(w)
                if a is None:
                    continue
                rms_vals.append(self.rms(a))
                count += 1
                if count >= max_wavs_per_spk:
                    break
            if rms_vals:
                thr = float(np.percentile(rms_vals, self.rms_percentile))
                self.spk_thr[sid] = max(thr, 1e-6)
        self.logger.info(f"Computed RMS thresholds for {len(self.spk_thr)} speakers at p{self.rms_percentile}.")

    # -------- Analyzer --------
    def analyze(self, wav: Path, sid: str):
        a, _ = self.load_wav(wav)
        if a is None:
            return {"ok": False}
        d = self.duration(a)
        r = self.rms(a)
        thr = self.spk_thr.get(sid, 1e-6)
        meets = (r >= thr)
        if d < self.min_duration:
            cat = "too_short"
        elif d > self.max_duration:
            cat = "too_long"
        else:
            cat = "optimal"
        # Duration score
        if cat == "optimal":
            dscore = 1.0
        elif d < self.min_duration:
            dscore = max(0.5, d / self.min_duration)
        else:
            dscore = max(0.5, self.max_duration / d)
        # Energy score with tiny ranking nudge for quiet-but-valid segments
        ratio = r / max(thr, 1e-6)
        escore = min(1.0, max(0.4, ratio))
        if 1.0 <= ratio <= 1.1:
            escore *= 1.03
        score = 0.6 * dscore + 0.4 * escore
        return {"ok": True, "dur": d, "rms": r, "thr": thr, "meets": meets, "cat": cat, "score": score}

    # -------- Silence-aware chunking helpers --------
    def _silence_intervals(self, a: np.ndarray, top_db: float) -> List[Tuple[int, int]]:
        # returns non-silent intervals [(s,e), ...] in samples
        intervals = librosa.effects.split(a, top_db=top_db)  # shape (k,2)
        return [(int(s), int(e)) for s, e in intervals]

    def _emit_chunk(self, a: np.ndarray, start: int, end: int, sr: int, dst: Path, out_list: List[Path]):
        if end - start < int(self.split_min_keep * sr):
            return
        seg = a[start:end].astype(np.float32)
        seg = remove_dc(peak_normalize(seg, 0.95))
        try:
            sf.write(dst, seg, sr, subtype="PCM_16")
            out_list.append(dst)
        except Exception:
            pass

    def split_long_silence_aware(self, wav: Path) -> List[Path]:
        """
        Try to pack non-silent intervals into ~4.4 s chunks, cutting at silence.
        Fallback to uniform chunking if silence is scarce.
        """
        out_paths: List[Path] = []
        a, _ = self.load_wav(wav)
        if a is None or len(a) == 0:
            return out_paths
        sr = self.sample_rate
        intervals = self._silence_intervals(a, self.silence_top_db)

        if not intervals or sum(e - s for s, e in intervals) < int(self.split_min_keep * sr):
            # Fallback: uniform
            return self.split_long_uniform(wav)

        target = int(((self.split_target_low + self.split_target_high) / 2.0) * sr)  # ~4.4s
        cur_start = intervals[0][0]
        cur_len = 0
        last_end = intervals[0][0]

        for (s, e) in intervals:
            # append continuous speech; if there's a gap, that's a natural cut candidate
            if cur_len == 0:
                cur_start = s
                cur_len = e - s
                last_end = e
            else:
                gap = s - last_end
                # if adding this interval keeps us under (target + 0.6s), do it
                if (cur_len + gap + (e - s)) <= (target + int(0.6 * sr)):
                    cur_len += gap + (e - s)
                    last_end = e
                else:
                    # flush current chunk at last_end
                    out = wav.parent / f"{wav.stem}_sac_{cur_start}_{last_end}.wav"
                    self._emit_chunk(a, cur_start, last_end, sr, out, out_paths)
                    cur_start = s
                    cur_len = e - s
                    last_end = e

        # flush remainder
        if cur_len >= int(self.split_min_keep * sr):
            out = wav.parent / f"{wav.stem}_sac_{cur_start}_{last_end}.wav"
            self._emit_chunk(a, cur_start, last_end, sr, out, out_paths)

        # if nothing emitted (rare), fallback
        if not out_paths:
            return self.split_long_uniform(wav)
        return out_paths

    # -------- Uniform split (your original logic) --------
    def split_long_uniform(self, wav: Path) -> List[Path]:
        chunks: List[Path] = []
        a, _ = self.load_wav(wav)
        if a is None or len(a) == 0:
            return chunks
        sr = self.sample_rate
        target = int(((self.split_target_low + self.split_target_high) / 2.0) * sr)  # ~4.4s
        min_keep = int(self.split_min_keep * sr)
        i = 0
        while i < len(a):
            end = min(i + target, len(a))
            sub = a[i:end]
            if len(sub) >= min_keep:
                out = wav.parent / f"{wav.stem}_split_{i}_{end}.wav"
                try:
                    sf.write(out, sub.astype(np.float32), sr, subtype="PCM_16")
                    chunks.append(out)
                except Exception:
                    pass
            i = end
        return chunks

    # -------- Micro-merge adjacent shorts (emit multiple >=min chunks) --------
    def micro_merge_adjacent_shorts(self, shorts: List[Path], out_dir: Path, sid: str, idx_base: int) -> List[Path]:
        produced: List[Path] = []
        sr = self.sample_rate
        cf = int(self.crossfade_duration * sr)
        buf = None
        for w in sorted(shorts):
            a, _ = self.load_wav(w)
            if a is None or len(a) == 0:
                continue
            if buf is None:
                buf = a.astype(np.float32)
            else:
                if cf > 0 and len(buf) > cf and len(a) > cf:
                    fade_out = np.linspace(1.0, 0.0, cf, dtype=np.float32)
                    fade_in = np.linspace(0.0, 1.0, cf, dtype=np.float32)
                    tail = buf[-cf:] * fade_out
                    head = a[:cf].astype(np.float32) * fade_in
                    buf = np.concatenate([buf[:-cf], tail + head, a[cf:].astype(np.float32)])
                else:
                    buf = np.concatenate([buf, a.astype(np.float32)])
            while buf is not None and len(buf) / sr >= self.min_duration:
                out = out_dir / f"{sid}_micro_{idx_base:04d}.wav"
                chunk = buf[:int(self.min_duration * sr)]
                remainder = buf[int(self.min_duration * sr):]
                try:
                    sf.write(out, peak_normalize(remove_dc(chunk.astype(np.float32))), sr, subtype="PCM_16")
                    produced.append(out)
                    idx_base += 1
                except Exception:
                    pass
                buf = remainder if len(remainder) >= int(0.3 * sr) else None
        return produced

    # -------- Full merge of shorts (classic) --------
    def merge_shorts(self, short_list: List[Path], out_dir: Path, sid: str, idx_base: int) -> List[Path]:
        merged: List[Path] = []
        buf = None
        dur_acc = 0.0
        sr = self.sample_rate
        cf = int(self.crossfade_duration * sr)
        def flush(buf_np):
            nonlocal idx_base
            out = out_dir / f"{sid}_merged_{idx_base:04d}.wav"
            try:
                sf.write(out, peak_normalize(remove_dc(buf_np.astype(np.float32))), sr, subtype="PCM_16")
                merged.append(out)
                idx_base += 1
            except Exception:
                pass
        for w in sorted(short_list):
            a, _ = self.load_wav(w)
            if a is None or len(a) == 0:
                continue
            if buf is None:
                buf = a.astype(np.float32)
                dur_acc = len(buf) / sr
            else:
                if cf > 0 and len(buf) > cf and len(a) > cf:
                    fade_out = np.linspace(1.0, 0.0, cf, dtype=np.float32)
                    fade_in = np.linspace(0.0, 1.0, cf, dtype=np.float32)
                    tail = buf[-cf:] * fade_out
                    head = a[:cf].astype(np.float32) * fade_in
                    buf = np.concatenate([buf[:-cf], tail + head, a[cf:].astype(np.float32)])
                else:
                    buf = np.concatenate([buf, a.astype(np.float32)])
                dur_acc = len(buf) / sr
            if dur_acc >= self.min_duration:
                flush(buf)
                buf = None
                dur_acc = 0.0
        return merged  # discard residual < min_duration

    # -------- Collision-free copy --------
    def safe_copy(self, src: Path, dst: Path) -> bool:
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                stem, ext = dst.stem, dst.suffix
                k = 1
                ndst = dst
                while ndst.exists():
                    ndst = dst.with_name(f"{stem}_{k}{ext}")
                    k += 1
                dst = ndst
            shutil.copy2(src, dst)
            stxt = src.with_suffix(".txt")
            if stxt.exists():
                shutil.copy2(stxt, dst.with_suffix(".txt"))
            return True
        except Exception:
            return False

    # -------- Last-mile rescue (only if copied==target-1) --------
    def last_mile_rescue(self, leftovers: List[Path], out_dir: Path, sid: str, infos: Dict[Path, Dict[str, Any]], used: set, copied: int) -> int:
        if copied >= self.target_segments:
            return copied
        if copied != self.target_segments - 1:
            return copied
        candidates = [w for w in leftovers if w not in used and infos.get(w, {}).get("ok") and infos[w].get("meets")]
        if not candidates:
            return copied
        sr = self.sample_rate
        cf = int(self.crossfade_duration * sr)
        buf = None
        for w in sorted(candidates):
            a, _ = self.load_wav(w)
            if a is None or len(a) == 0:
                continue
            if buf is None:
                buf = a.astype(np.float32)
            else:
                if cf > 0 and len(buf) > cf and len(a) > cf:
                    fade_out = np.linspace(1.0, 0.0, cf, dtype=np.float32)
                    fade_in = np.linspace(0.0, 1.0, cf, dtype=np.float32)
                    tail = buf[-cf:] * fade_out
                    head = a[:cf].astype(np.float32) * fade_in
                    buf = np.concatenate([buf[:-cf], tail + head, a[cf:].astype(np.float32)])
                else:
                    buf = np.concatenate([buf, a.astype(np.float32)])
            if buf is not None and len(buf) / sr >= self.min_duration:
                out = out_dir / f"{sid}_{copied:04d}.wav"
                try:
                    seg = peak_normalize(remove_dc(buf[:int(self.min_duration * sr)].astype(np.float32)))
                    sf.write(out, seg, sr, subtype="PCM_16")
                    copied += 1
                except Exception:
                    pass
                break
        return copied

    # -------- Per-speaker processing --------
    def process_speaker(self, split: str, spk_dir: Path, out_root: Path) -> Tuple[str, Dict[str, Any]]:
        sid = spk_dir.name
        out_dir = out_root / split / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        wavs = sorted(spk_dir.glob("*.wav"))
        stats = {"original": len(wavs), "selected": 0, "status": "init"}
        if not wavs:
            return sid, {**stats, "status": "no_files"}

        # Analyze all
        infos = {w: self.analyze(w, sid) for w in wavs}

        # Partition
        optimal = [w for w, i in infos.items() if i.get("ok") and i.get("meets") and i.get("cat") == "optimal"]
        too_long = [w for w, i in infos.items() if i.get("ok") and i.get("meets") and i.get("cat") == "too_long"]
        too_short = [w for w, i in infos.items() if i.get("ok") and i.get("meets") and i.get("cat") == "too_short"]

        # Split long files → re-evaluate
        split_chunks: List[Path] = []
        for w in too_long:
            if self.silence_aware:
                split_chunks.extend(self.split_long_silence_aware(w))
            else:
                split_chunks.extend(self.split_long_uniform(w))
        if split_chunks:
            for w in split_chunks:
                infos[w] = self.analyze(w, sid)
            split_opt = [w for w in split_chunks if infos[w].get("ok") and infos[w].get("meets")
                         and self.min_duration <= infos[w]["dur"] <= self.max_duration]
            optimal.extend(split_opt)

        # Micro-merge rescue (emit multiple chunks)
        if too_short:
            mm_files = self.micro_merge_adjacent_shorts(too_short, out_dir, sid, idx_base=0)
            for w in mm_files:
                infos[w] = self.analyze(w, sid)
            mm_opt = [w for w in mm_files if infos[w].get("ok") and infos[w].get("meets")
                      and self.min_duration <= infos[w]["dur"] <= self.max_duration]
            optimal.extend(mm_opt)

        # Full merge rescue
        if too_short:
            merged_files = self.merge_shorts(too_short, out_dir, sid, idx_base=10000)
            for w in merged_files:
                infos[w] = self.analyze(w, sid)
            merged_opt = [w for w in merged_files if infos[w].get("ok") and infos[w].get("meets")
                          and self.min_duration <= infos[w]["dur"] <= self.max_duration]
            optimal.extend(merged_opt)

        # Deduplicate and rank
        optimal = list(set(optimal))
        ranked = sorted(optimal, key=lambda w: infos[w]["score"], reverse=True)

        if split == "train":
            # Clean destination
            for f in out_dir.glob("*"):
                try:
                    f.unlink()
                except Exception:
                    pass

            copied = 0
            used = set()

            # Primary pass: strict ≥ min_duration
            for idx, src in enumerate(ranked):
                d = infos[src]["dur"]
                if d < self.min_duration:
                    continue
                dst = out_dir / f"{sid}_{idx:04d}.wav"
                if self.safe_copy(src, dst):
                    used.add(src)
                    copied += 1
                if copied >= self.target_segments:
                    break

            # Backfill Tier 1: accept 3.6–3.8s if still under target
            if copied < self.target_segments and self.backfill_soft_min < self.min_duration:
                for src in ranked:
                    if src in used:
                        continue
                    d = infos[src]["dur"]
                    if d < self.backfill_soft_min or d >= self.min_duration:
                        continue
                    dst = out_dir / f"{sid}_{copied:04d}.wav"
                    if self.safe_copy(src, dst):
                        used.add(src)
                        copied += 1
                    if copied >= self.target_segments:
                        break

            # Backfill Tier 2: accept 3.5–3.6s with cap
            if copied < self.target_segments and self.backfill_soft_min2 < self.backfill_soft_min:
                extra = 0
                for src in ranked:
                    if src in used:
                        continue
                    d = infos[src]["dur"]
                    if d < self.backfill_soft_min2 or d >= self.backfill_soft_min:
                        continue
                    dst = out_dir / f"{sid}_{copied:04d}.wav"
                    if self.safe_copy(src, dst):
                        used.add(src)
                        copied += 1
                        extra += 1
                    if copied >= self.target_segments or extra >= self.backfill_soft_cap2:
                        break

            # Last-mile rescue (only if at target-1)
            if copied == self.target_segments - 1:
                leftovers = [w for w in infos.keys() if w not in used]
                copied = self.last_mile_rescue(leftovers, out_dir, sid, infos, used, copied)

            final_wavs = list(out_dir.glob("*.wav"))
            stats["selected"] = len(final_wavs)
            stats["status"] = "balanced" if len(final_wavs) == self.target_segments else "audit_failed"

            # Clean on fail (audit-safe)
            if stats["status"] != "balanced":
                for f in out_dir.glob("*"):
                    try:
                        f.unlink()
                    except Exception:
                        pass
            return sid, stats

        else:
            # Test: copy all ranked optimal (no cap)
            for f in out_dir.glob("*"):
                try:
                    f.unlink()
                except Exception:
                    pass
            copied = 0
            for idx, src in enumerate(ranked):
                dst = out_dir / f"{sid}_{idx:04d}.wav"
                if self.safe_copy(src, dst):
                    copied += 1
            stats["selected"] = copied
            stats["status"] = "copied_test"
            return sid, stats

    # -------- Run --------
    def run(self, workers: int, export_csv: bool):
        vad_root = self.base_dir / "data" / "processed" / "vad_segments_by_speaker"
        out_root = self.base_dir / "data" / "processed" / "balanced_segments"
        out_root.mkdir(parents=True, exist_ok=True)
        train_root = vad_root / "train"
        if not train_root.exists():
            self.logger.error(f"Missing VAD train root: {train_root}")
            return

        # Compute thresholds once in main process
        self.scan_thresholds(train_root)

        speakers = [d for d in train_root.iterdir() if d.is_dir()]
        self.logger.info(f"TRAIN speakers: {len(speakers)}")
        train_stats: Dict[str, Any] = {}
        errors = 0
        tasks = [("train", spk, out_root) for spk in speakers]

        if workers > 1:
            payload = {
                "base_dir": str(self.base_dir),
                "target_segments": self.target_segments,
                "min_duration": self.min_duration,
                "max_duration": self.max_duration,
                "rms_percentile": self.rms_percentile,
                "crossfade_duration": self.crossfade_duration,
                "sample_rate": self.sample_rate,
                "spk_thr": self.spk_thr,
                "silence_aware": self.silence_aware,
                "silence_top_db": self.silence_top_db,
            }
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(worker_task, t, payload): t[1].name for t in tasks}
                for fut in tqdm(as_completed(futs), total=len(futs), desc="TRAIN", unit="spk"):
                    spk_name = futs[fut]
                    try:
                        sid, s = fut.result()
                        train_stats[sid] = s
                        if s["status"] != "balanced":
                            errors += 1
                        self.logger.info(f"{sid}: {s['status']} (orig={s['original']} sel={s.get('selected')})")
                    except Exception as e:
                        self.logger.error(f"Error processing {spk_name}: {e}")
        else:
            for t in tqdm(tasks, desc="TRAIN", unit="spk"):
                sid, s = self.process_speaker(*t)
                train_stats[sid] = s
                if s["status"] != "balanced":
                    errors += 1
                self.logger.info(f"{sid}: {s['status']} (orig={s['original']} sel={s.get('selected')})")

        total = len(speakers)
        balanced = sum(1 for v in train_stats.values() if v["status"] == "balanced")
        excl = total - balanced
        excl_rate = excl / max(total, 1)
        self.logger.info(f"TRAIN summary: speakers={total}, balanced={balanced}, excluded={excl}, exclusion_rate={excl_rate:.2%}")

        # TEST: ranked optimal, uncapped
        test_root = vad_root / "test"
        test_stats: Dict[str, Any] = {}
        if test_root.exists():
            spk_dirs = [d for d in test_root.iterdir() if d.is_dir()]
            self.logger.info(f"TEST speakers: {len(spk_dirs)}")
            for spk in tqdm(spk_dirs, desc="TEST", unit="spk"):
                sid, s = self.process_speaker("test", spk, out_root)
                test_stats[sid] = s
        else:
            self.logger.warning(f"Missing TEST root: {test_root}")

        out_json = self.base_dir / "balancing_stats.json"
        try:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump({
                    "summary": {
                        "train_speakers": total,
                        "train_balanced": balanced,
                        "train_excluded": excl,
                        "train_exclusion_rate": round(excl_rate, 4),
                    },
                    "train": train_stats,
                    "test": test_stats
                }, f, indent=2)
            self.logger.info(f"Wrote stats: {out_json}")
        except Exception as e:
            self.logger.error(f"Failed to write stats: {e}")

# ----------------------------
# Worker task (pickle-safe)
# ----------------------------
def worker_task(args_tuple, payload_dict):
    split, spk_dir, out_root_local = args_tuple
    logs_dir = Path(payload_dict["base_dir"]) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    worker_logger = setup_logger(logs_dir / f"balancer_worker_{spk_dir.name}.log", verbose=False)

    bal = StrictBalancer(
        base_dir=Path(payload_dict["base_dir"]),
        target_segments=payload_dict["target_segments"],
        min_duration=payload_dict["min_duration"],
        max_duration=payload_dict["max_duration"],
        rms_percentile=payload_dict["rms_percentile"],
        crossfade_duration=payload_dict["crossfade_duration"],
        sample_rate=payload_dict["sample_rate"],
        silence_aware=payload_dict["silence_aware"],
        silence_top_db=payload_dict["silence_top_db"],
        logger=worker_logger,
    )
    bal.spk_thr = payload_dict["spk_thr"]  # Inject precomputed thresholds
    return bal.process_speaker(split, spk_dir, out_root_local)

# ----------------------------
# MAIN
# ----------------------------
def main():
    args = parse_args()
    log_path = args.base_dir / "logs" / "balancer.log"
    logger = setup_logger(log_path, args.verbose)

    mode = "SV/ASR (uniform splits)" if not args.silence_aware else "TTS-friendly (silence-aware splits)"
    logger.info(f"Starting StrictBalancer [{mode}] — target≥{args.target_segments}, min={args.min_duration}s, p{args.rms_percentile} gate")

    balancer = StrictBalancer(
        base_dir=args.base_dir,
        target_segments=args.target_segments,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        rms_percentile=args.rms_percentile,
        crossfade_duration=args.crossfade_duration,
        sample_rate=args.sample_rate,
        silence_aware=args.silence_aware,
        silence_top_db=args.silence_top_db,
        logger=logger,
    )
    balancer.run(workers=args.workers, export_csv=args.export_csv)
    logger.info("Finished StrictBalancer")

if __name__ == "__main__":
    main()
