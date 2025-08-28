#!/usr/bin/env python3
import csv
import json
import logging
import warnings
import time
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import librosa
import numpy as np
import soundfile as sf
import webrtcvad
from tqdm import tqdm
import logging.handlers

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".pkg_resources is deprecated.")

def setup_logging(log_level: str) -> None:
    logger = logging.getLogger("AISHELL3_VAD_PROD")
    logger.propagate = False
    logger.setLevel(getattr(logging, log_level))
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] [PID:%(process)d] %(message)s',
            "%Y-%m-%d %H:%M:%S"
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_dir / "vad_processing.log", encoding="utf-8", maxBytes=1024*1024*1024, backupCount=3)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

class ProductionVADProcessor:
    """
    Identity-preserving VAD tuned for speaker verification:
    - 16 kHz end-to-end
    - frame_duration_ms = 30
    - silence_merge_gap = 0.60 (stitch micro-pauses)
    - energy_threshold = 'adaptive' at percentile 11 (with floor)
    - min_segment_length (train)=1.6s (prefilter)
    - merge short segments; split long into 3.4–5.4s chunks (max 12s)
    - 1-frame hysteresis to prevent flip-flop near threshold
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
    # VAD aggressiveness & frame
    'aggressiveness': 2,                 # moderate aggressiveness for clean speech detection
    'frame_duration_ms': 30,             # standard 30ms frames

    # Minimum segment and merging
    'min_segment_length': 1.6,           # ensures enough info per segment
    'merge_threshold_s': 1.6,            # merge short islands into useful segments
    'silence_merge_gap': 0.50,           # stitch micro-pauses, avoid merging too much

    # Sampling
    'vad_sample_rate': 16000,            # VAD expects 16kHz
    'target_sample_rate': 16000,         # resample all audio to 16kHz

    # Energy-based adaptive threshold
    'energy_threshold': 'adaptive',
    'adaptive_energy_percentile': 14,    # slightly strict to ignore very quiet/noisy frames
    'adaptive_floor': 1e-4,              # floor prevents admitting hum/background

    # Splitting
    'split_max_length_s': 12.0,          # cap long segments
    'split_chunk_min_s': 3.5,            # minimum chunk for verification
    'split_chunk_max_s': 5.5,            # max chunk length

    # Hysteresis to avoid flip-flops
    'hysteresis_keep_frames': 1,         
    'hysteresis_ratio': 0.92,            # 92% RMS relative to prior frame

    # Multiprocessing
    'multiprocessing': True,
    'max_workers': max(1, mp.cpu_count() - 1),

    # Logging & quality checks
    'log_level': 'INFO',
    'enable_quality_checks': True,
    'backup_on_overwrite': False,
}

        self.config = {**default_config, **(config or {})}
        setup_logging(self.config['log_level'])
        self.vad = webrtcvad.Vad(int(self.config['aggressiveness']))
        self.stats = defaultdict(lambda: defaultdict(int))
        self.file_vad_error_logged = False

    @property
    def logger(self):
        return logging.getLogger("AISHELL3_VAD_PROD")

    def _adaptive_threshold(self, segment: np.ndarray) -> float:
        perc = int(self.config.get('adaptive_energy_percentile', 11))
        return float(np.percentile(np.abs(segment), perc))

    def preprocess_audio(self, audio_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            audio, sr = librosa.load(
                str(audio_path),
                sr=self.config['target_sample_rate'],
                mono=True,
                dtype=np.float32
            )
            if self.config['enable_quality_checks']:
                if audio.size == 0: raise ValueError("Empty audio")
                if np.all(audio == 0): raise ValueError("All silence")
                if np.any(np.isnan(audio)) or np.any(np.isinf(audio)): raise ValueError("NaN/Inf")

            # Pre-emphasis with safe fallback
            try:
                audio = librosa.effects.preemphasis(audio, coef=0.97)
            except Exception:
                audio = np.append(audio[:1], audio[1:] - 0.97 * audio[:-1])

            # Peak normalize to 0.95
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if peak > 1e-6:
                audio = (audio / peak) * 0.95

            return audio, audio
        except Exception as e:
            self.logger.warning(f"Error preprocessing {audio_path}: {e}")
            return None, None

    def smooth_vad_decisions(self, decisions: List[bool], rms_series: Optional[np.ndarray], frame_len: int) -> List[bool]:
        # Merge micro-pauses between speech islands
        frame_ms = int(self.config['frame_duration_ms'])
        max_gap_frames = int(self.config['silence_merge_gap'] * 1000 / max(1, frame_ms))
        smoothed = decisions.copy()

        # Hysteresis: keep 1 more frame if near threshold to avoid flicker
        keep_frames = int(self.config.get('hysteresis_keep_frames', 1))
        near_ratio = float(self.config.get('hysteresis_ratio', 0.9))

        # Pass 1: stitch gaps
        for i in range(len(smoothed)):
            if not smoothed[i]:
                start = max(0, i - max_gap_frames)
                end = min(len(smoothed), i + max_gap_frames + 1)
                if any(smoothed[start:i]) and any(smoothed[i+1:end]):
                    smoothed[i] = True

        # Pass 2: hysteresis using RMS if provided
        if rms_series is not None and keep_frames > 0:
            i = 1
            while i < len(smoothed):
                if (not smoothed[i]) and smoothed[i-1]:
                    if rms_series[i] >= near_ratio * (rms_series[i-1] + 1e-12):
                        smoothed[i] = True
                        i += keep_frames
                i += 1

        return smoothed

    def extract_speech_segments(self, audio: np.ndarray, decisions: List[bool], frame_len: int) -> List[Tuple[int, int]]:
        segments: List[Tuple[int, int]] = []
        step = frame_len
        in_segment = False
        seg_start = None
        for i, sp in enumerate(decisions):
            if sp and not in_segment:
                in_segment = True
                seg_start = i
            elif not sp and in_segment:
                in_segment = False
                start_idx = seg_start * step
                end_idx = min(i * step, len(audio))
                segments.append((start_idx, end_idx))
        if in_segment and seg_start is not None:
            segments.append((seg_start * step, len(audio)))
        return segments

    def merge_short_segments(self, segments: List[Tuple[int,int]], sample_rate: int) -> List[Tuple[int,int]]:
        if not segments:
            return []
        merged: List[Tuple[int,int]] = []
        current_start, current_end = segments[0]
        merge_th = int(self.config['merge_threshold_s'] * sample_rate)
        gap_th = int(self.config['silence_merge_gap'] * sample_rate)
        for seg_start, seg_end in segments[1:]:
            current_duration = current_end - current_start
            gap = seg_start - current_end
            if current_duration < merge_th or gap <= gap_th:
                current_end = seg_end
            else:
                merged.append((current_start, current_end))
                current_start, current_end = seg_start, seg_end
        merged.append((current_start, current_end))
        return merged

    def split_long_segments(self, segments: List[Tuple[int,int]], sample_rate: int) -> List[Tuple[int,int]]:
        out: List[Tuple[int,int]] = []
        max_len = int(self.config['split_max_length_s'] * sample_rate)
        min_c = int(self.config['split_chunk_min_s'] * sample_rate)
        max_c = int(self.config['split_chunk_max_s'] * sample_rate)
        for s, e in segments:
            L = e - s
            if L <= max_len:
                out.append((s, e))
            else:
                # chunk around 3.4–5.4s
                num = max(1, int(np.ceil(L / max_c)))
                chunk = max(min_c, min(L // num, max_c))
                cur = s
                while cur < e:
                    nxt = min(cur + chunk, e)
                    out.append((cur, nxt))
                    cur = nxt
        return out

    def apply_vad(
        self,
        wav_path: Path,
        out_dir: Path,
        speaker_id: str,
        metadata: List[Dict[str, Any]]
    ) -> Tuple[int, int, int, int, int]:
        audio_raw, audio_16k = self.preprocess_audio(wav_path)
        if audio_16k is None:
            self.stats[speaker_id]['preproc_fail'] += 1
            return 0, 0, 0, 0, 1

        pcm = (audio_16k * 32767).astype(np.int16)
        frame_len = int(self.config['vad_sample_rate'] * self.config['frame_duration_ms'] / 1000)

        decisions: List[bool] = []
        rms_series: List[float] = []
        self.file_vad_error_logged = False
        for i in range(0, len(pcm) - frame_len + 1, frame_len):
            frame = pcm[i:i + frame_len]
            try:
                decisions.append(self.vad.is_speech(frame.tobytes(), self.config['vad_sample_rate']))
            except Exception as e:
                decisions.append(False)
                if not self.file_vad_error_logged:
                    self.logger.warning(f"VAD frame error in {wav_path}: {str(e)}")
                    self.file_vad_error_logged = True
            # RMS for hysteresis reference
            rms_series.append(float(np.sqrt(np.mean((frame.astype(np.float32)/32767.0) ** 2)) + 1e-12))

        decisions = self.smooth_vad_decisions(decisions, np.array(rms_series, dtype=np.float32), frame_len)
        segments_idx = self.extract_speech_segments(audio_16k, decisions, frame_len)
        segments_idx = self.merge_short_segments(segments_idx, self.config['vad_sample_rate'])
        segments_idx = self.split_long_segments(segments_idx, self.config['vad_sample_rate'])

        filename = wav_path.stem
        saved, rejected_short, rejected_energy, skipped = 0, 0, 0, 0

        for start, end in segments_idx:
            segment = audio_16k[start:end]
            duration = (end - start) / self.config['vad_sample_rate']
            start_time = start / self.config['vad_sample_rate']
            end_time = end / self.config['vad_sample_rate']

            if duration < self.config['min_segment_length']:
                rejected_short += 1
                continue

            if self.config['energy_threshold'] == 'adaptive':
                threshold = self._adaptive_threshold(segment)
                threshold = max(threshold, float(self.config.get('adaptive_floor', 8e-5)))
            else:
                threshold = float(self.config['energy_threshold'])

            rms = float(np.sqrt(np.mean(segment ** 2)))
            if rms < threshold:
                rejected_energy += 1
                continue

            output_file = out_dir / f"{filename}_seg{saved:04d}.wav"
            if output_file.exists():
                if self.config['backup_on_overwrite']:
                    backup_file = out_dir / f"{filename}_seg{saved:04d}_backup_{int(time.time())}.wav"
                    try:
                        output_file.rename(backup_file)
                        self.logger.info(f"Backed up existing file to {backup_file}")
                    except Exception:
                        pass
                else:
                    skipped += 1
                    continue

            sf.write(str(output_file), segment, self.config['vad_sample_rate'], subtype='PCM_16')
            metadata.append({
                "speaker_id": speaker_id,
                "original_file": wav_path.name,
                "segment_file": output_file.name,
                "duration": round(duration, 4),
                "rms_energy": round(rms, 6),
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
                "processing_timestamp": time.time()
            })
            saved += 1

        stats = self.stats[speaker_id]
        stats['processed_files'] += 1
        stats['segments_created'] += saved
        stats['rejected_short'] += rejected_short
        stats['rejected_energy'] += rejected_energy
        stats['skipped_exists'] += skipped

        return saved, rejected_short, rejected_energy, skipped, 0

def write_metadata_csv(metadata: List[Dict[str, Any]], csv_path: Path) -> None:
    if not metadata:
        return
    fieldnames = [
        "speaker_id", "original_file", "segment_file", "duration",
        "rms_energy", "start_time", "end_time", "processing_timestamp"
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in metadata:
            writer.writerow(row)

def process_speaker_production(args: Tuple[Path, Path, Dict[str, Any]]) -> Tuple[str, int, List[Dict[str, Any]], Dict[str, int]]:
    speaker_dir, out_dir, config = args
    speaker_id = speaker_dir.name
    speaker_out = out_dir / speaker_id
    speaker_out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("AISHELL3_VAD_PROD")

    wav_files = list(speaker_dir.glob("*.wav"))
    if not wav_files:
        logger.warning(f"No WAV files found in {speaker_dir}")
        return speaker_id, 0, [], {}

    metadata: List[Dict[str, Any]] = []
    vad_processor = ProductionVADProcessor(config)
    totals = {'segments': 0, 'short_rej': 0, 'energy_rej': 0, 'skipped': 0, 'preproc_fail': 0}

    try:
        for wav_file in wav_files:
            segments, short_rej, energy_rej, skip, preproc_fail = vad_processor.apply_vad(
                wav_file, speaker_out, speaker_id, metadata
            )
            totals['segments'] += segments
            totals['short_rej'] += short_rej
            totals['energy_rej'] += energy_rej
            totals['skipped'] += skip
            totals['preproc_fail'] += preproc_fail

        if metadata:
            metadata_csv_path = speaker_out / f"{speaker_id}_segments_metadata.csv"
            write_metadata_csv(metadata, metadata_csv_path)
        logger.info(
            f"{speaker_id}: {totals['segments']} segments | "
            f"rejected_short={totals['short_rej']}, rejected_energy={totals['energy_rej']}, "
            f"skipped={totals['skipped']} | preproc_fail={totals['preproc_fail']}"
        )
    except Exception as e:
        logger.error(f"Error processing speaker {speaker_id}: {e}")
        return speaker_id, 0, [], {}
    return speaker_id, totals['segments'], metadata, totals

def main():
    base_dir = Path(__file__).parent.parent.resolve()
    data_dir = base_dir / "data"
    # Input pattern: data/<split>/<split>/wav/<speaker>/*.wav
    input_template = "{split}/{split}/wav"
    out_base = base_dir / "data/processed/vad_segments"
    out_base.mkdir(parents=True, exist_ok=True)

    base_config = {
    # VAD aggressiveness & frame
    'aggressiveness': 2,
    'frame_duration_ms': 30,
    
    # Audio sample rates
    'vad_sample_rate': 16000,
    'target_sample_rate': 16000,
    
    # Energy thresholding
    'energy_threshold': 'adaptive',
    'adaptive_energy_percentile': 14,  # stricter to reject ultra-quiet/noisy frames
    'adaptive_floor': 1e-4,            # floor to prevent passing very low energy noise
    
    # Silence & segment handling
    'silence_merge_gap': 0.50,         # stitch short micro-pauses
    'min_segment_length': 1.6,         # drop very short fragments
    'merge_threshold_s': 1.6,          # let segments grow before splitting
    'split_max_length_s': 12.0,
    'split_chunk_min_s': 3.5,          # min chunk size
    'split_chunk_max_s': 5.4,          # max chunk size
    
    # Hysteresis to reduce flip-flopping near threshold
    'hysteresis_keep_frames': 1,
    'hysteresis_ratio': 0.92,          # tighter hysteresis
    
    # Multiprocessing & logging
    'multiprocessing': True,
    'max_workers': min(15, max(1, mp.cpu_count() - 1)),
    'log_level': 'INFO',
    
    # Quality & file handling
    'enable_quality_checks': True,
    'backup_on_overwrite': False,
}


    split_configs = {
        'train': {**base_config, 'min_segment_length': 1.6},
        'test':  {**base_config, 'min_segment_length': 1.5},
    }

    setup_logging(base_config['log_level'])
    logger = logging.getLogger("AISHELL3_VAD_PROD")
    logger.info("=" * 60)
    logger.info("Production VAD (identity-preserving) starting")
    logger.info(f"Paths: in=data/<split>/<split>/wav/<spk>/*.wav | out=data/processed/vad_segments/<split>/<spk>/*.wav")
    logger.info(f"16kHz | frame=30ms | merge_gap={base_config['silence_merge_gap']} | "
                f"energy={base_config['energy_threshold']}@{base_config['adaptive_energy_percentile']}pct")
    logger.info("=" * 60)

    start_time = time.time()
    summary: Dict[str, Any] = {"processing_start": start_time, "configurations": split_configs}

    for split in ["train", "test"]:
        cfg = split_configs[split]
        input_dir = data_dir / input_template.format(split=split)
        output_dir = out_base / split
        output_dir.mkdir(parents=True, exist_ok=True)
        if not input_dir.exists():
            logger.warning(f"Missing directory {input_dir}, skipping split {split}")
            continue

        speakers = [d for d in input_dir.iterdir() if d.is_dir()]
        logger.info(f"\nProcessing {split.upper()} split: {len(speakers)} speakers")
        logger.info(f"Configuration: min_segment_length={cfg['min_segment_length']}s, "
                    f"energy_threshold={cfg['energy_threshold']}@{cfg['adaptive_energy_percentile']}pct")

        processed_speakers = 0
        total_segments = 0
        split_metadata: List[Dict[str, Any]] = []
        split_stats = defaultdict(int)

        args_list = [(speaker_dir, output_dir, cfg) for speaker_dir in speakers]

        if cfg['multiprocessing'] and len(args_list) > 1:
            with ProcessPoolExecutor(max_workers=cfg['max_workers']) as executor:
                future_to_speaker = {
                    executor.submit(process_speaker_production, args): args[0].name
                    for args in args_list
                }
                with tqdm(total=len(args_list), desc=f"{split} speakers", unit="spk") as pbar:
                    for future in as_completed(future_to_speaker):
                        speaker_name = future_to_speaker[future]
                        try:
                            speaker_id, num_segments, metadata, stats = future.result()
                            if num_segments > 0:
                                processed_speakers += 1
                            total_segments += num_segments
                            split_metadata.extend(metadata)
                            for key, value in stats.items():
                                split_stats[key] += value
                        except Exception as e:
                            logger.error(f"Speaker {speaker_name} processing failed: {e}")
                        pbar.update(1)
        else:
            for args in tqdm(args_list, desc=f"{split} speakers", unit="spk"):
                speaker_id, num_segments, metadata, stats = process_speaker_production(args)
                if num_segments > 0:
                    processed_speakers += 1
                total_segments += num_segments
                split_metadata.extend(metadata)
                for key, value in stats.items():
                    split_stats[key] += value

        split_duration = time.time() - start_time
        # Acceptance includes both short and energy rejections in denominator
        denom = (total_segments + split_stats.get('short_rej', 0) + split_stats.get('energy_rej', 0))
        acceptance_rate = (total_segments / denom * 100) if denom > 0 else 0.0

        logger.info(f"\n{split.upper()} Summary:")
        logger.info(f"  Processed speakers: {processed_speakers}/{len(speakers)} "
                    f"({(processed_speakers/len(speakers)*100) if len(speakers)>0 else 0:.1f}%)")
        logger.info(f"  Total segments: {total_segments:,}")
        logger.info(f"  Average segments per speaker: {total_segments/len(speakers) if len(speakers)>0 else 0:.1f}")
        logger.info(f"  Segment acceptance rate: {acceptance_rate:.1f}%")
        logger.info(f"  Rejected (short): {split_stats.get('short_rej',0)}")
        logger.info(f"  Rejected (low energy): {split_stats.get('energy_rej',0)}")
        logger.info(f"  Skipped existing: {split_stats.get('skipped',0)}")
        logger.info(f"  Preprocessing failures: {split_stats.get('preproc_fail',0)}")
        logger.info(f"  Time taken: {split_duration:.1f} seconds")
        logger.info("="*60)

        split_csv_path = out_base / f"{split}_segments_metadata.csv"
        write_metadata_csv(split_metadata, split_csv_path)

        summary[split] = {
            'speakers_processed': processed_speakers,
            'segments_total': total_segments,
            'rejected_short': split_stats.get('short_rej', 0),
            'rejected_energy': split_stats.get('energy_rej', 0),
            'skipped': split_stats.get('skipped', 0),
            'preproc_fail': split_stats.get('preproc_fail', 0),
            'time_seconds': split_duration,
        }

    summary['processing_end'] = time.time()
    summary['total_time'] = summary['processing_end'] - summary['processing_start']
    with open(out_base / "vad_processing_summary.json", 'w', encoding='utf-8') as jf:
        json.dump(summary, jf, indent=2)

    logger.info("VAD processing complete.")
    logger.info(f"Total time: {summary['total_time']:.1f} seconds.")

if __name__ == "__main__":
    main()
