#!/usr/bin/env python3
import shutil
from pathlib import Path
from tqdm import tqdm
import re
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
import logging
from logging.handlers import RotatingFileHandler
from threading import Lock


class DetailedSegmentOrganizer:
    def __init__(self, log_path: Path):
        self.stats = defaultdict(lambda: defaultdict(int))
        self.speaker_pattern = re.compile(r'(SSB\d{4})')
        self.lock = Lock()
        self.logger = self.setup_logger(log_path)
        self.speaker_map = {}
        self.next_user_id = 1

    def setup_logger(self, log_path: Path) -> logging.Logger:
        log_dir = Path(log_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("VAD_ORGANIZER")
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        fh = RotatingFileHandler(str(log_path), encoding="utf-8", maxBytes=10 * 1024 * 1024, backupCount=3)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def extract_speaker_id(self, filename: str):
        match = self.speaker_pattern.search(filename)
        if match:
            return match.group(1)
        self.logger.warning(f"Could not extract speaker ID from {filename}")
        return None

    def get_user_id(self, speaker_id: str) -> str:
        if speaker_id not in self.speaker_map:
            user_id = f"user_{self.next_user_id:04d}"
            self.speaker_map[speaker_id] = user_id
            self.next_user_id += 1
        return self.speaker_map[speaker_id]

    def validate_segment(self, wav_path: Path):
        try:
            file_size = wav_path.stat().st_size
            if file_size < 1000:
                return False, "file_too_small"
            elif file_size > 50 * 1024 * 1024:
                return False, "file_too_large"
            try:
                with open(wav_path, 'rb') as f:
                    header = f.read(12)
                    if len(header) < 12:
                        return False, "incomplete_header"
                    if header[:4] != b'RIFF' or header[8:12] != b'WAVE':
                        return False, "invalid_wav_format"
            except Exception:
                return False, "file_access_error"
            return True, "valid"
        except Exception as e:
            return False, f"validation_error_{str(e)[:40]}"

    def organize_split(self, input_dir: Path, output_dir: Path, split_name: str):
        msg = f"\n📁 Organizing {split_name.upper()} split..."
        print(msg)
        self.logger.info(msg)

        split_output_dir = output_dir / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)

        wav_files = list(input_dir.glob("**/*.wav"))
        if not wav_files:
            print(f"❌ No wav files found in {input_dir}")
            self.logger.warning(f"No wav files found in {input_dir}")
            return {}

        print(f"📊 Found {len(wav_files)} segments to organize")
        self.logger.info(f"Found {len(wav_files)} segments to organize in {input_dir}")

        speaker_segments = defaultdict(list)
        invalid_files = defaultdict(list)
        validation_stats = defaultdict(int)
        validation_results = {}

        print("🔍 Analyzing and validating segments (parallel)...")
        self.logger.info("Validating segments in parallel...")
        with ThreadPoolExecutor() as executor:
            future_to_wav = {
                executor.submit(self.validate_and_extract, wav_file): wav_file for wav_file in wav_files
            }
            for future in tqdm(as_completed(future_to_wav), total=len(future_to_wav), desc="Validating"):
                wav_file, speaker_id, is_valid, reason = future.result()
                validation_results[wav_file] = (speaker_id, is_valid, reason)
                with self.lock:
                    if speaker_id and is_valid:
                        self.stats[split_name]['valid_segments'] += 1
                    elif not speaker_id:
                        self.stats[split_name]['invalid_speaker_id'] += 1
                        self.stats[split_name]['invalid_segments'] += 1
                    else:
                        self.stats[split_name]['invalid_segments'] += 1

        for wav_file, (speaker_id, is_valid, reason) in validation_results.items():
            if speaker_id and is_valid:
                speaker_segments[speaker_id].append(wav_file)
            elif not speaker_id:
                invalid_files["invalid_speaker_id"].append(wav_file)
                validation_stats["invalid_speaker_id"] += 1
            else:
                invalid_files[reason].append(wav_file)
                validation_stats[reason] += 1

        if invalid_files:
            print(f"\n⚠️  Validation Results:")
            for reason, files in invalid_files.items():
                print(f"  • {reason}: {len(files)} files")
                self.logger.warning(f"{len(files)} files failed validation for reason: {reason}")
                for file_path in files[:3]:
                    print(f"    - {file_path.name}")
                    self.logger.warning(f"Failed file example: {file_path}")
                if len(files) > 3:
                    print(f"    ... and {len(files) - 3} more")

        print(f"\n📂 Creating speaker directories and copying files (parallel)...")
        self.logger.info("Copying files into speaker directories (parallel)...")
        copy_stats = {'successful': 0, 'failed': 0, 'duplicates': 0}

        with ThreadPoolExecutor() as executor:
            all_tasks = []
            for speaker_id, segments in speaker_segments.items():
                user_id = self.get_user_id(speaker_id)
                speaker_dir = split_output_dir / user_id
                speaker_dir.mkdir(parents=True, exist_ok=True)
                for segment_file in segments:
                    all_tasks.append(
                        executor.submit(self.safe_copy, segment_file, speaker_dir, copy_stats, split_name, user_id)
                    )
            for _ in tqdm(as_completed(all_tasks), total=len(all_tasks), desc="Copying"):
                pass

        self.print_detailed_statistics(split_name, speaker_segments, validation_stats, copy_stats)

        mapping_path = output_dir / "speaker_id_map.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(self.speaker_map, f, indent=2)
        self.logger.info(f"Saved speaker ID map to {mapping_path}")

        return {
            "valid_segments": self.stats[split_name]['valid_segments'],
            "speakers": len(speaker_segments),
            "invalid_segments": self.stats[split_name]['invalid_segments']
        }

    def validate_and_extract(self, wav_file: Path):
        speaker_id = self.extract_speaker_id(wav_file.name)
        is_valid, reason = self.validate_segment(wav_file)
        return wav_file, speaker_id, is_valid, reason

    def safe_copy(self, segment_file: Path, speaker_dir: Path, copy_stats: dict, split_name: str, user_id: str):
        basename = segment_file.name
        if '.' in basename:
            stem, ext = basename.rsplit('.', 1)
            ext = '.' + ext
        else:
            stem, ext = basename, ''
        dest_path = speaker_dir / basename
        append_count = 1
        with self.lock:
            while dest_path.exists():
                new_basename = f"{stem}_{append_count}{ext}"
                dest_path = speaker_dir / new_basename
                append_count += 1
                copy_stats['duplicates'] += 1
                self.logger.info(f"Renamed duplicate file: {dest_path.name}")
        try:
            shutil.copy2(segment_file, dest_path)
            with self.lock:
                copy_stats['successful'] += 1
                self.stats[split_name]['speakers'].setdefault(
                    user_id, {'segments': 0, 'copied_successfully': 0, 'copy_failed': 0}
                )
                self.stats[split_name]['speakers'][user_id]['segments'] += 1
                self.stats[split_name]['speakers'][user_id]['copied_successfully'] += 1
        except Exception as e:
            with self.lock:
                copy_stats['failed'] += 1
                self.stats[split_name]['speakers'][user_id]['copy_failed'] += 1
            self.logger.error(f"Error copying {segment_file} to {dest_path}: {str(e)}")

    def print_detailed_statistics(self, split_name: str, speaker_segments: dict, validation_stats: dict, copy_stats: dict):
        print(f"\n📊 {split_name.upper()} SPLIT DETAILED STATISTICS:")
        valid = self.stats[split_name]['valid_segments']
        invalid = self.stats[split_name]['invalid_segments']
        print(f"  📈 Overall Numbers:")
        print(f"    • Total original speakers: {len(speaker_segments)}")
        print(f"    • Valid segments: {valid}")
        print(f"    • Invalid segments: {invalid}")
        total_segs = valid + invalid
        print(f"    • Success rate: {(valid / total_segs * 100) if total_segs else 0:.1f}%")

        print(f"  📁 File Operations:")
        print(f"    • Successfully copied: {copy_stats['successful']}")
        print(f"    • Copy failures: {copy_stats['failed']}")
        print(f"    • Duplicate names renamed: {copy_stats['duplicates']}")

        if speaker_segments:
            segment_counts = [len(segments) for segments in speaker_segments.values()]
            print(f"  👥 Speaker Distribution (original IDs):")
            print(f"    • Min segments per speaker: {min(segment_counts)}")
            print(f"    • Max segments per speaker: {max(segment_counts)}")
            print(f"    • Mean segments per speaker: {sum(segment_counts)/len(segment_counts):.1f}")
            print(f"    • Median segments per speaker: {statistics.median(segment_counts):.1f}")

            ranges = {
                '1-10': sum(1 for c in segment_counts if 1 <= c <= 10),
                '11-20': sum(1 for c in segment_counts if 11 <= c <= 20),
                '21-50': sum(1 for c in segment_counts if 21 <= c <= 50),
                '51+': sum(1 for c in segment_counts if c > 50)
            }
            print(f"  📊 Distribution Ranges:")
            for range_name, count in ranges.items():
                percentage = count / len(speaker_segments) * 100 if speaker_segments else 0
                print(f"    • {range_name} segments: {count} speakers ({percentage:.1f}%)")

            sorted_speakers = sorted(speaker_segments.items(), key=lambda x: len(x[1]), reverse=True)
            print(f"  🔝 Top 5 speakers by segment count:")
            for speaker_id, segments in sorted_speakers[:5]:
                print(f"    • {speaker_id}: {len(segments)} segments")
            print(f"  🔻 Bottom 5 speakers by segment count:")
            for speaker_id, segments in sorted_speakers[-5:]:
                print(f"    • {speaker_id}: {len(segments)} segments")

        self.logger.info(
            f"Split {split_name}: {valid} valid / {invalid} invalid segments | "
            f"{copy_stats['successful']} copied, {copy_stats['failed']} failed, "
            f"{copy_stats['duplicates']} renamed duplicates"
        )


def main():
    base_dir = Path(__file__).parent.parent.resolve()

    # Input and output directories
    input_root = base_dir / "data" / "processed" / "vad_segments"
    output_root = base_dir / "data" / "processed" / "vad_segments_by_speaker"

    logs_dir = base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "vad_organizer.log"

    print("📁 AISHELL-3 Enhanced Speaker Organization - Step 2")
    print("=" * 70)

    organizer = DetailedSegmentOrganizer(log_path)
    final_stats = {}

    splits = {
        "train": input_root / "train",
        "test": input_root / "test"
    }

    for split, split_input_dir in splits.items():
        if split_input_dir.exists():
            split_stats = organizer.organize_split(split_input_dir, output_root, split)
            final_stats[split] = split_stats
        else:
            print(f"⚠️  Warning: {split_input_dir} not found, skipping...")
            organizer.logger.warning(f"{split_input_dir} not found, skipping...")
            final_stats[split] = {"valid_segments": 0, "speakers": 0, "invalid_segments": 0}

    stats_file = base_dir / "data" / "processed" / "organization_detailed_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, indent=2)

    print(f"\n✅ Speaker organization complete!")
    print(f"📊 Detailed statistics saved to: {stats_file}")
    print(f"📁 Output saved to: {output_root}")
    print(f"🗒️  Log file: {log_path}")
    print(f"🗒️  Speaker ID mapping saved to: {output_root / 'speaker_id_map.json'}")
    print(f"\n➡️  Next step: Run 3_segment_balancer.py")


if __name__ == "__main__":
    main()
