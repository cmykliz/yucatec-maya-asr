"""
process_doreco.py
-----------------
Converts DoReCo Yucatec Maya EAF annotation files + WAV audio into a
HuggingFace-compatible dataset of utterance-level segments.

Usage:
    python process_doreco.py \
        --audio_dir doreco_yuca1254_audiofiles_v2.0 \
        --eaf_dir   doreco_yuca1254_core_v2.0 \
        --output_dir ./doreco_yuca1254_hf \
        --push_to_hub  (optional)

Output structure:
    doreco_yuca1254_hf/
        train/
            audio/     *.wav  (utterance-level clips, 16kHz mono)
            metadata.jsonl
        test/
            audio/     *.wav
            metadata.jsonl

Each metadata.jsonl line:
    {
        "file_name":    "train/audio/YUC-TXT-CD-00000-12_utt_001.wav",
        "transcription": "teʔej kʔiinoʔob bejlaʔakaʔ",
        "translation":   "ahora todos ya ...",
        "speaker_id":    "12",
        "genre":         "procedural",
        "start_ms":      26762,
        "end_ms":        29287,
        "duration_s":    2.525,
        "language":      "yua",
        "source":        "DoReCo 2.0"
    }

Segmentation strategy:
    We use the `ref@` tier (utterance reference boundaries) as the unit of
    segmentation — this gives natural utterance-length clips rather than
    single words, which is more suitable for ASR training.
"""

import os
import re
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import csv

# Optional: pydub for audio slicing (pip install pydub)
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("WARNING: pydub not installed. Audio slicing will be skipped.")
    print("Install with: pip install pydub")

# ─────────────────────────────────────────────
# EAF PARSING
# ─────────────────────────────────────────────

SKIP_VALUES = {"<p:>", "<x>", "<pencil>", "<笑>", ""}

def parse_eaf(eaf_path):
    """
    Parse a DoReCo .eaf file and return a list of utterance dicts.

    Each utterance dict contains:
        start_ms, end_ms, words (list), transcription (str),
        translation (str), speaker_id (str)
    """
    tree = ET.parse(eaf_path)
    root = tree.getroot()

    # 1. Build time slot lookup: ts_id -> milliseconds
    time_slots = {}
    for ts in root.findall(".//TIME_SLOT"):
        time_slots[ts.get("TIME_SLOT_ID")] = int(ts.get("TIME_VALUE"))

    # 2. Find speaker ID from tier names (e.g. "ref@12" → "12")
    ref_tiers = [t for t in root.findall(".//TIER")
                 if t.get("TIER_ID", "").startswith("ref@")]
    if not ref_tiers:
        print(f"  WARNING: No ref@ tier found in {eaf_path}")
        return []
    speaker_id = ref_tiers[0].get("TIER_ID").split("@")[1]

    # 3. Parse ref tier → utterance boundaries (time-aligned)
    ref_anns = {}  # ann_id -> {start_ms, end_ms}
    ref_tier = root.find(f'.//TIER[@TIER_ID="ref@{speaker_id}"]')
    if ref_tier is None:
        return []
    for ann in ref_tier.findall(".//ALIGNABLE_ANNOTATION"):
        ann_id = ann.get("ANNOTATION_ID")
        start = time_slots[ann.get("TIME_SLOT_REF1")]
        end   = time_slots[ann.get("TIME_SLOT_REF2")]
        val   = ann.find("ANNOTATION_VALUE").text or ""
        ref_anns[ann_id] = {"start_ms": start, "end_ms": end, "ref_val": val}

    # 4. Parse tx tier → utterance transcription (child of ref)
    tx_map = {}  # ref ann_id -> transcription text
    tx_tier = root.find(f'.//TIER[@TIER_ID="tx@{speaker_id}"]')
    if tx_tier is not None:
        for ann in tx_tier.findall(".//REF_ANNOTATION"):
            ref_id = ann.get("ANNOTATION_REF")
            val    = ann.find("ANNOTATION_VALUE").text or ""
            tx_map[ref_id] = val

    # 5. Parse ft tier → free translation (Spanish)
    ft_map = {}
    ft_tier = root.find(f'.//TIER[@TIER_ID="ft@{speaker_id}"]')
    if ft_tier is not None:
        for ann in ft_tier.findall(".//REF_ANNOTATION"):
            ref_id = ann.get("ANNOTATION_REF")
            val    = ann.find("ANNOTATION_VALUE").text or ""
            ft_map[ref_id] = val

    # 6. Parse wd tier → word-level timestamps (for word list per utterance)
    #    wd annotations are time-aligned and parented to ref via shared time slots
    wd_tier = root.find(f'.//TIER[@TIER_ID="wd@{speaker_id}"]')
    wd_annotations = []
    if wd_tier is not None:
        for ann in wd_tier.findall(".//ALIGNABLE_ANNOTATION"):
            val   = ann.find("ANNOTATION_VALUE").text or ""
            start = time_slots[ann.get("TIME_SLOT_REF1")]
            end   = time_slots[ann.get("TIME_SLOT_REF2")]
            if val.strip() not in SKIP_VALUES:
                wd_annotations.append((start, end, val.strip()))
        wd_annotations.sort(key=lambda x: x[0])

    # 7. Assemble utterances
    utterances = []
    for ann_id, bounds in sorted(ref_anns.items(),
                                  key=lambda x: x[1]["start_ms"]):
        start_ms = bounds["start_ms"]
        end_ms   = bounds["end_ms"]
        tx       = tx_map.get(ann_id, "").strip()
        ft       = ft_map.get(ann_id, "").strip()

        # Skip pause/noise utterances
        if not tx or tx in SKIP_VALUES:
            continue

        # Collect words that fall within this utterance's time window
        words = [w for (ws, we, w) in wd_annotations
                 if ws >= start_ms and we <= end_ms
                 and w not in SKIP_VALUES]

        # Use tx tier as primary transcription; fall back to joined words
        transcription = tx if tx and tx not in SKIP_VALUES else " ".join(words)

        if not transcription or transcription in SKIP_VALUES:
            continue

        utterances.append({
            "ann_id":        ann_id,
            "start_ms":      start_ms,
            "end_ms":        end_ms,
            "duration_s":    round((end_ms - start_ms) / 1000, 3),
            "transcription": transcription,
            "translation":   ft,
            "words":         words,
            "speaker_id":    speaker_id,
        })

    return utterances


# ─────────────────────────────────────────────
# AUDIO SLICING
# ─────────────────────────────────────────────

def slice_audio(wav_path, utterances, output_dir, file_stem):
    """
    Slice a WAV file into utterance-level clips.
    Returns list of output file paths (parallel to utterances list).
    """
    if not PYDUB_AVAILABLE:
        return [None] * len(utterances)

    os.makedirs(output_dir, exist_ok=True)
    audio = AudioSegment.from_wav(wav_path)
    # Resample to 16kHz mono (required for MMS/wav2vec2)
    audio = audio.set_frame_rate(16000).set_channels(1)

    clip_paths = []
    for i, utt in enumerate(utterances):
        clip = audio[utt["start_ms"]:utt["end_ms"]]
        # Skip very short clips (< 0.3s) — likely noise
        if len(clip) < 300:
            clip_paths.append(None)
            continue
        fname = f"{file_stem}_utt_{i:04d}.wav"
        fpath = os.path.join(output_dir, fname)
        clip.export(fpath, format="wav")
        clip_paths.append(fpath)

    return clip_paths


# ─────────────────────────────────────────────
# METADATA LOADING
# ─────────────────────────────────────────────

def load_metadata(metadata_csv):
    """Load file-level metadata from doreco_yuca1254_metadata.csv."""
    meta = {}
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Key by the file name stem (e.g. "YUC-TXT-CD-00000-12")
            name = row["name"]  # e.g. YUC-TXT-CD-00000-12
            meta[name] = {
                "speaker_id":    row["spk_code"],
                "speaker_age":   row["spk_age"],
                "speaker_sex":   row["spk_sex"],
                "genre":         row["genre"],
                "sound_quality": row["sound_quality"],
                "rec_date":      row["rec_date"],
            }
    return meta


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main(audio_dir, eaf_dir, output_dir, metadata_csv=None,
         test_split=0.2, push_to_hub=False, hub_repo=None):

    audio_dir  = Path(audio_dir)
    eaf_dir    = Path(eaf_dir)
    output_dir = Path(output_dir)

    # Load file-level metadata if available
    file_meta = {}
    if metadata_csv and Path(metadata_csv).exists():
        file_meta = load_metadata(metadata_csv)
        print(f"Loaded metadata for {len(file_meta)} files")

    # Find all EAF files
    eaf_files = sorted(eaf_dir.glob("*.eaf"))
    print(f"Found {len(eaf_files)} EAF files\n")

    all_records = []

    for eaf_path in eaf_files:
        # Derive file stem: doreco_yuca1254_YUC-TXT-CD-00000-12 → YUC-TXT-CD-00000-12
        stem = eaf_path.stem  # e.g. doreco_yuca1254_YUC-TXT-CD-00000-12
        short_stem = stem.replace("doreco_yuca1254_", "")  # YUC-TXT-CD-00000-12

        # Find matching WAV
        wav_path = audio_dir / f"{stem}.wav"
        if not wav_path.exists():
            # Try with prefix
            matches = list(audio_dir.glob(f"*{short_stem}.wav"))
            wav_path = matches[0] if matches else None

        if wav_path is None or not wav_path.exists():
            print(f"  WARNING: No WAV found for {stem}, skipping")
            continue

        print(f"Processing: {short_stem}")

        # Parse EAF
        utterances = parse_eaf(eaf_path)
        print(f"  {len(utterances)} utterances found")
        if not utterances:
            continue

        # Get file-level metadata
        fmeta = file_meta.get(short_stem, {})

        # Slice audio
        split_audio_dir = output_dir / "audio_clips" / short_stem
        clip_paths = slice_audio(str(wav_path), utterances,
                                  str(split_audio_dir), short_stem)

        # Build records
        for i, (utt, clip_path) in enumerate(zip(utterances, clip_paths)):
            if clip_path is None:
                continue  # skipped (too short or pydub unavailable)

            record = {
                "file_name":     os.path.relpath(clip_path, output_dir),
                "transcription": utt["transcription"],
                "translation":   utt["translation"],
                "speaker_id":    utt["speaker_id"],
                "speaker_age":   fmeta.get("speaker_age", ""),
                "speaker_sex":   fmeta.get("speaker_sex", ""),
                "genre":         fmeta.get("genre", ""),
                "sound_quality": fmeta.get("sound_quality", ""),
                "source_file":   short_stem,
                "start_ms":      utt["start_ms"],
                "end_ms":        utt["end_ms"],
                "duration_s":    utt["duration_s"],
                "language":      "yua",
                "language_name": "Yucatec Maya",
                "source":        "DoReCo 2.0",
                "license":       "CC BY-NC-ND 4.0",
            }
            all_records.append(record)

    print(f"\nTotal utterances: {len(all_records)}")

    # ── Train / test split (by file, not by utterance, to avoid leakage) ──
    source_files = sorted(set(r["source_file"] for r in all_records))
    n_test_files = max(1, int(len(source_files) * test_split))
    test_files   = set(source_files[-n_test_files:])  # last N files → test
    train_files  = set(source_files) - test_files

    print(f"Train files ({len(train_files)}): {sorted(train_files)}")
    print(f"Test files  ({len(test_files)}):  {sorted(test_files)}")

    train_records = [r for r in all_records if r["source_file"] in train_files]
    test_records  = [r for r in all_records if r["source_file"] in test_files]

    # ── Write metadata.jsonl ──
    for split, records in [("train", train_records), ("test", test_records)]:
        split_dir = output_dir / split
        os.makedirs(split_dir, exist_ok=True)
        jsonl_path = split_dir / "metadata.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(records)} records → {jsonl_path}")

    # ── Summary stats ──
    total_duration = sum(r["duration_s"] for r in all_records)
    print(f"\n{'='*50}")
    print(f"Dataset summary:")
    print(f"  Total utterances : {len(all_records)}")
    print(f"  Total duration   : {total_duration/3600:.2f} hours")
    print(f"  Train utterances : {len(train_records)}")
    print(f"  Test utterances  : {len(test_records)}")
    speakers = set(r["speaker_id"] for r in all_records)
    print(f"  Unique speakers  : {len(speakers)} ({', '.join(sorted(speakers))})")
    genres = set(r["genre"] for r in all_records)
    print(f"  Genres           : {', '.join(sorted(genres))}")
    print(f"{'='*50}")

    # ── Optional: push to HuggingFace Hub ──
    if push_to_hub:
        try:
            from datasets import load_dataset
            ds = load_dataset("audiofolder", data_dir=str(output_dir))
            ds.push_to_hub(hub_repo or "your-username/yucatec-maya-doreco")
            print(f"\nPushed to HuggingFace Hub: {hub_repo}")
        except Exception as e:
            print(f"\nHub push failed: {e}")
            print("Run manually with: datasets.load_dataset('audiofolder', data_dir='...')")

    return all_records


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert DoReCo Yucatec Maya EAF + WAV → HuggingFace dataset"
    )
    parser.add_argument("--audio_dir",    required=True,
                        help="Path to doreco_yuca1254_audiofiles_v2.0/")
    parser.add_argument("--eaf_dir",      required=True,
                        help="Path to doreco_yuca1254_core_v2.0/")
    parser.add_argument("--output_dir",   default="./doreco_yuca1254_hf",
                        help="Where to write the output dataset")
    parser.add_argument("--metadata_csv", default=None,
                        help="Path to doreco_yuca1254_metadata.csv (optional)")
    parser.add_argument("--test_split",   type=float, default=0.2,
                        help="Fraction of files to use for test split (default 0.2)")
    parser.add_argument("--push_to_hub",  action="store_true",
                        help="Push finished dataset to HuggingFace Hub")
    parser.add_argument("--hub_repo",     default=None,
                        help="HuggingFace repo name, e.g. username/yucatec-maya-doreco")
    args = parser.parse_args()

    main(
        audio_dir    = args.audio_dir,
        eaf_dir      = args.eaf_dir,
        output_dir   = args.output_dir,
        metadata_csv = args.metadata_csv,
        test_split   = args.test_split,
        push_to_hub  = args.push_to_hub,
        hub_repo     = args.hub_repo,
    )