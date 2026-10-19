#!/usr/bin/env python3
"""Auto rough-cut pipeline for repair/how-to footage.

Usage:
    python main.py --input clips/ --output out/ --config config.yaml

Point --input at a folder of source clips (iPhone + GoPro footage mixed is
fine - each file is analyzed independently). Clips are stitched in the
timeline in filename-sorted order by default; rename files with a numeric
prefix (01_intro.mov, 02_remove_bolts.mov, ...) if you want to control the
order, or pass --order-file with a text file listing filenames in order.
"""
import argparse
import glob
import os
import sys
import yaml

from pipeline.win_cuda_dlls import register_nvidia_dll_dirs
register_nvidia_dll_dirs()  # no-op on non-Windows; fixes cublas/cudnn DLL loading on Windows

from pipeline.transcribe import transcribe_clip
from pipeline.audio_energy import detect_energy_spikes
from pipeline.motion import analyze_motion
from pipeline.classify import classify_clip
from pipeline.merge import merge_segments
from pipeline.timeline_builder import build_timeline, export_timeline
from pipeline.audit_log import write_audit_log
from pipeline import cache as analysis_cache

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mts"}


def find_clips(input_dir: str, order_file: str = None):
    if order_file:
        with open(order_file) as f:
            names = [line.strip() for line in f if line.strip()]
        return [os.path.join(input_dir, n) for n in names]

    files = [
        f for f in glob.glob(os.path.join(input_dir, "*"))
        if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    ]
    return sorted(files)


def process_clip(path: str, cfg: dict, cache_dir: str):
    name = os.path.basename(path)

    speech = analysis_cache.load_transcript(cache_dir, path, cfg)
    if speech is not None:
        print(f"[{name}] transcript (cached)")
    else:
        print(f"[{name}] transcribing...")
        speech = transcribe_clip(
            path,
            model_name=cfg.get("whisper_model", "medium.en"),
            device=cfg.get("whisper_device", "cpu"),
            compute_type=cfg.get("whisper_compute_type", "int8"),
        )
        analysis_cache.save_transcript(cache_dir, path, cfg, speech)

    motion = analysis_cache.load_motion(cache_dir, path, cfg)
    if motion is not None:
        print(f"[{name}] motion (cached)")
    else:
        print(f"[{name}] analyzing motion...")
        motion = analyze_motion(
            path,
            sample_fps=cfg.get("motion_sample_fps", 3),
            resize=tuple(cfg.get("motion_resize", (320, 180))),
        )
        analysis_cache.save_motion(cache_dir, path, cfg, motion)

    spikes = analysis_cache.load_spikes(cache_dir, path, cfg)
    if spikes is not None:
        print(f"[{name}] audio spikes (cached)")
    else:
        print(f"[{name}] scanning for audio spikes...")
        spikes = detect_energy_spikes(
            path,
            zscore_threshold=cfg.get("reaction_zscore_threshold", 2.5),
            min_absolute_percentile=cfg.get("reaction_min_absolute_percentile", 60.0),
        )
        analysis_cache.save_spikes(cache_dir, path, cfg, spikes)

    # Not cached - cheap, and always reflects your current config.yaml thresholds.
    print(f"[{name}] classifying segments...")
    segments = classify_clip(path, speech, motion, spikes, cfg)
    return segments


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Directory of source clips")
    parser.add_argument("--output", default="out", help="Directory to write EDL/FCPXML/audit log")
    parser.add_argument("--config", default="config.example.yaml", help="Path to config YAML")
    parser.add_argument("--order-file", default=None, help="Optional text file listing filenames in desired order")
    parser.add_argument("--name", default="Auto Rough Cut", help="Timeline name")
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"Config file not found: {args.config}. Copy config.example.yaml to config.yaml and edit it.")
        sys.exit(1)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    clips = find_clips(args.input, args.order_file)
    if not clips:
        print(f"No video files found in {args.input}")
        sys.exit(1)

    print(f"Found {len(clips)} clip(s).")

    cache_dir = os.path.join(args.output, ".cache")
    failed_clips = []
    all_kept, all_cut = [], []

    for clip_path in clips:
        try:
            segments = process_clip(clip_path, cfg, cache_dir)
        except Exception as e:
            print(f"  !! ERROR processing {os.path.basename(clip_path)}: {e}")
            print(f"     Skipping this clip for now. Any earlier stages that finished "
                  f"before the error are cached, and this clip will only redo the "
                  f"stage that failed next time you run.")
            failed_clips.append(clip_path)
            continue

        kept, cut = merge_segments(segments, min_keep_seconds=cfg.get("min_keep_segment_seconds", 1.0))
        all_kept.extend(kept)
        all_cut.extend(cut)
        kept_time = sum(s.duration for s in kept)
        cut_time = sum(s.duration for s in cut)
        print(
            f"  -> kept {len(kept)} segments ({kept_time:.1f}s), "
            f"flagged {len(cut)} for cutting ({cut_time:.1f}s)"
        )

    if not all_kept:
        print("\nNo clips finished successfully - nothing to export.")
        sys.exit(1)

    timeline = build_timeline(all_kept, fps=cfg.get("timeline_fps", 30), name=args.name)
    edl_path, fcpxml_path = export_timeline(timeline, args.output)

    log_path = os.path.join(args.output, "audit_log.csv")
    write_audit_log(all_kept, all_cut, log_path)

    print("\nDone.")
    print(f"  EDL:        {edl_path}")
    if fcpxml_path:
        print(f"  FCPXML:     {fcpxml_path}  (import this one into Resolve for full clip names + markers)")
    else:
        print(f"  FCPXML:     not written (see warning above) - import the EDL instead, it's complete")
    print(f"  Audit log:  {log_path}  (review every keep/cut decision here)")
    if failed_clips:
        print(f"\n{len(failed_clips)} clip(s) failed and were skipped - not included in the timeline above:")
        for c in failed_clips:
            print(f"  - {c}")
        print("Fix the underlying issue and re-run the same command - clips that already "
              "succeeded will be served from cache, so only the failed ones will be reprocessed.")


if __name__ == "__main__":
    main()
