"""Combine speech, motion, and audio-spike signals into a sequence of
tagged ClipSegments for one source clip.
"""
from typing import List
import cv2
from .segment_types import ClipSegment, SpeechSegment, AudioSpike, MotionSample
from .motion import motion_score_in_range
from .repetition import find_repeated_clusters


def _get_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return frame_count / fps if fps else 0.0


def classify_clip(
    video_path: str,
    speech_segments: List[SpeechSegment],
    motion_samples: List[MotionSample],
    audio_spikes: List[AudioSpike],
    cfg: dict,
) -> List[ClipSegment]:
    duration = _get_duration(video_path)
    segments: List[ClipSegment] = []

    speech_segments = sorted(speech_segments, key=lambda s: s.start)

    # --- Speech segments: TALKING_HEAD vs NARRATION, based on camera stillness ---
    pad = cfg.get("pad_speech_seconds", 0.3)
    for sp in speech_segments:
        start = max(0.0, sp.start - pad)
        end = min(duration, sp.end + pad)
        m = motion_score_in_range(motion_samples, start, end)
        if m < cfg.get("static_camera_motion_threshold", 4.0):
            tag, reason = "TALKING_HEAD", "Speech with a still camera - likely a to-camera explanation"
        else:
            tag, reason = "NARRATION", "Speech while working"
        segments.append(
            ClipSegment(source_path=video_path, start=start, end=end, tag=tag, reason=reason, keep=True, text=sp.text)
        )

    # --- Gaps between speech segments: silent stretches, either dead air or silent work ---
    gap_bounds = []
    cursor = 0.0
    for sp in segments:
        if sp.start > cursor:
            gap_bounds.append((cursor, sp.start))
        cursor = max(cursor, sp.end)
    if cursor < duration:
        gap_bounds.append((cursor, duration))

    dead_air_min = cfg.get("dead_air_min_seconds", 4.0)
    active_threshold = cfg.get("active_work_motion_threshold", 8.0)

    work_candidates: List[ClipSegment] = []
    for g_start, g_end in gap_bounds:
        g_dur = g_end - g_start
        if g_dur <= 0:
            continue
        m = motion_score_in_range(motion_samples, g_start, g_end)

        if g_dur < dead_air_min:
            # too short to bother cutting - keep as connective tissue
            work_candidates.append(
                ClipSegment(video_path, g_start, g_end, "KEEP_WORK", "Short gap, kept for continuity", True)
            )
        elif m >= active_threshold:
            # real work happening, just no talking - keep, but this is exactly the
            # kind of segment repetition-detection will comb through for repeated bolts etc.
            work_candidates.append(
                ClipSegment(video_path, g_start, g_end, "KEEP_WORK", "Silent active work", True)
            )
        else:
            work_candidates.append(
                ClipSegment(
                    video_path, g_start, g_end, "DEAD_AIR",
                    f"{g_dur:.1f}s of low motion + no speech (searching/struggling)", False
                )
            )

    # Run repetition clustering only over the KEEP_WORK candidates
    keep_work_only = [s for s in work_candidates if s.tag == "KEEP_WORK"]
    dead_air_only = [s for s in work_candidates if s.tag == "DEAD_AIR"]
    clustered = find_repeated_clusters(
        keep_work_only,
        motion_lookup=lambda seg: motion_score_in_range(motion_samples, seg.start, seg.end),
        min_cluster_size=cfg.get("repeat_min_cluster_size", 3),
        duration_tolerance=cfg.get("repeat_duration_tolerance", 0.4),
        motion_tolerance=cfg.get("repeat_motion_tolerance", 0.35),
    )
    segments.extend(clustered)
    segments.extend(dead_air_only)

    # --- Audio spikes: reactions - but only if something is ALSO visibly
    # happening on screen. A loud audio spike with a static frame (background
    # noise, a tool clanging off-camera, ambient sound) is exactly the false
    # positive pattern where "nothing is happening on screen but a noise
    # triggered a keep" - gating on motion filters that out. Set
    # reaction_require_motion: false in config.yaml to disable this and
    # go back to pure audio-only reaction detection.
    pad_r = cfg.get("reaction_pad_seconds", 1.5)
    require_motion = cfg.get("reaction_require_motion", True)
    min_reaction_motion = cfg.get("reaction_min_motion_score", cfg.get("static_camera_motion_threshold", 4.0))

    for spike in audio_spikes:
        start = max(0.0, spike.time - pad_r)
        end = min(duration, spike.time + pad_r)

        if require_motion:
            m = motion_score_in_range(motion_samples, start, end)
            if m < min_reaction_motion:
                continue  # loud but nothing visibly happening - not a real "moment", skip it

        segments.append(
            ClipSegment(
                video_path, start, end, "REACTION",
                f"Audio energy spike (z={spike.zscore:.1f}) - reaction/interesting moment", True
            )
        )

    segments.sort(key=lambda s: s.start)
    return segments
