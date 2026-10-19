"""Cheap motion estimation via downsampled frame differencing.

We don't need optical flow precision here - just a relative signal that says
"camera/scene is basically still" (good candidate for a to-camera shot) vs
"lots of movement" (hands actively working). Sampling at a few fps instead
of full framerate keeps this fast on long clips.
"""
from typing import List, Tuple
import cv2
import numpy as np
from .segment_types import MotionSample


def analyze_motion(
    video_path: str,
    sample_fps: float = 3.0,
    resize: Tuple[int, int] = (320, 180),
) -> List[MotionSample]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, round(source_fps / sample_fps))

    samples: List[MotionSample] = []
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, resize)
            t = frame_idx / source_fps
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                score = float(diff.mean())
            else:
                score = 0.0
            samples.append(MotionSample(time=t, score=score))
            prev_gray = gray
        frame_idx += 1

    cap.release()
    return samples


def motion_score_in_range(samples: List[MotionSample], start: float, end: float) -> float:
    """Average motion score within a time window. Returns 0.0 if no samples fall in range."""
    vals = [s.score for s in samples if start <= s.time < end]
    if not vals:
        return 0.0
    return float(np.mean(vals))
