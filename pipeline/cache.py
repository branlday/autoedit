"""Disk cache for the expensive per-clip analysis stages (transcription,
motion analysis, audio spike detection) so a crash or interrupted run
doesn't force reprocessing clips that already finished.

Cache keys are based on: the source file's size + modified-time (so if you
replace/re-export a clip, it's treated as new), plus a fingerprint of the
subset of config values that actually affect that stage. Changing an
unrelated config value (e.g. a classification threshold) won't invalidate
the cache; changing something like `whisper_model` will.

Classification itself (classify.py) is NOT cached - it's cheap (pure
Python over already-computed signals) and always reruns against the
current config, so tuning thresholds and re-running is fast and always
reflects your latest config.yaml without needing to clear anything.
"""
import os
import json
import hashlib
from typing import List, Optional
from .segment_types import SpeechSegment, WordTiming, MotionSample, AudioSpike


def _file_fingerprint(path: str) -> str:
    st = os.stat(path)
    return f"{st.st_size}_{int(st.st_mtime)}"


def _cfg_fingerprint(cfg_subset: dict) -> str:
    return hashlib.sha1(json.dumps(cfg_subset, sort_keys=True).encode()).hexdigest()[:10]


def _cache_file(cache_dir: str, video_path: str, stage: str, cfg_subset: dict) -> str:
    base = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(
        cache_dir, f"{base}__{_file_fingerprint(video_path)}__{stage}__{_cfg_fingerprint(cfg_subset)}.json"
    )


def _atomic_write_json(path: str, data, cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    os.replace(tmp_path, path)  # atomic on both Windows and POSIX


# ---- Transcription cache ----

_TRANSCRIPT_KEYS = ("whisper_model", "whisper_device", "whisper_compute_type")


def load_transcript(cache_dir: str, video_path: str, cfg: dict) -> Optional[List[SpeechSegment]]:
    subset = {k: cfg.get(k) for k in _TRANSCRIPT_KEYS}
    path = _cache_file(cache_dir, video_path, "transcript", subset)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return [
            SpeechSegment(
                start=d["start"], end=d["end"], text=d["text"],
                words=[WordTiming(**w) for w in d["words"]],
            )
            for d in data
        ]
    except (json.JSONDecodeError, KeyError, OSError):
        return None  # corrupt/partial cache file - treat as a cache miss


def save_transcript(cache_dir: str, video_path: str, cfg: dict, segments: List[SpeechSegment]):
    subset = {k: cfg.get(k) for k in _TRANSCRIPT_KEYS}
    path = _cache_file(cache_dir, video_path, "transcript", subset)
    data = [
        {"start": s.start, "end": s.end, "text": s.text,
         "words": [{"word": w.word, "start": w.start, "end": w.end} for w in s.words]}
        for s in segments
    ]
    _atomic_write_json(path, data, cache_dir)


# ---- Motion analysis cache ----

_MOTION_KEYS = ("motion_sample_fps", "motion_resize")


def load_motion(cache_dir: str, video_path: str, cfg: dict) -> Optional[List[MotionSample]]:
    subset = {k: cfg.get(k) for k in _MOTION_KEYS}
    path = _cache_file(cache_dir, video_path, "motion", subset)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return [MotionSample(**d) for d in data]
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def save_motion(cache_dir: str, video_path: str, cfg: dict, samples: List[MotionSample]):
    subset = {k: cfg.get(k) for k in _MOTION_KEYS}
    path = _cache_file(cache_dir, video_path, "motion", subset)
    data = [{"time": s.time, "score": s.score} for s in samples]
    _atomic_write_json(path, data, cache_dir)


# ---- Audio spike cache ----

_SPIKE_KEYS = ("reaction_zscore_threshold", "reaction_min_absolute_percentile")


def load_spikes(cache_dir: str, video_path: str, cfg: dict) -> Optional[List[AudioSpike]]:
    subset = {k: cfg.get(k) for k in _SPIKE_KEYS}
    path = _cache_file(cache_dir, video_path, "spikes", subset)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return [AudioSpike(**d) for d in data]
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def save_spikes(cache_dir: str, video_path: str, cfg: dict, spikes: List[AudioSpike]):
    subset = {k: cfg.get(k) for k in _SPIKE_KEYS}
    path = _cache_file(cache_dir, video_path, "spikes", subset)
    data = [{"time": s.time, "zscore": s.zscore} for s in spikes]
    _atomic_write_json(path, data, cache_dir)
