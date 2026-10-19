"""Transcribe a clip and return word-level timestamped speech segments.

Uses faster-whisper (CTranslate2-backed Whisper) for speed. VAD filtering
is enabled so long silences don't get transcribed as garbage/hallucinated
text - this doubles as a first-pass silence detector too.
"""
from typing import List
from .segment_types import SpeechSegment, WordTiming


_model_cache = {}


def _get_model(model_name: str, device: str, compute_type: str):
    key = (model_name, device, compute_type)
    if key not in _model_cache:
        from faster_whisper import WhisperModel
        _model_cache[key] = WhisperModel(
            model_name, device=device, compute_type=compute_type
        )
    return _model_cache[key]


def transcribe_clip(
    audio_or_video_path: str,
    model_name: str = "medium.en",
    device: str = "cpu",
    compute_type: str = "int8",
) -> List[SpeechSegment]:
    """Returns speech segments with word-level timestamps.

    faster-whisper can read directly from video files that ffmpeg can decode,
    so you don't need to pre-extract audio.
    """
    model = _get_model(model_name, device, compute_type)

    segments, _info = model.transcribe(
        audio_or_video_path,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    result = []
    for seg in segments:
        words = [
            WordTiming(word=w.word.strip(), start=w.start, end=w.end)
            for w in (seg.words or [])
        ]
        result.append(
            SpeechSegment(start=seg.start, end=seg.end, text=seg.text.strip(), words=words)
        )
    return result
