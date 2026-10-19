"""Detect loud reaction moments (laughter, exclamations, sudden clangs)
via rolling z-score of RMS loudness. This is intentionally simple and fast -
it doesn't try to classify *what* the sound is, just that it's a notable
spike relative to the surrounding audio.
"""
from typing import List
import os
import subprocess
import tempfile
import numpy as np
from .segment_types import AudioSpike


def _extract_audio_to_wav(video_path: str) -> str:
    """Pre-extract audio to a temp mono WAV via ffmpeg.

    librosa/soundfile can't read video containers (.MOV/.MP4) directly -
    it only understands audio formats. librosa is *supposed* to fall back
    to a video-capable decoder automatically when soundfile fails, but that
    fallback path is broken on current soundfile releases (which renamed
    their error class out from under librosa's except clause), so instead
    of relying on it we just do the extraction ourselves with ffmpeg, which
    is already a hard requirement of this pipeline.
    """
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",                  # no video
        "-acodec", "pcm_s16le", # plain PCM, universally readable
        "-ar", "16000",         # 16kHz is plenty for RMS/energy analysis
        "-ac", "1",              # mono
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        os.remove(wav_path)
        raise RuntimeError(
            f"ffmpeg failed to extract audio from {video_path}:\n{result.stderr[-2000:]}"
        )
    return wav_path


def detect_energy_spikes(
    video_path: str,
    zscore_threshold: float = 2.5,
    rolling_window_seconds: float = 20.0,
    min_absolute_percentile: float = 60.0,
) -> List[AudioSpike]:
    """A moment counts as a "spike" only if BOTH:
      1. It's a local relative spike (loud compared to its ~20s neighborhood), AND
      2. It's loud compared to the clip as a whole (>= min_absolute_percentile of
         this clip's overall RMS distribution).

    Condition 2 exists specifically to stop quiet clips from generating false
    positives: in a near-silent clip, even a faint click can be a huge *relative*
    spike against its quiet surroundings without being loud in any absolute
    sense. Requiring it to also clear a whole-clip loudness bar filters those out.
    Set min_absolute_percentile to 0 to disable this and go back to pure
    relative detection.
    """
    import librosa

    wav_path = _extract_audio_to_wav(video_path)
    try:
        y, sr = librosa.load(wav_path, sr=None, mono=True)
    finally:
        os.remove(wav_path)

    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    if len(rms) < 3:
        return []  # clip too short to compute any meaningful rolling statistics

    frame_rate = sr / hop_length
    window_frames = max(1, int(rolling_window_seconds * frame_rate))
    # A short clip can be shorter than the rolling window itself. np.convolve's
    # mode="same" returns an array sized to the LARGER of the two inputs, so if
    # the kernel (window_frames) is longer than rms, the output no longer
    # matches rms's length and later arithmetic breaks with a shape mismatch.
    # Clamping the window to the signal's own length avoids that - for a very
    # short clip this just means the "rolling" window covers the whole thing.
    window_frames = min(window_frames, len(rms))

    # Rolling mean/std via simple convolution (fast enough for whole-video RMS arrays)
    kernel = np.ones(window_frames) / window_frames
    rolling_mean = np.convolve(rms, kernel, mode="same")
    rolling_sq_mean = np.convolve(rms ** 2, kernel, mode="same")
    rolling_std = np.sqrt(np.clip(rolling_sq_mean - rolling_mean ** 2, 1e-8, None))

    zscores = (rms - rolling_mean) / rolling_std

    absolute_floor = np.percentile(rms, min_absolute_percentile) if len(rms) else 0.0

    spikes = []
    above = (zscores > zscore_threshold) & (rms >= absolute_floor)
    # collapse consecutive above-threshold frames into single spike events
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            peak_idx = i
            while j < len(above) and above[j]:
                if zscores[j] > zscores[peak_idx]:
                    peak_idx = j
                j += 1
            spikes.append(AudioSpike(time=float(times[peak_idx]), zscore=float(zscores[peak_idx])))
            i = j
        else:
            i += 1

    return spikes
