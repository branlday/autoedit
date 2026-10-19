"""Resolve overlaps between tagged segments.

Priority when segments overlap (e.g. a reaction spike happens during what
was otherwise flagged as dead air): REACTION > TALKING_HEAD > NARRATION >
KEEP_WORK > (cut) DEAD_AIR / REPEATED_ACTION_TRIM.

Produces:
  - a clean, non-overlapping list of KEEP segments in chronological order
  - a list of CUT segments (for the audit log) with any portion that
    overlapped a kept interval removed
"""
from typing import List, Tuple
from .segment_types import ClipSegment

_PRIORITY = {
    "REACTION": 0,
    "TALKING_HEAD": 1,
    "NARRATION": 2,
    "KEEP_WORK": 3,
    "DEAD_AIR": 4,
    "REPEATED_ACTION_TRIM": 4,
}


def _subtract_intervals(interval: Tuple[float, float], cover: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Subtract a set of covering intervals from a single interval, return remaining pieces."""
    start, end = interval
    pieces = [(start, end)]
    for c_start, c_end in sorted(cover):
        new_pieces = []
        for p_start, p_end in pieces:
            if c_end <= p_start or c_start >= p_end:
                new_pieces.append((p_start, p_end))
                continue
            if c_start > p_start:
                new_pieces.append((p_start, c_start))
            if c_end < p_end:
                new_pieces.append((c_end, p_end))
        pieces = new_pieces
    return pieces


def merge_segments(
    segments: List[ClipSegment], min_keep_seconds: float = 1.0
) -> Tuple[List[ClipSegment], List[ClipSegment]]:
    segments = sorted(segments, key=lambda s: (s.start, _PRIORITY.get(s.tag, 9)))

    keep_segs = [s for s in segments if s.keep]
    cut_segs = [s for s in segments if not s.keep]

    # Merge overlapping keep segments, preferring the higher-priority tag/reason
    merged_keep: List[ClipSegment] = []
    for seg in sorted(keep_segs, key=lambda s: s.start):
        if merged_keep and seg.start <= merged_keep[-1].end:
            prev = merged_keep[-1]
            if _PRIORITY.get(seg.tag, 9) < _PRIORITY.get(prev.tag, 9):
                prev.tag, prev.reason, prev.text = seg.tag, seg.reason, seg.text or prev.text
            prev.end = max(prev.end, seg.end)
        else:
            merged_keep.append(
                ClipSegment(seg.source_path, seg.start, seg.end, seg.tag, seg.reason, True, seg.text)
            )

    merged_keep = [s for s in merged_keep if s.duration >= min_keep_seconds]

    keep_cover = [(s.start, s.end) for s in merged_keep]
    final_cuts: List[ClipSegment] = []
    for seg in cut_segs:
        remaining = _subtract_intervals((seg.start, seg.end), keep_cover)
        for r_start, r_end in remaining:
            if r_end - r_start >= 0.2:  # ignore tiny slivers
                final_cuts.append(
                    ClipSegment(seg.source_path, r_start, r_end, seg.tag, seg.reason, False, seg.text)
                )

    return merged_keep, final_cuts
