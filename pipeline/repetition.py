"""Heuristic detection of repeated near-identical action bursts.

This is the "10 bolts, only show the first and last" problem. There's no
object-level understanding here - instead we look for a run of consecutive
work bursts (silent, motion-active segments) that are similar to each other
in duration and average motion energy, on the theory that a repeated
mechanical action (loosen bolt, drop in tray, reach for next one) produces a
fairly consistent rhythm. This is a heuristic, not ground truth - always
spot check the flagged segments before trusting the auto-cut.
"""
from typing import List
from .segment_types import ClipSegment


def _similar(a: ClipSegment, b: ClipSegment, duration_tol: float, motion_tol: float, motion_lookup) -> bool:
    dur_a, dur_b = a.duration, b.duration
    if dur_a <= 0 or dur_b <= 0:
        return False
    dur_ratio = abs(dur_a - dur_b) / max(dur_a, dur_b)
    if dur_ratio > duration_tol:
        return False

    m_a = motion_lookup(a)
    m_b = motion_lookup(b)
    if max(m_a, m_b) <= 0:
        return False
    motion_ratio = abs(m_a - m_b) / max(m_a, m_b)
    if motion_ratio > motion_tol:
        return False

    return True


def find_repeated_clusters(
    work_segments: List[ClipSegment],
    motion_lookup,
    min_cluster_size: int = 3,
    duration_tolerance: float = 0.4,
    motion_tolerance: float = 0.35,
) -> List[ClipSegment]:
    """Given consecutive KEEP_WORK segments (no speech, active motion),
    find runs that look like repeated identical actions and mark all but
    the first and last of each run as REPEATED_ACTION_TRIM (cut candidates).

    Returns a new list of ClipSegment with tags/keep flags updated.
    Segments not part of a qualifying cluster are returned unchanged.
    """
    segments = sorted(work_segments, key=lambda s: s.start)
    if len(segments) < min_cluster_size:
        return segments

    output: List[ClipSegment] = []
    i = 0
    while i < len(segments):
        cluster = [segments[i]]
        j = i + 1
        while j < len(segments) and _similar(
            cluster[-1], segments[j], duration_tolerance, motion_tolerance, motion_lookup
        ):
            cluster.append(segments[j])
            j += 1

        if len(cluster) >= min_cluster_size:
            first, last = cluster[0], cluster[-1]
            output.append(first)  # keep first occurrence as-is
            for mid in cluster[1:-1]:
                mid.tag = "REPEATED_ACTION_TRIM"
                mid.keep = False
                mid.reason = (
                    f"Part of a repeated-action cluster of {len(cluster)} similar bursts "
                    f"(kept first + last only)"
                )
                output.append(mid)
            if last is not first:
                output.append(last)  # keep last occurrence as-is
        else:
            output.extend(cluster)

        i = j

    return output
