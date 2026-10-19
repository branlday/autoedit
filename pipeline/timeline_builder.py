"""Assemble kept segments into an OpenTimelineIO timeline and export to
formats DaVinci Resolve can import.

EDL (CMX3600) is exported for compatibility, but it's a very old format:
single video track, timecode-only, no per-clip naming beyond reel IDs.
FCPXML is exported alongside it and preserves clip names + markers, which
survives the round-trip into Resolve much better - import that one if you
want the tags/reasons visible as markers on the timeline.
"""
import os
from typing import List
import opentimelineio as otio
from .segment_types import ClipSegment

_MARKER_COLOR = {
    "TALKING_HEAD": otio.schema.MarkerColor.GREEN,
    "NARRATION": otio.schema.MarkerColor.BLUE,
    "REACTION": otio.schema.MarkerColor.RED,
    "KEEP_WORK": otio.schema.MarkerColor.CYAN,
}


def build_timeline(kept_segments: List[ClipSegment], fps: float = 30.0, name: str = "Auto Rough Cut"):
    timeline = otio.schema.Timeline(name=name)
    timeline.global_start_time = otio.opentime.RationalTime(0, fps)  # required by the FCPXML writer
    track = otio.schema.Track(name="V1")
    timeline.tracks.append(track)

    for seg in kept_segments:
        rate = fps
        start_rt = otio.opentime.RationalTime(round(seg.start * rate), rate)
        duration_rt = otio.opentime.RationalTime(round(seg.duration * rate), rate)
        if duration_rt.value <= 0:
            continue

        clip_basename = os.path.basename(seg.source_path)
        clip_name = f"{clip_basename} [{seg.tag}]"
        clip = otio.schema.Clip(
            name=clip_name,
            media_reference=otio.schema.ExternalReference(
                target_url="file://" + os.path.abspath(seg.source_path)
            ),
            source_range=otio.opentime.TimeRange(start_time=start_rt, duration=duration_rt),
        )

        marker = otio.schema.Marker(
            name=f"[{clip_basename}] {seg.tag}: {seg.reason}"[:100],
            marked_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, rate),
                duration=otio.opentime.RationalTime(1, rate),
            ),
            color=_MARKER_COLOR.get(seg.tag, otio.schema.MarkerColor.MAGENTA),
        )
        clip.markers.append(marker)

        track.append(clip)

    return timeline


def export_timeline(timeline, out_dir: str, base_name: str = "rough_cut"):
    """Writes EDL (required, via the otio-cmx3600-adapter package) and
    FCPXML (best-effort, via the community "lite" otio-fcpx-xml-lite-adapter
    package, which doesn't support the full FCPXML feature set). If FCPXML
    export fails for any reason, that's logged as a warning rather than
    crashing the run - the EDL alone is enough to import into Resolve.
    """
    os.makedirs(out_dir, exist_ok=True)
    edl_path = os.path.join(out_dir, f"{base_name}.edl")
    fcpxml_path = os.path.join(out_dir, f"{base_name}.fcpxml")

    otio.adapters.write_to_file(timeline, edl_path)

    fcpxml_written_path = None
    try:
        otio.adapters.write_to_file(
            timeline, fcpxml_path, adapter_name="otio_fcpx_xml_lite_adapter"
        )
        fcpxml_written_path = fcpxml_path
    except Exception as e:
        print(
            f"  Warning: FCPXML export failed ({e}). "
            f"EDL was written successfully and is enough to import into Resolve; "
            f"if you want FCPXML specifically (for full marker/name support), "
            f"make sure `otio-fcpx-xml-lite-adapter` is installed."
        )

    return edl_path, fcpxml_written_path
