import csv
import os
from typing import List
from .segment_types import ClipSegment


def write_audit_log(kept: List[ClipSegment], cut: List[ClipSegment], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    for s in kept:
        rows.append([s.source_path, f"{s.start:.2f}", f"{s.end:.2f}", f"{s.duration:.2f}", "KEEP", s.tag, s.reason, s.text or ""])
    for s in cut:
        rows.append([s.source_path, f"{s.start:.2f}", f"{s.end:.2f}", f"{s.duration:.2f}", "CUT", s.tag, s.reason, s.text or ""])

    rows.sort(key=lambda r: (r[0], float(r[1])))

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_file", "start_sec", "end_sec", "duration_sec", "decision", "tag", "reason", "transcript"])
        writer.writerows(rows)
