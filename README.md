# Auto Rough-Cut Pipeline

Automatically builds a rough-cut timeline from narrated repair / how-to footage (iPhone, GoPro, etc.) and exports it as an **EDL** and **FCPXML** you can import into **DaVinci Resolve** for the final edit.

It listens for speech, watches for motion, and detects loud reaction moments, then keeps the interesting parts and flags dead air and repetitive work as cut candidates. Nothing is deleted from your source files, and every keep/cut decision is written to an audit log so you can review it before trusting the timeline.

## How it works

For each clip in a folder, the pipeline runs three analysis passes and then classifies the results:

1. **Transcription** (`faster-whisper`): word-level timestamps for everything you say.
2. **Motion analysis** (OpenCV): downsampled frame differencing to tell "camera still" from "hands working."
3. **Audio spikes** (`librosa`): loud moments relative to both the local neighborhood and the whole clip.
4. **Classification**: combines the signals into tagged segments.
5. **Export** (OpenTimelineIO): stitches kept segments into a timeline and writes EDL + FCPXML.

### Segment tags

| Tag | Meaning | Default |
|---|---|---|
| `TALKING_HEAD` | Speech while the camera is basically still (to-camera explanation) | Keep |
| `NARRATION` | Speech while working | Keep |
| `REACTION` | Loud audio spike **and** visible motion at the same moment | Keep (padded) |
| `KEEP_WORK` | Silent but visibly active work | Keep |
| `DEAD_AIR` | No speech and low motion for several seconds (searching for a tool, stuck) | Cut |
| `REPEATED_ACTION_TRIM` | Middle of a run of similar silent work bursts (for example bolts 2 to 9 of 10) | Cut (first and last of the run are kept) |

When segments overlap, priority is `REACTION` > `TALKING_HEAD` > `NARRATION` > `KEEP_WORK` > cut candidates.

> **Repetition detection is a heuristic.** It has no idea what a bolt is. It looks for 3 or more back-to-back silent work bursts with similar duration and motion energy. It works well on steady, repetitive tasks and can misfire on others. Spot-check the audit log.

## Requirements

- Python 3.10+ (developed on 3.13, Windows 11)
- [FFmpeg](https://ffmpeg.org/download.html) installed and on your `PATH`
- Optional but strongly recommended: an NVIDIA GPU for transcription

```bash
pip install -r requirements.txt
```

The requirements include two OpenTimelineIO adapter packages (`otio-cmx3600-adapter` for EDL and `otio-fcpx-xml-lite-adapter` for FCPXML). Recent OpenTimelineIO releases no longer bundle these, so exporting fails without them.

### GPU transcription (NVIDIA)

Set these in `config.yaml`:

```yaml
whisper_device: "cuda"
whisper_compute_type: "float16"
```

On Windows, `ctranslate2` needs the CUDA 12 runtime libraries:

```powershell
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

`pipeline/win_cuda_dlls.py` finds those DLLs inside your Python environment and adds them to `PATH` at startup, so no manual PATH edits are needed. It does nothing on other platforms.

## Usage

1. Copy `config.example.yaml` to `config.yaml` and adjust it (see Tuning below).
2. Put your source clips in a folder, for example `clips/`.
3. Run:

```bash
python main.py --input clips/ --output out/ --config config.yaml
```

4. Import `out/rough_cut.fcpxml` (or `out/rough_cut.edl`) into Resolve, and review `out/audit_log.csv`.

### Options

| Flag | Description |
|---|---|
| `--input` | Folder of source clips (`.mov`, `.mp4`, `.m4v`, `.mts`) |
| `--output` | Output folder (default `out`) |
| `--config` | Path to config YAML |
| `--order-file` | Text file listing filenames, one per line, to control clip order |
| `--name` | Timeline name |

By default, clips are stitched in filename order. Use numeric prefixes (`01_intro.mov`, `02_bolts.mov`) or `--order-file` to control the order.

## Output files

| File | Purpose |
|---|---|
| `rough_cut.edl` | CMX3600 EDL. Mature and widely supported, but minimal: single track, no marker support. |
| `rough_cut.fcpxml` | FCPXML with clip names and colored markers. Written by a community "lite" adapter, so see Known issues. |
| `audit_log.csv` | Every keep/cut decision with source file, timestamps, tag, reason, and transcript text. |
| `.cache/` | Cached analysis results (see below). Safe to delete. |

## Caching and resilience

- The three expensive analysis stages (transcription, motion, audio spikes) are cached per clip in `out/.cache/`. Cache keys include the file's size and modified time plus the config values that affect that stage.
- Classification is **not** cached, so changing thresholds like `dead_air_min_seconds` and re-running takes seconds.
- Changing a setting that affects an analysis stage (for example `whisper_model` or `reaction_zscore_threshold`) re-runs only that stage.
- If one clip fails, it is logged and skipped. The rest of the run completes, and re-running only reprocesses the clips that failed.

## Tuning

All thresholds live in `config.yaml`. Rerun after each change and check `audit_log.csv`.

| Symptom | What to adjust |
|---|---|
| Quiet clips keep sections because of faint background noise | Raise `reaction_min_absolute_percentile` (try 75 to 85) and/or `reaction_zscore_threshold` |
| Noise-triggered keeps with nothing happening on screen | Raise `reaction_min_motion_score`; keep `reaction_require_motion: true` |
| Real reactions are being filtered out | Lower `reaction_min_motion_score` and `reaction_min_absolute_percentile` |
| Too much dead air survives | Lower `dead_air_min_seconds`, or raise `static_camera_motion_threshold` |
| To-camera shots tagged as narration | Raise `static_camera_motion_threshold` (handheld GoPro footage is rarely perfectly still) |
| Repetition trimming too aggressive or too timid | Adjust `repeat_min_cluster_size` and the two `repeat_*_tolerance` values |
| Transcription too slow | Use a smaller `whisper_model`, or switch to a GPU |

## Project structure

```
main.py                    CLI entry point
config.example.yaml        Annotated example config
requirements.txt
pipeline/
  transcribe.py            faster-whisper transcription
  audio_energy.py          Reaction-spike detection (extracts audio via ffmpeg first)
  motion.py                Frame-difference motion scoring
  classify.py              Combines signals into tagged segments
  repetition.py            Repeated-action clustering
  merge.py                 Overlap resolution and keep/cut separation
  timeline_builder.py      OpenTimelineIO timeline + EDL/FCPXML export
  audit_log.py             CSV decision log
  cache.py                 Per-stage disk cache
  segment_types.py         Shared dataclasses
  win_cuda_dlls.py         Windows CUDA DLL path helper
```

## Known issues and limitations

- **FCPXML import into Resolve may fail.** The FCPXML adapter is a community "lite" implementation and does not produce the full structure Resolve expects (one reported error is `Unable to find inherited value for key "library"`). The EDL import is the reliable path. A failed FCPXML export never blocks the EDL.
- **"Clamping offset to 0" messages** printed during export are harmless. Markers are anchored to the start of each clip rather than the exact spike time.
- **No multi-camera sync.** Each file is treated as an independent, sequential source. Picking the better angle for the same moment across iPhone and GoPro is not handled.
- **Repeated actions are cut, not sped up.** Speed-ramping repetitive work instead of removing it would be a natural extension of `repetition.py` and the timeline builder.
- **Reaction detection is loudness-based.** It cannot distinguish laughter from a power tool. The motion gate and absolute-loudness floor reduce, but do not eliminate, false positives.

## License

Add a license of your choice before publishing.
