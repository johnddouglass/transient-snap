#!/usr/bin/env python3
"""
drum_align — Band-limited spectral flux onset alignment for drum recordings.

Refines MIDI drum note positions to sample-accurate transient onsets.  Each
instrument preset applies a bandpass filter that isolates the drum's
characteristic frequency range before computing spectral flux, reducing bleed
interference from other kit elements.

Instrument presets and their default bands:

    kick        40–180 Hz
    snare       150–1200 Hz
    hihat       7000–18000 Hz
    ride        5000–16000 Hz
    crash       3000–16000 Hz
    tom         80–600 Hz
    overhead    wideband (no filter)
    room        wideband (no filter)

The search window is asymmetric by default: 3 ms before the MIDI position and
10 ms after.  The narrow back-window prevents latching onto bleed from the
preceding hit; the wider forward-window accommodates MIDI notes that landed
slightly before the actual strike.

Usage examples:

    # Kick close-mic; notes in track 1 of a multi-track MIDI file
    python drum_align.py kick.wav session.mid --instrument kick --tracks 1 -o kick_aligned.mid

    # Floor tom with a custom frequency band
    python drum_align.py floortom.wav session.mid --tracks 5 \\
        --low-hz 60 --high-hz 400 -o floortom_aligned.mid

    # Snare; widen the forward window for a session where MIDI arrived early
    python drum_align.py snare.wav session.mid --instrument snare \\
        --search-fwd-ms 15 -o snare_aligned.mid

    # All tracks in the MIDI; print a per-note CSV report to stdout
    python drum_align.py overhead.wav session.mid --instrument overhead \\
        -o overhead_aligned.mid --report
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from onset import INSTRUMENT_BANDS, OnsetResult, refine_all
from snap import extract_tempo_map, load_audio, load_markers_midi, save_markers_midi


# ── Helpers ──────────────────────────────────────────────────────────────────

def _band_label(low_hz: float, high_hz: float, instrument: str) -> str:
    if low_hz <= 0.0 and high_hz <= 0.0:
        return f"wideband ({instrument})"
    if low_hz <= 0.0:
        return f"lowpass {high_hz:.0f} Hz ({instrument})"
    if high_hz <= 0.0:
        return f"highpass {low_hz:.0f} Hz ({instrument})"
    return f"{low_hz:.0f}–{high_hz:.0f} Hz ({instrument})"


def _resolve_band(
    instrument: str,
    low_hz: Optional[float],
    high_hz: Optional[float],
) -> tuple:
    """Return (low_hz, high_hz) from CLI args, applying the preset when no
    explicit overrides are given."""
    if low_hz is not None or high_hz is not None:
        return (low_hz or 0.0), (high_hz or 0.0)
    return INSTRUMENT_BANDS.get(instrument, (0.0, 0.0))


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_summary(
    results: list[OnsetResult],
    sr: int,
    label: str,
    band_str: str,
    confidence_warn: float,
) -> None:
    """Print a human-readable alignment summary to stdout."""
    if not results:
        print("  No notes processed.")
        return

    offsets_ms   = np.array([r.offset_ms      for r in results])
    offsets_samp = np.array([r.offset_samples  for r in results])
    confidences  = np.array([r.confidence      for r in results])
    n            = len(results)
    n_moved      = int(np.sum(offsets_samp != 0))

    n_primary  = sum(1 for r in results if r.pass_used == "primary")
    n_wideband = sum(1 for r in results if r.pass_used == "wideband")
    n_frozen   = sum(1 for r in results if r.pass_used == "frozen")
    low_conf   = [r for r in results if r.confidence < confidence_warn
                  and r.pass_used != "frozen"]

    print(f"\n{'─' * 58}")
    print(f"  {label}")
    print(f"  Band:              {band_str}")
    print(f"  Notes processed:   {n}")
    print(f"  Moved:             {n_moved}/{n}  ({n_moved / n * 100:.0f}%)")
    print(f"  Mean offset:       {np.mean(offsets_ms):+.2f} ms  "
          f"({np.mean(offsets_samp):+.1f} samples)")
    print(f"  Std offset:        {np.std(offsets_ms):.2f} ms")
    print(f"  Max offset:        {np.max(np.abs(offsets_ms)):.2f} ms  "
          f"({int(np.max(np.abs(offsets_samp)))} samples)")
    print(f"  Mean confidence:   {np.mean(confidences):.1f}x  "
          f"(warning threshold: {confidence_warn:.1f}x)")
    print(f"  Pass breakdown:    {n_primary} primary  "
          f"/ {n_wideband} wideband retry  "
          f"/ {n_frozen} frozen")

    if n_frozen:
        frozen = [r for r in results if r.pass_used == "frozen"]
        show   = frozen[:10]
        print(f"\n  Frozen notes ({n_frozen}) — left at original MIDI position:")
        for r in show:
            t = r.original / sr
            print(f"    {t:9.3f} s   confidence {r.confidence:.1f}x")
        if n_frozen > 10:
            print(f"    … and {n_frozen - 10} more")

    if low_conf:
        show = low_conf[:20]
        print(f"\n  Low-confidence notes ({len(low_conf)}) — consider manual review:")
        for r in show:
            t = r.original / sr
            print(f"    {t:9.3f} s   offset {r.offset_ms:+6.2f} ms   "
                  f"confidence {r.confidence:.1f}x  [{r.pass_used}]")
        if len(low_conf) > 20:
            print(f"    … and {len(low_conf) - 20} more")

    print()


def write_csv_report(
    results: list[OnsetResult],
    sr: int,
    path: Path,
) -> None:
    """Write a per-note CSV with original position, refined position, offset,
    and confidence for downstream analysis."""
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "index", "original_sample", "refined_sample",
            "original_time_s", "refined_time_s",
            "offset_samples", "offset_ms", "confidence", "pass_used",
        ])
        for i, r in enumerate(results):
            writer.writerow([
                i,
                r.original, r.refined,
                f"{r.original / sr:.6f}", f"{r.refined / sr:.6f}",
                r.offset_samples, f"{r.offset_ms:.4f}", f"{r.confidence:.2f}",
                r.pass_used,
            ])
    print(f"Report: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="drum_align",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "audio",
        help="Close-mic drum audio WAV for this element.",
    )
    parser.add_argument(
        "midi",
        help="MIDI file containing the drum notes to align.",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        metavar="FILE",
        help="Output MIDI file path.",
    )

    band_group = parser.add_argument_group("instrument / band")
    band_group.add_argument(
        "--instrument", "-i",
        choices=sorted(INSTRUMENT_BANDS),
        default="kick",
        metavar="PRESET",
        help=(
            "Instrument preset (determines bandpass filter). "
            f"Choices: {', '.join(sorted(INSTRUMENT_BANDS))}. "
            "Default: kick"
        ),
    )
    band_group.add_argument(
        "--low-hz",
        type=float, default=None, metavar="HZ",
        help="Override lower bandpass cutoff (Hz). Overrides --instrument band.",
    )
    band_group.add_argument(
        "--high-hz",
        type=float, default=None, metavar="HZ",
        help="Override upper bandpass cutoff (Hz). Overrides --instrument band.",
    )

    window_group = parser.add_argument_group("search window")
    window_group.add_argument(
        "--search-back-ms",
        type=float, default=0.0, metavar="MS",
        help=(
            "How far before each MIDI position to search for the onset (ms). "
            "Default 0 works well for kick, snare, and toms where the trigger "
            "fires at or before the acoustic hit.  Use 3–5 ms for ghost notes "
            "or any track where MIDI was captured after the actual strike "
            "(trigger delay, quantisation overshoot, etc.).  Default: 0.0"
        ),
    )
    window_group.add_argument(
        "--search-fwd-ms",
        type=float, default=10.0, metavar="MS",
        help=(
            "How far after each MIDI position to search (ms). "
            "Widen if MIDI was captured ahead of the actual strike. Default: 10.0"
        ),
    )
    window_group.add_argument(
        "--onset-threshold",
        type=float, default=0.05, metavar="FRAC",
        help=(
            "Fraction of peak amplitude used for the CLOSE Stage 2 walk-back "
            "(Stage 1 peak within ~2.67 ms of MIDI position).  "
            "Smaller = earlier onset edge. Default: 0.05.  "
            "For snare, 0.10 reduces noise on notes placed close to the onset."
        ),
    )
    window_group.add_argument(
        "--onset-threshold-distant",
        type=float, default=None, metavar="FRAC",
        help=(
            "Separate fraction for the DISTANT Stage 2 walk-back "
            "(Stage 1 peak more than ~2.67 ms after MIDI position).  "
            "When omitted, defaults to --onset-threshold.  "
            "A higher value (e.g. 0.15) reduces the ~1 ms systematic residual "
            "on notes where MIDI was placed at a cymbal hit well before the "
            "actual snare stroke, without affecting close-detected notes."
        ),
    )

    parser.add_argument(
        "--confidence-warn",
        type=float, default=3.0, metavar="RATIO",
        help=(
            "Flag notes whose spectral flux peak-to-mean ratio is below this "
            "value in the alignment report. Default: 3.0"
        ),
    )
    parser.add_argument(
        "--confidence-min",
        type=float, default=2.0, metavar="RATIO",
        help=(
            "Peak-to-mean confidence threshold for the two-pass retry. "
            "Notes below this on the primary (band-filtered) pass are retried "
            "with a wideband (no filter) pass.  If confidence is still below "
            "this threshold after the retry, the note is frozen at its "
            "original MIDI position (not moved). "
            "Set to 0.0 to disable retry/freeze. Default: 2.0"
        ),
    )
    parser.add_argument(
        "--tracks",
        type=int, nargs="+", metavar="N",
        help=(
            "MIDI track indices to process (0-based). "
            "Omit to process all tracks."
        ),
    )
    parser.add_argument(
        "--report",
        metavar="FILE",
        nargs="?",
        const="_auto",
        help=(
            "Write a per-note CSV report. Optionally supply a path; "
            "if omitted, the report is written next to the output MIDI "
            "with a .csv extension."
        ),
    )

    args = parser.parse_args()

    audio_path  = Path(args.audio)
    midi_path   = Path(args.midi)
    output_path = Path(args.output)

    if not audio_path.exists():
        sys.exit(f"error: audio file not found: {audio_path}")
    if not midi_path.exists():
        sys.exit(f"error: MIDI file not found: {midi_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    instrument = args.instrument.lower()
    low_hz, high_hz = _resolve_band(instrument, args.low_hz, args.high_hz)
    band_str = _band_label(low_hz, high_hz, instrument)

    # ── Load audio ───────────────────────────────────────────────────────────
    print(f"Audio:  {audio_path}")
    audio, sr = load_audio(str(audio_path))
    print(f"        {len(audio) / sr:.1f} s @ {sr} Hz  ({len(audio):,} samples)")

    # ── Load MIDI ────────────────────────────────────────────────────────────
    print(f"MIDI:   {midi_path}")
    track_indices = args.tracks or None
    positions, amplitudes, track_assignments, note_values, track_names = \
        load_markers_midi(str(midi_path), sr, track_indices=track_indices)

    if len(positions) == 0:
        sys.exit(
            "error: no MIDI notes found. "
            "Check --tracks, or verify the file contains note_on events."
        )

    print(f"        {len(positions)} notes")
    if track_names:
        for idx, name in sorted(track_names.items()):
            count = int(np.sum(track_assignments == idx))
            print(f"        Track {idx}: {name!r}  ({count} notes)")

    # ── Refine ───────────────────────────────────────────────────────────────
    print(f"\nRefining with {band_str} …")

    results = refine_all(
        audio, sr, positions,
        low_hz=low_hz,
        high_hz=high_hz,
        search_back_ms=args.search_back_ms,
        search_fwd_ms=args.search_fwd_ms,
        onset_threshold=args.onset_threshold,
        onset_threshold_distant=args.onset_threshold_distant,
        confidence_min=args.confidence_min,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    print_summary(results, sr, audio_path.stem, band_str, args.confidence_warn)

    # ── Save MIDI ────────────────────────────────────────────────────────────
    refined_positions = np.array([r.refined for r in results], dtype=np.int64)

    save_markers_midi(
        refined_positions, sr, str(output_path),
        amplitudes=amplitudes,
        tempo_map=None,           # force_high_ppq will supply PPQ 28800 / 120 BPM
        track_assignments=track_assignments,
        note_values=note_values,
        track_names=track_names,
        force_high_ppq=True,      # PPQ 28800 — highest resolution Pro Tools accepts
    )
    print(f"Output: {output_path}")

    # ── Optional CSV report ──────────────────────────────────────────────────
    if args.report is not None:
        if args.report == "_auto":
            report_path = output_path.with_suffix(".csv")
        else:
            report_path = Path(args.report)
        write_csv_report(results, sr, report_path)


if __name__ == "__main__":
    main()
