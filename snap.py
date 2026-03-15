#!/usr/bin/env python3
"""
Transient snap refinement.

Takes approximate drum marker positions and snaps each one to the
onset of the nearest transient. Works by finding the highest peak
in a search window, then walking backwards to find where the
attack begins.

Usage:
    python snap.py --audio audio.wav --markers markers.wav --output refined.wav --tick tick.wav
"""

import argparse
import numpy as np
import soundfile as sf
import librosa
import mido
from scipy.ndimage import maximum_filter1d
from pathlib import Path


def load_audio(path, target_sr=None):
    """Load audio, convert to mono, optionally resample."""
    audio, sr = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if target_sr and sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio, sr


def detect_marker_positions(marker_audio, sample_rate, threshold_db=-40.0, min_distance_ms=50.0):
    """Find tick onset positions in a marker audio file."""
    threshold = 10 ** (threshold_db / 20)
    above = np.abs(marker_audio) > threshold
    if not np.any(above):
        return np.array([], dtype=np.int64)

    padded = np.concatenate([[False], above])
    edges = np.where(np.diff(padded.astype(int)) == 1)[0]
    if len(edges) == 0:
        return np.array([], dtype=np.int64)

    min_dist = int(sample_rate * min_distance_ms / 1000)
    filtered = [edges[0]]
    for pos in edges[1:]:
        if pos - filtered[-1] >= min_dist:
            filtered.append(pos)
    return np.array(filtered, dtype=np.int64)


def detect_marker_amplitudes(marker_audio, positions, window_samples=100):
    """Get peak amplitude of each marker tick."""
    amps = []
    for pos in positions:
        pos = int(pos)
        end = min(pos + window_samples, len(marker_audio))
        if pos < len(marker_audio):
            amps.append(np.max(np.abs(marker_audio[pos:end])))
        else:
            amps.append(0.0)
    return np.array(amps, dtype=np.float32)


def compute_envelope(audio, window=5):
    """Compute a smoothed amplitude envelope using a running max."""
    return maximum_filter1d(np.abs(audio), size=window)


def find_onset(audio, search_start, search_end, onset_threshold=0.05, envelope_window=5):
    """
    Find the onset of the most prominent transient in a search window.

    1. Find the highest peak in the window
    2. Walk backwards from the peak along the envelope
    3. The onset is where the envelope drops below onset_threshold * peak_amplitude

    Args:
        audio: Full audio array
        search_start: Start of search window (sample index)
        search_end: End of search window (sample index)
        onset_threshold: Fraction of peak amplitude that defines onset (0.03-0.10)
        envelope_window: Samples for envelope smoothing

    Returns:
        (onset_position, peak_position, confidence)
        confidence: ratio of peak to second-highest peak (higher = more certain)
    """
    search_start = max(0, search_start)
    search_end = min(len(audio), search_end)
    segment = audio[search_start:search_end]

    if len(segment) == 0:
        return search_start, search_start, 0.0

    # Compute envelope
    envelope = compute_envelope(segment, window=envelope_window)

    # Find the highest peak
    peak_idx = np.argmax(envelope)
    peak_amp = envelope[peak_idx]

    if peak_amp == 0:
        return search_start, search_start, 0.0

    # Threshold for onset detection
    threshold = peak_amp * onset_threshold

    # Walk backwards from peak to find onset
    onset_idx = peak_idx
    for i in range(peak_idx - 1, -1, -1):
        if envelope[i] < threshold:
            onset_idx = i + 1  # One sample after we drop below threshold
            break
    else:
        onset_idx = 0  # Reached beginning of window

    # Confidence: check if there's a competing peak
    # Suppress the main peak region and look for second peak
    suppressed = envelope.copy()
    suppress_start = max(0, peak_idx - 20)
    suppress_end = min(len(suppressed), peak_idx + 20)
    suppressed[suppress_start:suppress_end] = 0
    second_peak = np.max(suppressed) if len(suppressed) > 0 else 0

    if second_peak > 0:
        confidence = peak_amp / second_peak  # Higher = more confident
    else:
        confidence = float('inf')  # No competing peak

    onset_pos = search_start + onset_idx
    peak_pos = search_start + peak_idx

    return onset_pos, peak_pos, confidence


def snap_markers(audio, marker_positions, sample_rate, search_ms=5.0,
                 onset_threshold=0.05, envelope_window=5):
    """
    Snap all marker positions to the onset of the nearest transient.

    Args:
        audio: Audio array
        marker_positions: Array of approximate marker positions
        sample_rate: Sample rate
        search_ms: Search window radius in ms (searches ±search_ms around each marker)
        onset_threshold: Fraction of peak that defines onset
        envelope_window: Envelope smoothing window in samples

    Returns:
        (refined_positions, stats_per_marker)
    """
    search_samples = int(sample_rate * search_ms / 1000)

    refined = []
    stats = []

    for pos in marker_positions:
        pos = int(pos)
        search_start = pos - search_samples
        search_end = pos + search_samples

        onset, peak, confidence = find_onset(
            audio, search_start, search_end,
            onset_threshold=onset_threshold,
            envelope_window=envelope_window
        )

        offset = onset - pos
        offset_ms = (offset / sample_rate) * 1000

        refined.append(onset)
        stats.append({
            'original': pos,
            'refined': onset,
            'peak': peak,
            'offset_samples': offset,
            'offset_ms': offset_ms,
            'confidence': confidence,
        })

    return np.array(refined, dtype=np.int64), stats


def get_midi_track_info(path):
    """Get info about tracks in a MIDI file.

    Returns:
        List of dicts with 'index', 'name', 'note_count' for each track.
    """
    mid = mido.MidiFile(path)
    tracks = []
    for i, track in enumerate(mid.tracks):
        name = track.name or f"Track {i}"
        note_count = sum(1 for msg in track
                         if msg.type == 'note_on' and msg.velocity > 0)
        tracks.append({'index': i, 'name': name, 'note_count': note_count})
    return tracks


def extract_tempo_map(path):
    """Extract PPQ and tempo map from a MIDI file.

    Returns:
        dict with 'ppq' and 'tempo_events' (list of (abs_tick, tempo_us) tuples)
    """
    mid = mido.MidiFile(path)
    ppq = mid.ticks_per_beat
    tempo_events = []

    # Tempo events are typically on the first track
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'set_tempo':
                tempo_events.append((abs_tick, msg.tempo))

    if not tempo_events:
        tempo_events = [(0, 500000)]  # default 120 BPM

    return {'ppq': ppq, 'tempo_events': tempo_events}


def _tempo_map_to_samples(tempo_map, sample_rate):
    """Convert tempo map to a list of (abs_tick, abs_sample, tempo_us).

    Uses integer sample positions to avoid floating-point accumulation errors.
    """
    ppq = tempo_map['ppq']
    events = sorted(tempo_map['tempo_events'], key=lambda x: x[0])
    result = []

    # Use Fraction for exact arithmetic, convert to int samples at the end
    from fractions import Fraction
    abs_sample_frac = Fraction(0)
    prev_tick = 0
    prev_tempo = events[0][1] if events else 500000

    for tick, tempo in events:
        if tick > prev_tick:
            delta_ticks = tick - prev_tick
            # seconds = delta_ticks * tempo_us / (ppq * 1_000_000)
            # samples = seconds * sample_rate
            # Combined: samples = delta_ticks * tempo_us * sample_rate / (ppq * 1_000_000)
            abs_sample_frac += Fraction(delta_ticks * prev_tempo * sample_rate, ppq * 1_000_000)
        result.append((tick, int(round(float(abs_sample_frac))), tempo))
        prev_tick = tick
        prev_tempo = tempo

    return result


def _sample_to_tick(sample_pos, tempo_map, sample_rate):
    """Convert a sample position to an absolute tick using the tempo map.

    Uses exact rational arithmetic. Returns the tick that, when converted back
    to samples, gives the closest value to sample_pos (may be equal, +1, or -1).
    """
    from fractions import Fraction
    import math

    ppq = tempo_map['ppq']
    timeline = _tempo_map_to_samples(tempo_map, sample_rate)

    # Find which tempo segment this sample falls in
    seg_tick = 0
    seg_sample = 0
    seg_tempo = timeline[0][2] if timeline else 500000

    for tick, samp, tempo in timeline:
        if samp > sample_pos:
            break
        seg_tick = tick
        seg_sample = samp
        seg_tempo = tempo

    # Convert remaining samples to ticks using exact arithmetic
    # ticks = samples * ppq * 1_000_000 / (tempo_us * sample_rate)
    remaining_samples = sample_pos - seg_sample
    remaining_ticks_frac = Fraction(remaining_samples * ppq * 1_000_000, seg_tempo * sample_rate)

    # Try both floor and ceil, pick the one that gives closest round-trip
    tick_floor = seg_tick + int(math.floor(float(remaining_ticks_frac)))
    tick_ceil = seg_tick + int(math.ceil(float(remaining_ticks_frac)))

    # Compute what sample each tick maps back to (inline to avoid circular call)
    # sample = tick * tempo * sample_rate / (ppq * 1_000_000)
    def tick_to_sample_inline(t):
        delta = t - seg_tick
        return seg_sample + int(round(float(Fraction(delta * seg_tempo * sample_rate, ppq * 1_000_000))))

    sample_floor = tick_to_sample_inline(tick_floor)
    sample_ceil = tick_to_sample_inline(tick_ceil)

    # Pick the tick that gives the closest sample position
    if abs(sample_floor - sample_pos) <= abs(sample_ceil - sample_pos):
        return tick_floor
    else:
        return tick_ceil


def _tick_to_sample(abs_tick, tempo_map, sample_rate):
    """Convert an absolute tick to a sample position using exact arithmetic."""
    from fractions import Fraction

    ppq = tempo_map['ppq']
    events = sorted(tempo_map['tempo_events'], key=lambda x: x[0])

    # Find which tempo segment this tick falls in
    seg_tick = 0
    seg_sample_frac = Fraction(0)
    seg_tempo = events[0][1] if events else 500000

    for tick, tempo in events:
        if tick > abs_tick:
            break
        if tick > seg_tick:
            delta_ticks = tick - seg_tick
            seg_sample_frac += Fraction(delta_ticks * seg_tempo * sample_rate, ppq * 1_000_000)
        seg_tick = tick
        seg_tempo = tempo

    # Convert remaining ticks to samples
    remaining_ticks = abs_tick - seg_tick
    remaining_samples = Fraction(remaining_ticks * seg_tempo * sample_rate, ppq * 1_000_000)

    return int(round(float(seg_sample_frac + remaining_samples)))


def load_markers_midi(path, sample_rate, track_indices=None):
    """Load marker positions and velocities from a MIDI file.

    Args:
        path: Path to MIDI file
        sample_rate: Target sample rate
        track_indices: List of track indices to include (None = all tracks)

    Returns:
        (positions, amplitudes, track_assignments, note_values, track_names)
        - positions: sample positions (int64)
        - amplitudes: velocities mapped to 0.0-1.0 (float32)
        - track_assignments: source track index for each note (int32)
        - note_values: MIDI note number for each note (int32)
        - track_names: dict mapping track index to track name
    """
    mid = mido.MidiFile(path)
    tempo_map = extract_tempo_map(path)

    # Collect note_on events from selected tracks
    positions = []
    velocities = []
    track_assigns = []
    note_vals = []
    track_names = {}

    for i, track in enumerate(mid.tracks):
        if track_indices is not None and i not in track_indices:
            continue

        track_names[i] = track.name or f"Track {i}"

        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                sample_pos = _tick_to_sample(abs_tick, tempo_map, sample_rate)
                positions.append(sample_pos)
                velocities.append(msg.velocity / 127.0)
                track_assigns.append(i)
                note_vals.append(msg.note)

    if not positions:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.float32),
                np.array([], dtype=np.int32), np.array([], dtype=np.int32), {})

    positions = np.array(positions, dtype=np.int64)
    amplitudes = np.array(velocities, dtype=np.float32)
    track_assigns = np.array(track_assigns, dtype=np.int32)
    note_vals = np.array(note_vals, dtype=np.int32)

    # Sort by position
    sort_idx = np.argsort(positions)
    return (positions[sort_idx], amplitudes[sort_idx],
            track_assigns[sort_idx], note_vals[sort_idx], track_names)


def save_markers_midi(positions, sample_rate, output_path, amplitudes=None,
                      note=36, tempo_map=None, track_assignments=None,
                      note_values=None, track_names=None, debug=False,
                      force_high_ppq=True):
    """Save marker positions as a MIDI file.

    Args:
        positions: Sample positions
        sample_rate: Sample rate of the audio
        output_path: Output .mid file path
        amplitudes: Optional amplitudes (0.0-1.0) mapped to MIDI velocity
        note: Default MIDI note number (used when note_values is None)
        tempo_map: Tempo map dict from extract_tempo_map() (None for default)
        track_assignments: Per-note track index array (None = single track)
        note_values: Per-note MIDI note numbers (None = use `note` for all)
        track_names: Dict mapping track index to name (for multi-track)
        debug: If True, print round-trip verification for each note
        force_high_ppq: If True, use PPQ=48000/120BPM for exact sample mapping
    """
    if tempo_map is None or force_high_ppq:
        # PPQ 28800 at 120 BPM - compatible with Pro Tools import from Reaper
        tempo_map = {'ppq': 28800, 'tempo_events': [(0, 500000)]}

    ppq = tempo_map['ppq']

    # Compute a short note length (~5ms using first tempo)
    first_tempo = tempo_map['tempo_events'][0][1]
    tps = ppq * 1e6 / first_tempo
    note_len = max(1, int(tps * 0.005))

    if track_assignments is not None and len(set(track_assignments)) > 0:
        # Multi-track export (Type 1 MIDI)
        mid = mido.MidiFile(type=1, ticks_per_beat=ppq)

        # Track 0: tempo map only
        tempo_track = mido.MidiTrack()
        mid.tracks.append(tempo_track)
        prev_tick = 0
        for abs_tick, tempo_us in sorted(tempo_map['tempo_events'], key=lambda x: x[0]):
            delta = abs_tick - prev_tick
            tempo_track.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=delta))
            prev_tick = abs_tick

        # One track per unique source track
        unique_tracks = sorted(set(track_assignments))
        for trk_idx in unique_tracks:
            track = mido.MidiTrack()
            name = (track_names or {}).get(trk_idx, f"Track {trk_idx}")
            track.append(mido.MetaMessage('track_name', name=name, time=0))
            mid.tracks.append(track)

            # Gather notes for this track
            mask = np.array(track_assignments) == trk_idx
            trk_positions = np.array(positions)[mask]
            trk_amplitudes = amplitudes[mask] if amplitudes is not None else None
            trk_note_values = note_values[mask] if note_values is not None else None

            # Sort by position within track
            sort_idx = np.argsort(trk_positions)
            trk_positions = trk_positions[sort_idx]
            if trk_amplitudes is not None:
                trk_amplitudes = trk_amplitudes[sort_idx]
            if trk_note_values is not None:
                trk_note_values = trk_note_values[sort_idx]

            errors = []
            prev_tick = 0
            for i, pos in enumerate(trk_positions):
                pos_int = int(pos)
                abs_tick = _sample_to_tick(pos_int, tempo_map, sample_rate)
                delta = max(0, abs_tick - prev_tick)

                # Debug: verify round-trip
                if debug:
                    round_trip = _tick_to_sample(abs_tick, tempo_map, sample_rate)
                    if round_trip != pos_int:
                        errors.append((i, pos_int, abs_tick, round_trip, round_trip - pos_int))

                vel = 100
                if trk_amplitudes is not None and i < len(trk_amplitudes):
                    vel = max(1, min(127, int(trk_amplitudes[i] * 127)))

                n = note
                if trk_note_values is not None and i < len(trk_note_values):
                    n = int(trk_note_values[i])

                track.append(mido.Message('note_on', note=n, velocity=vel, time=delta))
                track.append(mido.Message('note_off', note=n, velocity=0, time=note_len))
                prev_tick = abs_tick + note_len

            if debug and errors:
                print(f"\n=== MIDI ROUND-TRIP ERRORS for track '{name}' ({len(errors)}) ===")
                print(f"PPQ: {ppq}, Sample rate: {sample_rate}")
                for idx, orig, tick, rt, diff in errors[:10]:
                    print(f"  #{idx}: sample {orig} -> tick {tick} -> sample {rt} (diff: {diff:+d})")

        if debug:
            print(f"\n=== MIDI Export Complete ===")
            print(f"PPQ: {ppq}, Tempo events: {len(tempo_map['tempo_events'])}")
            print(f"Tracks: {len(unique_tracks)}, Total notes: {len(positions)}")

        mid.save(output_path)
    else:
        # Single-track export
        mid = mido.MidiFile(ticks_per_beat=ppq)
        track = mido.MidiTrack()
        mid.tracks.append(track)

        # Write tempo events
        prev_tick = 0
        for abs_tick, tempo_us in sorted(tempo_map['tempo_events'], key=lambda x: x[0]):
            delta = abs_tick - prev_tick
            track.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=delta))
            prev_tick = abs_tick

        # Write notes
        errors = []
        for i, pos in enumerate(positions):
            pos_int = int(pos)
            abs_tick = _sample_to_tick(pos_int, tempo_map, sample_rate)
            delta = max(0, abs_tick - prev_tick)

            # Debug: verify round-trip
            if debug:
                round_trip = _tick_to_sample(abs_tick, tempo_map, sample_rate)
                if round_trip != pos_int:
                    errors.append((i, pos_int, abs_tick, round_trip, round_trip - pos_int))

            vel = 100
            if amplitudes is not None and i < len(amplitudes):
                vel = max(1, min(127, int(amplitudes[i] * 127)))

            n = note
            if note_values is not None and i < len(note_values):
                n = int(note_values[i])

            track.append(mido.Message('note_on', note=n, velocity=vel, time=delta))
            track.append(mido.Message('note_off', note=n, velocity=0, time=note_len))
            prev_tick = abs_tick + note_len

        if debug and errors:
            print(f"\n=== MIDI ROUND-TRIP ERRORS ({len(errors)}/{len(positions)}) ===")
            print(f"PPQ: {ppq}, Sample rate: {sample_rate}")
            print(f"Tempo events: {tempo_map['tempo_events'][:5]}...")
            for idx, orig, tick, rt, diff in errors[:20]:
                print(f"  #{idx}: sample {orig} -> tick {tick} -> sample {rt} (diff: {diff:+d})")
            if len(errors) > 20:
                print(f"  ... and {len(errors) - 20} more errors")

        mid.save(output_path)


def save_markers_wav(positions, sample_rate, output_path, tick_template,
                     total_length, amplitudes=None):
    """Generate a WAV file with tick markers at specified positions."""
    output = np.zeros(total_length, dtype=np.float32)
    tick_len = len(tick_template)

    tick_peak = np.max(np.abs(tick_template))
    tick_normalized = tick_template / tick_peak if tick_peak > 0 else tick_template

    for i, pos in enumerate(positions):
        pos = int(pos)
        end_pos = min(pos + tick_len, total_length)
        tick_samples = end_pos - pos

        if tick_samples > 0 and pos >= 0:
            if amplitudes is not None and i < len(amplitudes):
                output[pos:end_pos] = tick_normalized[:tick_samples] * amplitudes[i]
            else:
                output[pos:end_pos] = tick_template[:tick_samples]

    sf.write(output_path, output, sample_rate)


def main():
    parser = argparse.ArgumentParser(description='Snap drum markers to transient onsets')
    parser.add_argument('--audio', required=True, help='Input drum audio WAV')
    parser.add_argument('--markers', required=True, help='Input approximate markers WAV')
    parser.add_argument('--output', required=True, help='Output refined markers WAV')
    parser.add_argument('--tick', required=True, help='Tick template WAV for output')
    parser.add_argument('--search-ms', type=float, default=15.0,
                        help='Search window radius in ms (default: 15.0)')
    parser.add_argument('--onset-threshold', type=float, default=0.05,
                        help='Onset threshold as fraction of peak (default: 0.05)')
    parser.add_argument('--sample-rate', type=int, default=48000,
                        help='Target sample rate (default: 48000)')
    args = parser.parse_args()

    print(f"Loading audio: {args.audio}")
    audio, sr = load_audio(args.audio, target_sr=args.sample_rate)
    print(f"  {len(audio)} samples, {sr}Hz, {len(audio)/sr:.1f}s")

    print(f"Loading markers: {args.markers}")
    marker_audio, _ = load_audio(args.markers, target_sr=args.sample_rate)
    positions = detect_marker_positions(marker_audio, sr)
    amplitudes = detect_marker_amplitudes(marker_audio, positions)
    print(f"  Found {len(positions)} markers")

    print(f"Loading tick: {args.tick}")
    tick, _ = load_audio(args.tick, target_sr=args.sample_rate)

    print(f"\nSnapping markers (±{args.search_ms}ms window, {args.onset_threshold} threshold)...")
    refined, stats = snap_markers(
        audio, positions, sr,
        search_ms=args.search_ms,
        onset_threshold=args.onset_threshold
    )

    # Report
    offsets_ms = [s['offset_ms'] for s in stats]
    offsets_samples = [s['offset_samples'] for s in stats]
    moved = [s for s in stats if s['offset_samples'] != 0]
    low_conf = [s for s in stats if s['confidence'] < 3.0]

    print(f"\nResults:")
    print(f"  Markers moved: {len(moved)}/{len(stats)}")
    print(f"  Mean offset: {np.mean(offsets_ms):.2f}ms ({np.mean(offsets_samples):.1f} samples)")
    print(f"  Std offset:  {np.std(offsets_ms):.2f}ms")
    print(f"  Max offset:  {np.max(np.abs(offsets_ms)):.2f}ms ({np.max(np.abs(offsets_samples))} samples)")

    if low_conf:
        print(f"\n  Low confidence markers: {len(low_conf)} (possible bleed interference)")
        for s in low_conf[:10]:
            t = s['original'] / sr
            print(f"    @ {t:.2f}s - offset {s['offset_ms']:+.2f}ms, confidence {s['confidence']:.1f}")

    print(f"\nSaving refined markers: {args.output}")
    save_markers_wav(refined, sr, args.output, tick, len(audio), amplitudes)
    print("Done.")


if __name__ == '__main__':
    main()
