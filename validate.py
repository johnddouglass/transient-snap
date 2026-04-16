import sys, pathlib
import numpy as np
import mido
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from onset import refine_all, refine_position, bandpass

PROJECT = pathlib.Path("/Users/johndouglass/Desktop/new song")

def get_tempo(mid):
    for t in mid.tracks:
        for m in t:
            if m.type == 'set_tempo': return m.tempo
    return 500000

def load_notes(mid, tempo):
    notes = {}
    for track in mid.tracks:
        tick, nl = 0, []
        for m in track:
            tick += m.time
            if m.type == 'note_on' and m.velocity > 0:
                nl.append(int(round(mido.tick2second(tick, mid.ticks_per_beat, tempo) * 48000)))
        if nl: notes[track.name] = sorted(nl)
    return notes

ref  = mido.MidiFile(str(PROJECT / "refined_markers.mid"))
orig = mido.MidiFile(str(PROJECT / "new song.mid"))
rn = load_notes(ref, get_tempo(ref))
on = load_notes(orig, get_tempo(orig))

SR = 48000
MATCH_WIN = int(0.020 * SR)
CONFIGS = [
    # (orig_track, ref_track, wav_file, low_hz, high_hz, label,
    #  clamp_to_midi, min_shift_ms, onset_thresh, onset_thresh_dist, fwd_ms)
    #
    # clamp_to_midi=True for toms: MIDI is placed at stick impact and must
    # not be moved later. The LF body (80–600 Hz) builds 2–5 ms after the
    # click; without clamping, Stage 2 walk-back latches onto the quiet gap
    # between the decayed click and the rising LF resonance.
    #
    # clamp_to_midi=False for kick/snare: kick/snare MIDI is often placed at
    # beat grid and the refined position may legitimately be slightly later
    # (drummer slightly behind beat) or earlier (snare placed at cymbal hit).
    #
    # min_shift_ms=0.2 for snare: the detection produces 1-7 sample noise
    # shifts (0.02-0.15ms) on correctly-placed notes. All 70 WORSE snare
    # cases had shifts <= 0.146ms. A 0.2ms floor eliminates all noise moves
    # while preserving every meaningful correction (>= 0.2ms).
    #
    # onset_thresh=0.10 for snare CLOSE path: threshold sweep confirms 0.10
    # gives the best <=5samp count with 0 WORSE on close-detected notes.
    # Values above 0.10 cause regressions on well-placed CLOSE notes such as
    # notes with a cymbal 2–3ms after MIDI (stage1 peak is CLOSE, walk-back
    # stops inside the onset frame).
    #
    # onset_thresh_dist=0.15 for snare DISTANT path: separate threshold for
    # notes where MIDI was placed 5–15ms before the acoustic onset (e.g.
    # snare MIDI placed at a cymbal hit). The DISTANT walk-back benefits from
    # a higher threshold — it stops closer to the onset peak rather than at
    # the inter-onset gap caused by cymbal decay. Empirically 0.15 improves
    # large-shift notes (idx 14: 38→8 samp, idx 90: 36→4 samp) without
    # affecting CLOSE-detected notes.
    #
    # fwd_ms=12.0 for snare: one note (idx 155) has its onset at 10.73 ms
    # after MIDI, outside the 10 ms window. At 12 ms, conf=9.99 and the
    # note lands within 1 ms of reference. 10 ms is enough for all others.
    ("Kick MIDI",       "Kick MIDI",       "Kick In_01.wav",   40.0,  180.0, "Kick",   False, 0.0, 0.05, None, 10.0),
    ("Snare MIDI",      "Snare MIDI",      "Sn Top D_01.wav", 150.0, 1200.0, "Snare",  False, 0.2, 0.10, 0.15, 12.0),
    ("Snare Fill MIDI", "Snare Fill MIDI", "Sn Top D_01.wav", 150.0, 1200.0, "SnFill", False, 0.2, 0.10, 0.15, 12.0),
    ("T1 MIDI",         "T1 MIDI",         "F Tom 1_01.wav",   80.0,  600.0, "T1",     True,  0.0, 0.05, None, 10.0),
    ("T2 MIDI",         "T2 MIDI",         "R Tom_01.wav",     80.0,  600.0, "T2",     True,  0.0, 0.05, None, 10.0),
]

def match_notes(orig_pos, ref_pos, window):
    used, pairs = set(), []
    for op in sorted(orig_pos):
        best_i, best_d = None, window + 1
        for i, rp in enumerate(ref_pos):
            if i in used: continue
            d = abs(op - rp)
            if d < best_d: best_d, best_i = d, i
        if best_i is not None and best_d <= window:
            pairs.append((op, ref_pos[best_i])); used.add(best_i)
    return pairs

print(f"\n{'Track':<10} {'N':>4}  {'mean_err':>9}  {'|mean|':>6}  {'std':>6}  {'<=5smp':>7}  {'better':>6}  {'worse':>5}  passes")
print("-" * 91)

audio_cache = {}
for orig_tk, ref_tk, wav_file, low_hz, high_hz, label, clamp_to_midi, min_shift_ms, onset_thresh, onset_thresh_dist, fwd_ms in CONFIGS:
    if wav_file not in audio_cache:
        a, _ = sf.read(str(PROJECT / wav_file), dtype='float32', always_2d=False)
        audio_cache[wav_file] = a[:, 0] if a.ndim > 1 else a
    audio = audio_cache[wav_file]

    pairs = match_notes(on[orig_tk], rn[ref_tk], MATCH_WIN)
    mo = np.array([p[0] for p in pairs], dtype=int)
    mr = np.array([p[1] for p in pairs], dtype=int)

    results = refine_all(audio, SR, mo, low_hz=low_hz, high_hz=high_hz,
        search_back_ms=3.0, search_fwd_ms=fwd_ms, onset_threshold=onset_thresh,
        onset_threshold_distant=onset_thresh_dist, confidence_min=2.0,
        clamp_to_midi=clamp_to_midi, min_shift_ms=min_shift_ms)

    refined   = np.array([r.refined for r in results], dtype=int)
    err_ms    = (refined - mr) / SR * 1000.0
    err_samp  = np.abs(refined - mr)
    orig_ms   = (mo - mr) / SR * 1000.0
    n_within5 = int(np.sum(err_samp <= 5))
    n_better  = int(np.sum(np.abs(err_ms) < np.abs(orig_ms)))
    n_worse   = int(np.sum(np.abs(err_ms) > np.abs(orig_ms)))
    passes    = {}
    for r in results: passes[r.pass_used] = passes.get(r.pass_used, 0) + 1

    print(f"{label:<10} {len(pairs):>4}  {np.mean(err_ms):>+9.2f}  "
          f"{np.mean(np.abs(err_ms)):>6.2f}  {np.std(err_ms):>6.2f}  "
          f"{n_within5:>7}  {n_better:>6}  {n_worse:>5}  {passes}")
