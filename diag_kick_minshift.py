"""
Diagnostic sweep: min_shift_ms for kick drum.

Tests min_shift_ms at 0, 1, 2, and 3 sample granularity (0.0, 0.021, 0.042, 0.063 ms
at 48 kHz) to find the optimal noise-suppression floor for kick onset refinement.
"""
import sys, pathlib
import numpy as np
import mido
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from onset import refine_all

PROJECT = pathlib.Path("/Users/johndouglass/Desktop/_desktop/new song")
SR = 48000
MATCH_WIN = int(0.020 * SR)

# 1 sample at 48 kHz = 0.020833 ms
SWEEP = [0.0, 0.021, 0.042, 0.063]   # 0, 1, 2, 3 samples

def get_tempo(mid):
    for t in mid.tracks:
        for m in t:
            if m.type == 'set_tempo':
                return m.tempo
    return 500000

def load_notes(mid, tempo):
    notes = {}
    for track in mid.tracks:
        tick, nl = 0, []
        for m in track:
            tick += m.time
            if m.type == 'note_on' and m.velocity > 0:
                nl.append(int(round(mido.tick2second(tick, mid.ticks_per_beat, tempo) * SR)))
        if nl:
            notes[track.name] = sorted(nl)
    return notes

def match_notes(orig_pos, ref_pos, window):
    used, pairs = set(), []
    for op in sorted(orig_pos):
        best_i, best_d = None, window + 1
        for i, rp in enumerate(ref_pos):
            if i in used:
                continue
            d = abs(op - rp)
            if d < best_d:
                best_d, best_i = d, i
        if best_i is not None and best_d <= window:
            pairs.append((op, ref_pos[best_i]))
            used.add(best_i)
    return pairs

ref  = mido.MidiFile(str(PROJECT / "refined_markers.mid"))
orig = mido.MidiFile(str(PROJECT / "new song.mid"))
rn = load_notes(ref, get_tempo(ref))
on = load_notes(orig, get_tempo(orig))

# Kick config (matches validate.py)
KICK_WAV     = "Kick In_01.wav"
KICK_ORIG_TK = "Kick MIDI"
KICK_REF_TK  = "Kick MIDI"
LOW_HZ       = 40.0
HIGH_HZ      = 180.0
ONSET_THR    = 0.05
FWD_MS       = 10.0

audio, _ = sf.read(str(PROJECT / KICK_WAV), dtype='float32', always_2d=False)
audio = audio[:, 0] if audio.ndim > 1 else audio

pairs = match_notes(on[KICK_ORIG_TK], rn[KICK_REF_TK], MATCH_WIN)
mo    = np.array([p[0] for p in pairs], dtype=int)
mr    = np.array([p[1] for p in pairs], dtype=int)
n     = len(pairs)

print(f"\nKick: {n} notes matched\n")
print(f"{'min_shift_ms':>14}  {'(samples)':>9}  {'n_within5':>9}  {'n_moved':>7}  {'n_frozen':>8}  {'n_worse':>7}  {'mean_err_ms':>11}  {'passes'}")
print("-" * 90)

orig_ms = (mo - mr) / SR * 1000.0

for ms in SWEEP:
    samp_approx = ms * SR / 1000.0
    results = refine_all(
        audio, SR, mo,
        low_hz=LOW_HZ, high_hz=HIGH_HZ,
        search_back_ms=3.0, search_fwd_ms=FWD_MS,
        onset_threshold=ONSET_THR, onset_threshold_distant=None,
        confidence_min=2.0, clamp_to_midi=False,
        min_shift_ms=ms,
    )
    refined  = np.array([r.refined for r in results], dtype=int)
    err_ms   = (refined - mr) / SR * 1000.0
    err_samp = np.abs(refined - mr)
    n_within5 = int(np.sum(err_samp <= 5))
    n_moved   = int(np.sum(np.array([r.offset_samples for r in results]) != 0))
    n_frozen  = sum(1 for r in results if r.pass_used == 'frozen')
    n_worse   = int(np.sum(np.abs(err_ms) > np.abs(orig_ms)))
    passes    = {}
    for r in results:
        passes[r.pass_used] = passes.get(r.pass_used, 0) + 1

    print(f"{ms:>14.3f}  {samp_approx:>9.2f}  {n_within5:>9}  {n_moved:>7}  {n_frozen:>8}  {n_worse:>7}  {np.mean(err_ms):>+11.3f}  {passes}")
