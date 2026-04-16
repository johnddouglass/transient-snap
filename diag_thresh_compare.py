"""
Compare threshold=0.10 vs 0.15 to identify which 2 notes become WORSE at 0.15.
"""
import sys, pathlib
import numpy as np
import soundfile as sf
import mido

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from onset import refine_all

PROJECT = pathlib.Path("/Users/johndouglass/Desktop/new song")
SR = 48000

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
                nl.append(int(round(mido.tick2second(tick, mid.ticks_per_beat, tempo) * SR)))
        if nl: notes[track.name] = sorted(nl)
    return notes

def match_notes(orig_pos, ref_pos, window=int(0.020 * SR)):
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

ref  = mido.MidiFile(str(PROJECT / "refined_markers.mid"))
orig = mido.MidiFile(str(PROJECT / "new song.mid"))
rn   = load_notes(ref,  get_tempo(ref))
on   = load_notes(orig, get_tempo(orig))

audio, _ = sf.read(str(PROJECT / "Sn Top D_01.wav"), dtype='float32', always_2d=False)
if audio.ndim > 1: audio = audio[:, 0]

pairs = match_notes(on["Snare MIDI"], rn["Snare MIDI"])
mo = np.array([p[0] for p in pairs], dtype=int)
mr = np.array([p[1] for p in pairs], dtype=int)
orig_err = np.abs(mo - mr)

def run(thresh, fwd=12.0):
    return refine_all(
        audio, SR, mo, low_hz=150.0, high_hz=1200.0,
        search_back_ms=3.0, search_fwd_ms=fwd,
        onset_threshold=thresh, confidence_min=2.0,
        clamp_to_midi=False, min_shift_ms=0.2
    )

r10 = run(0.10)
r15 = run(0.15)

ref10 = np.array([r.refined for r in r10], dtype=int)
ref15 = np.array([r.refined for r in r15], dtype=int)

err10 = np.abs(ref10 - mr)
err15 = np.abs(ref15 - mr)

print("Notes that change status between thresh=0.10 and thresh=0.15:")
print(f"{'idx':>4}  {'orig':>8}  {'err10':>8}  {'err15':>8}  {'delta':>7}  change")
print("-" * 55)

changed = []
for i in range(len(mo)):
    if err10[i] != err15[i]:
        delta = int(err15[i]) - int(err10[i])
        changed.append((i, int(orig_err[i]), int(err10[i]), int(err15[i]), delta))

# Sort by delta descending (worst regressions first)
for i, oe, e10, e15, delta in sorted(changed, key=lambda x: -x[4]):
    tag = "WORSE" if delta > 0 else "better"
    print(f"{i:>4}  {oe:>8}smp  {e10:>7}smp  {e15:>7}smp  {delta:>+7}  {tag}")

n_worse = sum(1 for _, _, e10, e15, _ in changed if e15 > e10)
n_better_cross = sum(1 for _, _, e10, e15, _ in changed if e15 < e10)
print(f"\nTotal changes: {len(changed)}  ({n_worse} worse, {n_better_cross} better)")
print(f"Within ±5 samp: thresh=0.10 → {int(np.sum(err10<=5))}, thresh=0.15 → {int(np.sum(err15<=5))}")

# Also check 0.12 as possible sweet spot
r12 = run(0.12)
ref12 = np.array([r.refined for r in r12], dtype=int)
err12 = np.abs(ref12 - mr)
n_worse_12 = int(np.sum(err12 > orig_err))
print(f"\nthresh=0.12: within ±5 samp = {int(np.sum(err12<=5))}, worse vs orig = {n_worse_12}")

r13 = run(0.13)
ref13 = np.array([r.refined for r in r13], dtype=int)
err13 = np.abs(ref13 - mr)
n_worse_13 = int(np.sum(err13 > orig_err))
print(f"thresh=0.13: within ±5 samp = {int(np.sum(err13<=5))}, worse vs orig = {n_worse_13}")

r14 = run(0.14)
ref14 = np.array([r.refined for r in r14], dtype=int)
err14 = np.abs(ref14 - mr)
n_worse_14 = int(np.sum(err14 > orig_err))
print(f"thresh=0.14: within ±5 samp = {int(np.sum(err14<=5))}, worse vs orig = {n_worse_14}")
