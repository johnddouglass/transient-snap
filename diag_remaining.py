"""
Show the remaining snare notes outside ±5 samples after the threshold/fwd_ms changes.
Categorise each: frozen-needs-correction vs moved-residual vs frozen-correctly.
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

results = refine_all(
    audio, SR, mo, low_hz=150.0, high_hz=1200.0,
    search_back_ms=3.0, search_fwd_ms=12.0,
    onset_threshold=0.10, confidence_min=2.0,
    clamp_to_midi=False, min_shift_ms=0.2
)

refined  = np.array([r.refined for r in results], dtype=int)
err_samp = refined - mr          # positive = after ref, negative = before ref
orig_err = mo      - mr          # positive = MIDI after ref (late), neg = MIDI before ref (early)

print(f"Snare notes outside ±5 samples of reference:")
print(f"{'idx':>4}  {'orig_err':>10}  {'refined':>9}  {'err_samp':>9}  {'pass':>8}  {'conf':>6}  category")
print("-" * 72)

outside = np.where(np.abs(err_samp) > 5)[0]
cats = {'frozen-needs-fix': 0, 'moved-residual': 0, 'frozen-correct': 0}

for idx in sorted(outside, key=lambda i: abs(err_samp[i]), reverse=True):
    r = results[idx]
    oe_ms  = orig_err[idx] / SR * 1000
    es_ms  = err_samp[idx] / SR * 1000
    shift  = r.offset_samples

    if r.pass_used == 'frozen' and abs(orig_err[idx]) > 5:
        cat = 'frozen-needs-fix'   # frozen but MIDI was also wrong — should have moved
    elif r.pass_used != 'frozen' and abs(err_samp[idx]) > 5:
        cat = 'moved-residual'     # moved but didn't land close enough
    else:
        cat = 'frozen-correct'     # frozen, MIDI was fine (orig ≤ 5 samp), stayed put
    cats[cat] += 1

    print(f"{idx:>4}  {oe_ms:>+9.2f}ms  {shift:>+8}smp  {es_ms:>+8.2f}ms  "
          f"{r.pass_used:>8}  {r.confidence:>6.2f}  {cat}")

print()
print(f"Total outside ±5 samples: {len(outside)}")
for k, v in cats.items():
    print(f"  {k}: {v}")

print(f"\nNotes within ±5 samples: {len(mo) - len(outside)} / {len(mo)}")

# Additional breakdown: of the frozen-needs-fix, what's their original error?
frozen_fix = [outside[i] for i, idx in enumerate(outside) if
              results[idx].pass_used == 'frozen' and abs(orig_err[idx]) > 5]
if frozen_fix:
    print(f"\nFrozen-needs-fix details:")
    for idx in sorted(frozen_fix):
        r = results[idx]
        oe = orig_err[idx] / SR * 1000
        print(f"  idx={idx}  orig_err={oe:+.2f}ms  conf={r.confidence:.2f}")
