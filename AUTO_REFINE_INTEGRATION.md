# Feature: Auto-Refine Integration for transient_snap.py

## Context

`/Users/johndouglass/Documents/Coding/transient_snap/onset.py` contains a batch
drum-onset detection function `refine_all()` that takes a close-mic audio array
and an array of MIDI sample positions and returns sample-accurate refined
positions. It has been fully tuned and tested. The goal is to integrate it into
`transient_snap.py` as an optional "Auto-Refine" step that runs before the user
does manual review.

---

## Algorithm API — exactly how to call it

```python
from onset import refine_all, INSTRUMENT_BANDS

results = refine_all(
    audio,                            # np.ndarray, float32 mono, at the element's SR
    sr,                               # int, sample rate (e.g. 48000)
    positions,                        # np.ndarray of int sample positions from MIDI
    low_hz=low_hz,                    # float, lower bandpass cutoff
    high_hz=high_hz,                  # float, upper bandpass cutoff
    search_back_ms=3.0,               # float, look-back window in ms
    search_fwd_ms=fwd_ms,             # float, look-forward window in ms
    onset_threshold=onset_thr,        # float, CLOSE path walk-back threshold
    onset_threshold_distant=dist_thr, # float or None, DISTANT path threshold
    confidence_min=2.0,               # float, below this → wideband retry or freeze
    clamp_to_midi=clamp,              # bool, True for toms
    min_shift_ms=min_shift,           # float, suppress sub-threshold noise moves
)
```

Each element of `results` is an `OnsetResult` dataclass:

```python
r.original        # int, original sample position
r.refined         # int, refined sample position  ← use this
r.offset_samples  # int, refined - original
r.offset_ms       # float, offset in ms
r.confidence      # float, detection confidence (peak-to-mean ratio)
r.pass_used       # str: 'primary', 'wideband', or 'frozen'
```

`INSTRUMENT_BANDS` is a dict mapping instrument name → `(low_hz, high_hz)`:

```python
INSTRUMENT_BANDS = {
    "kick":      (40.0,   180.0),
    "snare":    (150.0,  1200.0),
    "hihat":   (7000.0, 18000.0),
    "ride":    (5000.0, 16000.0),
    "crash":   (3000.0, 16000.0),
    "tom":      (80.0,   600.0),
    "overhead":  (0.0,     0.0),
    "room":      (0.0,     0.0),
}
```

---

## Per-instrument presets

Build a dict of per-instrument parameters. These are empirically tuned — do not
change them:

```python
# (low_hz, high_hz, onset_thr, onset_thr_dist, fwd_ms, clamp_to_midi, min_shift_ms)
REFINE_PRESETS = {
    "kick":     (40.0,   180.0,  0.05, None, 10.0, False, 0.0),
    "snare":    (150.0, 1200.0,  0.10, 0.15, 12.0, False, 0.2),
    "hihat":   (7000.0,18000.0,  0.05, None, 10.0, False, 0.0),
    "ride":    (5000.0,16000.0,  0.05, None, 10.0, False, 0.0),
    "crash":   (3000.0,16000.0,  0.05, None, 10.0, False, 0.0),
    "tom":      (80.0,   600.0,  0.05, None, 10.0, True,  0.0),  # clamp_to_midi!
    "overhead":  (0.0,     0.0,  0.05, None, 10.0, False, 0.0),
    "room":      (0.0,     0.0,  0.05, None, 10.0, False, 0.0),
}
REFINE_PRESET_DEFAULT = (0.0, 0.0, 0.05, None, 10.0, False, 0.0)
```

**Preset lookup:** use the `KitElement`'s name/type, matched case-insensitively
by checking if any key in `REFINE_PRESETS` appears as a substring of the element
name (e.g. `"Snare Top"` → `"snare"`, `"Floor Tom"` → `"tom"`). Fall back to
`REFINE_PRESET_DEFAULT` if no key matches.

---

## What the feature should do

### Trigger

Add an **"Auto-Refine"** button in the sidebar, or a menu item under File/Tools —
your call based on what fits the existing UI. Also consider offering it as an
optional step at the end of MIDI import ("Run auto-refine after import?" checkbox
in the import dialog).

### Behaviour

1. For each `KitElement` that has **both audio loaded and MIDI markers**:
   - Look up its preset from the element name (see above).
   - Call `refine_all()` with `element.audio` (mono float32), `element.sr`, and
     **`element.original_positions`** — the raw MIDI sample positions, NOT
     `final_positions`. Refinement must always start from the MIDI ground truth,
     not a previous manual edit.
   - Write the refined positions into `element.final_positions`.
   - Mark those markers' status as **`'pending'`** — not `'approved'`. The user
     still reviews each one in the normal workflow.
   - **Skip markers already marked `'manual'`** (explicitly edited by the user).
     Do not overwrite manual edits.

2. After all elements are processed, show a brief summary — toast notification or
   status bar update — for example:
   ```
   Auto-refine complete — Kick: 14 moved  Snare: 52 moved / 104 frozen
   ```
   The `r.pass_used` field (`'primary'`, `'wideband'`, `'frozen'`) gives the
   per-note breakdown for this summary.

3. Run the refinement in a **background thread** — it's fast (typically < 1 s per
   element for a full song) but should not block the UI. Show a simple progress
   indicator (label or progress bar) while running. Disable edit controls during
   processing; keep the window responsive.

### Non-destructive

`original_positions` is never touched. Refinement only writes to
`final_positions`. The user can always revert by re-importing the MIDI.

---

## Loading the audio

`refine_all()` needs the close-mic audio as a **mono float32 numpy array at the
correct sample rate**. The `KitElement` already stores the path to its audio
file. Load it with:

```python
import soundfile as sf
audio, sr = sf.read(str(element.audio_path), dtype='float32', always_2d=False)
if audio.ndim > 1:
    audio = audio[:, 0]   # take first channel if stereo
```

Pass the **full-length file** — `refine_all()` handles windowing internally per
note.

---

## Edge cases to handle

| Situation | Action |
|-----------|--------|
| No audio linked to element | Skip silently; show "skipped (no audio)" in summary |
| Stereo audio file | Take channel 0 (handled by snippet above) |
| SR mismatch | Not an issue — pass the actual file SR and `refine_all()` uses it correctly; it does not resample |
| Empty element (0 markers) | Skip |
| Exception from `refine_all()` | Catch, log, mark element as "failed" in summary; do not crash the app |
| All markers already `'manual'` | Skip element; note in summary |

---

## Files involved

| File | Action |
|------|--------|
| `onset.py` | **Read-only** — import from it, do not modify |
| `transient_snap.py` | Add the feature here |
| No new files needed | — |
