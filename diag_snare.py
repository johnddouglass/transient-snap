"""
Diagnostic: two snare issues
  A) Threshold sweep — find onset_threshold that minimises residual on
     DISTANT-detected moved notes (currently land ~1ms early vs reference)
  B) CLOSE-failure analysis — for notes that stay frozen at MIDI, attempt
     a restricted DISTANT walk within the Stage-1 peak frame only, using
     filtered audio, across multiple bands
"""
import sys, pathlib
import numpy as np
import soundfile as sf
import mido

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from onset import (
    INSTRUMENT_BANDS, CLOSE_DETECTION_SAMP, HOP_LENGTH, ONSET_N_FFT,
    OnsetResult, bandpass, refine_all, refine_position,
)
import librosa

PROJECT = pathlib.Path("/Users/johndouglass/Desktop/new song")
SR = 48000

# ── helpers ──────────────────────────────────────────────────────────────────

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

# ── load data ─────────────────────────────────────────────────────────────────

ref  = mido.MidiFile(str(PROJECT / "refined_markers.mid"))
orig = mido.MidiFile(str(PROJECT / "new song.mid"))
rn   = load_notes(ref,  get_tempo(ref))
on   = load_notes(orig, get_tempo(orig))

audio_raw, _ = sf.read(str(PROJECT / "Sn Top D_01.wav"), dtype='float32', always_2d=False)
if audio_raw.ndim > 1:
    audio_raw = audio_raw[:, 0]

pairs = match_notes(on["Snare MIDI"], rn["Snare MIDI"])
mo = np.array([p[0] for p in pairs], dtype=int)
mr = np.array([p[1] for p in pairs], dtype=int)
N  = len(pairs)

print(f"Snare notes: {N}")

# ── Part A: onset_threshold sweep ─────────────────────────────────────────────
print("\n" + "="*70)
print("PART A — onset_threshold sweep (snare DISTANT detection)")
print("Fixed params: low=150 high=1200 back=3ms fwd=10ms conf_min=2.0 min_shift=0.2")
print("="*70)

THRESHOLDS = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

print(f"\n{'thresh':>8}  {'n_moved':>7}  {'mean_err':>9}  {'|mean|':>7}  {'std':>6}  {'<=5samp':>8}  {'worse':>6}")
print("-" * 65)

audio_filt_150_1200 = bandpass(audio_raw, 150.0, 1200.0, SR)

baseline_results = refine_all(
    audio_raw, SR, mo,
    low_hz=150.0, high_hz=1200.0,
    search_back_ms=3.0, search_fwd_ms=10.0,
    onset_threshold=0.05, confidence_min=2.0,
    clamp_to_midi=False, min_shift_ms=0.2
)

for thr in THRESHOLDS:
    results = refine_all(
        audio_raw, SR, mo,
        low_hz=150.0, high_hz=1200.0,
        search_back_ms=3.0, search_fwd_ms=10.0,
        onset_threshold=thr, confidence_min=2.0,
        clamp_to_midi=False, min_shift_ms=0.2
    )
    refined  = np.array([r.refined for r in results], dtype=int)
    err_ms   = (refined - mr) / SR * 1000.0
    orig_ms  = (mo      - mr) / SR * 1000.0
    err_samp = np.abs(refined - mr)

    moved_mask = np.abs(refined - mo) >= int(0.2 / 1000 * SR)  # actually moved (>= min_shift)
    n_moved    = int(np.sum(moved_mask))
    n_within5  = int(np.sum(err_samp <= 5))
    n_worse    = int(np.sum(np.abs(err_ms) > np.abs(orig_ms)))

    print(f"  {thr:6.2f}  {n_moved:>7}  {np.mean(err_ms):>+9.3f}  "
          f"{np.mean(np.abs(err_ms)):>7.3f}  {np.std(err_ms):>6.3f}  "
          f"{n_within5:>8}  {n_worse:>6}")

# ── Part B: CLOSE-failure investigation ──────────────────────────────────────
print("\n" + "="*70)
print("PART B — CLOSE-failure analysis: restricted DISTANT walk within peak frame")
print("="*70)

# Identify notes that are frozen (stay at MIDI) but have large reference error
baseline_refined = np.array([r.refined for r in baseline_results], dtype=int)
baseline_err_samp = baseline_refined - mr  # positive = after ref, negative = before ref

# CLOSE failure = result ≈ MIDI but reference shows large positive offset needed
frozen_mask = np.abs(baseline_refined - mo) < int(0.2 / 1000 * SR)  # frozen
large_needed = (mr - mo) > 50  # reference is 50+ samples after MIDI

problem_idx = np.where(frozen_mask & large_needed)[0]
print(f"\nNotes frozen at MIDI but needing large positive correction: {len(problem_idx)}")
print(f"  (expected: idx 44, 68, 91, 121, 155 from prior trace)")

def stage1_peak(audio_raw, audio_filt, sr, position,
                back_ms=3.0, fwd_ms=10.0):
    """Return (peak_sample_abs, peak_distance, confidence, weighted_env_peak_frame)."""
    back_samp    = int(sr * back_ms / 1000)
    fwd_samp     = int(sr * fwd_ms / 1000)
    context_samp = HOP_LENGTH * 8
    seg_start    = max(0, position - back_samp - context_samp)
    seg_end      = min(len(audio_raw), position + fwd_samp)
    seg_len      = seg_end - seg_start

    S_raw = np.abs(librosa.stft(
        audio_raw[seg_start:seg_end],
        n_fft=ONSET_N_FFT, hop_length=HOP_LENGTH, center=False,
    ))
    onset_env = librosa.onset.onset_strength(
        S=S_raw, sr=sr, hop_length=HOP_LENGTH,
        aggregate=np.median, center=False,
    )
    seg_f  = audio_filt[seg_start:seg_end]
    n_fr   = min(len(onset_env), seg_len // HOP_LENGTH)
    amp_env = np.array([
        float(np.max(np.abs(seg_f[i*HOP_LENGTH: min((i+2)*HOP_LENGTH, seg_len)])))
        for i in range(n_fr)
    ], dtype=np.float64)

    win_start_samp  = (position - back_samp) - seg_start
    win_end_samp    = seg_end - seg_start
    win_frame_start = max(0, win_start_samp // HOP_LENGTH)
    win_frame_end   = min(n_fr, (win_end_samp + HOP_LENGTH - 1) // HOP_LENGTH)
    search_amp_max  = float(np.max(amp_env[win_frame_start:win_frame_end]))
    if search_amp_max > 1e-10:
        amp_env_norm = amp_env / search_amp_max
    else:
        amp_env_norm = amp_env
    weighted_env = onset_env[:n_fr] * (amp_env_norm ** 2)
    search_env   = weighted_env[win_frame_start:win_frame_end]
    peak_local   = int(np.argmax(search_env))
    peak_val     = float(search_env[peak_local])
    mean_val     = float(np.mean(search_env))
    peak_abs_fr  = win_frame_start + peak_local
    peak_sample  = seg_start + peak_abs_fr * HOP_LENGTH
    return peak_sample, peak_sample - position, peak_val / (mean_val + 1e-10)


def restricted_distant_walk(audio_raw, audio_filt, sr, position, peak_sample,
                             back_samp, onset_threshold):
    """
    Walk back from (peak_sample + HOP) through filtered audio, but only
    within [peak_sample - HOP, peak_sample + HOP] (2 HOP window centred
    on peak).  Returns onset_sample.
    """
    walk_lo  = max(0, peak_sample - HOP_LENGTH)
    wb_start = min(peak_sample + HOP_LENGTH, len(audio_filt) - 1)
    amp_end  = min(len(audio_filt), wb_start + HOP_LENGTH)
    peak_amp = float(np.max(np.abs(audio_filt[wb_start:amp_end])))
    onset_sample = peak_sample  # fallback
    if peak_amp > 1e-10:
        threshold = peak_amp * onset_threshold
        ENV_WIN = 32
        for i in range(wb_start - 1, walk_lo - 1, -1):
            win_lo    = max(0, i - ENV_WIN)
            local_max = float(np.max(np.abs(audio_filt[win_lo:i + 1])))
            if local_max < threshold:
                onset_sample = i + 1
                break
        # else: fallback = peak_sample (start of the onset frame)
    return onset_sample


BANDS_TO_TRY = [
    (150.0, 1200.0, "150-1200 Hz"),
    (200.0,  800.0, "200-800 Hz"),
    (300.0, 1500.0, "300-1500 Hz"),
    (600.0, 2400.0, "600-2400 Hz"),
    (1200.0, 4000.0, "1200-4000 Hz"),
]

for idx in problem_idx:
    pos = int(mo[idx])
    ref_pos = int(mr[idx])
    orig_err_samp = pos - ref_pos
    orig_err_ms   = orig_err_samp / SR * 1000
    print(f"\n  idx={idx}  orig_err={orig_err_ms:+.2f}ms  ref_offset={ref_pos-pos:+d}samp")

    for low_hz, high_hz, band_label in BANDS_TO_TRY:
        af = bandpass(audio_raw, low_hz, high_hz, SR)
        peak_samp, peak_dist, conf = stage1_peak(audio_raw, af, SR, pos)
        path = "CLOSE" if peak_dist <= CLOSE_DETECTION_SAMP else "DISTANT"
        # Try restricted walk for THIS band
        back_samp = int(3.0 / 1000 * SR)
        onset = restricted_distant_walk(
            audio_raw, af, SR, pos, peak_samp, back_samp, onset_threshold=0.05
        )
        result_err_samp = onset - ref_pos
        result_err_ms   = result_err_samp / SR * 1000
        print(f"    {band_label:<20}  peak={peak_dist:+d}samp path={path}  conf={conf:.2f}  "
              f"restricted_walk→onset={onset-pos:+d}samp  err={result_err_ms:+.2f}ms vs ref")

# ── Part C: search_fwd_ms sweep for idx 155 ──────────────────────────────────
print("\n" + "="*70)
print("PART C — search_fwd_ms sweep for idx 155 (onset outside 10ms window)")
print("="*70)

# Find idx 155 (large negative orig_err, currently frozen)
idx155_candidates = np.where((mr - mo) > 400)[0]  # needs >400 samples forward
print(f"\nNotes needing >400 sample forward correction: {idx155_candidates}")

for idx in idx155_candidates:
    pos     = int(mo[idx])
    ref_pos = int(mr[idx])
    orig_err_ms = (pos - ref_pos) / SR * 1000
    print(f"\n  idx={idx}  orig_err={orig_err_ms:+.2f}ms  needed={ref_pos-pos:+d}samp ({(ref_pos-pos)/SR*1000:.2f}ms)")
    print(f"  {'fwd_ms':>8}  {'peak_dist':>10}  {'conf':>6}  {'result':>10}  {'err_ms':>8}")
    for fwd_ms in [10.0, 12.0, 15.0, 20.0]:
        results_single = refine_all(
            audio_raw, SR, np.array([pos]), low_hz=150.0, high_hz=1200.0,
            search_back_ms=3.0, search_fwd_ms=fwd_ms,
            onset_threshold=0.05, confidence_min=2.0,
            clamp_to_midi=False, min_shift_ms=0.2
        )
        r = results_single[0]
        err_ms = (r.refined - ref_pos) / SR * 1000
        print(f"  {fwd_ms:>8.1f}  shift={r.offset_ms:+.2f}ms  conf={r.confidence:.2f}  "
              f"result={r.offset_samples:+d}samp  err={err_ms:+.2f}ms [{r.pass_used}]")

print("\nDone.")
