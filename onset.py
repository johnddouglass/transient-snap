"""
Band-limited spectral flux onset detection for drum recordings.

Two-stage refinement strategy:

  Stage 1 — Frame-level onset detection (HOP_LENGTH resolution, ~1.3 ms):

    The onset frame is located by multiplying two complementary signals:

      (a) Spectral flux of the FULL-BANDWIDTH (raw) audio computed via
          a direct STFT with n_fft = 2 × HOP_LENGTH (128 samples = 2.67 ms
          window at 48 kHz).

          Raw audio must be used here — not the filtered signal — because at
          n_fft=128 / sr=48 kHz the FFT frequency resolution is 375 Hz/bin.
          Low-frequency bands (kick 40–180 Hz, snare 150–1200 Hz) span fewer
          than 4 of the 65 available bins.  aggregate=median of 65 bins where
          < 4 have content returns 0 for every frame, giving a completely flat
          onset envelope.  Raw audio distributes energy across all 65 bins and
          produces a valid, sharp onset envelope.

          Small n_fft: the default n_fft=2048 creates 97% frame overlap,
          smearing every onset across 30+ frames.  With n_fft=128 the
          spectral change is localised to the actual transient.

          STFT passed as S= (not y=): passing raw audio via y= triggers
          librosa's internal mel filterbank, which at n_fft=128 / sr=48 kHz
          also produces a flat envelope (same bin-sparsity reason).

      (b) Per-frame amplitude of the BAND-FILTERED signal, normalised by
          the search-window peak and then CUBED.

          Cubing amp_env_norm gives much more aggressive bleed suppression
          than linear weighting while keeping the target onset at weight 1.0:

            Frame at 40% of window peak: linear = 0.40, cubed = 0.064

          A cymbal at the MIDI position can have 5–8× the raw spectral flux
          of the snare that follows, while its energy in the snare band
          (150–1200 Hz) is only 30–40% of the snare's.  With cubed weighting
          even the worst case (8× flux, 40% amp) loses:
            8 × 0.064 = 0.51  <  1.0 × 1.0 = 1.0  → snare wins ✓

          Normalising by the window peak means ghost notes are unaffected —
          the ghost note IS the window maximum → its weight is 1.0³ = 1.0.

  Stage 2 — Sample-accurate onset placement:

    Two strategies based on how far Stage 1 found the onset from the MIDI
    note position (not from the backward-search boundary):

    Close detection (peak_sample − position ≤ 2.67 ms):
      Envelope walk-back on RAW audio from (peak_sample + HOP_LENGTH).
      At each step, tests the backward-looking max over ENV_WIN = 32
      samples against onset_threshold × local peak.  The envelope max
      (rather than individual sample values) prevents spurious threshold
      crossings from zero-crossings in sustained bleed signals (common on
      tom close-mics with kick/snare bleed).  When the envelope drops below
      threshold the onset edge is at i + 1.

      Three fallback rules:
        • Walk exhausted (bleed above threshold throughout):
            onset_sample = min(peak_sample, position).  For toms, whose LF
            onset builds 2–3 ms after the stick click, this stays at MIDI.
        • Silent at wb_start:
            onset_sample = min(peak_sample, position).
        • Post-walk clamp:
            onset_sample = max(onset_sample, position).  Prevents close
            detection from pushing a note earlier than MIDI.  When Stage 1
            detects a brief pre-hit contact (e.g. snare stick-scrape) the
            walk would otherwise anchor to that contact instead of the main
            attack.  Genuine large early offsets (> 2.67 ms) exit the close
            window and are handled by distant detection.

    Distant detection (peak_sample − position > 2.67 ms):
      Threshold walk-back on FILTERED audio from (peak_sample + HOP_LENGTH).
      Stage 1 found the onset well after MIDI (e.g. cymbal-before-snare).
      Filtered audio is used here because raw audio's wideband cymbal tail
      would give a false early crossing.  The +HOP start compensates for the
      one-frame-early detection bias — onset_strength peaks at the frame
      whose window contains the transient; the actual onset may arrive up to
      HOP_LENGTH later.

    Why measure from position, not walk_limit:
      With search_back_ms = 3 ms, walk_limit is 144 samples before position.
      A tom onset 2.33 ms after MIDI is 112 samples past position but 256
      samples past walk_limit — incorrectly classified as "distant" if
      measured from walk_limit.  Measuring from position correctly routes all
      near-MIDI onsets through the close path.

  Back-window guidance:
    Most MIDI drum notes fire at or slightly before the acoustic onset.
    The default search_back_ms=0 works well for kick, snare, and toms.

    Ghost notes and notes captured with trigger delay may need backward
    search (e.g. search_back_ms=5.0) so the walk-back has room to find
    the true onset edge before the MIDI position.

Public API:
    INSTRUMENT_BANDS   — preset (low_hz, high_hz) pairs by instrument name
    bandpass()         — 4th-order Butterworth filter
    refine_position()  — refine a single MIDI note position
    refine_all()       — batch-refine an array of positions
    OnsetResult        — dataclass returned by both refine functions
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt


# Samples per hop for the onset strength computation.
# 64 samples ≈ 1.3 ms at 48 kHz; fine enough for frame-level onset location.
HOP_LENGTH: int = 64

# STFT window for onset_strength.
# MUST be small (= 2 × HOP_LENGTH) to avoid the 97%-overlap smearing
# artefact that occurs with librosa's default n_fft=2048.  With n_fft=128
# consecutive frames share only 50% of their audio content, giving the
# spectral-flux peak sharp temporal localisation at the drum transient.
ONSET_N_FFT: int = HOP_LENGTH * 2   # 128 samples = 2.67 ms at 48 kHz

# Stage 2 split threshold (samples).
# If Stage 1 found the onset peak within this many samples of the search
# start (walk_limit), the detection is "close" and we refine with a raw-audio
# argmin over a small region.  If the peak is further away (e.g. MIDI was
# placed at a cymbal hit 8 ms before the actual snare onset), we use
# threshold walk-back from the detected frame to find the precise onset edge.
CLOSE_DETECTION_SAMP: int = HOP_LENGTH * 2  # 128 samples ≈ 2.67 ms

# Instrument frequency bands for bandpass filtering.
# (low_hz, high_hz): set both to 0.0 for wideband (no filter applied).
# These are starting points — real drum tunings vary, and custom bands can
# be supplied directly to refine_all() / refine_position().
INSTRUMENT_BANDS: dict[str, tuple] = {
    # Kick: 40–180 Hz.  Upper cutoff at 180 Hz (not 250 Hz) removes snare-bleed
    # overtones in the 180–250 Hz range that sit above the kick fundamental and
    # body while contributing nothing to onset detection accuracy.  Empirically
    # confirmed on test data: 40–180 Hz reduces |mean_err| vs 40–250 Hz and
    # drops WORSE-case count from 2 to 1 on a 265-note reference set.
    "kick":     (40.0,   180.0),
    "snare":    (150.0, 1200.0),
    "hihat":   (7000.0, 18000.0),
    "ride":    (5000.0, 16000.0),
    "crash":   (3000.0, 16000.0),
    "tom":      (80.0,   600.0),
    "overhead":  (0.0,    0.0),   # wideband — no filter
    "room":      (0.0,    0.0),   # wideband — no filter
}


@dataclass
class OnsetResult:
    """Result of refining a single drum hit position.

    Attributes:
        original:       Original sample position supplied as input.
        refined:        Refined sample position after onset detection.
        offset_samples: refined - original (negative = moved earlier).
        offset_ms:      Offset expressed in milliseconds.
        confidence:     Weighted onset peak-to-mean ratio within the search
                        window.  Higher values indicate a more prominent,
                        well-isolated onset.  Values below ~3.0 often
                        indicate bleed competition and warrant review.
        pass_used:      Which detection pass produced this result:
                        'primary'  — band-filtered pass (normal).
                        'wideband' — low-confidence retry using raw audio.
                        'frozen'   — confidence too low even wideband; note
                                     left at original MIDI position.
    """
    original: int
    refined: int
    offset_samples: int
    offset_ms: float
    confidence: float
    pass_used: str = "primary"


# ── Filtering ────────────────────────────────────────────────────────────────

def bandpass(audio: np.ndarray, low_hz: float, high_hz: float, sr: int) -> np.ndarray:
    """Apply a 4th-order Butterworth bandpass filter using zero-phase filtering.

    Zero-phase filtering (sosfiltfilt) is used instead of causal filtering
    (sosfilt) for two reasons critical to onset detection accuracy:

      1. No group delay.  A causal 4th-order Butterworth bandpass at 40 Hz has
         a time constant of ~4 ms, so the filtered amplitude reaches its peak
         several milliseconds after the acoustic onset.  This causes the onset
         frame's amplitude to appear much smaller than later frames, severely
         biasing the weighted_env argmax away from the correct onset.  With
         zero-phase filtering the filtered amplitude peaks at the true onset
         time, so the onset frame wins the argmax correctly.

      2. Accurate Stage 2 threshold walk-back.  When Stage 2 walks backward
         through the filtered signal to find the onset edge, a causal filter's
         delayed response causes false late threshold crossings.  Zero-phase
         output is centred on the actual onset, so threshold crossings are
         at the correct onset boundary.

    This is an offline batch process, so zero-phase (non-causal) filtering
    is appropriate.  sosfiltfilt runs the filter forward and backward,
    doubling compute cost but producing an output with the correct temporal
    alignment and no passband ripple amplification.

    Edge cases:
      - low_hz <= 0           → lowpass at high_hz
      - high_hz <= 0 or >= Nyquist → highpass at low_hz
      - both <= 0             → return audio unchanged (wideband)
    """
    if low_hz <= 0.0 and high_hz <= 0.0:
        return audio

    nyq = sr / 2.0

    if low_hz <= 0.0:
        sos = butter(4, high_hz / nyq, btype="low", output="sos")
    elif high_hz <= 0.0 or high_hz >= nyq:
        sos = butter(4, low_hz / nyq, btype="high", output="sos")
    else:
        sos = butter(4, [low_hz / nyq, high_hz / nyq], btype="band", output="sos")

    return sosfiltfilt(sos, audio).astype(audio.dtype)


# ── Core refinement ──────────────────────────────────────────────────────────

def refine_position(
    audio_raw: np.ndarray,
    audio_filtered: np.ndarray,
    sr: int,
    position: int,
    search_back_ms: float = 0.0,
    search_fwd_ms: float = 10.0,
    onset_threshold: float = 0.05,
    onset_threshold_distant: float | None = None,
    clamp_to_midi: bool = False,
    _pass_label: str = "primary",
) -> OnsetResult:
    """Refine a single drum hit position using weighted onset detection.

    Stage 1 (frame-level): spectral flux of RAW audio (via direct STFT,
    bypassing the mel filterbank) × normalised per-frame filtered amplitude³.
    Raw audio is required because at n_fft=128 / sr=48 kHz the frequency
    resolution is 375 Hz/bin, making filtered-signal STFTs useless for
    low-frequency bands.  Cubing the amplitude weight aggressively suppresses
    bleed even when the bleed's raw spectral flux is 5–8× the target onset.

    Stage 2 (sample-level): find the amplitude minimum of the RAW audio
    between the walk-back limit and the onset frame (argmin).  Raw audio
    captures the initial wideband beater click, which arrives 0.2–0.8 ms
    earlier than the filtered-signal valley for kick.  For wide backward
    windows (ghost notes), threshold walk-back on the filtered signal is
    used instead, as argmin can roam into long pre-onset silences.

    If walk-back exhausts the search window without a crossing, onset_sample
    stays at peak_sample (no clamping to the window edge).

    Args:
        audio_raw:                Full-bandwidth audio.
        audio_filtered:           Bandpass-filtered audio (call bandpass() first).
                                  May equal audio_raw when no filter is applied.
        sr:                       Sample rate in Hz.
        position:                 Approximate sample position from MIDI.
        search_back_ms:           How far before position to search (ms).
                                  Default 0 suits most drums where MIDI fires at
                                  or before the acoustic onset.  Use 3–5 ms for
                                  ghost notes or sessions with consistent trigger
                                  delay.
        search_fwd_ms:            How far after position to search (ms).
        onset_threshold:          Fraction of peak amplitude defining the onset
                                  edge for the CLOSE Stage 2 walk-back
                                  (0.01–0.30).  Applied when Stage 1 found the
                                  onset within CLOSE_DETECTION_SAMP of the MIDI
                                  position.  Default 0.05.
        onset_threshold_distant:  Separate threshold for the DISTANT Stage 2
                                  walk-back (Stage 1 peak more than
                                  CLOSE_DETECTION_SAMP from MIDI).  When None
                                  (default), falls back to onset_threshold.
                                  Setting this higher than onset_threshold
                                  causes DISTANT walk-back to stop closer to the
                                  onset peak — useful for notes where MIDI was
                                  placed at a cymbal hit 5–15 ms before the
                                  snare stroke, where the default 5 % threshold
                                  latches onto the inter-onset gap instead of
                                  the snare onset edge.  Empirically, 0.15
                                  reduces residual error on such notes without
                                  affecting notes handled by the close path.
        clamp_to_midi:            When True, the refined position is clamped to
                                  ``position`` if detection would move the note
                                  *later* than the MIDI event.  Use for
                                  instruments where the MIDI note is placed at
                                  the physical impact (e.g. toms): the
                                  bandpass-filtered LF body builds 2–5 ms after
                                  the stick click, and the threshold walk-back
                                  can latch onto the start of that gradual rise
                                  rather than the click itself.  Leave False
                                  (default) for instruments where MIDI may be
                                  placed before the actual acoustic onset (e.g.
                                  snare notes placed at a cymbal hit that
                                  precedes the stroke by 5–15 ms).
        _pass_label:              Internal label stored in OnsetResult.pass_used.

    Returns:
        OnsetResult for this position.
    """
    # Resolve the distant threshold: fall back to onset_threshold when not set.
    if onset_threshold_distant is None:
        onset_threshold_distant = onset_threshold
    back_samp = int(sr * search_back_ms / 1000.0)
    fwd_samp  = int(sr * search_fwd_ms  / 1000.0)

    # Pre-context gives onset_strength stable spectral history before the
    # search window starts; avoids artefacts in the first few frames.
    context_samp = HOP_LENGTH * 8

    seg_start = max(0, position - back_samp - context_samp)
    seg_end   = min(len(audio_raw), position + fwd_samp)
    seg_len   = seg_end - seg_start

    if seg_len <= 0:
        return OnsetResult(position, position, 0, 0.0, 0.0)

    # Stage 1a: onset_strength on the RAW (wideband) audio using a direct
    # STFT (S=) to bypass the mel filterbank.
    #
    # Why raw, not filtered:
    #   With n_fft = 128 at sr = 48 kHz the FFT frequency resolution is
    #   48000 / 128 = 375 Hz / bin.  Low-frequency bands (kick 40–180 Hz,
    #   snare 150–1200 Hz) span < 4 bins out of 65.  aggregate=median of 65
    #   bins where < 4 have content returns 0 for every frame — the same
    #   pathological flat envelope as the mel-filterbank bug.  Raw audio
    #   has energy across all 65 bins and produces a valid onset envelope.
    #
    # Why S= not y=:
    #   Passing y= triggers librosa's mel filterbank, which at n_fft=128 /
    #   sr=48 kHz leaves most mel channels empty → aggregate=median = 0 for
    #   every frame.  Pre-computing S = |STFT(raw)| and passing S= gives raw
    #   per-bin spectral flux, which is sharp and correct at this hop length.
    #
    # center=False: frame i covers samples [seg_start + i*HOP, ...], giving
    # a predictable sample-to-frame mapping with no edge padding.
    S_raw = np.abs(librosa.stft(
        audio_raw[seg_start:seg_end],
        n_fft=ONSET_N_FFT,
        hop_length=HOP_LENGTH,
        center=False,
    ))
    onset_env = librosa.onset.onset_strength(
        S=S_raw,
        sr=sr,
        hop_length=HOP_LENGTH,
        aggregate=np.median,
        center=False,
    )

    if len(onset_env) == 0:
        return OnsetResult(position, position, 0, 0.0, 0.0)

    # Stage 1b: per-frame amplitude of the band-filtered signal.
    # Using a 2-hop window per frame to capture transients that straddle
    # frame boundaries.
    seg_f   = audio_filtered[seg_start:seg_end]
    n_fr    = min(len(onset_env), seg_len // HOP_LENGTH)
    amp_env = np.array([
        float(np.max(np.abs(
            seg_f[i * HOP_LENGTH : min((i + 2) * HOP_LENGTH, seg_len)]
        )))
        for i in range(n_fr)
    ], dtype=np.float64)

    # Map search window to frame indices (needed before weighting).
    win_start_samp  = (position - back_samp) - seg_start
    win_end_samp    = seg_end - seg_start

    win_frame_start = max(0, win_start_samp // HOP_LENGTH)
    win_frame_end   = min(n_fr, (win_end_samp + HOP_LENGTH - 1) // HOP_LENGTH)

    if win_frame_start >= win_frame_end:
        return OnsetResult(position, position, 0, 0.0, 0.0)

    # Normalise amp_env by the search-window peak so the weight is always a
    # relative bleed discriminator (0–1), not an absolute amplitude gate.
    # Without this, ghost notes (low absolute amplitude) produce a near-zero
    # weighted_env across the entire window and onset detection fails entirely.
    search_amp_max = float(np.max(amp_env[win_frame_start:win_frame_end]))
    if search_amp_max > 1e-10:
        amp_env_norm = amp_env / search_amp_max
    else:
        amp_env_norm = amp_env  # all-zero window; fallback to onset_env alone

    # Weighted envelope: spectral flux × normalised filtered amplitude².
    #
    # Squaring amp_env_norm gives stronger bleed suppression than linear
    # weighting while keeping the onset frame's weight near 1.0 (because
    # zero-phase filtering now correctly aligns the filtered amplitude peak
    # with the acoustic onset, not several ms later).
    #
    #   Frame at 40% of window peak: linear = 0.40, squared = 0.16
    #
    # Why this matters for snare:
    #   A crash or hi-hat at the MIDI position can have 5–8× the raw spectral
    #   flux of the snare onset that follows it, while its energy in the snare
    #   band (150–1200 Hz) is only ~35% of the snare's amplitude.
    #   With squared weighting the snare almost always wins:
    #     Cymbal (6× onset flux, 35% filtered amp): 6 × 0.123 = 0.74 < 1.0 ✓
    #
    # Why NOT cubed: with zero-phase filtering, the onset frame already has
    # amp_norm ≈ 0.7–0.9 (filter aligned to onset, not delayed).  Cubing would
    # give 0.8³ = 0.51, unnecessarily suppressing legitimate onset frames.
    #
    # Ghost notes: the ghost note IS the window maximum → amp_norm = 1.0 →
    # weight 1.0² = 1.0 — unaffected by the exponent choice.
    weighted_env = onset_env[:n_fr] * (amp_env_norm ** 2)

    search_env  = weighted_env[win_frame_start:win_frame_end]
    peak_local  = int(np.argmax(search_env))
    peak_val    = float(search_env[peak_local])
    mean_val    = float(np.mean(search_env))
    confidence  = peak_val / (mean_val + 1e-10)

    # Absolute sample at the start of the peak onset frame.
    peak_abs_frame = win_frame_start + peak_local
    peak_sample    = min(
        seg_start + peak_abs_frame * HOP_LENGTH,
        len(audio_filtered) - 1,
    )

    # Stage 2: sample-accurate onset placement.
    #
    # The strategy depends on how far Stage 1's detected onset frame is from
    # the MIDI note position:
    #
    #   Close detection  (peak_sample - position ≤ CLOSE_DETECTION_SAMP):
    #     The onset was found near the MIDI note.  Walk back through the RAW
    #     wideband signal from (peak_sample + HOP_LENGTH) until amplitude drops
    #     below onset_threshold × local peak.  This finds the last quiet sample
    #     immediately before the initial beater/stick click.
    #     Raw audio is used so the sharp wideband transient is captured rather
    #     than the slower band-limited onset build-up.
    #
    #     If the walk-back exhausts (pre-onset bleed keeps the signal above
    #     threshold throughout), the note is not moved later than the original
    #     MIDI position — onset_sample = min(peak_sample, position).  This
    #     prevents tom hits (whose LF onset builds 2–3 ms after the stick click)
    #     from being pushed later when the close-mic has residual kick/snare
    #     bleed that obscures the onset edge.
    #
    #   Distant detection  (peak_sample - position > CLOSE_DETECTION_SAMP):
    #     Stage 1 found the onset well after the MIDI note.  This happens when
    #     MIDI was placed early (e.g. at a cymbal hit 5–15 ms before the actual
    #     snare stroke).  Threshold walk-back on the FILTERED signal from
    #     (peak_sample + HOP_LENGTH) finds the last quiet sample before the new
    #     attack.  The +HOP_LENGTH shift compensates for the one-frame-early
    #     detection bias (onset_strength peaks at the frame whose STFT window
    #     contains the transient, which may not arrive until the end of the
    #     window).
    #
    # CLOSE_DETECTION_SAMP = 2 × HOP_LENGTH ≈ 2.67 ms.
    #
    # Why measure from position, not walk_limit:
    #   walk_limit = position - back_samp.  With search_back_ms = 3 ms,
    #   walk_limit is 144 samples before position.  A tom onset 2.33 ms after
    #   the MIDI note is 112 samples past position but 256 samples past
    #   walk_limit — incorrectly classified as "distant" under the old rule,
    #   routing it to filtered walk-back which fails due to LF onset delay.
    #   Measuring from position correctly classifies such hits as "close".
    walk_limit   = max(0, position - back_samp)
    onset_sample = position             # fallback: stay at MIDI if detection skipped

    if peak_sample > walk_limit:
        peak_distance = peak_sample - position   # distance from MIDI note

        if peak_distance <= CLOSE_DETECTION_SAMP:
            # ── Close detection: raw-audio envelope walk-back ─────────────────
            # Stage 1 found the onset within ~2.67 ms of the MIDI note.
            #
            # Walk backward from (peak_sample + HOP_LENGTH) through the RAW
            # wideband signal.  At each step, test the backward-looking max
            # (ENV_WIN samples) against onset_threshold × local peak.  When
            # the max drops below threshold, the onset edge is at i + 1.
            #
            # Envelope max, not individual samples:
            #   Raw audio oscillates through zero even when the sustained
            #   envelope is high (e.g. kick/snare bleed on a tom close-mic).
            #   Individual-sample comparisons find spurious zero-crossings
            #   inside the bleed region.  The backward-window max (ENV_WIN =
            #   32 samp = 0.67 ms) correctly represents the signal level and
            #   stays above threshold throughout sustained bleed.
            #
            # Walk-back exhaustion fallback:
            #   When bleed keeps the level above threshold throughout the
            #   window, the loop exhausts → onset_sample = min(peak_sample,
            #   position).  For toms (peak > position) this gives position.
            #
            # Pre-MIDI clamp (applied after walk-back):
            #   If the walk-back finds an onset before MIDI (e.g. snare stick
            #   contact 1–2 ms before the main crack), the acoustic event
            #   found is not the reference point.  We clamp onset_sample to
            #   max(onset_sample, position) so the note is never moved earlier
            #   than its original MIDI position by close detection.
            #   This is safe because:
            #     • Genuine early onsets within the close window (≤ 2.67 ms
            #       before MIDI) are rare and small enough that staying at
            #       MIDI costs ≤ 2.67 ms — less than the tom error we're
            #       fixing.
            #     • Large genuine offsets (> 2.67 ms) exit the close window
            #       and are handled by distant detection, which can move
            #       notes in either direction.
            wb_start = min(peak_sample + HOP_LENGTH, len(audio_raw) - 1)
            amp_end  = min(len(audio_raw), wb_start + HOP_LENGTH)
            peak_amp = float(np.max(np.abs(audio_raw[wb_start:amp_end])))
            if peak_amp > 1e-10:
                threshold = peak_amp * onset_threshold
                ENV_WIN = 32  # backward-looking envelope window (samples)
                for i in range(wb_start - 1, walk_limit - 1, -1):
                    win_lo    = max(0, i - ENV_WIN)
                    local_max = float(np.max(np.abs(audio_raw[win_lo:i + 1])))
                    if local_max < threshold:
                        onset_sample = i + 1
                        break
                else:
                    # Walk exhausted — bleed throughout, or walk_limit reached.
                    # Don't move the note later than MIDI.
                    onset_sample = min(peak_sample, position)
            else:
                # Silent region at wb_start — no transient content.
                onset_sample = min(peak_sample, position)

            # Clamp: never move the note from MIDI in close detection.
            #
            # Later clamp (clamp_to_midi only):
            #   When clamp_to_midi is set the walk-back must not move the
            #   note *after* MIDI.  For toms, Stage 1 often finds the LF
            #   body peak several ms after the stick click; Stage 2 then
            #   walks back to the quiet gap between the decayed click and
            #   the rising LF body (1–2 ms after MIDI).  Clamping here
            #   prevents that spurious late placement.
            if clamp_to_midi:
                onset_sample = min(onset_sample, position)

            # Earlier clamp (always):
            #   Prevents moving the note before MIDI.  The walk can find
            #   the acoustic onset before the main transient (e.g. stick
            #   contact 1–2 ms before the snare crack), which is not the
            #   intended reference point for drum quantisation.
            onset_sample = max(onset_sample, position)

        else:
            # ── Distant detection: envelope walk-back on filtered signal ─────
            # Stage 1 found the onset more than 2.67 ms past the MIDI note.
            # This happens when MIDI was placed early (e.g. at a cymbal hit
            # several ms before the actual snare stroke).  Threshold walk-back
            # on the FILTERED signal from (peak_sample + HOP_LENGTH) finds the
            # last quiet sample before the new attack.
            #
            # ENV_WIN = 32 backward-looking envelope window (same as close
            # detection).  This prevents individual sample comparisons from
            # triggering on zero-crossings of the bandpass-filtered signal,
            # which is particularly problematic for tom drums whose LF onset
            # (80–600 Hz) builds gradually over 3–5 ms rather than starting
            # with a sharp edge.
            #
            # Exhaustion fallback:
            #   If the walk reaches walk_limit without finding below-threshold
            #   envelope, the onset edge is obscured by bleed throughout (as
            #   on a rack-tom close-mic where kick/snare bleed keeps the
            #   filtered envelope continuously above 5% of peak).  In that
            #   case onset_sample = min(peak_sample, position) — the note is
            #   not moved forward past MIDI, preserving the original timing.
            #
            # Why FILTERED audio (not raw):
            #   Filtered audio suppresses out-of-band bleed, helping to
            #   distinguish the onset band of the target instrument from
            #   adjacent instrument bleed.  For example, on the snare mic,
            #   a crash cymbal (mostly HF) has a much smaller envelope in
            #   150–1200 Hz than the snare onset — the filtered walk can find
            #   the quiet transition between the cymbal and the snare attack.
            wb_start = min(peak_sample + HOP_LENGTH, len(audio_filtered) - 1)
            amp_end  = min(len(audio_filtered), wb_start + HOP_LENGTH)
            peak_amp = float(np.max(np.abs(audio_filtered[wb_start:amp_end])))
            if peak_amp > 1e-10:
                threshold = peak_amp * onset_threshold_distant
                ENV_WIN   = 32  # same envelope window as close detection
                for i in range(wb_start - 1, walk_limit - 1, -1):
                    win_lo    = max(0, i - ENV_WIN)
                    local_max = float(np.max(np.abs(audio_filtered[win_lo:i + 1])))
                    if local_max < threshold:
                        onset_sample = i + 1
                        break
                else:
                    # Walk exhausted — filtered onset edge not found.
                    # The onset is likely obscured by bleed or the LF energy
                    # builds gradually (e.g. rack tom), making the individual-
                    # sample threshold crossing unreliable.  Don't push the
                    # note to a spurious position; stay at MIDI.
                    onset_sample = min(peak_sample, position)

    # clamp_to_midi: override walk-back result and stay at MIDI.
    #
    # When clamp_to_midi is True (toms), the MIDI note was placed at the
    # physical stick impact, which is the correct reference point.  The
    # bandpass-filtered LF onset builds 2–5 ms AFTER the stick click, so:
    #
    #   Late placement (onset > MIDI):
    #     DISTANT walk finds the start of the gradual LF rise, which is
    #     still after MIDI.  This incorrectly moves the note to the LF
    #     onset rather than the click.
    #
    #   Early placement (onset < MIDI):
    #     DISTANT walk overshoots past MIDI into the pre-onset bleed
    #     region.  The threshold (5 % of the LF body peak) can be as low
    #     as 0.0017, comparable to the noise floor of kick/snare bleed on
    #     the rack-tom mic.  The walk finds a spurious "quiet" transition
    #     in the slowly-decaying bleed 1–3 ms before MIDI.
    #
    # In both cases the walk-back result is unreliable.  The safest outcome
    # is onset_sample = position (no movement).  This is confirmed
    # empirically: the T2 instrument yields 0 BETTER and 3 WORSE when early
    # movement is permitted, showing no early move is genuinely helpful.
    #
    # This clamp is NOT applied to instruments such as snare where MIDI may
    # be placed at a pre-cursor event (cymbal hit 5–15 ms before the stroke)
    # and the correct refined position IS after MIDI.
    if clamp_to_midi:
        onset_sample = position

    offset_samples = onset_sample - position
    offset_ms      = float(offset_samples) / sr * 1000.0

    return OnsetResult(
        original=position,
        refined=onset_sample,
        offset_samples=offset_samples,
        offset_ms=offset_ms,
        confidence=confidence,
        pass_used=_pass_label,
    )


def refine_all(
    audio: np.ndarray,
    sr: int,
    positions: np.ndarray,
    low_hz: float,
    high_hz: float,
    search_back_ms: float = 0.0,
    search_fwd_ms: float = 10.0,
    onset_threshold: float = 0.05,
    onset_threshold_distant: float | None = None,
    confidence_min: float = 2.0,
    clamp_to_midi: bool = False,
    min_shift_ms: float = 0.0,
) -> list:
    """Refine a batch of drum hit positions with weighted onset detection.

    Three-outcome strategy per note:

      1. Primary pass — band-filtered amplitude weighting.  If confidence ≥
         confidence_min the result is accepted.

      2. Wideband retry — if the primary pass is below confidence_min, a
         second pass runs with audio_filtered = audio_raw (no band filter).
         The raw wideband transient (beater click, stick attack) is often
         much more prominent than the band-limited version and will yield
         higher confidence.  If the retry confidence exceeds the primary
         confidence, the retry result is used.

      3. Freeze — if neither pass reaches confidence_min, the note is left
         at its original MIDI position (offset = 0, pass_used = 'frozen').
         Moving a note with no confident onset detection risks making the
         timing worse, so the conservative choice is to not touch it.

    The wideband retry still uses audio_raw for Stage 2 argmin (same as the
    primary pass), so the sample-accurate placement is identical in character.

    Applies the bandpass filter once across the full audio array (not
    per-segment) to avoid Butterworth edge-ringing at every hit boundary.

    Args:
        audio:                    Raw (unfiltered) audio array.
        sr:                       Sample rate in Hz.
        positions:                1-D array of approximate sample positions from
                                  MIDI.
        low_hz:                   Lower bandpass cutoff (Hz).  0 to skip.
        high_hz:                  Upper bandpass cutoff (Hz).  0 to skip.
        search_back_ms:           Backward search window per note (ms).
        search_fwd_ms:            Forward search window per note (ms).
        onset_threshold:          Amplitude fraction for the CLOSE Stage 2
                                  walk-back (Stage 1 peak within
                                  CLOSE_DETECTION_SAMP of MIDI position).
                                  Default 0.05.
        onset_threshold_distant:  Separate threshold for the DISTANT Stage 2
                                  walk-back.  When None (default), falls back to
                                  onset_threshold so the behaviour is identical
                                  to the single-threshold case.  Setting this
                                  higher (e.g. 0.15) reduces the systematic
                                  ~1 ms residual on notes where MIDI was placed
                                  at a cymbal hit well before the actual snare
                                  stroke, without affecting the more numerous
                                  close-detected notes.  See refine_position for
                                  full details.
        confidence_min:           Peak-to-mean threshold below which a wideband
                                  retry is attempted.  Notes where even the retry
                                  falls below this value are frozen at their
                                  original position.  Default 2.0.  Set to 0.0
                                  to disable retry/freeze (original single-pass
                                  behaviour).
        clamp_to_midi:            Forwarded to refine_position.  Set True for
                                  instruments (e.g. toms) where MIDI is placed
                                  at the physical impact and the refined position
                                  must not exceed the original MIDI position.
        min_shift_ms:             Minimum absolute shift (ms) required to apply
                                  the detection result.  Moves smaller than this
                                  are discarded and the note is left at its
                                  original MIDI position (pass_used = 'frozen').
                                  Use this to suppress sub-millisecond noise
                                  moves on instruments where the MIDI is already
                                  well-placed (e.g. snare notes at the actual
                                  stroke, as opposed to cymbal-placed notes that
                                  need a large correction).  Default 0.0 (no
                                  minimum — every detected move is applied).
                                  A value of 0.2 ms eliminates the 1–7 sample
                                  noise shifts seen on correctly-placed snare
                                  notes while preserving all meaningful
                                  corrections (≥ 0.2 ms).

    Returns:
        List of OnsetResult, one per input position, in the same order.
    """
    audio_filtered = bandpass(audio, low_hz, high_hz, sr)

    # Pre-filter check: if no band filter was applied (wideband instrument),
    # audio_filtered IS audio_raw.  In that case skip the retry entirely —
    # it would be identical to the primary pass.
    is_wideband = low_hz <= 0.0 and high_hz <= 0.0

    results = []
    for pos in positions:
        p = int(pos)

        r = refine_position(
            audio, audio_filtered, sr, p,
            search_back_ms=search_back_ms,
            search_fwd_ms=search_fwd_ms,
            onset_threshold=onset_threshold,
            onset_threshold_distant=onset_threshold_distant,
            clamp_to_midi=clamp_to_midi,
            _pass_label="primary",
        )

        if confidence_min > 0.0 and r.confidence < confidence_min and not is_wideband:
            # ── Wideband retry ────────────────────────────────────────────────
            # Run a second pass with the raw (unfiltered) audio for both the
            # amplitude weight and the Stage 2 argmin.  The wideband transient
            # is sharper and more energetic than the band-limited version for
            # most drum hits, giving a higher confidence ratio when the target
            # frequency band has significant bleed competition.
            r2 = refine_position(
                audio, audio,   # audio_filtered = audio_raw → wideband
                sr, p,
                search_back_ms=search_back_ms,
                search_fwd_ms=search_fwd_ms,
                onset_threshold=onset_threshold,
                onset_threshold_distant=onset_threshold_distant,
                clamp_to_midi=clamp_to_midi,
                _pass_label="wideband",
            )

            if r2.confidence > r.confidence:
                # Wideband found a more confident onset — use it.
                r = r2
            # else: keep primary result (even if still below threshold)

            if r.confidence < confidence_min:
                # Neither pass is confident enough — freeze at original.
                r = OnsetResult(
                    original=p,
                    refined=p,
                    offset_samples=0,
                    offset_ms=0.0,
                    confidence=r.confidence,
                    pass_used="frozen",
                )

        # min_shift_ms: suppress sub-threshold noise moves.
        #
        # On correctly-placed notes the detection can return a shift of
        # 1–7 samples (0.02–0.15 ms) that represents onset-detection
        # noise rather than a genuine timing error.  Applying these tiny
        # moves makes already-correct notes measurably worse while adding
        # no perceptible improvement.  Any shift smaller than min_shift_ms
        # is discarded and the note stays at its original MIDI position.
        if min_shift_ms > 0.0 and abs(r.offset_ms) < min_shift_ms:
            r = OnsetResult(
                original=p,
                refined=p,
                offset_samples=0,
                offset_ms=0.0,
                confidence=r.confidence,
                pass_used="frozen",
            )

        results.append(r)

    return results
