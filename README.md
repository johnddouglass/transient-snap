# Transient Snap

A drum sample alignment tool. Import a MIDI file, link audio files to each kit element, and place transient markers that export as sample-accurate WAV cue points and MIDI notes.

Designed for the workflow of aligning multi-mic drum recordings to a "perfect" MIDI grid — place one transient marker per hit, and the tool handles converting sample positions to MIDI note timing.

---

## Features

- **MIDI-first workflow** — import a MIDI file and auto-detect kit elements (Kick, Snare, Toms, HiHat, etc.) from track names
- **Merge tracks** — combine multiple MIDI tracks into a single kit element (e.g. Snare + Snare Fill)
- **Per-element audio** — link a separate audio file to each kit element
- **Transient marker placement** — click or press `q` to place a marker at the current cursor position
- **WAV + MIDI export** — exports cue-pointed WAVs and aligned MIDI files for each element
- **Fixed 120 BPM export** — optional mode to avoid tempo map inaccuracies when importing into DAWs
- **Tick sample** — audible click on marker placement, configurable in Settings

---

## Setup

See [SETUP.md](SETUP.md) for full installation instructions for macOS and Windows.

**Quick start:**

```bash
pip install -r requirements.txt
python transient_snap.py        # macOS: python3 transient_snap.py
```

---

## Usage

1. **File > New Project** — choose an output folder, then select your MIDI file
2. The app detects kit elements from MIDI track names — confirm names and merge any tracks you want combined
3. Click the **⊕** button next to each kit element to link an audio file
4. Navigate hits with `1` / `2`, click the waveform (or press `q`) to place a transient marker
5. **File > Export All** when done — outputs aligned WAVs and MIDI files to your project folder

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1` | Previous hit |
| `2` | Next hit |
| `-` | Zoom out |
| `=` / `+` | Zoom in |
| `q` | Place transient marker at mouse cursor |
| `w` *(hold)* | Zoom to ~1 second view; release to restore |
| Scroll wheel | Zoom in / out |

---

## Files

| File | Description |
|------|-------------|
| `transient_snap.py` | Main application |
| `snap.py` | Audio/MIDI processing backend |
| `Tick_44k.wav` | Default tick sample (44.1kHz) |
| `Tick_48k.wav` | Default tick sample (48kHz) |
| `requirements.txt` | Python dependencies |
| `SETUP.md` | Detailed setup instructions |
