# Transient Snap v2 - Setup Instructions

## What You Need

- **Python 3.8+** (with tkinter)
- **pip** (Python package manager, included with Python)
- The Transient Snap project files (`snap.py`, `transient_snap.py`)

---

## macOS

### 1. Install Python

Download the latest Python from [python.org](https://www.python.org/downloads/):

1. Go to https://www.python.org/downloads/
2. Click "Download Python 3.x.x" (the big yellow button)
3. Open the downloaded `.pkg` file and follow the installer
4. **Important:** On the final installer screen, click "Install Certificates" if prompted

Verify it worked — open **Terminal** (Applications > Utilities > Terminal):

```bash
python3 --version
```

You should see something like `Python 3.12.x`.

### 2. Install Dependencies

In Terminal, navigate to the transient_snap folder and install:

```bash
cd /path/to/transient_snap
pip3 install -r requirements.txt
```

Or manually:

```bash
pip3 install numpy soundfile librosa mido scipy matplotlib customtkinter
```

If you get a permissions error, use:

```bash
pip3 install --user numpy soundfile librosa mido scipy matplotlib customtkinter
```

### 3. Run

```bash
cd /path/to/transient_snap
python3 transient_snap.py
```

---

## Windows

### 1. Install Python

Download the latest Python from [python.org](https://www.python.org/downloads/):

1. Go to https://www.python.org/downloads/
2. Click "Download Python 3.x.x"
3. Run the installer
4. **IMPORTANT: Check the box that says "Add Python to PATH"** at the bottom of the first installer screen
5. Click "Install Now"

Verify it worked — open **Command Prompt** (search "cmd" in Start menu):

```cmd
python --version
```

You should see something like `Python 3.12.x`.

> If you see "Python was not found" or it opens the Microsoft Store, close and reopen Command Prompt, or re-run the installer and make sure "Add Python to PATH" is checked.

### 2. Install Dependencies

In Command Prompt, navigate to the transient_snap folder:

```cmd
cd C:\path\to\transient_snap
pip install -r requirements.txt
```

Or manually:

```cmd
pip install numpy soundfile librosa mido scipy matplotlib customtkinter
```

### 3. Run

```cmd
cd C:\path\to\transient_snap
python transient_snap.py
```

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1` | Previous sample |
| `2` | Next sample |
| `-` | Zoom out |
| `=` / `+` | Zoom in |
| `q` | Place transient marker at mouse cursor position |
| `w` *(hold)* | Temporarily zoom out to ~1 second view; release to restore |
| Scroll wheel | Zoom in / out |

---

## Troubleshooting

### "No module named tkinter"

**macOS:** Reinstall Python from python.org (the Homebrew version sometimes omits tkinter). The official installer includes it.

**Windows:** Re-run the Python installer, click "Modify", and make sure "tcl/tk and IDLE" is checked.

### "No module named soundfile" / libsndfile errors

**macOS:**
```bash
brew install libsndfile
```
If you don't have Homebrew: https://brew.sh

**Windows:** The `soundfile` pip package bundles libsndfile on Windows, so this usually just works. If not, try:
```cmd
pip install soundfile --force-reinstall
```

### "No module named customtkinter"

```bash
pip install customtkinter
```
(use `pip3` on macOS)

### matplotlib backend errors

If you see errors about matplotlib backends, try:
```bash
pip install PyQt5
```
or on macOS:
```bash
pip3 install PyQt5
```

### Default tick file not found

On first use, if the default tick WAV isn't found at the expected path, go to **Settings > Tick Sample...** to select one. Any short click/tick WAV file will work (mono, 48kHz preferred).

---

## Quick Reference

| | macOS | Windows |
|---|---|---|
| Python command | `python3` | `python` |
| Pip command | `pip3` | `pip` |
| Terminal | Terminal.app | Command Prompt (cmd) |
| Run the app | `python3 transient_snap.py` | `python transient_snap.py` |
