#!/usr/bin/env python3
"""
drum_align_gui — Multi-element batch onset alignment.

Workflow:
  1. Load MIDI → assign each MIDI track to a named kit element.
  2. Link a close-mic WAV to each element; choose its instrument preset.
  3. Set an output MIDI path and run.

The tool refines every element's note positions using band-limited spectral
flux onset detection, then writes a single aligned multitrack MIDI.

Assumes constant 120 BPM for both input and output.
"""

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import customtkinter as ctk
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

from onset import INSTRUMENT_BANDS, refine_all
from snap import get_midi_track_info, load_audio, load_markers_midi, save_markers_midi
from drum_align import write_csv_report


# 120 BPM, PPQ 28800 — used for all sample↔tick conversions.
FIXED_TEMPO_MAP = {'ppq': 28800, 'tempo_events': [(0, 500_000)]}
DEFAULT_SR = 48_000

INSTRUMENT_KEYS = sorted(INSTRUMENT_BANDS.keys())


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class DrumElement:
    name: str
    track_indices: list[int]
    instrument: str = "kick"
    audio_path: str | None = None
    # Per-element search window overrides. None = use global UI values.
    search_back_ms: float | None = None
    search_fwd_ms:  float | None = None


# ── Name / instrument guessing ────────────────────────────────────────────────

def _guess_element_name(track_name: str) -> str:
    n = track_name.lower()
    if any(p in n for p in ("kick", "kik", "bass drum")) or n.startswith("bd") or " bd" in n:
        return "Kick"
    if "snare" in n or n.startswith("sd") or " sd" in n:
        return "Snare"
    if "floor" in n or "tom 3" in n or "tom3" in n:
        return "Tom 3"
    if "hi tom" in n or "tom 1" in n or "tom1" in n:
        return "Tom 1"
    if "mid tom" in n or "tom 2" in n or "tom2" in n:
        return "Tom 2"
    if "tom 4" in n or "tom4" in n:
        return "Tom 4"
    if "rack" in n:
        return "Tom 1"
    if "tom" in n:
        return "Tom"
    # Shorthand like "T1 MIDI", "T2 MIDI", "T3 MIDI" (common Reaper export names)
    if n.startswith("t1") or " t1" in n:
        return "Tom 1"
    if n.startswith("t2") or " t2" in n:
        return "Tom 2"
    if n.startswith("t3") or " t3" in n:
        return "Tom 3"
    if n.startswith("t4") or " t4" in n:
        return "Tom 4"
    if any(p in n for p in ("hihat", "hi-hat", "hi hat")) or n.startswith("hh") or " hh" in n:
        return "HiHat"
    if "ride" in n:
        return "Ride"
    if "crash" in n:
        return "Crash"
    if "overhead" in n or n.startswith("oh") or " oh" in n:
        return "Overhead"
    if "room" in n or "amb" in n:
        return "Room"
    return track_name.strip() or "Track"


def _guess_instrument(element_name: str) -> str:
    n = element_name.lower()
    if "kick" in n:                                        return "kick"
    if "snare" in n:                                       return "snare"
    if "hi" in n and "hat" in n:                           return "hihat"
    if "hihat" in n or n == "hh":                          return "hihat"
    if "ride" in n:                                        return "ride"
    if "crash" in n:                                       return "crash"
    if "tom" in n or "floor" in n or "rack" in n:          return "tom"
    if n.startswith("t1") or n.startswith("t2") or \
       n.startswith("t3") or n.startswith("t4"):           return "tom"
    if "overhead" in n or n == "oh":                       return "overhead"
    if "room" in n or "amb" in n:                          return "room"
    return "kick"


# Search-window overrides applied automatically to ghost tracks.
GHOST_BACK_MS = 5.0
GHOST_FWD_MS  = 5.0


def _is_ghost_track(name: str) -> bool:
    """Return True if the element or track name contains the word 'ghost'."""
    return "ghost" in name.lower()


# ── Main window ───────────────────────────────────────────────────────────────

class DrumAlignGUI(ctk.CTk):

    # Palette — matches TransientSnap
    _BG     = '#1a1b1e'
    _PANEL  = '#16171a'
    _CARD   = '#0f1013'
    _ENTRY  = '#202225'
    _FG     = '#e2e4e8'
    _FG_DIM = '#6e7280'
    _BORDER = '#2a2b2f'
    _GREEN  = '#10b981'
    _AMBER  = '#f59e0b'
    _BLUE   = '#3b8ed0'
    _RED    = '#ef4444'

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title("Drum Align")
        self.geometry("920x660")
        self.minsize(800, 520)

        self.midi_path: str | None = None
        self.elements: list[DrumElement] = []

        self._log_queue: queue.Queue = queue.Queue()
        self._running = False
        # Kept to prevent GC of StringVars inside rebuilt element rows
        self._inst_vars: list[tk.StringVar] = []

        self._build_ui()
        self._poll_log()

    # ── Top-level layout ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_content()

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self) -> None:
        sb = ctk.CTkFrame(self, width=264, corner_radius=0, fg_color=self._PANEL)
        sb.grid(row=0, column=0, sticky='nsew')
        sb.grid_propagate(False)
        sb.grid_columnconfigure(0, weight=1)
        sb.grid_rowconfigure(5, weight=1)  # element list

        # Header
        hdr = ctk.CTkFrame(sb, fg_color='transparent', height=46)
        hdr.grid(row=0, column=0, sticky='ew')
        hdr.grid_propagate(False)
        ctk.CTkLabel(hdr, text="∿", font=ctk.CTkFont(size=20),
                     text_color=self._BLUE).pack(side='left', padx=(14, 6), pady=10)
        ctk.CTkLabel(hdr, text="Drum Align",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=self._FG).pack(side='left')
        ctk.CTkFrame(sb, height=1, fg_color=self._BORDER).grid(row=1, column=0, sticky='ew')

        # MIDI source
        midi_sec = ctk.CTkFrame(sb, fg_color='transparent')
        midi_sec.grid(row=2, column=0, sticky='ew', padx=12, pady=(10, 8))
        midi_sec.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(midi_sec, text="MIDI SOURCE",
                     font=ctk.CTkFont(size=9, weight="bold"),
                     text_color=self._FG_DIM).grid(row=0, column=0, sticky='w', pady=(0, 4))
        ctk.CTkButton(
            midi_sec, text="Load MIDI…",
            height=30, fg_color=self._ENTRY, hover_color='#303338',
            border_width=1, border_color=self._BORDER,
            text_color=self._FG, font=ctk.CTkFont(size=11),
            command=self._load_midi,
        ).grid(row=1, column=0, sticky='ew', pady=(0, 4))

        self._midi_lbl = ctk.CTkLabel(
            midi_sec, text="No MIDI loaded",
            font=ctk.CTkFont(family='Menlo', size=9),
            text_color=self._FG_DIM, anchor='w', wraplength=220,
        )
        self._midi_lbl.grid(row=2, column=0, sticky='w')

        ctk.CTkFrame(sb, height=1, fg_color=self._BORDER).grid(row=3, column=0, sticky='ew')

        # Element list
        elem_hdr = ctk.CTkFrame(sb, fg_color='transparent')
        elem_hdr.grid(row=4, column=0, sticky='ew', padx=12, pady=(8, 4))
        ctk.CTkLabel(elem_hdr, text="ELEMENTS",
                     font=ctk.CTkFont(size=9, weight="bold"),
                     text_color=self._FG_DIM).pack(side='left')

        self._elem_scroll = ctk.CTkScrollableFrame(
            sb, fg_color='transparent', corner_radius=0)
        self._elem_scroll.grid(row=5, column=0, sticky='nsew', padx=4)
        self._elem_scroll.grid_columnconfigure(0, weight=1)

        ctk.CTkFrame(sb, height=1, fg_color=self._BORDER).grid(row=6, column=0, sticky='ew')

        # Load MIDI hint at bottom
        ctk.CTkLabel(
            sb, text="Load a MIDI file to begin",
            font=ctk.CTkFont(size=10), text_color=self._FG_DIM,
        ).grid(row=7, column=0, pady=10)

    def _rebuild_elements(self) -> None:
        """Rebuild the scrollable element list from self.elements."""
        for w in self._elem_scroll.winfo_children():
            w.destroy()
        self._inst_vars.clear()

        for i, elem in enumerate(self.elements):
            self._build_element_row(i, elem)

        self._update_run_button()

    def _build_element_row(self, idx: int, elem: DrumElement) -> None:
        has_audio = elem.audio_path is not None

        outer = ctk.CTkFrame(
            self._elem_scroll,
            fg_color=self._CARD, corner_radius=6,
            border_width=1, border_color=self._BORDER,
        )
        outer.grid(row=idx, column=0, sticky='ew', padx=4, pady=(0, 4))
        outer.grid_columnconfigure(0, weight=1)

        # ── Row 1: name + ghost badge + remove button ────────────────────────
        name_row = ctk.CTkFrame(outer, fg_color='transparent')
        name_row.grid(row=0, column=0, sticky='ew', padx=(10, 4), pady=(6, 2))
        name_row.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            name_row, text=elem.name,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=self._FG, anchor='w',
        ).grid(row=0, column=0, sticky='w')

        # Show a compact badge when per-element window overrides are active
        if elem.search_back_ms is not None or elem.search_fwd_ms is not None:
            back = elem.search_back_ms if elem.search_back_ms is not None else "—"
            fwd  = elem.search_fwd_ms  if elem.search_fwd_ms  is not None else "—"
            badge_text = f"⟵{back}ms / {fwd}ms→"
            ctk.CTkLabel(
                name_row, text=badge_text,
                font=ctk.CTkFont(family='Menlo', size=9),
                text_color=self._BLUE,
            ).grid(row=0, column=1, padx=(6, 0))

        ctk.CTkButton(
            name_row, text="×",
            width=22, height=22,
            fg_color='transparent', hover_color=self._ENTRY,
            text_color=self._FG_DIM, font=ctk.CTkFont(size=13),
            command=lambda i=idx: self._remove_element(i),
        ).grid(row=0, column=2, padx=(4, 0))

        # ── Row 2: instrument menu ────────────────────────────────────────────
        inst_var = tk.StringVar(value=elem.instrument)
        self._inst_vars.append(inst_var)

        inst_menu = ctk.CTkOptionMenu(
            outer,
            values=INSTRUMENT_KEYS,
            width=190, height=26,
            fg_color=self._ENTRY,
            button_color=self._BLUE, button_hover_color='#2b7ec5',
            text_color=self._FG, dropdown_fg_color=self._PANEL,
            dropdown_text_color=self._FG, dropdown_hover_color=self._ENTRY,
            font=ctk.CTkFont(size=11),
            command=lambda val, i=idx: self._set_instrument(i, val),
        )
        inst_menu.set(elem.instrument)
        inst_menu.grid(row=1, column=0, sticky='w', padx=(8, 8), pady=(0, 4))

        # ── Row 3: audio link ─────────────────────────────────────────────────
        audio_row = ctk.CTkFrame(outer, fg_color='transparent')
        audio_row.grid(row=2, column=0, sticky='ew', padx=(8, 8), pady=(0, 8))
        audio_row.grid_columnconfigure(0, weight=1)

        if has_audio:
            short = Path(elem.audio_path).name
            if len(short) > 28:
                short = short[:25] + "…"
            ctk.CTkLabel(
                audio_row, text=f"● {short}",
                font=ctk.CTkFont(family='Menlo', size=9),
                text_color=self._GREEN, anchor='w',
            ).grid(row=0, column=0, sticky='w')
            ctk.CTkButton(
                audio_row, text="relink",
                width=46, height=22,
                fg_color='transparent', hover_color=self._ENTRY,
                text_color=self._FG_DIM, font=ctk.CTkFont(size=10),
                command=lambda i=idx: self._link_audio(i),
            ).grid(row=0, column=1)
        else:
            ctk.CTkButton(
                audio_row, text="⊕  Link audio WAV…",
                height=28, fg_color='transparent',
                hover_color=self._ENTRY,
                border_width=1, border_color=self._AMBER,
                text_color=self._AMBER, font=ctk.CTkFont(size=11),
                command=lambda i=idx: self._link_audio(i),
            ).grid(row=0, column=0, sticky='ew')

    # ── Content area ──────────────────────────────────────────────────────────

    def _build_content(self) -> None:
        content = ctk.CTkFrame(self, fg_color=self._BG, corner_radius=0)
        content.grid(row=0, column=1, sticky='nsew')
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(6, weight=1)  # log area expands

        def _sep(r):
            ctk.CTkFrame(content, height=1, fg_color=self._BORDER,
                         corner_radius=0).grid(row=r, column=0, sticky='ew')

        self._build_output_section(content, row=0)
        _sep(1)
        self._build_params_section(content, row=2)
        _sep(3)
        self._build_run_row(content, row=4)
        _sep(5)
        self._build_log_area(content, row=6)

    def _build_output_section(self, parent, row: int) -> None:
        self._output_var = tk.StringVar()
        self._csv_var    = tk.BooleanVar(value=False)

        outer = ctk.CTkFrame(parent, fg_color='transparent')
        outer.grid(row=row, column=0, sticky='ew', padx=20, pady=(16, 8))
        outer.grid_columnconfigure(1, weight=1)

        _section_label(outer, "OUTPUT", row=0)

        _field_label(outer, "MIDI file", row=1)
        ctk.CTkEntry(
            outer, textvariable=self._output_var,
            fg_color=self._ENTRY, border_color=self._BORDER,
            text_color=self._FG, font=ctk.CTkFont(family='Menlo', size=10),
            placeholder_text="(click Save…)",
            placeholder_text_color=self._FG_DIM,
        ).grid(row=1, column=1, sticky='ew', padx=(6, 4), pady=2)
        ctk.CTkButton(
            outer, text="Save…", width=68, height=28,
            fg_color=self._ENTRY, hover_color=self._BORDER,
            border_color=self._BORDER, border_width=1,
            text_color=self._FG, font=ctk.CTkFont(size=11),
            command=self._browse_output,
        ).grid(row=1, column=2, pady=2)

        ctk.CTkCheckBox(
            outer, text="Save CSV report per element",
            variable=self._csv_var,
            text_color=self._FG_DIM, font=ctk.CTkFont(size=11),
            fg_color=self._BLUE, hover_color='#2b7ec5',
            checkmark_color='#ffffff',
        ).grid(row=2, column=0, columnspan=3, sticky='w', pady=(6, 0))

    def _build_params_section(self, parent, row: int) -> None:
        self._back_var      = tk.StringVar(value="0.0")
        self._fwd_var       = tk.StringVar(value="10.0")
        self._thresh_var    = tk.StringVar(value="0.05")
        self._conf_var      = tk.StringVar(value="3.0")
        self._conf_min_var  = tk.StringVar(value="2.0")

        outer = ctk.CTkFrame(parent, fg_color='transparent')
        outer.grid(row=row, column=0, sticky='ew', padx=20, pady=(12, 8))

        _section_label(outer, "SEARCH WINDOW", row=0)

        params_row = ctk.CTkFrame(outer, fg_color='transparent')
        params_row.grid(row=1, column=0, sticky='w', pady=(4, 0))

        for col, (label, var, unit) in enumerate([
            ("Back",         self._back_var,     "ms"),
            ("Forward",      self._fwd_var,      "ms"),
            ("Threshold",    self._thresh_var,   ""),
            ("Conf. warn",   self._conf_var,     "×"),
            ("Conf. min",    self._conf_min_var, "×"),
        ]):
            cell = ctk.CTkFrame(params_row, fg_color='transparent')
            cell.grid(row=0, column=col, padx=(0, 22))
            ctk.CTkLabel(cell, text=label, text_color=self._FG_DIM,
                         font=ctk.CTkFont(size=10)).pack(anchor='w')
            unit_f = ctk.CTkFrame(cell, fg_color='transparent')
            unit_f.pack(anchor='w')
            ctk.CTkEntry(
                unit_f, textvariable=var, width=62,
                fg_color=self._ENTRY, border_color=self._BORDER,
                text_color=self._FG, font=ctk.CTkFont(family='Menlo', size=12),
            ).pack(side='left')
            if unit:
                ctk.CTkLabel(unit_f, text=f" {unit}", text_color=self._FG_DIM,
                             font=ctk.CTkFont(size=10)).pack(side='left')

    def _build_run_row(self, parent, row: int) -> None:
        run_f = ctk.CTkFrame(parent, fg_color='transparent')
        run_f.grid(row=row, column=0, sticky='ew', padx=20, pady=(10, 10))

        self._run_btn = ctk.CTkButton(
            run_f, text="Run All Elements",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=36, width=170,
            fg_color=self._BLUE, hover_color='#2b7ec5',
            state='disabled',
            command=self._on_run,
        )
        self._run_btn.pack(side='left')

        self._status_lbl = ctk.CTkLabel(
            run_f, text="Load a MIDI file to begin",
            font=ctk.CTkFont(size=11), text_color=self._FG_DIM,
        )
        self._status_lbl.pack(side='left', padx=(14, 0))

    def _build_log_area(self, parent, row: int) -> None:
        parent.grid_rowconfigure(row, weight=1)

        ctk.CTkLabel(
            parent, text="OUTPUT",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color=self._FG_DIM, fg_color='transparent',
        ).grid(row=row, column=0, sticky='nw', padx=20, pady=(8, 0))

        log_outer = ctk.CTkFrame(parent, fg_color=self._CARD, corner_radius=6)
        log_outer.grid(row=row, column=0, sticky='nsew', padx=16, pady=(24, 16))
        log_outer.grid_rowconfigure(0, weight=1)
        log_outer.grid_columnconfigure(0, weight=1)

        self._log_text = tk.Text(
            log_outer,
            bg=self._CARD, fg=self._FG,
            font=('Menlo', 10),
            insertbackground=self._FG,
            selectbackground=self._BLUE,
            relief='flat', padx=10, pady=8,
            wrap='word', state='disabled',
        )
        self._log_text.grid(row=0, column=0, sticky='nsew')

        sb = ctk.CTkScrollbar(log_outer, command=self._log_text.yview)
        sb.grid(row=0, column=1, sticky='ns')
        self._log_text.configure(yscrollcommand=sb.set)

        self._log_text.tag_configure('dim',   foreground=self._FG_DIM)
        self._log_text.tag_configure('green', foreground=self._GREEN)
        self._log_text.tag_configure('amber', foreground=self._AMBER)
        self._log_text.tag_configure('blue',  foreground=self._BLUE)
        self._log_text.tag_configure('red',   foreground=self._RED)

    # ── Element management ────────────────────────────────────────────────────

    def _set_instrument(self, idx: int, value: str) -> None:
        if 0 <= idx < len(self.elements):
            self.elements[idx].instrument = value

    def _remove_element(self, idx: int) -> None:
        if 0 <= idx < len(self.elements):
            self.elements.pop(idx)
            self._rebuild_elements()

    def _link_audio(self, idx: int) -> None:
        if not (0 <= idx < len(self.elements)):
            return
        initial = None
        if self.midi_path:
            initial = str(Path(self.midi_path).parent)
        path = filedialog.askopenfilename(
            title=f"Link audio WAV — {self.elements[idx].name}",
            initialdir=initial,
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if path:
            self.elements[idx].audio_path = path
            self._rebuild_elements()
            self._update_run_button()

    def _update_run_button(self) -> None:
        ready = (
            self.elements
            and self._output_var.get().strip()
            and all(e.audio_path for e in self.elements)
            and not self._running
        )
        self._run_btn.configure(state='normal' if ready else 'disabled')
        if not self.elements:
            self._status_lbl.configure(text="Load a MIDI file to begin",
                                       text_color=self._FG_DIM)
        elif not all(e.audio_path for e in self.elements):
            missing = sum(1 for e in self.elements if not e.audio_path)
            self._status_lbl.configure(
                text=f"{missing} element{'s' if missing > 1 else ''} need audio linked",
                text_color=self._AMBER)
        elif not self._output_var.get().strip():
            self._status_lbl.configure(text="Set an output MIDI path",
                                       text_color=self._FG_DIM)
        else:
            self._status_lbl.configure(text="Ready", text_color=self._GREEN)

    # ── Browse ────────────────────────────────────────────────────────────────

    def _browse_output(self) -> None:
        initial_dir = None
        if self.midi_path:
            initial_dir = str(Path(self.midi_path).parent)
        stem = Path(self.midi_path).stem if self.midi_path else "session"
        path = filedialog.asksaveasfilename(
            title="Save aligned MIDI as…",
            defaultextension=".mid",
            initialdir=initial_dir,
            initialfile=f"{stem}_aligned.mid",
            filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")],
        )
        if path:
            self._output_var.set(path)
            self._update_run_button()

    # Trace output var so Run button enables as soon as a path is typed in
    def _on_output_var_change(self, *_) -> None:
        self._update_run_button()

    # ── Load MIDI + assignment dialog ─────────────────────────────────────────

    def _load_midi(self) -> None:
        initial = str(Path(self.midi_path).parent) if self.midi_path else None
        path = filedialog.askopenfilename(
            title="Select MIDI file",
            initialdir=initial,
            filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            tracks = get_midi_track_info(path)
        except Exception as exc:
            messagebox.showerror("MIDI Error", f"Could not read MIDI file:\n{exc}")
            return

        tracks_with_notes = [t for t in tracks if t['note_count'] > 0]
        if not tracks_with_notes:
            messagebox.showwarning("MIDI", "No tracks with notes found in this file.")
            return

        if self.elements:
            if not messagebox.askyesno(
                "Load MIDI",
                "Replace current elements with tracks from the new MIDI file?",
            ):
                return

        groups = self._show_track_assignment_dialog(path, tracks_with_notes)
        if groups is None:  # cancelled
            return

        self.midi_path = path
        self._midi_lbl.configure(text=Path(path).name)
        self.elements = []
        for g in groups:
            name = g['name']
            ghost = _is_ghost_track(name)
            self.elements.append(DrumElement(
                name=name,
                track_indices=g['track_indices'],
                instrument=_guess_instrument(name),
                search_back_ms=GHOST_BACK_MS if ghost else None,
                search_fwd_ms=GHOST_FWD_MS  if ghost else None,
            ))
        self._rebuild_elements()

        # Suggest output path
        if not self._output_var.get():
            self._output_var.set(
                str(Path(path).with_name(Path(path).stem + "_aligned.mid"))
            )
        self._update_run_button()

    def _show_track_assignment_dialog(
        self, midi_path: str, tracks: list[dict]
    ) -> list[dict] | None:
        """Modal dialog: assign each MIDI track to a kit element name.

        Returns list of {'name': str, 'track_indices': [int]}, or None if
        the user cancels.  Tracks sharing a name are merged.  Blank = skip.
        """
        dialog = ctk.CTkToplevel(self)
        dialog.title("Assign MIDI Tracks to Kit Elements")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(True, True)
        dialog.minsize(520, 200)
        dialog.after(100, dialog.lift)

        ctk.CTkLabel(
            dialog,
            text="Assign each MIDI track to a kit element name.\n"
                 "Tracks with the same name are merged.  Leave blank to skip.",
            text_color=self._FG_DIM, justify='left',
            font=ctk.CTkFont(size=11),
        ).pack(anchor='w', padx=20, pady=(18, 10))

        # Column headers
        hdr = ctk.CTkFrame(dialog, fg_color=self._CARD, corner_radius=6)
        hdr.pack(fill='x', padx=20, pady=(0, 2))
        for col, (text, width) in enumerate([
            ("MIDI Track",  200),
            ("Notes",        52),
            ("Kit Element",   0),
        ]):
            ctk.CTkLabel(
                hdr, text=text,
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=self._FG_DIM,
                width=width, anchor='w',
            ).grid(row=0, column=col,
                   sticky='w', padx=(12 if col == 0 else 4, 4), pady=6)
        hdr.grid_columnconfigure(2, weight=1)

        # Track rows
        scroll_h = min(340, len(tracks) * 44 + 12)
        scroll = ctk.CTkScrollableFrame(
            dialog, fg_color=self._PANEL, height=scroll_h, corner_radius=6)
        scroll.pack(fill='x', padx=20, pady=(0, 12))
        scroll.grid_columnconfigure(2, weight=1)

        entry_vars: list[tuple[dict, tk.StringVar]] = []
        for i, track in enumerate(tracks):
            var = tk.StringVar(value=_guess_element_name(track['name']))
            entry_vars.append((track, var))

            ctk.CTkLabel(
                scroll, text=track['name'],
                font=ctk.CTkFont(family='Menlo', size=11),
                text_color=self._FG, anchor='w', width=200,
            ).grid(row=i, column=0, sticky='w', padx=(12, 4), pady=4)

            ctk.CTkLabel(
                scroll, text=str(track['note_count']),
                font=ctk.CTkFont(family='Menlo', size=11),
                text_color=self._FG_DIM, anchor='e', width=52,
            ).grid(row=i, column=1, sticky='e', padx=4, pady=4)

            ctk.CTkEntry(
                scroll, textvariable=var, height=28,
                fg_color=self._ENTRY, border_color=self._BORDER,
                text_color=self._FG, font=ctk.CTkFont(size=11),
            ).grid(row=i, column=2, sticky='ew', padx=(4, 12), pady=4)

        result: list[dict] | None = None

        def _on_import():
            nonlocal result
            merged: dict[str, list[int]] = {}
            for track, var in entry_vars:
                name = var.get().strip()
                if name:
                    merged.setdefault(name, []).append(track['index'])
            if not merged:
                messagebox.showwarning("No Elements",
                                       "Assign at least one track to an element.")
                return
            result = [{'name': n, 'track_indices': idx}
                      for n, idx in merged.items()]
            dialog.destroy()

        def _on_cancel():
            dialog.destroy()

        btn_row = ctk.CTkFrame(dialog, fg_color='transparent')
        btn_row.pack(pady=(0, 16))
        ctk.CTkButton(
            btn_row, text="Import",
            width=100, height=32,
            fg_color=self._BLUE, hover_color='#2b7ec5',
            command=_on_import,
        ).pack(side='left', padx=(0, 8))
        ctk.CTkButton(
            btn_row, text="Cancel",
            width=80, height=32,
            fg_color=self._ENTRY, hover_color=self._BORDER,
            border_width=1, border_color=self._BORDER,
            text_color=self._FG_DIM,
            command=_on_cancel,
        ).pack(side='left')

        dialog.wait_window()
        return result

    # ── Log ───────────────────────────────────────────────────────────────────

    def _log(self, text: str, tag: str = "") -> None:
        self._log_queue.put((text, tag))

    def _poll_log(self) -> None:
        while not self._log_queue.empty():
            text, tag = self._log_queue.get_nowait()
            self._log_text.configure(state='normal')
            self._log_text.insert('end', text, (tag,) if tag else ())
            self._log_text.see('end')
            self._log_text.configure(state='disabled')
        self.after(40, self._poll_log)

    def _clear_log(self) -> None:
        self._log_text.configure(state='normal')
        self._log_text.delete('1.0', 'end')
        self._log_text.configure(state='disabled')

    # ── Validation ────────────────────────────────────────────────────────────

    def _parse_params(self) -> dict:
        def _float(var, name, lo=None, hi=None):
            try:
                v = float(var.get())
            except ValueError:
                raise ValueError(f"{name} must be a number")
            if lo is not None and v < lo:
                raise ValueError(f"{name} must be ≥ {lo}")
            if hi is not None and v > hi:
                raise ValueError(f"{name} must be ≤ {hi}")
            return v

        output = self._output_var.get().strip()
        if not output:
            raise ValueError("Output MIDI path is required")
        if not self.midi_path:
            raise ValueError("No MIDI file loaded")
        if not self.elements:
            raise ValueError("No elements defined")
        missing = [e.name for e in self.elements if not e.audio_path]
        if missing:
            raise ValueError(f"Missing audio for: {', '.join(missing)}")

        return {
            'output_path':     Path(output),
            'search_back_ms':  _float(self._back_var,     "Search back",  0.0, 500.0),
            'search_fwd_ms':   _float(self._fwd_var,     "Search fwd",   0.1, 500.0),
            'onset_threshold': _float(self._thresh_var,  "Threshold",    0.001, 1.0),
            'confidence_warn': _float(self._conf_var,    "Conf. warn",   0.0),
            'confidence_min':  _float(self._conf_min_var,"Conf. min",    0.0),
            'save_csv':        self._csv_var.get(),
        }

    # ── Run ───────────────────────────────────────────────────────────────────

    def _on_run(self) -> None:
        if self._running:
            return
        self._clear_log()
        try:
            params = self._parse_params()
        except ValueError as exc:
            self._log(f"Error: {exc}\n", 'red')
            return

        self._running = True
        self._run_btn.configure(state='disabled', text="Running…")
        self._status_lbl.configure(text="", text_color=self._FG_DIM)

        # Snapshot element state so the worker isn't affected by UI changes mid-run
        elements_snapshot = [
            DrumElement(e.name, list(e.track_indices), e.instrument, e.audio_path,
                        e.search_back_ms, e.search_fwd_ms)
            for e in self.elements
        ]
        threading.Thread(
            target=self._worker,
            args=(params, elements_snapshot, self.midi_path),
            daemon=True,
        ).start()

    def _worker(self, params: dict, elements: list[DrumElement],
                midi_path: str) -> None:
        t0 = time.perf_counter()
        try:
            self._run_alignment(params, elements, midi_path)
            elapsed = time.perf_counter() - t0
            self._log(f"\nAll done in {elapsed:.1f} s\n", 'green')
            self.after(0, self._on_success)
        except Exception as exc:
            self._log(f"\nError: {exc}\n", 'red')
            self.after(0, self._on_failure)

    def _run_alignment(self, params: dict, elements: list[DrumElement],
                       midi_path: str) -> None:
        output_path     = params['output_path']
        search_back_ms  = params['search_back_ms']
        search_fwd_ms   = params['search_fwd_ms']
        onset_threshold = params['onset_threshold']
        confidence_warn = params['confidence_warn']
        confidence_min  = params['confidence_min']

        # Accumulate all refined notes across elements for the combined output
        all_positions:        list[np.ndarray] = []
        all_amplitudes:       list[np.ndarray] = []
        all_track_assignments:list[np.ndarray] = []
        all_note_values:      list[np.ndarray] = []
        track_names:          dict[int, str]   = {}

        for elem in elements:
            low_hz, high_hz = INSTRUMENT_BANDS.get(elem.instrument, (0.0, 0.0))

            if low_hz <= 0 and high_hz <= 0:
                band_str = f"wideband ({elem.instrument})"
            else:
                band_str = f"{low_hz:.0f}–{high_hz:.0f} Hz ({elem.instrument})"

            # Resolve per-element window overrides (e.g. ghost tracks)
            # falling back to the global UI values when not set.
            elem_back_ms = elem.search_back_ms if elem.search_back_ms is not None \
                           else search_back_ms
            elem_fwd_ms  = elem.search_fwd_ms  if elem.search_fwd_ms  is not None \
                           else search_fwd_ms

            self._log(f"\n{'─' * 50}\n")
            self._log(f"  {elem.name}  [{band_str}]\n", 'blue')
            if elem.search_back_ms is not None or elem.search_fwd_ms is not None:
                self._log(
                    f"  Window: ⟵{elem_back_ms} ms / {elem_fwd_ms} ms→"
                    f"  (ghost override)\n", 'blue'
                )

            # Load audio at the fixed session sample rate so all elements share
            # the same time base, and MIDI tick→sample conversion is consistent.
            self._log(f"  Audio:  {Path(elem.audio_path).name}\n", 'dim')
            audio, sr = load_audio(str(elem.audio_path), target_sr=DEFAULT_SR)
            self._log(f"          {len(audio) / sr:.1f} s @ {sr} Hz\n", 'dim')

            # Load MIDI notes using the same sample rate.
            # load_markers_midi reads the file's tempo map; since we assume
            # 120 BPM throughout, the result is correct as-is.
            positions, amplitudes, track_assignments, note_values, names = \
                load_markers_midi(midi_path, DEFAULT_SR,
                                  track_indices=elem.track_indices)

            if len(positions) == 0:
                self._log(f"  No notes found in tracks {elem.track_indices} — skipping\n",
                          'amber')
                continue

            self._log(f"  {len(positions)} notes (tracks: {elem.track_indices})\n", 'dim')
            track_names.update(names)

            # Refine
            results = refine_all(
                audio, sr, positions,
                low_hz=low_hz, high_hz=high_hz,
                search_back_ms=elem_back_ms,
                search_fwd_ms=elem_fwd_ms,
                onset_threshold=onset_threshold,
                confidence_min=confidence_min,
            )

            # Stats
            offsets_ms   = np.array([r.offset_ms     for r in results])
            offsets_samp = np.array([r.offset_samples for r in results])
            confidences  = np.array([r.confidence     for r in results])
            n            = len(results)
            n_moved      = int(np.sum(offsets_samp != 0))
            n_primary    = sum(1 for r in results if r.pass_used == "primary")
            n_wideband   = sum(1 for r in results if r.pass_used == "wideband")
            n_frozen     = sum(1 for r in results if r.pass_used == "frozen")
            low_conf     = [r for r in results if r.confidence < confidence_warn
                            and r.pass_used != "frozen"]

            self._log(f"  Moved:           {n_moved}/{n}  "
                      f"({n_moved / n * 100:.0f}%)\n")
            self._log(f"  Mean offset:     {np.mean(offsets_ms):+.2f} ms  "
                      f"({np.mean(offsets_samp):+.1f} samples)\n")
            self._log(f"  Std offset:      {np.std(offsets_ms):.2f} ms\n")
            self._log(f"  Max offset:      {np.max(np.abs(offsets_ms)):.2f} ms\n")
            self._log(f"  Mean confidence: {np.mean(confidences):.1f}×\n")
            self._log(f"  Passes:          {n_primary} primary  "
                      f"/ {n_wideband} wideband  "
                      f"/ {n_frozen} frozen\n",
                      'amber' if n_frozen else None)

            if n_frozen:
                frozen_list = [r for r in results if r.pass_used == "frozen"]
                self._log(f"  Frozen notes ({n_frozen}) — left at MIDI position:\n",
                          'amber')
                for r in frozen_list[:8]:
                    t = r.original / DEFAULT_SR
                    self._log(f"    {t:.3f} s  conf {r.confidence:.1f}×\n", 'amber')
                if n_frozen > 8:
                    self._log(f"    … and {n_frozen - 8} more\n", 'amber')

            if low_conf:
                self._log(f"  Low-confidence: {len(low_conf)} notes\n", 'amber')
                for r in low_conf[:10]:
                    t = r.original / DEFAULT_SR
                    self._log(f"    {t:.3f} s  {r.offset_ms:+.2f} ms  "
                              f"conf {r.confidence:.1f}×  [{r.pass_used}]\n", 'amber')
                if len(low_conf) > 10:
                    self._log(f"    … and {len(low_conf) - 10} more\n", 'amber')

            refined = np.array([r.refined for r in results], dtype=np.int64)
            all_positions.append(refined)
            all_amplitudes.append(amplitudes)
            all_track_assignments.append(track_assignments)
            all_note_values.append(note_values)

            if params['save_csv']:
                csv_path = output_path.with_name(
                    output_path.stem + f"_{elem.name.replace(' ', '_')}.csv")
                write_csv_report(results, sr, csv_path)
                self._log(f"  Report: {csv_path.name}\n", 'dim')

        if not all_positions:
            raise ValueError("No notes were processed — check MIDI track assignments")

        # Combine and save
        combined_positions        = np.concatenate(all_positions)
        combined_amplitudes       = np.concatenate(all_amplitudes)
        combined_track_assignments= np.concatenate(all_track_assignments)
        combined_note_values      = np.concatenate(all_note_values)

        sort_idx = np.argsort(combined_positions)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_markers_midi(
            combined_positions[sort_idx],
            DEFAULT_SR,
            str(output_path),
            amplitudes=combined_amplitudes[sort_idx],
            tempo_map=FIXED_TEMPO_MAP,
            track_assignments=combined_track_assignments[sort_idx],
            note_values=combined_note_values[sort_idx],
            track_names=track_names,
            force_high_ppq=False,  # we supply FIXED_TEMPO_MAP explicitly
        )
        self._log(f"\nOutput: {output_path}\n", 'green')

    def _on_success(self) -> None:
        self._running = False
        self._run_btn.configure(state='normal', text="Run All Elements")
        self._status_lbl.configure(text="Done", text_color=self._GREEN)

    def _on_failure(self) -> None:
        self._running = False
        self._run_btn.configure(state='normal', text="Run All Elements")
        self._status_lbl.configure(text="Failed", text_color=self._RED)
        self._update_run_button()


# ── Layout helpers ────────────────────────────────────────────────────────────

def _section_label(parent, text: str, row: int) -> None:
    ctk.CTkLabel(
        parent, text=text,
        font=ctk.CTkFont(size=9, weight="bold"),
        text_color='#6e7280',
    ).grid(row=row, column=0, columnspan=3, sticky='w', pady=(0, 4))


def _field_label(parent, text: str, row: int) -> None:
    ctk.CTkLabel(
        parent, text=text,
        font=ctk.CTkFont(size=11), text_color='#6e7280',
        width=80, anchor='w',
    ).grid(row=row, column=0, sticky='w', pady=2)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    app = DrumAlignGUI()
    # Enable run button when output path is typed directly
    app._output_var.trace_add('write', app._on_output_var_change)
    app.mainloop()


if __name__ == "__main__":
    main()
