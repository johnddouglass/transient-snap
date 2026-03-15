#!/usr/bin/env python3
"""
Transient Snap v2 - Human refinement tool for drum marker alignment.

Batch processing with project save/load. Supports multiple kit elements
(kick, snare, toms, etc.) each with their own audio + MIDI track pairing.

Usage:
    python gui_v2.py
"""

import json
import re
import tkinter as tk
from tkinter import messagebox, filedialog
import customtkinter as ctk
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from snap import (load_audio, save_markers_wav, save_markers_midi,
                  load_markers_midi, get_midi_track_info, extract_tempo_map)

DEFAULT_TICK = str(Path(__file__).parent / "Tick_48k.wav")
DEFAULT_SR = 48000


def _guess_element_name(track_name: str) -> str:
    """Pattern-match a MIDI track name to a suggested kit element name."""
    n = track_name.lower().strip()
    if any(p in n for p in ["kick", "kik", "bass drum", "bassdrum"]):
        return "Kick"
    if re.search(r'\bbd\b', n):
        return "Kick"
    if "snare" in n or re.search(r'\bsd\b', n):
        return "Snare"
    if "tom 1" in n or "tom1" in n or "hi tom" in n or "high tom" in n or re.search(r'\bt1\b', n):
        return "Tom 1"
    if "tom 2" in n or "tom2" in n or "mid tom" in n or re.search(r'\bt2\b', n):
        return "Tom 2"
    if any(p in n for p in ["tom 3", "tom3", "floor tom", "lo tom", "low tom"]) or re.search(r'\bt3\b', n):
        return "Tom 3"
    if "tom 4" in n or "tom4" in n or re.search(r'\bt4\b', n):
        return "Tom 4"
    if "rack" in n:
        return "Tom 1"
    if "floor" in n:
        return "Tom 3"
    if "tom" in n:
        return "Tom"
    if any(p in n for p in ["hihat", "hi-hat", "hi hat", "hat"]) or re.search(r'\bhh\b', n):
        return "HiHat"
    if "ride" in n:
        return "Ride"
    if "crash" in n:
        return "Crash"
    if "overhead" in n or re.search(r'\boh\b', n):
        return "Overhead"
    if "room" in n or "amb" in n:
        return "Room"
    return track_name.strip() or "Track"


class KitElement:
    """One kit element (e.g., kick, snare) with audio + markers."""
    def __init__(self, name="Untitled"):
        self.name = name
        self.audio_path = None
        self.audio = None
        self.sr = DEFAULT_SR

        # From MIDI
        self.midi_path = None
        self.track_indices = []
        self.track_names = {}

        # Marker data
        self.original_positions = None
        self.final_positions = None
        self.amplitudes = None
        self.track_assignments = None
        self.note_values = None

        # Review state
        self.status = None  # 'pending', 'approved', 'manual'
        self.current_idx = 0

    @property
    def n_markers(self):
        return len(self.original_positions) if self.original_positions is not None else 0

    def to_dict(self):
        return {
            'name': self.name,
            'audio_path': self.audio_path,
            'midi_path': self.midi_path,
            'track_indices': self.track_indices,
            'track_names': self.track_names,
            'original_positions': self.original_positions.tolist() if self.original_positions is not None else None,
            'final_positions': self.final_positions.tolist() if self.final_positions is not None else None,
            'amplitudes': self.amplitudes.tolist() if self.amplitudes is not None else None,
            'track_assignments': self.track_assignments.tolist() if self.track_assignments is not None else None,
            'note_values': self.note_values.tolist() if self.note_values is not None else None,
            'status': self.status,
            'current_idx': self.current_idx,
        }

    @classmethod
    def from_dict(cls, d, sr=DEFAULT_SR):
        elem = cls(d.get('name', 'Untitled'))
        elem.audio_path = d.get('audio_path')
        elem.midi_path = d.get('midi_path')
        elem.track_indices = d.get('track_indices', [])
        elem.track_names = d.get('track_names', {})
        elem.track_names = {int(k): v for k, v in elem.track_names.items()}
        elem.sr = sr

        if d.get('original_positions'):
            elem.original_positions = np.array(d['original_positions'], dtype=np.int64)
        if d.get('final_positions'):
            elem.final_positions = np.array(d['final_positions'], dtype=np.int64)
        if d.get('amplitudes'):
            elem.amplitudes = np.array(d['amplitudes'], dtype=np.float32)
        if d.get('track_assignments'):
            elem.track_assignments = np.array(d['track_assignments'], dtype=np.int32)
        if d.get('note_values'):
            elem.note_values = np.array(d['note_values'], dtype=np.int32)

        elem.status = d.get('status')
        elem.current_idx = d.get('current_idx', 0)
        return elem


class TransientSnapV2(ctk.CTk):
    # Palette — shared with matplotlib and tk.Listbox
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

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title("Transient Snap")
        self.geometry("1280x720")
        self.minsize(900, 560)

        # Project state
        self.project_path = None
        self.elements = []
        self.current_element_idx = 0
        self.tempo_map = None
        self.tick = None
        self.tick_path = DEFAULT_TICK
        self._fixed_bpm_var = tk.BooleanVar(value=True)
        self.display_ms = 5.0
        self.output_ppq = 28800
        self._track_idx_map = {}
        self._track_combo_values = []
        self._sidebar_btns = []
        self._resize_pending = False
        self.cursor_line = None
        self.canvas_bg = None
        self._last_mouse_xdata = None

        self._blink_state = False
        self._blink_btns = []

        self._build_ui()
        self._new_project()
        self.after(700, self._start_blink)

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main()
        self._build_menu()
        self._bind_keys()

    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=230, corner_radius=0, fg_color=self._PANEL)
        sb.grid(row=0, column=0, sticky='nsew')
        sb.grid_propagate(False)
        sb.grid_columnconfigure(0, weight=1)
        sb.grid_rowconfigure(2, weight=1)

        # Header
        hdr = ctk.CTkFrame(sb, fg_color='transparent', height=48)
        hdr.grid(row=0, column=0, sticky='ew')
        hdr.grid_propagate(False)
        ctk.CTkLabel(hdr, text="∿", font=ctk.CTkFont(size=20),
                     text_color=self._BLUE).pack(side='left', padx=(14, 6), pady=12)
        ctk.CTkLabel(hdr, text="Transient Snap",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=self._FG).pack(side='left')

        ctk.CTkFrame(sb, height=1, fg_color=self._BORDER).grid(row=1, column=0, sticky='ew')

        # Element list
        elem_outer = ctk.CTkFrame(sb, fg_color='transparent')
        elem_outer.grid(row=2, column=0, sticky='nsew')
        elem_outer.grid_rowconfigure(1, weight=1)
        elem_outer.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(elem_outer, text="ELEMENTS",
                     font=ctk.CTkFont(size=9, weight="bold"),
                     text_color=self._FG_DIM).grid(row=0, column=0, sticky='w',
                                                    padx=14, pady=(10, 4))

        self._elem_scroll = ctk.CTkScrollableFrame(elem_outer, fg_color=self._PANEL,
                                                    corner_radius=0)
        self._elem_scroll.grid(row=1, column=0, sticky='nsew', padx=4)
        self._elem_scroll.grid_columnconfigure(0, weight=1)

        ctk.CTkFrame(sb, height=1, fg_color=self._BORDER).grid(row=3, column=0, sticky='ew')

        # Controls panel
        ctrl = ctk.CTkFrame(sb, fg_color='transparent')
        ctrl.grid(row=4, column=0, sticky='ew', padx=12, pady=10)
        ctrl.grid_columnconfigure(1, weight=1)

        row = 0

        # Zoom
        ctk.CTkLabel(ctrl, text="ZOOM", font=ctk.CTkFont(size=9, weight="bold"),
                     text_color=self._FG_DIM).grid(row=row, column=0, sticky='w', pady=(0, 2))
        self.zoom_label = ctk.CTkLabel(ctrl, text=f"±{self.display_ms:.1f}ms",
                                       font=ctk.CTkFont(family='Menlo', size=10),
                                       text_color=self._BLUE)
        self.zoom_label.grid(row=row, column=1, sticky='e', pady=(0, 2))
        row += 1

        self.zoom_var = tk.DoubleVar(value=self.display_ms)
        ctk.CTkSlider(ctrl, from_=1.0, to=200.0, variable=self.zoom_var,
                      command=self._on_zoom,
                      fg_color=self._BORDER, button_color=self._BLUE,
                      button_hover_color='#5aa3e0').grid(
            row=row, column=0, columnspan=2, sticky='ew', pady=(0, 2))
        row += 1

        ctk.CTkLabel(ctrl, text="⊙ scroll wheel to zoom",
                     font=ctk.CTkFont(size=9), text_color=self._FG_DIM).grid(
            row=row, column=0, columnspan=2, sticky='w', pady=(0, 8))
        row += 1

        # Marker jump
        jump_row = ctk.CTkFrame(ctrl, fg_color='transparent')
        jump_row.grid(row=row, column=0, columnspan=2, sticky='ew', pady=(0, 6))
        jump_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(jump_row, text="Go:", text_color=self._FG_DIM,
                     font=ctk.CTkFont(size=11)).grid(row=0, column=0, padx=(0, 6))
        self.marker_var = tk.StringVar()
        self.marker_entry = ctk.CTkEntry(jump_row, textvariable=self.marker_var,
                                         width=52, height=26,
                                         fg_color=self._ENTRY, border_color=self._BORDER)
        self.marker_entry.grid(row=0, column=1, sticky='ew', padx=(0, 6))
        self.marker_entry.bind('<Return>', self._on_marker_jump)
        ctk.CTkButton(jump_row, text="GO", command=self._on_marker_jump,
                      width=36, height=26,
                      fg_color=self._ENTRY, hover_color='#303338',
                      border_width=1, border_color=self._BLUE,
                      text_color=self._BLUE,
                      font=ctk.CTkFont(size=10, weight="bold")).grid(row=0, column=2)
        row += 1

        ctk.CTkButton(ctrl, text="FIRST PENDING", command=self._goto_first_pending,
                      height=28, fg_color=self._ENTRY, hover_color='#303338',
                      border_width=1, border_color=self._BORDER,
                      text_color=self._FG_DIM,
                      font=ctk.CTkFont(size=10, weight="bold")).grid(
            row=row, column=0, columnspan=2, sticky='ew', pady=(0, 8))
        row += 1

        # Track selector
        ctk.CTkLabel(ctrl, text="TRACK", font=ctk.CTkFont(size=9, weight="bold"),
                     text_color=self._FG_DIM).grid(row=row, column=0, columnspan=2,
                                                    sticky='w', pady=(0, 2))
        row += 1
        self.track_var = tk.StringVar()
        self.track_combo = ctk.CTkComboBox(ctrl, variable=self.track_var,
                                            values=[], height=26,
                                            fg_color=self._ENTRY,
                                            border_color=self._BORDER,
                                            button_color=self._BORDER,
                                            button_hover_color=self._BLUE,
                                            dropdown_fg_color=self._CARD,
                                            command=lambda _: self._on_track_change())
        self.track_combo.grid(row=row, column=0, columnspan=2, sticky='ew', pady=(0, 10))
        self.track_combo.set('')
        row += 1

        ctk.CTkFrame(ctrl, height=1, fg_color=self._BORDER).grid(
            row=row, column=0, columnspan=2, sticky='ew', pady=(0, 8))
        row += 1

        # Info + progress
        self.info_label = ctk.CTkLabel(ctrl, text="",
                                        font=ctk.CTkFont(family='Menlo', size=10),
                                        text_color=self._FG_DIM,
                                        anchor='w', justify='left')
        self.info_label.grid(row=row, column=0, columnspan=2, sticky='ew', pady=(0, 4))
        row += 1

        self.progress_label = ctk.CTkLabel(ctrl, text="",
                                            font=ctk.CTkFont(family='Menlo', size=10),
                                            text_color=self._FG_DIM,
                                            anchor='w')
        self.progress_label.grid(row=row, column=0, columnspan=2, sticky='ew', pady=(0, 8))
        row += 1

        ctk.CTkFrame(sb, height=1, fg_color=self._BORDER).grid(row=5, column=0, sticky='ew')

        ctk.CTkButton(sb, text="↓  Import MIDI", command=self._import_midi,
                      height=36, fg_color='transparent',
                      hover_color='#202225',
                      border_width=1, border_color=self._BORDER,
                      text_color=self._FG_DIM,
                      font=ctk.CTkFont(size=11)).grid(
            row=6, column=0, padx=12, pady=10, sticky='ew')

    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color=self._BG, corner_radius=0)
        main.grid(row=0, column=1, sticky='nsew')
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(0, weight=1)

        # Canvas
        canvas_frame = ctk.CTkFrame(main, fg_color=self._BG, corner_radius=0)
        canvas_frame.grid(row=0, column=0, sticky='nsew')
        canvas_frame.grid_columnconfigure(0, weight=1)
        canvas_frame.grid_rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(10, 4), dpi=100, facecolor=self._BG)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self._BG)
        self.ax.tick_params(colors=self._FG_DIM, labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color(self._BORDER)
        self.fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.14)

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('resize_event', self._on_resize)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)

        # Bottom controls — pill style
        bottom = ctk.CTkFrame(main, fg_color=self._BG, height=68, corner_radius=0)
        bottom.grid(row=1, column=0, sticky='ew')
        bottom.grid_propagate(False)

        pill = ctk.CTkFrame(bottom, fg_color=self._CARD, corner_radius=40,
                             border_width=1, border_color=self._BORDER)
        pill.place(relx=0.5, rely=0.42, anchor='center')

        ctk.CTkButton(pill, text="◀  PREV  (1)", command=self._prev,
                      width=110, height=36,
                      fg_color='transparent', hover_color='#202225',
                      text_color=self._FG_DIM,
                      font=ctk.CTkFont(size=11),
                      corner_radius=40).pack(side='left', padx=(6, 2), pady=6)

        ctk.CTkButton(pill, text="ACCEPT  (space)", command=self._accept,
                      width=130, height=36,
                      font=ctk.CTkFont(size=12, weight="bold"),
                      corner_radius=40).pack(side='left', padx=4)

        ctk.CTkButton(pill, text="NEXT  (2)  ▶", command=self._next,
                      width=110, height=36,
                      fg_color='transparent', hover_color='#202225',
                      text_color=self._FG_DIM,
                      font=ctk.CTkFont(size=11),
                      corner_radius=40).pack(side='left', padx=(2, 6), pady=6)

        ctk.CTkButton(bottom, text="ACCEPT ALL REMAINING", command=self._accept_all,
                      width=180, height=22,
                      fg_color='transparent', hover_color='#202225',
                      text_color=self._FG_DIM,
                      font=ctk.CTkFont(size=9, weight="bold")).place(
            relx=0.5, rely=0.88, anchor='center')

        # Status bar
        status_bar = ctk.CTkFrame(main, fg_color=self._PANEL, height=28, corner_radius=0)
        status_bar.grid(row=2, column=0, sticky='ew')
        status_bar.grid_propagate(False)

        self._st_pending = ctk.CTkLabel(status_bar, text="● Pending: --",
                                         font=ctk.CTkFont(family='Menlo', size=10),
                                         text_color=self._AMBER)
        self._st_pending.pack(side='left', padx=(14, 12), pady=6)

        self._st_approved = ctk.CTkLabel(status_bar, text="● Approved: --",
                                          font=ctk.CTkFont(family='Menlo', size=10),
                                          text_color=self._GREEN)
        self._st_approved.pack(side='left', padx=12)

        self._st_manual = ctk.CTkLabel(status_bar, text="● Manual: --",
                                        font=ctk.CTkFont(family='Menlo', size=10),
                                        text_color=self._FG_DIM)
        self._st_manual.pack(side='left', padx=12)

    def _build_menu(self):
        menubar = tk.Menu(self)
        self.configure(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Project", command=self._new_project)
        file_menu.add_command(label="Open Project...", command=self._open_project)
        file_menu.add_command(label="Save Project", command=self._save_project)
        file_menu.add_command(label="Save Project As...", command=self._save_project_as)
        file_menu.add_separator()
        file_menu.add_command(label="Import MIDI...", command=self._import_midi)
        file_menu.add_separator()
        file_menu.add_command(label="Export All...", command=self._export_all)

        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Output PPQ...", command=self._set_ppq)
        settings_menu.add_command(label="Tick Sample...", command=self._set_tick)
        settings_menu.add_separator()
        settings_menu.add_checkbutton(label="Fixed 120 BPM MIDI Export",
                                      variable=self._fixed_bpm_var)

    def _bind_keys(self):
        self.bind_all('<space>', self._on_key_accept)
        self.bind_all('<Return>', self._on_key_return)
        self.bind_all('<Left>', self._on_key_prev)
        self.bind_all('<Right>', self._on_key_next)
        self.bind_all('1', self._on_key_prev)
        self.bind_all('2', self._on_key_next)
        self.bind_all('<minus>', self._on_key_zoom_out)
        self.bind_all('<equal>', self._on_key_zoom_in)
        self.bind_all('<plus>', self._on_key_zoom_in)
        self.bind_all('q', self._on_key_place)
        self.bind_all('w', self._on_key_wide_press)
        self.bind_all('<KeyRelease-w>', self._on_key_wide_release)

    # ── Sidebar element list ─────────────────────────────────────────────

    def _start_blink(self):
        self._blink_state = not self._blink_state
        color = self._AMBER if self._blink_state else '#7a5500'
        for btn in self._blink_btns:
            try:
                if btn.winfo_exists():
                    btn.configure(text_color=color)
            except Exception:
                pass
        self.after(700, self._start_blink)

    def _rebuild_sidebar(self):
        for w in self._elem_scroll.winfo_children():
            w.destroy()
        self._sidebar_btns = []
        self._blink_btns = []

        for i, elem in enumerate(self.elements):
            is_sel = (i == self.current_element_idx)

            if elem.status:
                total = len(elem.status)
                done  = sum(1 for s in elem.status if s in ('approved', 'manual'))
                badge = f"✓ " if done == total else f"{done}/{total}  "
                badge_color = self._GREEN if done == total else self._AMBER
            else:
                badge = ""
                badge_color = self._FG_DIM

            row_frame = ctk.CTkFrame(self._elem_scroll, fg_color='transparent')
            row_frame.grid(row=i, column=0, sticky='ew', pady=1)
            row_frame.grid_columnconfigure(0, weight=1)

            btn = ctk.CTkButton(
                row_frame, text=f"  {elem.name}", anchor='w', height=32,
                fg_color='#1e2028' if is_sel else 'transparent',
                hover_color='#1e2028',
                text_color=self._FG if is_sel else '#9ea3b0',
                border_width=1 if is_sel else 0,
                border_color=self._BLUE if is_sel else self._PANEL,
                font=ctk.CTkFont(size=12),
                command=lambda idx=i: self._select_element(idx)
            )
            btn.grid(row=0, column=0, sticky='ew')

            has_audio = elem.audio_path is not None
            audio_btn = ctk.CTkButton(
                row_frame, text="●" if has_audio else "⊕",
                width=30, height=32,
                fg_color='transparent',
                hover_color='#2a1a00' if not has_audio else '#202225',
                text_color=self._GREEN if has_audio else self._AMBER,
                border_width=1 if not has_audio else 0,
                border_color=self._AMBER if not has_audio else self._PANEL,
                font=ctk.CTkFont(size=12 if not has_audio else 11),
                command=lambda idx=i: self._link_audio(idx)
            )
            audio_btn.grid(row=0, column=1, padx=(0, 2))
            if not has_audio:
                self._blink_btns.append(audio_btn)

            if badge:
                ctk.CTkLabel(row_frame, text=badge,
                             font=ctk.CTkFont(family='Menlo', size=9),
                             text_color=badge_color,
                             fg_color='transparent').grid(row=0, column=2, padx=(0, 6))

            self._sidebar_btns.append(btn)

    def _select_element(self, idx):
        if 0 <= idx < len(self.elements):
            self.current_element_idx = idx
            self._update_track_combo()
            self._show_current()
            self.after(0, self._rebuild_sidebar)

    # ── Key handlers ────────────────────────────────────────────────────

    def _is_typing(self, event):
        w = event.widget
        return isinstance(w, (tk.Entry, tk.Text))

    def _on_key_accept(self, event):
        if not self._is_typing(event): self._accept()

    def _on_key_return(self, event):
        if not self._is_typing(event): self._accept()

    def _on_key_prev(self, event):
        if not self._is_typing(event): self._prev()

    def _on_key_next(self, event):
        if not self._is_typing(event): self._next()

    def _on_key_zoom_out(self, event):
        if not self._is_typing(event): self._zoom_out()

    def _on_key_zoom_in(self, event):
        if not self._is_typing(event): self._zoom_in()

    def _on_key_wide_press(self, event):
        if self._is_typing(event):
            return
        if hasattr(self, '_wide_zoom_active') and self._wide_zoom_active:
            return  # already active (key repeat)
        self._wide_zoom_active = True
        self._saved_display_ms = self.display_ms
        self.display_ms = 1000.0
        self.zoom_var.set(min(self.display_ms, 200.0))
        self.zoom_label.configure(text=f"±{self.display_ms:.0f}ms")
        self._show_current()

    def _on_key_wide_release(self, _event):
        if not hasattr(self, '_wide_zoom_active') or not self._wide_zoom_active:
            return
        self._wide_zoom_active = False
        self.display_ms = self._saved_display_ms
        self.zoom_var.set(self.display_ms)
        self.zoom_label.configure(text=f"±{self.display_ms:.1f}ms")
        self._show_current()

    def _on_key_place(self, event):
        if self._is_typing(event):
            return
        elem = self._current_element()
        if elem is None or self._last_mouse_xdata is None:
            return
        mi = elem.current_idx
        current_pos  = int(elem.final_positions[mi])
        click_offset = int(self._last_mouse_xdata / 1000 * elem.sr)
        new_pos = max(0, min(len(elem.audio) - 1, current_pos + click_offset))
        elem.final_positions[mi] = new_pos
        elem.status[mi] = 'manual'
        if elem.current_idx < elem.n_markers - 1:
            self._next()
        else:
            self._show_current()

    # ── Current element helpers ──────────────────────────────────────────

    def _current_element(self):
        if not self.elements or self.current_element_idx >= len(self.elements):
            return None
        return self.elements[self.current_element_idx]

    # ── Project management ───────────────────────────────────────────────

    def _new_project(self):
        if self.elements:
            if not messagebox.askyesno("New Project", "Discard current project?"):
                return
        self.project_path = None
        self.elements = []
        self.current_element_idx = 0
        self.tempo_map = None
        self.tick = None
        self.output_ppq = 28800
        self._track_idx_map = {}
        self._track_combo_values = []
        self._rebuild_sidebar()
        self.track_combo.configure(values=[])
        self.track_combo.set('')
        self.marker_var.set('')
        self.display_ms = 5.0
        self.zoom_var.set(self.display_ms)
        self.zoom_label.configure(text=f"±{self.display_ms:.1f}ms")
        self.ax.clear()
        self.canvas.draw()
        self.info_label.configure(text="")
        self.progress_label.configure(text="")
        self._st_pending.configure(text="● Pending: --")
        self._st_approved.configure(text="● Approved: --")
        self._st_manual.configure(text="● Manual: --")
        self.title("Transient Snap — New Project")
        self._import_midi()

    def _open_project(self):
        path = filedialog.askopenfilename(
            title="Open Project",
            filetypes=[("Transient Snap Project", "*.tsproj"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            self.elements = [KitElement.from_dict(d) for d in data.get('elements', [])]
            self.current_element_idx = data.get('current_element_idx', 0)
            self.tempo_map = data.get('tempo_map')
            self.output_ppq = data.get('output_ppq', 28800)
            self.project_path = path

            for elem in self.elements:
                if elem.audio_path and Path(elem.audio_path).exists():
                    elem.audio, elem.sr = load_audio(elem.audio_path, target_sr=DEFAULT_SR)

            tick_path = data.get('tick_path', DEFAULT_TICK)
            self.tick_path = tick_path
            if Path(tick_path).exists():
                self.tick, _ = load_audio(tick_path, target_sr=DEFAULT_SR)

            self._rebuild_sidebar()
            if self.elements:
                self._show_current()

            self.title(f"Transient Snap — {Path(path).stem}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load project:\n{e}")

    def _save_project(self):
        if not self.project_path:
            self._save_project_as()
            return
        self._do_save(self.project_path)

    def _save_project_as(self):
        path = filedialog.asksaveasfilename(
            title="Save Project",
            defaultextension=".tsproj",
            filetypes=[("Transient Snap Project", "*.tsproj")])
        if not path:
            return
        self._do_save(path)
        self.project_path = path
        self.title(f"Transient Snap — {Path(path).stem}")

    def _do_save(self, path):
        data = {
            'elements': [e.to_dict() for e in self.elements],
            'current_element_idx': self.current_element_idx,
            'tempo_map': self.tempo_map,
            'tick_path': self.tick_path,
            'output_ppq': self.output_ppq,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        messagebox.showinfo("Saved", f"Project saved to:\n{path}")

    # ── Import MIDI / link audio ──────────────────────────────────────────

    def _import_midi(self):
        """Import a MIDI file and auto-generate kit elements from its tracks."""
        if self.tempo_map is not None:
            use_existing = messagebox.askyesnocancel(
                "MIDI Source",
                "Use existing MIDI file?\n\nYes = Use existing\nNo = Select new file")
            if use_existing is None:
                return
            if use_existing:
                midi_path = next((e.midi_path for e in self.elements if e.midi_path), None)
                if not midi_path:
                    messagebox.showwarning("No MIDI", "No MIDI file in project.")
                    return
            else:
                midi_path = filedialog.askopenfilename(
                    title="Select MIDI file",
                    filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")])
                if not midi_path:
                    return
                self.tempo_map = extract_tempo_map(midi_path)
        else:
            midi_path = filedialog.askopenfilename(
                title="Select MIDI file",
                filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")])
            if not midi_path:
                return
            self.tempo_map = extract_tempo_map(midi_path)

        tracks = get_midi_track_info(midi_path)
        tracks_with_notes = [t for t in tracks if t['note_count'] > 0]
        if not tracks_with_notes:
            messagebox.showwarning("MIDI", "No tracks with notes found.")
            return

        groups = self._show_import_midi_dialog(tracks_with_notes)
        if not groups:
            return

        for group in groups:
            elem = KitElement(group['name'])
            elem.midi_path = midi_path
            elem.track_indices = group['track_indices']
            self.elements.append(elem)

        self.current_element_idx = max(0, len(self.elements) - len(groups))
        self._rebuild_sidebar()
        self._show_current()

    def _show_import_midi_dialog(self, tracks_with_notes):
        """Show dialog to assign MIDI tracks to kit elements.

        Returns list of {'name': str, 'track_indices': [int]} or None if cancelled.
        Tracks sharing the same element name are merged into one element.
        """
        dialog = ctk.CTkToplevel(self)
        dialog.title("Import MIDI — Assign Kit Elements")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, True)
        dialog.after(100, dialog.lift)

        ctk.CTkLabel(dialog,
                     text="Assign tracks to kit elements. Tracks with the same element\n"
                          "name will be merged. Leave blank to skip a track.",
                     text_color=self._FG_DIM, justify='left').pack(
            anchor='w', padx=20, pady=(20, 10))

        # Column header
        hdr = ctk.CTkFrame(dialog, fg_color=self._CARD, corner_radius=6)
        hdr.pack(fill='x', padx=20, pady=(0, 2))
        hdr.grid_columnconfigure(0, minsize=210)
        hdr.grid_columnconfigure(1, minsize=60)
        hdr.grid_columnconfigure(2, weight=1)
        for col, text in enumerate(["MIDI Track", "Notes", "Kit Element  (blank = skip)"]):
            ctk.CTkLabel(hdr, text=text,
                         font=ctk.CTkFont(size=10, weight="bold"),
                         text_color=self._FG_DIM).grid(
                row=0, column=col,
                sticky='w', padx=(12 if col == 0 else 4, 12 if col == 2 else 4),
                pady=6)

        # Track rows
        scroll_h = min(360, len(tracks_with_notes) * 44 + 16)
        scroll = ctk.CTkScrollableFrame(dialog, fg_color=self._PANEL,
                                        height=scroll_h, corner_radius=6)
        scroll.pack(fill='x', padx=20, pady=(0, 14))
        scroll.grid_columnconfigure(0, minsize=210)
        scroll.grid_columnconfigure(1, minsize=60)
        scroll.grid_columnconfigure(2, weight=1)

        entry_vars = []
        for i, track in enumerate(tracks_with_notes):
            var = tk.StringVar(value=_guess_element_name(track['name']))
            entry_vars.append((track, var))

            ctk.CTkLabel(scroll, text=track['name'],
                         font=ctk.CTkFont(family='Menlo', size=11),
                         text_color=self._FG, anchor='w').grid(
                row=i, column=0, sticky='w', padx=(12, 4), pady=4)

            ctk.CTkLabel(scroll, text=str(track['note_count']),
                         font=ctk.CTkFont(family='Menlo', size=11),
                         text_color=self._FG_DIM, anchor='e').grid(
                row=i, column=1, sticky='e', padx=4, pady=4)

            ctk.CTkEntry(scroll, textvariable=var, height=28,
                         fg_color=self._ENTRY, border_color=self._BORDER,
                         font=ctk.CTkFont(size=11)).grid(
                row=i, column=2, sticky='ew', padx=(4, 12), pady=4)

        result = [None]

        def on_import():
            groups_dict = {}
            for track, var in entry_vars:
                name = var.get().strip()
                if name:
                    groups_dict.setdefault(name, []).append(track['index'])
            if not groups_dict:
                messagebox.showwarning("No Elements", "No kit elements defined.", parent=dialog)
                return
            result[0] = [{'name': n, 'track_indices': idxs}
                         for n, idxs in groups_dict.items()]
            dialog.destroy()

        btn_frame = ctk.CTkFrame(dialog, fg_color='transparent')
        btn_frame.pack(pady=(0, 20))
        ctk.CTkButton(btn_frame, text="Cancel", command=dialog.destroy,
                      width=90, fg_color=self._ENTRY, hover_color='#303338',
                      text_color=self._FG).pack(side='left', padx=6)
        ctk.CTkButton(btn_frame, text="Import →", command=on_import,
                      width=100).pack(side='left', padx=6)

        dialog.wait_window()
        return result[0]

    def _set_ppq(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Output PPQ")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)
        dialog.after(100, dialog.lift)

        ctk.CTkLabel(dialog, text="MIDI output PPQ (pulses per quarter note):",
                     text_color=self._FG).pack(padx=20, pady=(20, 8))

        ppq_var = tk.StringVar(value=str(self.output_ppq))
        entry = ctk.CTkEntry(dialog, textvariable=ppq_var, width=140,
                             fg_color=self._ENTRY, border_color=self._BORDER)
        entry.pack(padx=20, pady=(0, 10))
        entry.select_range(0, tk.END)
        entry.focus_set()

        preset_frame = ctk.CTkFrame(dialog, fg_color='transparent')
        preset_frame.pack(pady=(0, 10))
        ctk.CTkLabel(preset_frame, text="Presets:", text_color=self._FG_DIM).pack(side='left', padx=(0, 6))
        for val in [480, 960, 9600, 28800]:
            ctk.CTkButton(preset_frame, text=str(val),
                          command=lambda v=val: ppq_var.set(str(v)),
                          width=62, height=26,
                          fg_color=self._ENTRY, hover_color='#303338',
                          text_color=self._FG,
                          font=ctk.CTkFont(size=11)).pack(side='left', padx=2)

        def on_ok():
            try:
                val = int(ppq_var.get())
                if val < 1:
                    raise ValueError("PPQ must be positive")
                self.output_ppq = val
                dialog.destroy()
            except ValueError as e:
                messagebox.showerror("Invalid", f"Invalid PPQ value: {e}")

        entry.bind('<Return>', lambda _: on_ok())

        btn_frame = ctk.CTkFrame(dialog, fg_color='transparent')
        btn_frame.pack(pady=(0, 16))
        ctk.CTkButton(btn_frame, text="OK", command=on_ok, width=80).pack(side='left', padx=6)
        ctk.CTkButton(btn_frame, text="Cancel", command=dialog.destroy, width=80,
                      fg_color=self._ENTRY, hover_color='#303338',
                      text_color=self._FG).pack(side='left', padx=6)

        dialog.wait_window()

    def _set_tick(self):
        """Choose the tick template WAV used for WAV export."""
        path = filedialog.askopenfilename(
            title="Select Tick Template WAV",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.configure(cursor="watch")
            self.update()
            self.tick, _ = load_audio(path, target_sr=DEFAULT_SR)
            self.tick_path = path
            messagebox.showinfo("Tick Sample", f"Tick set to:\n{Path(path).name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load tick:\n{e}")
        finally:
            self.configure(cursor="")

    # ── Track combo ──────────────────────────────────────────────────────

    def _update_track_combo(self):
        elem = self._current_element()
        if elem is None or elem.track_assignments is None:
            self._track_combo_values = []
            self.track_combo.configure(values=[])
            self.track_combo.set('')
            return

        unique_tracks = []
        seen = set()
        for trk_idx in elem.track_assignments:
            if trk_idx not in seen:
                seen.add(trk_idx)
                unique_tracks.append(trk_idx)

        if len(unique_tracks) <= 1:
            self._track_combo_values = []
            self.track_combo.configure(values=[])
            self.track_combo.set('')
            return

        track_names = [elem.track_names.get(i, f"Track {i}") for i in unique_tracks]
        self._track_combo_values = track_names
        self.track_combo.configure(values=track_names)
        self._track_idx_map = {name: idx for name, idx in zip(track_names, unique_tracks)}

    def _on_track_change(self, _=None):
        elem = self._current_element()
        if elem is None or not hasattr(self, '_track_idx_map'):
            return
        track_name = self.track_var.get()
        if track_name not in self._track_idx_map:
            return
        target_track = self._track_idx_map[track_name]
        for i, trk_idx in enumerate(elem.track_assignments):
            if trk_idx == target_track:
                elem.current_idx = i
                self._show_current()
                return

    # ── Display ──────────────────────────────────────────────────────────

    def _show_current(self):
        elem = self._current_element()
        if elem is None or elem.audio is None or elem.n_markers == 0:
            self.ax.clear()
            self.ax.set_facecolor(self._BG)
            for spine in self.ax.spines.values():
                spine.set_color(self._BORDER)
            if elem is not None and elem.audio_path is None:
                self.ax.set_title(f"No audio linked for '{elem.name}'",
                                  color=self._AMBER, fontsize=12, pad=10)
                self.ax.text(0.5, 0.5,
                             "Click  ○  next to the element name to link an audio file.",
                             transform=self.ax.transAxes, ha='center', va='center',
                             color=self._FG_DIM, fontsize=10)
            else:
                self.ax.set_title("No markers loaded", color=self._FG_DIM, fontsize=11)
            self.canvas.draw()
            return

        mi   = elem.current_idx
        pos  = int(elem.final_positions[mi])
        orig = int(elem.original_positions[mi])
        display_samples = int(elem.sr * self.display_ms / 1000)

        audio_start = max(0, pos - display_samples)
        audio_end   = min(len(elem.audio), pos + display_samples)
        samples  = np.arange(audio_start, audio_end)
        time_ms  = (samples - pos) / elem.sr * 1000

        self.ax.clear()
        self.ax.set_facecolor(self._BG)

        self.ax.plot(time_ms, elem.audio[audio_start:audio_end],
                     color='#4e5870', linewidth=0.6)

        orig_ms = (orig - pos) / elem.sr * 1000
        self.ax.axvline(orig_ms, color='#ef4444', linewidth=1.5,
                        linestyle='--', alpha=0.75, label='Original')
        self.ax.axvline(0, color=self._GREEN, linewidth=2, alpha=0.65, label='Current')

        self.ax.set_xlabel('ms', color=self._FG_DIM, fontsize=8)
        self.ax.set_xlim(-self.display_ms, self.display_ms)
        self.ax.tick_params(colors=self._FG_DIM, labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color(self._BORDER)
        self.ax.legend(loc='upper right', fontsize=8,
                       facecolor=self._CARD, edgecolor=self._BORDER,
                       labelcolor=self._FG_DIM)

        STATUS_COLORS = {'pending': self._AMBER, 'approved': self._GREEN, 'manual': self._BLUE}
        status = elem.status[mi]
        track_info = ""
        if elem.track_assignments is not None and len(set(elem.track_assignments)) > 1:
            trk_idx  = elem.track_assignments[mi]
            trk_name = elem.track_names.get(trk_idx, f"Track {trk_idx}")
            track_info = f" [{trk_name}]"
        self.ax.set_title(
            f"{elem.name}{track_info}  ·  Marker {mi+1}/{elem.n_markers}  ·  {status.upper()}",
            color=STATUS_COLORS.get(status, self._FG), fontsize=11, pad=6
        )

        self.fig.tight_layout(rect=[0, 0, 1, 1])
        self.canvas.draw()
        self.canvas_bg   = self.canvas.copy_from_bbox(self.ax.bbox)
        self.cursor_line = self.ax.axvline(0, color='white', linewidth=0.5, alpha=0.3)
        self.cursor_line.set_visible(False)

        if self._last_mouse_xdata is not None:
            self.cursor_line.set_xdata([self._last_mouse_xdata])
            self.cursor_line.set_visible(True)
            self.canvas.restore_region(self.canvas_bg)
            self.ax.draw_artist(self.cursor_line)
            self.canvas.blit(self.ax.bbox)

        # Sidebar info + progress
        offset_ms = (pos - orig) / elem.sr * 1000
        time_s    = orig / elem.sr
        track_label = ""
        if elem.track_assignments is not None and len(set(elem.track_assignments)) > 1:
            trk_idx  = elem.track_assignments[mi]
            trk_name = elem.track_names.get(trk_idx, f"Track {trk_idx}")
            track_label = f"\n{trk_name}"
        self.info_label.configure(
            text=f"@ {time_s:.2f}s  |  {offset_ms:+.2f}ms{track_label}")

        reviewed = sum(1 for s in elem.status if s in ('approved', 'manual'))
        if reviewed == elem.n_markers:
            self.progress_label.configure(
                text=f"✓ All {elem.n_markers} reviewed", text_color=self._GREEN)
        else:
            self.progress_label.configure(
                text=f"Reviewed: {reviewed}/{elem.n_markers}", text_color=self._FG_DIM)

        # Status bar
        pending  = sum(1 for s in elem.status if s == 'pending')
        approved = sum(1 for s in elem.status if s == 'approved')
        manual   = sum(1 for s in elem.status if s == 'manual')
        self._st_pending.configure(text=f"● Pending: {pending}")
        self._st_approved.configure(text=f"● Approved: {approved}")
        self._st_manual.configure(text=f"● Manual: {manual}")

        # Marker entry + track combo
        self.marker_var.set(str(mi + 1))

        if elem.track_assignments is not None and len(set(elem.track_assignments)) > 1:
            if not self._track_combo_values:
                self._update_track_combo()
            trk_idx  = elem.track_assignments[mi]
            trk_name = elem.track_names.get(trk_idx, f"Track {trk_idx}")
            if trk_name in self._track_combo_values:
                self.track_combo.set(trk_name)
        else:
            self.track_combo.set('')
            self._track_combo_values = []
            self.track_combo.configure(values=[])

        # Rebuild sidebar to update badges
        self._rebuild_sidebar()

    def _link_audio(self, idx):
        """Link an audio file to a kit element and load its markers."""
        if idx >= len(self.elements):
            return
        elem = self.elements[idx]

        audio_path = filedialog.askopenfilename(
            title=f"Select audio for '{elem.name}'",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if not audio_path:
            return

        if self.tick is None:
            tick_path = self.tick_path
            if not Path(tick_path).exists():
                tick_path = filedialog.askopenfilename(
                    title="Select tick template WAV",
                    filetypes=[("WAV files", "*.wav")])
                if not tick_path:
                    return
            self.tick_path = tick_path
            self.tick, _ = load_audio(tick_path, target_sr=DEFAULT_SR)

        self.configure(cursor="watch")
        self.update()
        try:
            elem.audio_path = audio_path
            elem.audio, elem.sr = load_audio(audio_path, target_sr=DEFAULT_SR)
            (elem.original_positions, elem.amplitudes,
             elem.track_assignments, elem.note_values,
             elem.track_names) = load_markers_midi(
                elem.midi_path, elem.sr, track_indices=elem.track_indices)

            if elem.track_assignments is not None and len(elem.track_assignments) > 0:
                sort_idx = np.lexsort((elem.original_positions, elem.track_assignments))
                elem.original_positions = elem.original_positions[sort_idx]
                elem.amplitudes         = elem.amplitudes[sort_idx]
                elem.track_assignments  = elem.track_assignments[sort_idx]
                elem.note_values        = elem.note_values[sort_idx]

            elem.final_positions = elem.original_positions.copy()
            elem.status          = ['pending'] * elem.n_markers
            elem.current_idx     = 0

            self.current_element_idx = idx
            self._update_track_combo()
            self._rebuild_sidebar()
            self._show_current()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load audio:\n{e}")
        finally:
            self.configure(cursor="")

    # ── Zoom ─────────────────────────────────────────────────────────────

    def _on_zoom(self, val):
        self.display_ms = float(val)
        self.zoom_label.configure(text=f"±{self.display_ms:.1f}ms")
        self._show_current()

    def _zoom_in(self):
        self.display_ms = max(1.0, self.display_ms - 1.0)
        self.zoom_var.set(self.display_ms)
        self._on_zoom(self.display_ms)

    def _zoom_out(self):
        self.display_ms = min(200.0, self.display_ms + 1.0)
        self.zoom_var.set(self.display_ms)
        self._on_zoom(self.display_ms)

    # ── Canvas events ────────────────────────────────────────────────────

    def _on_resize(self, event):
        if self._current_element() is None or self._resize_pending:
            return
        self._resize_pending = True
        self.after(100, self._do_resize_redraw)

    def _do_resize_redraw(self):
        self._resize_pending = False
        if self._current_element() is not None:
            self._show_current()

    def _on_scroll(self, event):
        if self._current_element() is None:
            return
        if event.button == 'up':
            self._zoom_in()
        elif event.button == 'down':
            self._zoom_out()

    def _on_motion(self, event):
        if self.cursor_line is None or self.canvas_bg is None:
            return
        if event.inaxes != self.ax or event.xdata is None:
            self._last_mouse_xdata = None
            if self.cursor_line.get_visible():
                self.cursor_line.set_visible(False)
                self.canvas.restore_region(self.canvas_bg)
                self.canvas.blit(self.ax.bbox)
            return
        self._last_mouse_xdata = event.xdata
        self.cursor_line.set_xdata([event.xdata])
        self.cursor_line.set_visible(True)
        self.canvas.restore_region(self.canvas_bg)
        self.ax.draw_artist(self.cursor_line)
        self.canvas.blit(self.ax.bbox)

    def _on_click(self, event):
        elem = self._current_element()
        if elem is None or event.inaxes != self.ax or event.xdata is None:
            self.focus_force()
            return

        mi          = elem.current_idx
        current_pos = int(elem.final_positions[mi])
        click_offset = int(event.xdata / 1000 * elem.sr)
        new_pos     = max(0, min(len(elem.audio) - 1, current_pos + click_offset))

        elem.final_positions[mi] = new_pos
        elem.status[mi] = 'manual'

        if elem.current_idx < elem.n_markers - 1:
            self._next()
        else:
            self._show_current()
        self.focus_force()

    # ── Navigation ───────────────────────────────────────────────────────

    def _next(self):
        elem = self._current_element()
        if elem is not None and elem.current_idx < elem.n_markers - 1:
            elem.current_idx += 1
            self._show_current()

    def _prev(self):
        elem = self._current_element()
        if elem is not None and elem.current_idx > 0:
            elem.current_idx -= 1
            self._show_current()

    def _accept(self):
        elem = self._current_element()
        if elem is None:
            return
        mi = elem.current_idx
        if elem.status[mi] == 'pending':
            elem.status[mi] = 'approved'
        self._next()

    def _accept_all(self):
        elem = self._current_element()
        if elem is None:
            return
        for i in range(elem.n_markers):
            if elem.status[i] == 'pending':
                elem.status[i] = 'approved'
        self._show_current()

    def _on_marker_jump(self, _=None):
        elem = self._current_element()
        if elem is None:
            return
        try:
            idx = int(self.marker_var.get()) - 1
            if 0 <= idx < elem.n_markers:
                elem.current_idx = idx
                self._show_current()
        except ValueError:
            pass

    def _goto_first_pending(self):
        elem = self._current_element()
        if elem is None:
            return
        for i, s in enumerate(elem.status):
            if s == 'pending':
                elem.current_idx = i
                self._show_current()
                return
        messagebox.showinfo("Done", f"All markers for '{elem.name}' have been reviewed!")

    # ── Export ───────────────────────────────────────────────────────────

    def _export_all(self):
        if not self.elements:
            messagebox.showwarning("Export", "No elements to export.")
            return

        output_path = filedialog.asksaveasfilename(
            title="Export Combined MIDI",
            defaultextension=".mid",
            initialfile="refined_markers.mid",
            filetypes=[("MIDI files", "*.mid")])
        if not output_path:
            return

        output_dir = Path(output_path).parent

        try:
            all_positions       = []
            all_amplitudes      = []
            all_track_assignments = []
            all_note_values     = []
            combined_track_names = {}

            output_track_idx = 0
            track_mapping = {}

            for elem_idx, elem in enumerate(self.elements):
                if elem.n_markers == 0:
                    continue
                if elem.track_assignments is not None:
                    unique_orig_tracks = []
                    seen = set()
                    for t in elem.track_assignments:
                        if t not in seen:
                            seen.add(t)
                            unique_orig_tracks.append(t)
                else:
                    unique_orig_tracks = [0]

                for orig_track in unique_orig_tracks:
                    track_mapping[(elem_idx, orig_track)] = output_track_idx
                    if elem.track_names and orig_track in elem.track_names:
                        combined_track_names[output_track_idx] = elem.track_names[orig_track]
                    else:
                        combined_track_names[output_track_idx] = elem.name
                    output_track_idx += 1

            for elem_idx, elem in enumerate(self.elements):
                if elem.n_markers == 0:
                    continue
                for i in range(elem.n_markers):
                    all_positions.append(elem.final_positions[i])
                    all_amplitudes.append(elem.amplitudes[i] if elem.amplitudes is not None else 1.0)
                    orig_track = elem.track_assignments[i] if elem.track_assignments is not None else 0
                    out_track  = track_mapping.get((elem_idx, orig_track), 0)
                    all_track_assignments.append(out_track)
                    all_note_values.append(elem.note_values[i] if elem.note_values is not None else 36)

            if all_positions:
                if self._fixed_bpm_var.get() or self.tempo_map is None:
                    # Fixed 120 BPM — positions converted purely from sample time,
                    # avoids inaccuracies from complex tempo maps in the DAW.
                    output_tempo_map = {'ppq': self.output_ppq, 'tempo_events': [(0, 500000)]}
                else:
                    orig_ppq = self.tempo_map['ppq']
                    scale    = self.output_ppq / orig_ppq
                    rescaled_events = [
                        (int(round(tick * scale)), tempo_us)
                        for tick, tempo_us in self.tempo_map['tempo_events']
                    ]
                    output_tempo_map = {'ppq': self.output_ppq, 'tempo_events': rescaled_events}

                save_markers_midi(
                    np.array(all_positions, dtype=np.int64),
                    DEFAULT_SR, output_path,
                    amplitudes=np.array(all_amplitudes, dtype=np.float32),
                    tempo_map=output_tempo_map,
                    track_assignments=np.array(all_track_assignments, dtype=np.int32),
                    note_values=np.array(all_note_values, dtype=np.int32),
                    track_names=combined_track_names,
                    debug=True, force_high_ppq=False)

            wav_files = []
            for elem in self.elements:
                if elem.n_markers == 0:
                    continue
                wav_name = f"{elem.name.replace(' ', '_')}_refined.wav"
                wav_path = output_dir / wav_name
                save_markers_wav(
                    elem.final_positions, elem.sr, str(wav_path),
                    self.tick, len(elem.audio), elem.amplitudes)
                wav_files.append(wav_name)

            wav_list = '\n'.join(wav_files) if wav_files else "(none)"
            messagebox.showinfo("Exported",
                f"Exported to:\n{output_path}\n\nWAV files:\n{wav_list}")

        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{e}")


def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = TransientSnapV2()
    app.mainloop()


if __name__ == '__main__':
    main()
