#!/usr/bin/env python3
"""
generate_references.py
─────────────────────────────────────────────────────────────────────────────
  Туршилтын reference audio файл үүсгэх скрипт.
  Жинхэнэ файл байгаа бол энийг ажиллуулах шаардлагагүй.

  Хэрэглэх:
    pip install soundfile numpy scipy
    python generate_references.py

  references/ хавтсанд дараах файлууд үүснэ:
    references/riptide.wav         — Am-G-C strumming, 110 BPM
    references/acoustic.wav        — Bm-G-D-A fingerpicking, 76 BPM
"""

import os
import numpy as np
import soundfile as sf

os.makedirs("references", exist_ok=True)

SR = 22050

def guitar_string(freq, dur, brightness=0.55, sr=SR):
    """Harmonic summation + exponential decay."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    sig = np.zeros_like(t)
    for k in range(1, 9):
        amp = (brightness ** (k - 1)) / k
        sig += amp * np.sin(2 * np.pi * freq * k * t)
    decay = np.exp(-3.0 * t / dur)
    # ADSR envelope
    env = np.ones(len(sig)) * 0.7
    ai = int(0.01 * sr); di = int(0.05 * sr); ri = int(0.12 * sr)
    env[:ai] = np.linspace(0, 1, ai)
    env[ai:ai+di] = np.linspace(1, 0.7, di)
    env[max(0, len(sig)-ri):] = np.linspace(0.7, 0, min(ri, len(sig)))
    return sig * decay * env

def strum_chord(freqs, beat_dur, amp=0.7, sr=SR):
    out = np.zeros(int(sr * beat_dur))
    spread = 0.015
    for i, f in enumerate(freqs):
        offset = int(i * spread * sr)
        seg = guitar_string(f, beat_dur * 0.9) * amp
        end = min(offset + len(seg), len(out))
        out[offset:end] += seg[:end - offset]
    return out

def pick(freq, dur, amp=0.5, sr=SR):
    return guitar_string(freq, dur, brightness=0.45) * amp

def shift(freqs, semitones):
    return [f * (2 ** (semitones / 12)) for f in freqs]

# ── RIPTIDE  110 BPM, Am-G-C strumming ──────────────────────────────────────
BPM1 = 110
beat1 = 60.0 / BPM1
bar1  = beat1 * 4
Am = shift([110.00, 164.81, 220.00, 261.63, 329.63], 1)
G  = shift([98.00,  123.47, 196.00, 246.94, 293.66, 392.00], 1)
C  = shift([130.81, 164.81, 196.00, 261.63, 329.63], 1)

prog1 = [Am, G, C, C]
n_bars1 = int(3 * 60 / bar1)
rng = np.random.default_rng(42)
audio1 = []
for b in range(n_bars1):
    chord = prog1[b % len(prog1)]
    bar_audio = np.zeros(int(SR * bar1))
    for st in [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]:
        pos = int(st * beat1 * SR) + rng.integers(-200, 200)
        pos = max(0, pos)
        amp = 0.75 if st in [0, 2] else 0.50
        seg = strum_chord(chord, beat1 * 0.5, amp=amp)
        end = min(pos + len(seg), len(bar_audio))
        bar_audio[pos:end] += seg[:end - pos]
    audio1.append(bar_audio)

audio1 = np.concatenate(audio1)
audio1 /= np.max(np.abs(audio1)) + 1e-9
audio1 *= 0.85
sf.write("references/riptide.wav", audio1, SR)
print(f"✓ references/riptide.wav  ({len(audio1)/SR:.1f}s, {BPM1} BPM)")

# ── ACOUSTIC  76 BPM, Bm-G-D-A fingerpicking ────────────────────────────────
BPM2 = 76
beat2 = 60.0 / BPM2
bar2  = beat2 * 4
Bm   = [123.47, 185.00, 246.94, 293.66, 370.00]
Dmaj = [146.83, 220.00, 293.66, 369.99]
G2   = [98.00,  123.47, 196.00, 246.94, 293.66]
A    = [110.00, 138.59, 220.00, 277.18, 329.63]

prog2 = [Bm, G2, Dmaj, A]
n_bars2 = int(3 * 60 / bar2)
rng2 = np.random.default_rng(7)
audio2 = []
note_dur = beat2 * 0.5
pat_offsets = [0, 0.33, 0.66, 1.0, 1.33, 1.66, 2.0, 2.33, 2.66, 3.0, 3.33, 3.66]
pat_strings = [0, 2, -1, 2, 1, 2, -1, 2, 0, 1, -1, 2]

for b in range(n_bars2):
    chord = prog2[b % len(prog2)]
    bar_audio = np.zeros(int(SR * bar2))
    for off, sidx in zip(pat_offsets, pat_strings):
        pos = int(off * beat2 * SR) + rng2.integers(-150, 150)
        pos = max(0, pos)
        sidx = sidx % len(chord)
        seg = pick(chord[sidx], note_dur * 1.4, amp=0.6 if sidx == 0 else 0.4)
        end = min(pos + len(seg), len(bar_audio))
        bar_audio[pos:end] += seg[:end - pos]
    audio2.append(bar_audio)

audio2 = np.concatenate(audio2)
audio2 /= np.max(np.abs(audio2)) + 1e-9
audio2 *= 0.85
sf.write("references/acoustic.wav", audio2, SR)
print(f"✓ references/acoustic.wav  ({len(audio2)/SR:.1f}s, {BPM2} BPM)")
print("\nReference файлууд бэлэн боллоо. Одоо сервер ажиллуулна уу:")
print("  uvicorn main:app --host 0.0.0.0 --port 8000 --reload")