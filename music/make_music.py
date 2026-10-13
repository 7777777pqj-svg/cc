#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_music.py -- original instrumental soundtrack for 「AI 四棒接力」 (45.1 s).
(Final version, grown from gen_electro.py: same key, palette and motif.)

Style  : bright, glossy synth-pop / future-funk.  Detuned saw stabs and synth
         brass through resonant filters, plucky arps, funky synth bass with
         glides, claps, crisp metallic hats, supersaw swell, resonant sweeps.
         Purely instrumental: no voices, no vocal-formant sounds.
Key    : A major (home); parallel A minor / diminished colour for the bills.
Motif  : "the baton"  A - C# - E - F#  (1-3-5-6).  Title sparkle, the four app
         shouts (one note per app, synth brass + scooped lead), the bounce
         plinks, Clawd's checks, Gemini's bell fragments, the PASS sparkle,
         the plug run and the pull-back lead.  With the bills it turns minor:
         A - C - E - F.  At the end it deflates: woodblock tiles F - E - C - A.

One master tempo: 96 BPM (beat 0.625 s) from the apps section to the stop.
Only the beat PHASE is reset at the room cuts (11.80, 17.05*, 21.70, 25.70
PASS, 27.80, 30.95, 32.20), so the middle plays as one song.  Gemini is
half-time; its grid sits 30 ms before the cut so the three stamps fall on
16ths (beats 1.25 / 2.5 / 3.75, a dotted-8th+16th hemiola).  Off-grid story
events (checks, plugs, typed lines) are caught by stingers over the groove.
After the 35.40 tape-stop a 0.225 s minor pulse grid puts the bills on pulses
5, 9 and 13.

Everything is synthesised here (oscillators, noise, envelopes, block-wise
time-varying resonant biquads, FM, additive).  Deterministic: every random
source is seeded from a CRC of its name.
Run:  python3 make_music.py [out.wav]   -> default: soundtrack.wav next to this script.
"""
import os
import sys
import time
import zlib
from functools import lru_cache

import numpy as np
from scipy import signal as ss
from scipy.io import wavfile
from scipy.ndimage import minimum_filter1d

T_START = time.time()
SR = 48000
DUR = 45.1
N = int(round(DUR * SR))            # 2,164,800 frames
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'soundtrack.wav')
TAU = 2 * np.pi
STOP_T = 35.40                      # tape-stop / abrupt turn
BEAT = 0.625                        # 96 BPM
E8 = BEAT / 2                       # 8th note
DOT8 = BEAT * 0.75                  # dotted 8th (echo time)


# ----------------------------------------------------------------------------
# basic helpers
# ----------------------------------------------------------------------------
def rng_for(*keys):
    return np.random.default_rng(zlib.crc32(repr(keys).encode()))


def mtof(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=float) - 69.0) / 12.0)


@lru_cache(None)
def _sos(kind, f, order):
    nyq = SR / 2
    if isinstance(f, tuple):
        f = (max(f[0], 10.0), min(f[1], nyq * 0.95))
    else:
        f = min(max(f, 10.0), nyq * 0.95)
    return ss.butter(order, f, btype=kind, fs=SR, output='sos')


def lp(x, f, order=2):
    return ss.sosfilt(_sos('lowpass', float(f), order), x, axis=0)


def hp(x, f, order=2):
    return ss.sosfilt(_sos('highpass', float(f), order), x, axis=0)


def bp(x, f1, f2, order=2):
    return ss.sosfilt(_sos('bandpass', (float(f1), float(f2)), order), x, axis=0)


def fade(x, a=0.002, r=0.006):
    """raised-cosine attack / release ramps (>= 2 ms) on a mono or stereo note"""
    x = np.array(x, dtype=float, copy=True)
    n = x.shape[0]
    na = min(n, max(96, int(a * SR)))
    nr = min(n, max(96, int(r * SR)))
    w = np.ones(n)
    w[:na] = 0.5 - 0.5 * np.cos(np.pi * np.arange(na) / na)
    w[n - nr:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(1, nr + 1) / nr)
    return x * (w[:, None] if x.ndim == 2 else w)


def env_ad(n, a=0.003, d=0.2, s=0.0, gate=None, r=0.03):
    """attack (raised cosine), exponential decay to sustain s, release after gate"""
    t = np.arange(n) / SR
    a = max(a, 0.002)
    e = s + (1.0 - s) * np.exp(-np.maximum(t - a, 0.0) / max(d, 1e-4))
    na = min(n, int(a * SR))
    e[:na] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(na) / na)
    if gate is not None:
        u = np.clip((t - gate) / max(r, 0.002), 0, 1)
        e *= 0.5 + 0.5 * np.cos(np.pi * u)
    nr = min(n, 96)
    e[n - nr:] *= np.linspace(1, 0, nr)
    return e


def saw(f, n, ph0=0.0):
    """polyBLEP band-limited sawtooth; f scalar or per-sample array (Hz)"""
    dt = np.broadcast_to(np.asarray(f, dtype=float), (n,)) / SR
    p = (ph0 + np.cumsum(dt)) % 1.0
    y = 2.0 * p - 1.0
    m = p < dt
    x = p[m] / dt[m]
    y[m] -= x + x - x * x - 1.0
    m = p > 1.0 - dt
    x = (p[m] - 1.0) / dt[m]
    y[m] -= x * x + x + x + 1.0
    return y


def pulse(f, n, duty=0.5, ph0=0.0):
    return 0.5 * (saw(f, n, ph0) - saw(f, n, ph0 + duty))


def sine_f(f, n, ph0=0.0):
    f = np.broadcast_to(np.asarray(f, dtype=float), (n,))
    return np.sin(TAU * (ph0 + np.cumsum(f) / SR))


def tvf(x, fc, Q=0.707, kind='lp', block=64):
    """time-varying resonant biquad (RBJ), coefficients updated every `block`
    samples with the filter state carried across blocks (a real resonant sweep,
    not a crossfade of static filters).  fc: scalar or per-sample cutoff (Hz)."""
    n = x.shape[0]
    fc = np.broadcast_to(np.asarray(fc, dtype=float), (n,))
    nb = (n + block - 1) // block
    mid = np.minimum(np.arange(nb) * block + block // 2, n - 1)
    f = np.clip(fc[mid], 20.0, SR * 0.45)
    w0 = TAU * f / SR
    cw = np.cos(w0)
    al = np.sin(w0) / (2.0 * Q)
    a0 = 1.0 + al
    if kind == 'lp':
        b0 = (1 - cw) / 2 / a0
        b1 = (1 - cw) / a0
        b2 = b0
    elif kind == 'hp':
        b0 = (1 + cw) / 2 / a0
        b1 = -(1 + cw) / a0
        b2 = b0
    else:                                      # band-pass, 0 dB peak
        b0 = al / a0
        b1 = np.zeros_like(al)
        b2 = -al / a0
    a1 = -2 * cw / a0
    a2 = (1 - al) / a0
    y = np.empty_like(x, dtype=float)
    zi = np.zeros((2,) + x.shape[1:])
    for k in range(nb):
        i = k * block
        j = min(n, i + block)
        y[i:j], zi = ss.lfilter((b0[k], b1[k], b2[k]), (1.0, a1[k], a2[k]), x[i:j], axis=0, zi=zi)
    return y


def pan_gains(p):
    a = (np.clip(p, -1, 1) + 1) * np.pi / 4
    return np.sqrt(2) * np.cos(a), np.sqrt(2) * np.sin(a)


class Grid:
    """beat -> seconds through anchor points (linear, extrapolates)."""

    def __init__(self, pts):
        self.b = np.array([p[0] for p in pts], float)
        self.t = np.array([p[1] for p in pts], float)

    def __call__(self, beat):
        b = float(beat)
        if b <= self.b[0]:
            s = (self.t[1] - self.t[0]) / (self.b[1] - self.b[0])
            return float(self.t[0] + (b - self.b[0]) * s)
        if b >= self.b[-1]:
            s = (self.t[-1] - self.t[-2]) / (self.b[-1] - self.b[-2])
            return float(self.t[-1] + (b - self.b[-1]) * s)
        return float(np.interp(b, self.b, self.t))


def g96(t0):
    """the master 96 BPM grid with its phase reset to t0"""
    return Grid([(0, t0), (1, t0 + BEAT)])


# ----------------------------------------------------------------------------
# buses and placement
# ----------------------------------------------------------------------------
BUS = {k: np.zeros((N, 2)) for k in ('kick', 'drums', 'bass', 'music', 'pad', 'fx', 'boom', 'sendA', 'sendB')}
KICKS = []          # (time, depth) for the side-chain envelope
GAPS = []           # (time, depth, length): short suck-outs right before big hits
KICK_LEVEL = 0.56
I_STOP = int(round(STOP_T * SR))


def add(bus, sig, t, pan=0.0, gain=1.0, send=0.0, origin=None):
    """mix a mono/stereo signal into a bus at absolute time t.
    Material that starts before the 35.40 stop is cut at the stop (so the
    tape-stop really kills it); its reverb goes to sendA, later stuff to sendB."""
    if gain == 0:
        return
    origin = t if origin is None else origin
    partA = origin < STOP_T - 1e-9
    i0 = int(round(t * SR))
    n = sig.shape[0]
    if partA and i0 + n > I_STOP:
        cut = I_STOP - i0
        if cut <= 96:
            return
        sig = fade(sig[:cut], 0.0, 0.003)
        n = cut
    s0, s1 = max(i0, 0), min(i0 + n, N)
    if s1 <= s0:
        return
    seg = sig[s0 - i0:s1 - i0]
    sb = 'sendA' if partA else 'sendB'
    if seg.ndim == 1:
        gl, gr = pan_gains(pan)
        B = BUS[bus]
        B[s0:s1, 0] += seg * (gl * gain)
        B[s0:s1, 1] += seg * (gr * gain)
        if send:
            S = BUS[sb]
            S[s0:s1, 0] += seg * (gl * gain * send)
            S[s0:s1, 1] += seg * (gr * gain * send)
    else:
        if pan:
            gl, gr = pan_gains(pan)
            seg = seg * np.array([gl, gr])
        BUS[bus][s0:s1] += seg * gain
        if send:
            BUS[sb][s0:s1] += seg * (gain * send)


def add_echo(bus, sig, t, delay=DOT8, fb=0.33, taps=3, pan=0.0, gain=1.0, send=0.0, lp_hz=4800):
    """note + ping-pong echoes (each tap a little darker), tempo-synced"""
    add(bus, sig, t, pan, gain, send)
    y = sig
    for k in range(1, taps + 1):
        y = lp(y, lp_hz)
        p = (0.7 if k % 2 else -0.7) if sig.ndim == 1 else 0.0
        add(bus, y, t + k * delay, pan=p, gain=gain * fb ** k, send=send, origin=t)


# ----------------------------------------------------------------------------
# drums
# ----------------------------------------------------------------------------
@lru_cache(None)
def kick(dur=0.36, f_hi=170., f_lo=55., pdec=0.030, adec=0.13, click=0.30, drive=2.0):
    """short punchy kick; the click is band-limited to 1.5-5 kHz (no hard tick)"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f_lo + (f_hi - f_lo) * np.exp(-t / pdec)
    body = np.sin(TAU * np.cumsum(f) / SR) * np.exp(-t / adec)
    nz = rng_for('kick', f_hi).standard_normal(n) * np.exp(-t / 0.003)
    y = body + click * lp(bp(nz, 1500, 5000), 5000)
    y = np.tanh(drive * y) / np.tanh(drive)
    return fade(y, 0.002, 0.02)


def kick_at(t, gain=0.9, depth=1.0, **kw):
    add('kick', kick(**kw), t, gain=gain * KICK_LEVEL)
    KICKS.append((t, depth))


@lru_cache(None)
def snare(dur=0.26, tone=195., ndec=0.09, tdec=0.055, top=8000., key=0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    body = np.sin(TAU * np.cumsum(tone * (1 + 0.6 * np.exp(-t / 0.008))) / SR) * np.exp(-t / tdec)
    nz = rng_for('snare', key).standard_normal(n) * np.exp(-t / ndec)
    y = 0.5 * body + 0.95 * bp(nz, 1600, top) + 0.25 * bp(nz, 4000, 7500)
    return fade(y, 0.002, 0.02)


@lru_cache(None)
def clap(dur=0.36, dec=0.10, key=0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    env = np.zeros(n)
    for k, off in enumerate((0.0, 0.010, 0.021)):
        tt = t - off
        env += np.clip(tt / 0.002, 0, 1) * np.exp(-np.maximum(tt, 0) / 0.0045) * (1.0 if k == 0 else 0.75)
    tt = t - 0.030
    env += np.clip(tt / 0.002, 0, 1) * np.exp(-np.maximum(tt, 0) / dec) * 0.8
    nz = rng_for('clap', key).standard_normal(n)
    y = bp(nz * env, 900, 6500) + 0.5 * bp(nz * env, 1100, 1700) + 0.3 * bp(nz * env, 3000, 6000)
    return fade(y, 0.002, 0.02)


@lru_cache(None)
def hat(open_=False, key=0):
    """metallic hat: noise + six inharmonic pulse partials, 4.5-8 kHz band,
    gentle roll-off above 8 kHz (crisp but not fizzy)"""
    dur, dec = (0.34, 0.11) if open_ else (0.07, 0.016)
    n = int(dur * SR)
    t = np.arange(n) / SR
    rg = rng_for('hat', open_, key)
    met = np.zeros(n)
    for f in (317., 465., 541., 806., 1047., 1219.):
        met += pulse(f * 1.0 * (1 + 0.002 * key), n, 0.5, rg.random())
    x = 0.8 * rg.standard_normal(n) + 0.35 * met
    y = lp(hp(x, 4500, 2), 8200, 2)
    y = y * np.exp(-t / dec)
    return fade(y, 0.002, 0.01)


@lru_cache(None)
def shaker(key=0):
    n = int(0.09 * SR)
    t = np.arange(n) / SR
    env = np.clip(t / 0.010, 0, 1) * np.exp(-np.maximum(t - 0.010, 0) / 0.024)
    y = bp(rng_for('shk', key).standard_normal(n) * env, 3000, 7500)
    return fade(y, 0.003, 0.01)


@lru_cache(None)
def tom(f=110., dur=0.4, dec=0.16, noise=0.15):
    n = int(dur * SR)
    t = np.arange(n) / SR
    ff = f * (1 + 0.8 * np.exp(-t / 0.02))
    y = np.sin(TAU * np.cumsum(ff) / SR) * np.exp(-t / dec)
    y += noise * lp(rng_for('tom', f).standard_normal(n), 2500) * np.exp(-t / 0.02)
    return fade(np.tanh(1.5 * y), 0.002, 0.02)


@lru_cache(None)
def crash(dur=1.8, dec=0.55, key=0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    rg = rng_for('crash', key)
    a = lp(bp(rg.standard_normal(n), 3000, 9000), 8000) * np.exp(-t / dec)
    b = bp(rg.standard_normal(n), 1200, 4200) * np.exp(-t / (dec * 0.4))
    part = sum(np.sin(TAU * f * t + i) for i, f in enumerate((2950., 3710., 4630., 5390.))) * np.exp(-t / (dec * 1.4)) * 0.05
    L = a + 0.6 * b + part
    a2 = lp(bp(rg.standard_normal(n), 3000, 9000), 8000) * np.exp(-t / dec)
    R = 0.55 * a + 0.45 * a2 + 0.6 * b + part
    return fade(np.stack([L, R], 1), 0.002, 0.05)


@lru_cache(None)
def impact(dur=0.5, dec=0.12, f_top=10000., f_end=900., key=0):
    """broadband noise slam whose low-pass closes quickly"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    rg = rng_for('impact', key)
    c = rg.standard_normal(n)
    x = np.stack([c + 0.35 * rg.standard_normal(n), c + 0.35 * rg.standard_normal(n)], 1)
    cut = f_end + (f_top - f_end) * np.exp(-t / 0.05)
    y = tvf(x, cut, 0.8) * np.exp(-t / dec)[:, None]
    return fade(y, 0.002, 0.03)


@lru_cache(None)
def sub_boom(f0=70., f1=42., dur=1.2, pdec=0.30, adec=0.40):
    """pitch-dropping sine boom; never goes below ~40 Hz (audible, no wasted headroom)"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t / pdec)
    y = np.sin(TAU * np.cumsum(f) / SR) * np.exp(-t / adec)
    y += 0.25 * np.sin(2 * TAU * np.cumsum(f) / SR) * np.exp(-t / (adec * 0.6))
    return fade(np.tanh(1.4 * y), 0.003, 0.08)


@lru_cache(None)
def boom_body(dur=0.9, key=0):
    """mid-range body of the explosion: distorted 90-450 Hz noise + a falling tom"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    rg = rng_for('body', key)
    c = rg.standard_normal(n)
    x = np.stack([c + 0.5 * rg.standard_normal(n), c + 0.5 * rg.standard_normal(n)], 1)
    y = bp(x, 90, 480, 2) * 3.0
    ff = 70 + 110 * np.exp(-t / 0.05)
    y += 0.9 * np.sin(TAU * np.cumsum(ff) / SR)[:, None]
    y = np.tanh(2.5 * y * np.exp(-t / 0.20)[:, None])
    y += 0.5 * bp(x, 500, 2500) * np.exp(-t / 0.08)[:, None]
    return fade(y, 0.002, 0.05)


# ----------------------------------------------------------------------------
# tonal instruments
# ----------------------------------------------------------------------------
def stack(midis, n, voices=5, det=0.14, spread=0.8, key=0, bend=None):
    """detuned polyBLEP saw stack, voices spread across the stereo field"""
    rg = rng_for('stack', key, tuple(midis), voices)
    L = np.zeros(n)
    R = np.zeros(n)
    for m in midis:
        for v in range(voices):
            o = (v - (voices - 1) / 2) / max((voices - 1) / 2, 1)
            semis = m + o * det
            f = mtof(semis) if bend is None else mtof(semis + bend)
            y = saw(f, n, rg.random())
            gl, gr = pan_gains(o * spread)
            L += y * gl
            R += y * gr
    sc = 1.0 / np.sqrt(voices * len(midis))
    return np.stack([L * sc, R * sc], 1)


@lru_cache(None)
def stab(midis, dur=0.32, rel=0.10, fmin=900., fmax=9000., fdec=0.08, d=0.25, s=0.3,
         voices=5, det=0.14, key=0, bend=0.0, bend_t=0.4, Q=1.4, hpf=150.):
    """supersaw chord stab through a resonant filter envelope"""
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    b = None if not bend else bend * np.clip(t / bend_t, 0, 1) ** 1.5
    x = stack(midis, n, voices, det, 0.8, key, b)
    cut = fmin + (fmax - fmin) * np.exp(-t / fdec)
    y = tvf(x, cut, Q)
    if hpf:
        y = hp(y, hpf)
    return y * env_ad(n, 0.003, d, s, gate=dur, r=rel)[:, None]


@lru_cache(None)
def brass(midis, dur=0.2, rel=0.12, f_lo=700., f_hi=5200., fatk=0.012, fdec=0.11, Q=2.2,
          scoop=1.0, voices=3, det=0.12, drive=1.8, key=0, s=0.55):
    """synth brass: detuned saws, pitch scoop and a resonant 'bwah' filter blat"""
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    bend = -scoop * np.exp(-t / 0.028)
    x = stack(midis, n, voices, det, 0.7, key, bend)
    fe = (1 - np.exp(-t / fatk)) * np.exp(-np.maximum(t - 2 * fatk, 0) / fdec)
    cut = f_lo + (f_hi - f_lo) * fe
    y = tvf(x, cut, Q)
    y = np.tanh(drive * y) / np.tanh(drive)
    y = hp(y, 160)
    return y * env_ad(n, 0.004, 0.16, s, gate=dur, r=rel)[:, None]


@lru_cache(None)
def pad(midis, dur, a=0.25, rel=0.4, c0=2500., c1=2500., voices=7, det=0.20, key=0, curve=1.0,
        hpf=180., Q=0.8, lfo=0.0, lfo_rate=0.0):
    """supersaw pad; resonant low-pass cutoff glides c0 -> c1 over the note"""
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    x = stack(midis, n, voices, det, 0.95, key)
    u = np.clip(t / max(dur, 1e-3), 0, 1) ** curve
    cut = c0 * (c1 / c0) ** u
    if lfo:
        cut = cut * 2.0 ** (lfo * np.sin(TAU * lfo_rate * t))
    y = tvf(x, cut, Q)
    if hpf:
        y = hp(y, hpf)
    e = env_ad(n, a, 10.0, 1.0, gate=dur, r=rel)
    return y * e[:, None]


def res_sweep(midis, dur, f0, f1, Q=3.0, curve=1.5, amp=None, voices=5, det=0.18, key=0,
              a=0.05, r=0.008, hpf=150., bend=None):
    """resonant supersaw sweep (risers / whip beds); ends with a short ramp"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    u = t / dur
    x = stack(tuple(midis), n, voices, det, 0.9, key, bend)
    cut = f0 * (f1 / f0) ** (u ** curve)
    y = tvf(x, cut, Q)
    if hpf:
        y = hp(y, hpf)
    if amp is not None:
        y = y * amp(u)[:, None]
    return fade(y, a, r)


@lru_cache(None)
def pluck(m, dur=0.45, decay=0.30, bright=0.65, fmax=9000.):
    """additive pluck: high harmonics decay faster"""
    f = float(mtof(m))
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    K = int(min(40, fmax // f))
    for k in range(1, K + 1):
        rate = (1.0 / decay) * (1.0 + (k - 1) * (1.25 - bright))
        y += (1.0 / k) * np.exp(-t * rate) * np.sin(TAU * k * f * t + 0.7 * k)
    return fade(0.7 * y, 0.002, 0.02)


@lru_cache(None)
def bell(m, dur=1.0, idx=2.2, idec=0.12, adec=0.45, ratio=2.0):
    """FM bell / glass"""
    f = float(mtof(m))
    n = int(dur * SR)
    t = np.arange(n) / SR
    I = idx * np.exp(-t / idec) * min(1.0, 1400.0 / f)
    y = np.sin(TAU * f * t + I * np.sin(TAU * ratio * f * t)) * np.exp(-t / adec)
    y += 0.25 * np.sin(TAU * 2 * f * t) * np.exp(-t / (adec * 0.35))
    return fade(0.6 * y, 0.002, 0.03)


@lru_cache(None)
def epiano(m, dur=1.0, rel=0.3):
    """FM electric piano with a glassy tine"""
    f = float(mtof(m))
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    I = 1.5 * np.exp(-t / 0.30) + 0.3
    y = np.sin(TAU * f * t + I * np.sin(TAU * f * t))
    y += 0.18 * np.sin(TAU * 4.0 * f * t + 0.8 * np.sin(TAU * f * t)) * np.exp(-t / 0.12)
    y += 0.10 * np.sin(TAU * 7.0 * f * t) * np.exp(-t / 0.025) * (f < 900)
    return 0.5 * y * env_ad(n, 0.004, 1.2, 0.35, gate=dur, r=rel)


@lru_cache(None)
def chip(m, dur=0.12, duty=0.25, dec=0.09, s=0.45, grace=None, vib=0.0, cut=6500.):
    n = int((dur + 0.03) * SR)
    t = np.arange(n) / SR
    mm = np.full(n, float(m))
    if grace is not None:
        mm[:int(0.035 * SR)] = grace
    if vib:
        mm += vib * np.sin(TAU * 7.0 * t) * np.clip((t - 0.06) / 0.08, 0, 1)
    y = lp(pulse(mtof(mm), n, duty), min(cut, 7000.), 4)
    return y * env_ad(n, 0.002, dec, s, gate=dur, r=0.02)


@lru_cache(None)
def bass_note(m, dur, m_from=None, glide=0.03, cut_hi=2600., cut_lo=420., fdec=0.07, s=0.7,
              drive=1.5, sub=0.22, wob=None, sq=0.45, Q=1.9):
    """synth bass: saw+square through a resonant filter envelope ('squelch'),
    plus a clean sine sub at the fundamental"""
    n = int((dur + 0.035) * SR)
    t = np.arange(n) / SR
    if m_from is None:
        mm = np.full(n, float(m))
    else:
        mm = m + (m_from - m) * np.exp(-t / glide)
    f = mtof(mm)
    y = 0.65 * saw(f, n) + sq * pulse(f, n, 0.5)
    cut = cut_lo + (cut_hi - cut_lo) * np.exp(-t / fdec)
    if wob is not None:
        cut = cut * 2.0 ** (wob[0] * (0.5 - 0.5 * np.cos(TAU * wob[1] * t)))
    y = tvf(y, cut, Q)
    y += sub * np.sin(TAU * np.cumsum(f) / SR)
    e = env_ad(n, 0.003, 0.22, s, gate=dur, r=0.03)
    return np.tanh(drive * y * e) / np.tanh(drive)


@lru_cache(None)
def lead(m, dur, scoop=0.0, vib=0.25, cut=6500., det=0.07, bright=1.0, rel=0.12, Q=1.3):
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    bend = -scoop * np.exp(-t / 0.025) + vib * np.sin(TAU * 5.5 * t) * np.clip((t - 0.12) / 0.15, 0, 1)
    x = stack((m,), n, 3, det, 0.35, key=m, bend=bend)
    x += 0.30 * pulse(mtof(m - 12 + bend), n, 0.35)[:, None]
    c = cut * (0.55 + 0.45 * bright * np.exp(-t / 0.15))
    y = tvf(x, c, Q)
    y = hp(y, 220)
    return y * env_ad(n, 0.004, 0.35, 0.65, gate=dur, r=rel)[:, None]


@lru_cache(None)
def wah(m, dur, droop=0.0, vib=0.0, wobble=0.0):
    """muted-trumpet 'wah' (additive saw through a moving resonant low-pass)"""
    n = int((dur + 0.09) * SR)
    t = np.arange(n) / SR
    mm = m - 0.7 * np.exp(-t / 0.035) - droop * np.clip((t - dur * 0.55) / (dur * 0.45), 0, 1) ** 2
    mm = mm + vib * np.sin(TAU * 5.2 * t) * np.clip((t - 0.15) / 0.2, 0, 1)
    f0 = mtof(mm)
    fc = 360 + 1500 * np.clip(t / 0.09, 0, 1) * np.exp(-np.maximum(t - 0.09, 0) / 0.22)
    if wobble:
        fc = fc + wobble * (0.5 - 0.5 * np.cos(TAU * 4.6 * np.maximum(t - 0.2, 0))) * (t > 0.2)
    ph = TAU * np.cumsum(f0) / SR
    y = np.zeros(n)
    for k in range(1, 40):
        fk = k * f0
        if fk.min() > 7000:
            break
        r = fk / fc
        H = 1.0 / np.sqrt((1 - r * r) ** 2 + (r / 3.5) ** 2)
        y += (1.0 / k) * H * np.sin(k * ph) * (fk < 9000)
    y = y / 3.0
    return np.tanh(1.2 * y * env_ad(n, 0.025, 0.8, 0.85, gate=dur, r=0.08))


# ----------------------------------------------------------------------------
# sound effects (all synthetic)
# ----------------------------------------------------------------------------
_BANDS = np.geomspace(180, 12000, 14)


def noise_sweep(dur, f0, f1, amp, key, width=0.55, stereo=0.5, curve=1.0, pan=None):
    """noise whose spectral centre glides f0 -> f1 (log), amplitude curve amp(u)"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    u = t / dur
    rg = rng_for('sweep', key)
    c = rg.standard_normal(n)
    x = np.stack([c + stereo * rg.standard_normal(n), c + stereo * rg.standard_normal(n)], 1)
    centre = f0 * (f1 / f0) ** (u ** curve)
    y = np.zeros((n, 2))
    for fc in _BANDS:
        w = np.exp(-0.5 * (np.log2(fc / centre) / width) ** 2)
        if w.max() < 1e-3:
            continue
        y += bp(x, fc / 1.3, fc * 1.3) * w[:, None]
    y *= amp(u)[:, None]
    if pan is not None:
        p = pan(u)
        gl, gr = pan_gains(p)
        y = y * np.stack([gl, gr], 1)
    return fade(y, 0.01, 0.006)


def zap(big=False):
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    f = 300 + (3200 if big else 2400) * np.exp(-t / 0.012)
    y = np.sin(TAU * np.cumsum(f) / SR) * np.exp(-t / 0.03)
    y += 0.5 * bp(rng_for('zap', big).standard_normal(n), 2000, 7000) * np.exp(-t / 0.006)
    return fade(y, 0.002, 0.01)


@lru_cache(None)
def thunk():
    """cable plugging into a socket"""
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    f = 75 + 140 * np.exp(-t / 0.012)
    y = np.sin(TAU * np.cumsum(f) / SR) * np.exp(-t / 0.035)
    y += 0.35 * bp(rng_for('thunk').standard_normal(n), 1200, 4000) * np.exp(-t / 0.003)
    return fade(y, 0.002, 0.01)


@lru_cache(None)
def tick(f=2600., dec=0.012, key=0):
    n = int(0.05 * SR)
    t = np.arange(n) / SR
    y = np.sin(TAU * f * t) * np.exp(-t / dec)
    y += 0.4 * bp(rng_for('tick', key).standard_normal(n), 2000, 6000) * np.exp(-t / 0.003)
    return fade(y, 0.002, 0.005)


@lru_cache(None)
def woodblock(m, dec=0.03):
    """dry pitched woodblock / plink"""
    f = float(mtof(m))
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    y = np.sin(TAU * f * t) * np.exp(-t / dec)
    y += 0.35 * np.sin(TAU * 2.73 * f * t) * np.exp(-t / (dec * 0.35))
    y += 0.25 * bp(rng_for('wb', m).standard_normal(n), 1500, 5000) * np.exp(-t / 0.002)
    return fade(y, 0.002, 0.01)


@lru_cache(None)
def keyclick(key=0):
    n = int(0.03 * SR)
    t = np.arange(n) / SR
    rg = rng_for('kc', key)
    y = bp(rg.standard_normal(n), 1800, 6000) * np.exp(-t / 0.004)
    y += 0.4 * np.sin(TAU * (160 + 40 * rg.random()) * t) * np.exp(-t / 0.008)
    return fade(y, 0.002, 0.004)


@lru_cache(None)
def blip(m, dur=0.07, duty=0.5):
    n = int((dur + 0.03) * SR)
    y = lp(pulse(mtof(m), n, duty), 6000, 4) + 0.5 * sine_f(mtof(m), n)
    return y * env_ad(n, 0.002, 0.05, 0.3, gate=dur, r=0.025)


@lru_cache(None)
def drop_blip(m):
    n = int(0.22 * SR)
    t = np.arange(n) / SR
    mm = m - 7 * np.clip(t / 0.18, 0, 1) ** 1.5
    y = 0.6 * lp(pulse(mtof(mm), n, 0.3), 5500) + 0.6 * sine_f(mtof(mm), n)
    return y * env_ad(n, 0.002, 0.12, 0.0)


@lru_cache(None)
def boing():
    n = int(0.42 * SR)
    t = np.arange(n) / SR
    mm = 62 + 12 * (1 - np.exp(-t / 0.05)) + 1.6 * np.sin(TAU * 17 * t) * np.exp(-t / 0.15)
    y = sine_f(mtof(mm), n) + 0.3 * lp(pulse(mtof(mm), n, 0.5), 3500)
    return y * env_ad(n, 0.002, 0.16, 0.0)


@lru_cache(None)
def gulp(key=0):
    n = int(0.2 * SR)
    t = np.arange(n) / SR
    f = 150 + 330 * np.exp(-t / 0.035)
    y = np.sin(TAU * np.cumsum(f) / SR) * np.exp(-t / 0.07)
    y += 0.6 * bp(rng_for('gulp', key).standard_normal(n), 1500, 6000) * np.exp(-t / 0.004)
    return fade(np.tanh(1.6 * y), 0.002, 0.02)


@lru_cache(None)
def pop():
    n = int(0.16 * SR)
    t = np.arange(n) / SR
    f = 160 + 620 * (1 - np.exp(-t / 0.02))
    y = np.sin(TAU * np.cumsum(f) / SR) * np.exp(-t / 0.05)
    y += 0.5 * bp(rng_for('pop').standard_normal(n), 1500, 7000) * np.exp(-t / 0.003)
    return fade(y, 0.002, 0.02)


@lru_cache(None)
def coin(f, dec=0.18):
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    y = (np.sin(TAU * f * t) + 0.5 * np.sin(TAU * 2.76 * f * t) * np.exp(-t / 0.05)
         + 0.25 * np.sin(TAU * 5.40 * f * t) * np.exp(-t / 0.02))
    return fade(y * np.exp(-t / dec), 0.002, 0.02)


@lru_cache(None)
def paper(key=0):
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    env = np.clip(t / 0.01, 0, 1) * np.exp(-t / 0.035)
    return fade(bp(rng_for('paper', key).standard_normal(n), 1500, 5500) * env, 0.003, 0.01)


# ----------------------------------------------------------------------------
# convenience: a "hit" = kick + clap + noise slam + chord stab (+ crash)
# ----------------------------------------------------------------------------
def big_hit(t, chord=None, size=1.0, clap_g=0.42, noise_g=0.22, stab_g=0.42, crash_g=0.0,
            kick_kw=None, stab_kw=None, send=0.22, depth=1.0):
    kick_at(t, gain=0.92 * size, depth=depth, **(kick_kw or {}))
    if clap_g:
        add('drums', clap(), t, gain=clap_g * size, send=0.10)
    if noise_g:
        add('fx', impact(), t, gain=noise_g * size)
    if chord is not None:
        add('music', stab(tuple(chord), **(stab_kw or {})), t, gain=stab_g * size, send=send)
    if crash_g:
        add('drums', crash(), t, gain=crash_g)


def whip_tom_fill(t1, f0=180., gain=0.30, step=BEAT / 4):
    """four toms on the 16ths of the NEXT section's grid, ending one 16th
    before its downbeat (graft from the chip draft's whip_fill)"""
    for k in range(4):
        tt = t1 - (4 - k) * step
        add('drums', tom(f0 - 22 * k, 0.25, 0.08, 0.3), tt, pan=(-0.4, -0.15, 0.15, 0.4)[k],
            gain=gain * (0.75 + 0.1 * k), send=0.08)


# ----------------------------------------------------------------------------
# voicings (mid register), chord qualities and the motif
# ----------------------------------------------------------------------------
V = {
    'Amaj9': (57, 61, 64, 68, 71), 'F#m9': (54, 57, 61, 64, 68), 'Dmaj9': (54, 57, 61, 64, 69),
    'E11': (57, 59, 62, 66, 71), 'E7': (56, 59, 62, 64, 68), 'Bm9': (57, 61, 62, 66, 69),
    'C#m7': (56, 59, 61, 64, 68), 'E': (56, 59, 64, 68, 71), 'Aadd9': (57, 61, 64, 69, 71, 73),
    'F#m7': (57, 61, 64, 66), 'B9': (57, 61, 63, 66), 'Amaj7c': (57, 61, 64, 68),
    'Bm7c': (57, 59, 62, 66), 'C#m7c': (56, 59, 61, 64), 'Dmaj7c': (57, 61, 62, 66),
    'F#m7/C#': (61, 64, 66, 69), 'D69': (57, 62, 64, 66, 71), 'Esus': (57, 59, 64, 69, 71),
    'sour': (54, 60, 65, 67), 'Am_hit': (57, 60, 64, 69, 72), 'Cm_hit': (60, 63, 67, 72, 75),
    'Ebm_hit': (63, 66, 70, 75, 78),
}
# 7th-step of the bass pattern per chord quality (chord-aware bass)
QUAL = {'Amaj9': 11, 'Dmaj9': 11, 'F#m9': 10, 'Bm9': 10, 'C#m7': 10, 'E11': 10, 'E7': 10,
        'E': 7, 'Aadd9': 7, 'F#m7': 10, 'B9': 10, 'F#m7/C#': 7}
MOTIF = (81, 85, 88, 90)            # A5 C#6 E6 F#6
FUNK = [(0, 0, 1.6, 0), (2, 12, 0.55, 1), (3, 0, 0.9, 0), (5, 12, 0.5, 1), (6, 7, 0.9, 0), (7, 's', 0.7, 0)]


def bass_seq(G, blocks, start, end, gain=0.42, gfun=None, kw=None):
    """blocks: list of (beat, chord, root midi).  The funk pattern's 7th step is
    chosen from the chord (maj7 on maj chords, b7 on minor/dominant, 5th on triads)."""
    kw = kw or {}
    prev = None
    for bi, (b0, ch, root) in enumerate(blocks):
        b_end = blocks[bi + 1][0] if bi + 1 < len(blocks) else end
        for (p, off, d16, gl) in FUNK:
            b = b0 + p / 4.0
            if b >= b_end - 1e-9 or b < start:
                continue
            t0 = G(b)
            t1 = G(min(b + d16 / 4.0, b_end))
            m = root + (QUAL[ch] if off == 's' else off)
            g = gain * (gfun(t0) if gfun else 1.0)
            add('bass', bass_note(m, max(t1 - t0, 0.05), prev if gl else None, **kw), t0, gain=g)
            prev = m


# ============================================================================
# SCORE
# ============================================================================
def sec_intro():
    G = g96(5.40)
    # 0.10 title sting -------------------------------------------------------
    t = 0.10
    big_hit(t, V['Amaj9'] + (76,), size=1.0, crash_g=0.20, stab_kw=dict(dur=0.5, d=0.45, s=0.25, fmax=9500))
    add('music', brass((57, 61, 64, 69), 0.30, key=1), t, gain=0.22, send=0.2)
    add('bass', bass_note(33, 0.9, cut_hi=2000, s=0.6), t, gain=0.40)
    add('pad', pad(V['Amaj9'], 1.2, a=0.04, rel=0.5, c0=8000, c1=1500, Q=1.0), t, gain=0.10, send=0.3)
    for i, m in enumerate(MOTIF):
        add_echo('music', pluck(m, 0.6, 0.35, 0.8), t + 0.075 * i, E8, 0.3, 2,
                 pan=-0.45 + 0.3 * i, gain=0.22, send=0.25)
    add('music', bell(93, 1.0, 1.5, 0.08, 0.45), t + 0.30, pan=0.35, gain=0.07, send=0.35)
    # 1.35-1.80 four colour stripes drop away: motif falling, one blip each ---
    for i, (tt, m) in enumerate(zip((1.35, 1.50, 1.65, 1.80), (78, 76, 73, 69))):
        add('music', drop_blip(m), tt, pan=-0.6 + 0.4 * i, gain=0.30, send=0.15)
        add('drums', tom(150. - 18 * i, 0.3, 0.09, 0.3), tt, gain=0.30 - 0.04 * i)
    add('drums', tick(3000., key=1), 1.35, gain=0.25)
    # 1.80-3.80 idea bubble: curious E11 colour, bubble rising ----------------
    add('pad', pad(V['E11'], 2.0, a=0.35, rel=0.05, c0=1400, c1=3200, Q=1.0), 1.80, gain=0.11, send=0.35)
    add('bass', bass_note(40, 1.9, cut_hi=800, cut_lo=380, s=0.8, sub=0.3), 1.80, gain=0.18)
    n = int(0.65 * SR)
    tt = np.arange(n) / SR
    mm = 67 + 12 * (tt / 0.65) ** 0.7 + 0.6 * np.sin(TAU * 9 * tt)
    bub = sine_f(mtof(mm), n) * env_ad(n, 0.05, 10, 1.0, gate=0.58, r=0.07)
    add('music', bub, 1.80, gain=0.12, send=0.3)
    # 2.45 bubble settles
    add('fx', pop(), 2.45, gain=0.35)
    add('music', bell(88, 1.2, 1.6, 0.1, 0.5), 2.45, pan=0.2, gain=0.22, send=0.35)
    add('drums', tom(95., 0.3, 0.08, 0.2), 2.45, gain=0.3)
    # typing: one soft key click every 0.05 s, 2.50-3.80
    rg = rng_for('typing-intro')
    for k in range(27):
        add('fx', keyclick(k % 7), 2.50 + 0.05 * k, pan=float(rg.uniform(-0.3, 0.3)),
            gain=0.05 + 0.03 * rg.random())
    # a questioning pluck figure on the 96 BPM grid
    for b, m in zip((-4.5, -4.0, -3.5, -3.0), (76, 78, 81, 83)):
        add_echo('music', pluck(m, 0.5, 0.3, 0.6), G(b), E8, 0.3, 2, pan=0.3, gain=0.11, send=0.25)
    # 3.80-5.40 groove pick-up: a resonant E11 sweep opening into the first icon
    sw = res_sweep(V['E11'], 1.58, 500., 7000., Q=3.0, curve=1.8, amp=lambda u: 0.35 + 0.65 * u ** 1.5, key=3,
                   a=0.08)
    add('pad', sw, 3.80, gain=0.10, send=0.3)
    for k in range(10):
        b = -2.5 + 0.25 * k
        if G(b) >= 3.82:
            add('drums', shaker(k % 3), G(b), pan=0.25, gain=0.10 + 0.02 * k)
    for b in (-2, -1):
        add('music', stab(V['E11'], 0.18, fmin=700, fmax=4500, d=0.15, key=5), G(b), gain=0.20, send=0.2)
        add('bass', bass_note(40, 0.25, cut_hi=1800), G(b), gain=0.34)
        kick_at(G(b), gain=0.45, depth=0.5)
    add('bass', bass_note(47, 0.28, 40, 0.04), G(-0.5), gain=0.28)
    for i, b in enumerate((-0.75, -0.5, -0.25)):
        add('drums', snare(key=i), G(b), gain=0.10 + 0.06 * i, send=0.1)
    up = noise_sweep(1.0, 500, 7000, lambda u: 0.05 + 0.25 * u ** 2.5, 'pickup')
    add('fx', fade(up[:int(0.99 * SR)], 0.01, 0.006), 4.40, gain=0.22)


def sec_apps():
    G = g96(5.40)                                  # 96 BPM; icons on beats 0, 2, 4, 6
    blocks = [(0, 'Amaj9', 33), (2, 'F#m9', 42), (4, 'Dmaj9', 38), (6, 'E11', 40),
              (8, 'Bm9', 35), (9, 'E7', 40)]
    # shout voicings: synth brass whose top note is the motif note an octave down
    BRASS = [(57, 61, 64, 69), (61, 64, 69, 73), (66, 69, 73, 76), (66, 71, 74, 78)]
    for i, (b, ch, root) in enumerate(blocks[:4]):
        t = G(b)
        big_hit(t, V[ch], size=0.86 + 0.03 * i, stab_kw=dict(dur=0.30, fmax=9000, key=i))
        add('fx', tom(200. + 30 * i, 0.25, 0.07, 0.4), t, gain=0.16)
        ts = t + 0.12                               # the shout: brass 'hah!' + scooped lead
        add_echo('music', lead(MOTIF[i], 0.26, scoop=3.0, vib=0.2, cut=7000), ts, DOT8, 0.26, 2,
                 gain=0.30, send=0.2)
        add('music', brass(BRASS[i], 0.15, rel=0.10, key=10 + i), ts, pan=0.1 * (i - 1.5), gain=0.28, send=0.15)
        add('drums', tick(3400., 0.006, key=i), ts, gain=0.18)
        add('music', pluck(MOTIF[i], 0.5, 0.15, 0.9), ts, gain=0.10)
    # four-on-the-floor + backbeat claps + bright 16th hats
    for b in range(1, 9):
        if b not in (2, 4, 6):
            kick_at(G(b), gain=0.80)
    for b in (1, 3, 5, 7):
        add('drums', clap(), G(b), gain=0.34, send=0.08)
        add('drums', snare(key=b), G(b), gain=0.10)
    for k in range(36):
        b = k * 0.25
        if k % 8 == 6:
            add('drums', hat(True), G(b), pan=0.3, gain=0.10)
        else:
            add('drums', hat(False, k % 4), G(b), pan=0.3, gain=0.10 if k % 2 == 0 else 0.15)
    # bass: funky octave line, chord-aware
    bass_seq(G, blocks[:5], 0, 9)
    add('bass', bass_note(40, 0.75, 35, 0.05, cut_hi=2200, s=0.85), G(9), gain=0.36)
    # off-beat 'chk' stabs and the plucky arp
    for b0, ch, _ in blocks[:5]:
        for p in (3, 6):
            b = b0 + p / 4
            if b < 9:
                add('music', stab(V[ch], 0.07, fmin=900, fmax=6500, fdec=0.04, d=0.06, s=0.2, Q=1.8),
                    G(b), pan=-0.2, gain=0.15, send=0.12)
    arp_notes = {ch: sorted(set(V[ch][1:] + tuple(x + 12 for x in V[ch][1:3]))) for _, ch, _ in blocks}
    for k in range(32):
        b = k * 0.25
        ch = [c for bb, c, _ in blocks if bb <= b][-1]
        notes = arp_notes[ch]
        m = notes[(k * 2 + (k // 4)) % len(notes)] + 12
        if abs(G(b) - 10.10) < 0.3:          # leave room for the bounce plinks
            continue
        add('music', pluck(m, 0.25, 0.12, 0.75), G(b), pan=(-0.45 if k % 2 else 0.45), gain=0.09, send=0.15)
    # 10.10 / 10.18 / 10.26 / 10.34 the icons bounce together: motif plinks
    for i, (tt, m) in enumerate(zip((10.10, 10.18, 10.26, 10.34), MOTIF)):
        add_echo('music', pluck(m + 12 if i == 3 else m, 0.5, 0.25, 0.85), tt, E8, 0.25, 2,
                 pan=-0.6 + 0.4 * i, gain=0.26, send=0.2)
        add('drums', tick(2600. + 300 * i, 0.008, key=10 + i), tt, gain=0.14)
    # Bm9 chord stab on beat 8, then the E7 resonant riser 11.0 -> 11.8
    add('music', stab(V['Bm9'], 0.4, fmax=8000, d=0.35, key=9), G(8), gain=0.28, send=0.2)
    add('music', brass((61, 66, 69, 74), 0.35, key=19, s=0.5), G(8), gain=0.16, send=0.2)
    rz = res_sweep(V['E7'], 0.80, 400., 8000., Q=3.5, curve=1.4, amp=lambda u: 0.25 + 0.75 * u ** 1.5,
                   key=21, a=0.2)
    add('pad', rz, 11.0, gain=0.15, send=0.15)
    up = noise_sweep(0.80, 400, 9000, lambda u: 0.03 + u ** 2.2, 'riser1', curve=1.2)
    add('fx', up, 11.0, gain=0.28)
    for k in range(10):            # snare roll: 32nds of the Clawd grid, crescendo, ends on 11.8
        tt = 11.8 - (BEAT / 8) * (k + 1)
        add('drums', snare(0.12, key=k % 3), tt, gain=0.06 + 0.022 * (10 - k), send=0.06)
    whip_tom_fill(11.80, 190., 0.22)


def sec_clawd():
    G = g96(11.80)
    chords = [(0, 'Amaj7c', 33), (2, 'Bm7c', 35), (4, 'C#m7c', 37), (6, 'Dmaj7c', 38)]
    checks = (12.92, 13.64, 14.36, 15.08)
    big_hit(11.80, V['Amaj9'], size=1.0, crash_g=0.16, stab_kw=dict(dur=0.35, key=7, fmax=9000))
    add('music', brass((57, 61, 64, 69), 0.25, key=17), 11.80, gain=0.14, send=0.2)
    # light, bouncy 96 groove (chip snare on the backbeat, bright 8th hats)
    for b in (2, 3.5, 4, 6):
        kick_at(G(b), gain=0.72, depth=0.8)
    for b in (1, 3, 5):
        tb = 13.64 if b == 3 else G(b)          # beat 3 is pulled onto check 2 (no flam)
        add('drums', snare(0.16, 230., 0.05, 0.03, 7500., key=4), tb, gain=0.26, send=0.06)
        add('drums', clap(0.25, 0.05, key=1), tb, gain=0.10)
    for k in range(28):
        b = 0.25 * k
        if G(b) > 16.12:
            break
        if k % 2 == 1:
            add('drums', hat(False, k % 3), G(b), pan=-0.2, gain=0.13)
        elif k % 4 == 2:
            add('drums', shaker(k % 3), G(b), pan=0.3, gain=0.08)
    for ci, (b0, ch, root) in enumerate(chords):
        b1 = chords[ci + 1][0] if ci + 1 < len(chords) else 7.04
        dur = G(b1) - G(b0)
        # square-wave chord pad (two detuned pulses per note)
        n = int((dur + 0.1) * SR)
        x = np.zeros(n)
        for m in V[ch]:
            x += pulse(mtof(m + 12) * 1.003, n, 0.5) + pulse(mtof(m + 12) * 0.997, n, 0.25, 0.3)
        x = lp(x, 3800, 4) * env_ad(n, 0.01, 0.5, 0.45, gate=dur, r=0.08) * 0.25
        add('pad', x, G(b0), gain=0.14, send=0.2)
        # chip bass: straight 8ths, root/octave bounce
        for k in range(int(round((b1 - b0) * 2))):
            b = b0 + 0.5 * k
            if G(b) > 16.15:
                break
            m = root + (12 if k % 2 else 0)
            add('bass', chip(m, 0.16, 0.25, 0.12, 0.6, cut=2600), G(b), gain=0.30)
            nn = int(0.2 * SR)
            add('bass', sine_f(mtof(root), nn) * env_ad(nn, 0.003, 0.1, 0.4, 0.16, 0.03), G(b), gain=0.22)
        # chip arpeggio: 16ths through the chord, an octave up
        tones = list(V[ch]) + [V[ch][0] + 12]
        for k in range(int(round((b1 - b0) * 4))):
            b = b0 + 0.25 * k
            if G(b) > 16.15:
                break
            if any(abs(G(b) - c) < 0.07 for c in checks + (15.50,)):
                continue
            m = tones[(k if (k // 4) % 2 == 0 else -k - 1) % len(tones)] + 12
            add('music', chip(m, 0.06, 0.5, 0.05, 0.3), G(b), pan=(0.35 if k % 2 else -0.35), gain=0.07, send=0.1)
    # the four check marks: chip 'bling' playing the motif, plus a tick
    for i, (tt, m) in enumerate(zip(checks, MOTIF)):
        add_echo('music', chip(m, 0.16, 0.125, 0.12, 0.5, vib=0.25, cut=7000), tt, DOT8, 0.3, 2,
                 pan=0.15, gain=0.30, send=0.15)
        add('music', pluck(m + 12, 0.3, 0.10, 0.9), tt, pan=0.25, gain=0.10, send=0.15)
        add('drums', tick(3800., 0.006, key=20 + i), tt, gain=0.30)
        add('drums', tom(260., 0.15, 0.05, 0.6), tt, gain=0.16)
    # 15.50 Clawd hops
    add('music', boing(), 15.50, pan=-0.1, gain=0.30, send=0.15)
    add('drums', tom(140., 0.25, 0.07, 0.3), 15.50, gain=0.35)
    add('drums', tick(3000., 0.008, key=25), 15.50, gain=0.25)
    # 16.20 -> 17.05 whip pan: whoosh L->R + accelerating chip arp riser
    wh = noise_sweep(0.85, 600, 9000, lambda u: 0.02 + u ** 2.0, 'whip1', curve=1.3,
                     pan=lambda u: -0.8 + 1.6 * u)
    add('fx', wh, 16.20, gain=0.30)
    tt, k = 16.20, 0
    while tt < 17.0:
        m = 74 + [0, 4, 7, 11][k % 4] + 12 * (k // 8)
        add('music', chip(min(m, 100), 0.05, 0.5, 0.04, 0.3), tt, pan=-0.6 + 1.2 * (tt - 16.2) / 0.85,
            gain=0.05 + 0.07 * (tt - 16.2) / 0.85)
        tt += max(0.035, 0.11 - 0.09 * (tt - 16.2) / 0.85)
        k += 1
    # whip bed: the D chord swelling through a resonant sweep, held D bass, tom fill
    bed = res_sweep(V['D69'], 0.83, 700., 6000., Q=2.5, curve=1.3, amp=lambda u: 0.2 + 0.8 * u ** 1.6, key=33,
                    a=0.1)
    add('pad', bed, 16.20, gain=0.12, send=0.2)
    add('bass', bass_note(38, 0.78, 50, 0.08, cut_hi=1400, cut_lo=500, s=0.8), 16.20, gain=0.26)
    whip_tom_fill(17.05, 200., 0.24)


def sec_gemini():
    G = Grid([(0, 17.02), (1, 17.02 + BEAT)])    # half-time 96; stamps on beats 1.25 / 2.5 / 3.75
    stamps = (17.80, 18.58, 19.36)
    chords = [(0, 'Dmaj9', 38), (1.25, 'Bm9', 35), (2.5, 'C#m7', 37), (3.75, 'F#m9', 42), (5, 'Dmaj9', 38)]
    END_B = (20.70 - 17.02) / BEAT
    big_hit(17.05, V['Dmaj9'], size=1.0, crash_g=0.16, stab_kw=dict(dur=0.35, key=11, fmax=9000))
    for i, m in enumerate((86, 90, 93)):
        add('music', bell(m, 1.0), 17.05 + 0.06 * i, pan=-0.3 + 0.3 * i, gain=0.15, send=0.35)
    # three research stamps: heavy, each a new chord, bell fragments climbing
    for i, t in enumerate(stamps):
        b = (1.25, 2.5, 3.75)[i]
        ch = [c for bb, c, _ in chords if bb <= b + 1e-6][-1]
        big_hit(t, V[ch], size=0.88 + 0.04 * i, clap_g=0.42, noise_g=0.26,
                stab_kw=dict(dur=0.25, fmax=8500, key=12 + i))
        add('drums', tom(85., 0.45, 0.13, 0.5), t, gain=0.40)
        frag = [MOTIF[i], MOTIF[i + 1] if i + 1 < 4 else 93, (MOTIF + (93, 97))[i + 2]]
        for j, m in enumerate(frag):
            add_echo('music', bell(m, 0.8, 2.4, 0.1, 0.35), G(b + 0.25 * (j + 1)), DOT8, 0.3, 2,
                     pan=0.4 - 0.4 * j, gain=0.16, send=0.3)
    # half-time groove: backbeat clap on beat 2, shaker 16ths, hats on the off-8ths
    add('drums', clap(), G(2), gain=0.30, send=0.15)
    add('drums', snare(key=6), G(2), gain=0.14, send=0.1)
    kick_at(G(5), gain=0.70, depth=0.6)
    add('drums', clap(0.2, 0.04, key=3), G(4), pan=0.2, gain=0.12, send=0.15)
    for k in range(1, int(END_B * 4)):
        b = 0.25 * k
        tt = G(b)
        if tt > 20.66:
            break
        if any(abs(tt - c) < 0.04 for c in stamps):
            continue
        add('drums', shaker(k % 3), tt, pan=-0.3, gain=0.09 + 0.05 * (k % 2))
        if k % 4 == 2:
            add('drums', hat(False, k % 3), tt, pan=0.3, gain=0.10)
    for ci, (b0, ch, root) in enumerate(chords):
        b1 = chords[ci + 1][0] if ci + 1 < len(chords) else END_B
        t0 = 17.05 if ci == 0 else G(b0)
        dur = G(b1) - t0
        for m in V[ch]:
            add('music', epiano(m, dur, 0.25), t0, pan=0.25 * np.sin(m), gain=0.10, send=0.3)
        add('pad', pad(V[ch], dur, a=0.15, rel=0.2, c0=2600, c1=3600, voices=5, det=0.12, key=50 + ci, Q=0.9),
            t0, gain=0.035, send=0.3)
        # round bass: root on the change, octave answer, fifth pick-up (all on the grid)
        add('bass', bass_note(root, min(dur * 0.45, 0.30), cut_hi=1300, cut_lo=320, sq=0.2, sub=0.30, Q=1.4),
            t0, gain=0.40)
        if b1 - b0 >= 1:
            add('bass', bass_note(root + 12, 0.14, root, 0.02, cut_hi=1500, sq=0.2, Q=1.4), G(b0 + 0.5), gain=0.28)
            add('bass', bass_note(root + 7, 0.13, root + 12, 0.02, cut_hi=1400, sq=0.2, Q=1.4), G(b0 + 0.75),
                gain=0.24)
    for j, m in enumerate((93, 90, 88)):
        add_echo('music', bell(m, 0.6, 2.0, 0.1, 0.25), G(5 + 0.25 * (j + 1)), DOT8, 0.25, 1, pan=-0.3 + 0.3 * j,
                 gain=0.12, send=0.2)
    # 20.70 -> 21.70 rotating drop: falling dive tone, closing resonant E11 sweep, rising rush
    n = int(1.0 * SR)
    tt = np.arange(n) / SR
    mm = 96 - 60 * (tt / 1.0) ** 0.8
    dive = stack((0,), n, 3, 0.25, 0.9, key=99, bend=mm)
    dive = lp(dive, 2200) * (0.5 * env_ad(n, 0.05, 10, 1.0, gate=0.97, r=0.03))[:, None]
    add('music', dive, 20.70, gain=0.12, send=0.2)
    wh = noise_sweep(1.0, 9000, 3000, lambda u: 0.01 + u ** 3.0, 'drop', curve=1.0,
                     pan=lambda u: 0.6 * np.sin(TAU * 1.5 * u))
    add('fx', wh, 20.70, gain=0.40)
    body = res_sweep(V['E11'], 0.995, 2600., 500., Q=3.0, curve=0.8, amp=lambda u: 1.0 - 0.3 * u, key=21, a=0.05)
    add('pad', body, 20.70, gain=0.13, send=0.25)
    add('bass', bass_note(40, 0.95, 52, 0.35, cut_hi=1200, s=0.9), 20.70, gain=0.30)
    whip_tom_fill(21.70, 170., 0.24)


def sec_codex():
    G = g96(21.70)
    s = lambda p: G(p / 4.0)
    big_hit(21.70, V['F#m9'], size=1.0, crash_g=0.14, stab_kw=dict(dur=0.3, key=21, fmax=9000))
    # bar 1: a full funk bar (kick, snare backbeat on 2 and 4, 16th hats)
    for p in (6, 8, 11):
        kick_at(s(p), gain=0.78)
    for p in (4, 12):
        add('drums', snare(key=1), s(p), gain=0.38, send=0.1)
        add('drums', clap(), s(p), gain=0.18)
    for p in (7, 14):
        add('drums', snare(0.12, key=2), s(p), gain=0.07)                 # ghost notes
    for p in range(16):
        if p == 1:
            continue
        if p == 14:
            add('drums', hat(True, 1), s(p), pan=0.3, gain=0.09)
        else:
            add('drums', hat(False, p % 3), s(p), pan=0.3, gain=0.11 if p % 2 == 0 else 0.17)
    # keyboard typing texture: light clicks on the 32nds
    rg = rng_for('typing-codex')
    for k in range(int((24.15 - 21.70) / (BEAT / 8))):
        add('fx', keyclick(k % 7), 21.70 + (BEAT / 8) * k, pan=float(rg.uniform(-0.4, 0.4)),
            gain=0.03 + 0.03 * rg.random())
    # slap-ish bass, F# dorian; B under the B9 half (16th 8 on)
    riff = [(0, 42, 1.5), (2, 54, 0.5), (3, 52, 0.5), (4, 42, 1.0), (6, 45, 0.8), (7, 47, 0.8),
            (8, 35, 1.5), (10, 47, 0.5), (11, 45, 0.5), (12, 35, 1.0), (14, 37, 0.8), (15, 40, 0.8)]
    prev = None
    for p, m, d in riff:
        add('bass', bass_note(m, d * BEAT / 4, prev if p in (2, 7, 11) else None, 0.02, cut_hi=3000, fdec=0.05,
                              s=0.5, drive=1.8, Q=2.2), s(p), gain=0.40)
        prev = m
    # clav off-beats: F#m7 then B9
    for p in (3, 6, 9, 11, 14):
        ch = 'F#m7' if p < 8 else 'B9'
        add('music', stab(V[ch], 0.06, fmin=1500, fmax=7500, fdec=0.03, d=0.05, s=0.1, voices=2, det=0.05, Q=2.0),
            s(p), pan=-0.3, gain=0.20, send=0.08)
    # typed lines: rising pentatonic blips
    for tt, m in zip((21.90, 22.50, 22.70, 22.90, 23.10), (76, 78, 81, 83, 85)):
        add('music', blip(m), tt, pan=0.35, gain=0.30, send=0.12)
        add('drums', tom(330., 0.1, 0.025, 0.6), tt, pan=0.35, gain=0.20)
        add('drums', tick(3000., 0.005, key=30), tt, gain=0.36)
        add('fx', keyclick(3), tt, gain=0.24)
    # 23.60 pytest: quick arpeggio, then a nervous ticker
    for j, m in enumerate((85, 88, 93)):
        add('music', blip(m, 0.05), 23.60 + 0.05 * j, pan=-0.2, gain=0.24, send=0.1)
    add('drums', tick(2400., 0.01, key=31), 23.60, gain=0.25)
    for k in range(3):
        add('music', blip(83 if k % 2 else 85, 0.04, 0.25), s(13 + k), pan=0.2, gain=0.08 + 0.02 * k)
    rz = noise_sweep(0.58, 800, 3000, lambda u: 0.02 + 0.2 * u ** 2, 'pytest')
    add('fx', rz, 23.60, gain=0.16)
    # 24.20 TEST FAILS (the bar-2 downbeat): sour bent cluster + buzzer, groove stumbles
    t = 24.20
    kick_at(t, gain=0.95)
    add('fx', impact(0.4, 0.1, key=3), t, gain=0.26)
    add('music', stab(V['sour'], 0.55, fmin=500, fmax=5500, fdec=0.12, d=0.4, s=0.4, det=0.45,
                      bend=-2.0, bend_t=0.55, key=24, Q=1.8), t, gain=0.34, send=0.2)
    n = int(0.60 * SR)
    tt = np.arange(n) / SR
    fb = mtof(54 - 1.5 * (tt / 0.6) ** 1.5)
    buzz = lp(pulse(fb, n, 0.5) + pulse(fb * 1.06, n, 0.3), 2400) * env_ad(n, 0.003, 10, 1, gate=0.5, r=0.08)
    buzz *= (0.75 + 0.25 * np.tanh(5.0 * np.sin(TAU * 7.5 * tt)))   # gated 'eh-eh-eh'
    add('music', hp(buzz, 150), t, gain=0.20)
    add('bass', bass_note(42, 0.6, 54, 0.18, cut_hi=1400, s=0.8), t, gain=0.38)   # bass deflates an octave
    add('drums', snare(0.2, 160., 0.08, 0.05, 5000., key=5), t + 0.70, gain=0.12)  # late, wrong backbeat
    add('pad', pad((54, 57, 60, 66), 1.35, a=0.15, rel=0.05, c0=900, c1=450, voices=4, det=0.35, hpf=120.),
        24.30, gain=0.13, send=0.2)
    for k, tt in enumerate((24.48, 24.62, 24.70, 24.88)):                       # hesitant typing
        add('fx', keyclick(k), tt, pan=0.2 - 0.15 * k, gain=0.06)
    # 25.00 fixing: tentative tick, an Esus swell and blips on the PASS grid
    add('drums', tick(2200., 0.01, key=32), 25.00, gain=0.45)
    add('drums', tom(240., 0.15, 0.04, 0.5), 25.00, gain=0.22)
    add('fx', keyclick(5), 25.00, gain=0.30)
    add('music', blip(76, 0.05), 25.00, gain=0.24)
    for k, m in enumerate((76, 78, 80, 81)):
        add('music', blip(m, 0.04), 25.70 - (4 - k) * BEAT / 4, pan=0.3 - 0.15 * k, gain=0.08 + 0.025 * k)
    sus = res_sweep(V['Esus'], 0.69, 600., 5000., Q=2.2, curve=1.2, amp=lambda u: 0.3 + 0.7 * u ** 1.5, key=25,
                    a=0.1)
    add('pad', sus, 25.00, gain=0.10, send=0.2)
    rz = noise_sweep(0.70, 700, 7000, lambda u: 0.02 + u ** 2.5, 'fixing')
    add('fx', rz, 25.00, gain=0.18)
    GAPS.append((25.70, 0.35, 0.03))
    # 25.70 TESTS PASS: bright resolution, motif sparkle, groove resumes in A
    G2 = g96(25.70)
    s2 = lambda p: G2(p / 4.0)
    big_hit(25.70, V['Aadd9'], size=1.05, crash_g=0.18, stab_kw=dict(dur=0.45, fmax=9500, d=0.4, key=25))
    add('music', brass((61, 64, 69, 73), 0.35, key=26), 25.70, gain=0.20, send=0.2)
    for j, m in enumerate(MOTIF + (93,)):
        add('music', bell(m, 0.9, 2.0, 0.08, 0.35), 25.70 + 0.05 * j, pan=-0.5 + 0.25 * j, gain=0.13, send=0.3)
    for p in (6, 8):
        kick_at(s2(p), gain=0.78)
    add('drums', snare(key=1), s2(4), gain=0.38, send=0.1)
    add('drums', clap(), s2(4), gain=0.18)
    for p in range(8):
        add('drums', hat(False, p % 3), s2(p), pan=0.3, gain=0.11 if p % 2 == 0 else 0.17)
    riff2 = [(0, 33, 1.5), (2, 45, 0.5), (3, 44, 0.5), (4, 33, 1.0), (6, 40, 0.8), (7, 42, 0.8)]
    prev = None
    for p, m, d in riff2:
        add('bass', bass_note(m, d * BEAT / 4, prev if p in (2, 7) else None, 0.02, cut_hi=3000, fdec=0.05,
                              s=0.5, drive=1.8, Q=2.2), s2(p), gain=0.40)
        prev = m
    for p in (1, 3, 6):
        add('music', stab(V['Amaj7c'], 0.06, fmin=1200, fmax=6000, fdec=0.03, d=0.05, s=0.1, voices=2, det=0.05,
                          Q=2.0), s2(p), pan=-0.3, gain=0.20, send=0.08)
    # 26.90 -> 27.80 whip pan left: whoosh, D bed through a resonant sweep, snare + tom fill
    wh = noise_sweep(0.90, 600, 9000, lambda u: 0.02 + u ** 2.0, 'whip2', curve=1.3,
                     pan=lambda u: 0.8 - 1.6 * u)
    add('fx', wh, 26.90, gain=0.30)
    bed = res_sweep(V['D69'], 0.88, 700., 6000., Q=2.5, curve=1.3, amp=lambda u: 0.2 + 0.8 * u ** 1.6, key=34,
                    a=0.1)
    add('pad', bed, 26.90, gain=0.12, send=0.2)
    add('bass', bass_note(38, 0.83, 33, 0.1, cut_hi=1400, cut_lo=500, s=0.8), 26.90, gain=0.26)
    for k in range(5):
        tt = 27.80 - (BEAT / 8) * (5 - k)
        add('drums', snare(0.12, key=k % 3), tt, gain=0.08 + 0.045 * k, send=0.06)
    whip_tom_fill(27.80 - BEAT / 8, 190., 0.20)


def sec_openrouter():
    G = g96(27.80)
    chords = [(0, 'Dmaj9', 38), (2, 'Bm9', 35), (3, 'F#m7/C#', 37), (4, 'E', 40)]
    plugs = (28.42, 28.76, 29.10, 29.44, 29.78, 30.12)
    big_hit(27.80, V['Dmaj9'], size=1.0, crash_g=0.14, stab_kw=dict(dur=0.3, key=31, fmax=9000))
    for b in (1, 2, 3, 4):
        kick_at(G(b), gain=0.74, depth=0.9)
    for b in (1, 3):
        add('drums', clap(), G(b), gain=0.28, send=0.1)
    for k in range(20):
        b = 0.25 * k
        if G(b) > 30.88:
            break
        if k % 4 == 2:
            add('drums', hat(True, k % 2), G(b), pan=0.3, gain=0.11)
        else:
            add('drums', hat(False, k % 4), G(b), pan=0.3, gain=0.10 + 0.05 * (k % 2))
        add('drums', shaker(k % 3), G(b), pan=-0.35, gain=0.05 + 0.03 * (k % 2))
    # purple pad: supersaw with a slow resonant filter drift
    for ci, (b0, ch, root) in enumerate(chords):
        t1 = G(chords[ci + 1][0]) if ci + 1 < len(chords) else 30.95
        add('pad', pad(V[ch], t1 - G(b0), a=0.06, rel=0.15, c0=3000, c1=4500, key=ci, Q=1.6, lfo=0.3, lfo_rate=1.6),
            G(b0), gain=0.10, send=0.3)
    # rubbery octopus bass: glides and an 8th-synced resonant wobble
    line = [(0, 38, 0.9, None), (1, 38, 0.3, None), (1.5, 50, 0.3, 38), (2, 35, 0.4, 50), (2.5, 47, 0.3, 35),
            (3, 37, 0.4, 47), (3.5, 49, 0.3, 37), (4, 40, 0.45, 49), (4.5, 40, 0.3, None)]
    for b, m, d, mf in line:
        add('bass', bass_note(m, d, mf, 0.06, cut_hi=1900, cut_lo=480, fdec=0.15, s=0.8, wob=(1.3, 3.2), Q=3.5),
            G(b), gain=0.36)
    # six cable plugs: an ascending run, one note per plug (A B C# E F# A)
    for i, (tt, m) in enumerate(zip(plugs, (69, 71, 73, 76, 78, 81))):
        add('fx', zap(), tt, pan=0.5, gain=0.14)
        add('drums', thunk(), tt, pan=0.2, gain=0.20)
        add_echo('music', pluck(m, 0.5, 0.25, 0.9), tt, DOT8, 0.28, 2, pan=0.15, gain=0.32, send=0.2)
        add('music', bell(m + 12, 0.5, 1.5, 0.06, 0.2), tt, pan=0.3, gain=0.07, send=0.2)
    # 30.95 '+ new model' plugged: big E chord, the run continues to C#6
    t = 30.95
    big_hit(t, V['E'], size=0.88, crash_g=0.16, stab_kw=dict(dur=0.4, fmax=9500, d=0.4, key=32))
    add('music', brass((64, 68, 71, 76), 0.3, key=33), t, gain=0.15, send=0.2)
    add('fx', zap(True), t, gain=0.22)
    add_echo('music', bell(85, 1.0, 2.2, 0.1, 0.45), t, E8, 0.3, 3, pan=0.2, gain=0.18, send=0.3)
    add_echo('music', pluck(85, 0.6, 0.3, 0.85), t, E8, 0.3, 2, gain=0.22, send=0.2)
    add('bass', bass_note(40, 0.55, cut_hi=2000, s=0.8), t, gain=0.40)
    # build into the pull-back: two beats of the 96 grid re-phased on 30.95
    G3 = g96(30.95)
    kick_at(G3(1), gain=0.6, depth=0.6)
    add('bass', bass_note(40, 0.55, 28, 0.08, cut_hi=1800, s=0.8), G3(1), gain=0.34)
    rz = res_sweep(V['E11'], 1.13, 450., 9000., Q=3.5, curve=1.5, amp=lambda u: 0.15 + 0.85 * u ** 2.0, key=35,
                   a=0.3)
    add('pad', rz, 31.05, gain=0.11, send=0.2)
    for k in range(12):
        b = 1.0 + (k * 0.25 if k < 4 else 1.0 + (k - 4) * 0.125)
        if G3(b) > 32.17:
            break
        add('drums', snare(0.12, key=k % 3), G3(b), gain=0.06 + 0.02 * k, send=0.06)
    up = noise_sweep(0.85, 600, 7000, lambda u: 0.02 + 0.6 * u ** 2.5, 'build')
    add('fx', up, 31.35, gain=0.12)
    GAPS.append((32.20, 0.25, 0.035))


def sec_pullback():
    G = g96(32.20)                                  # 96 BPM
    blocks = [(0, 'Amaj9', 33), (2, 'F#m9', 42), (3, 'Dmaj9', 38), (4, 'E', 40)]
    u_of = lambda t: float(np.clip((t - 32.2) / 2.2, 0, 1))
    # 32.20 the reveal lands FULL: kick, crash, brass + supersaw Amaj9, bass
    kick_at(32.20, gain=1.08, depth=1.0)
    add('drums', crash(key=2), 32.20, gain=0.28)
    add('drums', clap(), 32.20, gain=0.42, send=0.1)
    add('drums', tom(110., 0.35, 0.10, 0.3), 32.20, gain=0.30)
    add('fx', impact(0.4, 0.08, key=6), 32.20, gain=0.26)
    add('music', stab(V['Amaj9'] + (76,), 0.28, fmax=9500, d=0.2, s=0.2, key=40), 32.20, gain=0.44, send=0.25)
    add('music', brass((57, 61, 64, 69, 73), 0.25, key=41, s=0.35), 32.20, gain=0.28, send=0.25)
    add('bass', bass_note(45, 0.2, 33, 0.03, cut_hi=2600, s=0.5), 32.20, gain=0.20)
    # groove: four-on-the-floor, claps on 2 and 4, hats growing, open hats join
    for b in range(1, 6):
        if G(b) < 35.35:
            kick_at(G(b), gain=0.74 if b == 1 else 0.88)
    for b in (1, 3, 5):
        if G(b) < 35.35:
            add('drums', clap(), G(b), gain=0.24 + 0.14 * u_of(G(b)), send=0.1)
            add('drums', snare(key=b), G(b), gain=0.06 + 0.08 * u_of(G(b)))
    add('drums', crash(key=1), G(2), gain=0.12)
    add('drums', crash(key=4), G(4), gain=0.20)
    for k in range(21):                                # hats: 8ths, then 16ths, then open hats join
        b = 0.25 * k
        if k % 2 == 1 and k < 4:
            continue
        v = 0.07 + 0.07 * min(1.0, k / 10)
        if k % 4 == 2 and k >= 6:
            add('drums', hat(True, k % 3), G(b), pan=0.3, gain=0.10)
        else:
            add('drums', hat(False, k % 4), G(b), pan=0.3, gain=v * (1.4 if k % 2 else 1.0))
    for k in range(12):                                # 16th -> 32nd snare/clap lift into 34.4
        b = 2.25 + 0.25 * k if k < 4 else 3.25 + 0.125 * (k - 4)
        add('drums', snare(0.12, key=k % 3), G(b), gain=0.05 + 0.014 * k, send=0.05)
        if k % 2 == 0:
            add('drums', clap(0.2, 0.05, key=2), G(b), pan=-0.2, gain=0.06 + 0.012 * k)
    # the swell: supersaws opening (resonant), an octave layer fading in, 32.2 -> 34.4
    for ci, (b0, ch, root) in enumerate(blocks):
        t0 = G(b0)
        t1 = G(blocks[ci + 1][0]) if ci + 1 < len(blocks) else 35.5
        u0, u1 = u_of(t0), u_of(t1)
        c0, c1 = 1600 * (5.0 ** u0), 1600 * (5.0 ** u1)
        p = pad(V[ch], t1 - t0, a=0.02, rel=0.1, c0=c0, c1=c1, voices=7, det=0.22, key=40 + ci, Q=1.3)
        nn = p.shape[0]
        tt = t0 + np.arange(nn) / SR
        g = 0.08 + 0.26 * np.clip((tt - 32.2) / 2.2, 0, 1) ** 1.3
        add('pad', p * g[:, None], t0, gain=1.0, send=0.25)
        if t1 > 33.2:                                  # octave-up layer joins from 33.2
            ts = max(t0, 33.2)
            hi = pad(tuple(m + 12 for m in V[ch][1:4]), t1 - ts, a=0.25 if ts == 33.2 else 0.02, rel=0.1,
                     c0=3000, c1=6000, voices=5, det=0.16, key=60 + ci, Q=1.0, hpf=400.)
            nn = hi.shape[0]
            tt = ts + np.arange(nn) / SR
            g = 0.02 + 0.11 * np.clip((tt - 33.2) / 1.2, 0, 1)
            add('pad', hi * g[:, None], ts, gain=1.0, send=0.25)
    wash = noise_sweep(2.2, 1800, 6000, lambda u: 0.01 + 0.06 * u ** 2, 'wash', stereo=1.0)
    add('fx', wash, 32.20, gain=0.40)
    rev_cym = noise_sweep(1.0, 2500, 8000, lambda u: u ** 2.2, 'revcym', stereo=1.0)   # reverse crash into 34.70
    add('drums', rev_cym, 33.70, gain=0.20)
    bass_seq(G, blocks, 0, 5.2, gain=0.42, gfun=lambda t: 0.62 + 0.45 * u_of(t))
    # lead: the motif, then again a third higher landing on the caption (33.90)
    seq = [(32.20, 81, 0.15), (32.356, 85, 0.15), (32.513, 88, 0.15), (32.669, 90, 0.7),
           (33.45, 85, 0.15), (33.606, 88, 0.15), (33.762, 90, 0.13), (33.90, 93, 0.7),
           (34.70, 92, 0.15), (34.856, 90, 0.15), (35.0125, 88, 0.4)]
    for tt, m, d in seq:
        add_echo('music', lead(m, d, scoop=0.6 if d > 0.3 else 0.0, vib=0.22, cut=7500), tt, DOT8, 0.25, 2,
                 gain=0.34 * (0.80 + 0.20 * u_of(tt)), send=0.22)
    add('fx', tick(4000., 0.01, key=50), 33.90, gain=0.2)
    add('music', bell(93, 0.6, 1.0, 0.05, 0.25), 33.90, pan=-0.4, gain=0.07, send=0.3)
    swp = noise_sweep(0.3, 2000, 6000, lambda u: 0.3 * np.sin(np.pi * u) ** 2, 'caption', pan=lambda u: -0.7 + 1.4 * u)
    add('fx', swp, 33.78, gain=0.12)
    # plucky 16th arp across the top, growing
    for k in range(20):
        b = 0.25 * k
        ch = [c for bb, c, _ in blocks if bb <= b][-1]
        notes = sorted(V[ch][1:]) + [V[ch][1] + 12]
        m = notes[(k * 3) % len(notes)] + 12
        add('music', pluck(m, 0.25, 0.12, 0.75), G(b), pan=(-0.5 if k % 2 else 0.5),
            gain=0.06 + 0.05 * u_of(G(b)), send=0.15)


def sec_bills():
    P = Grid([(0, 35.40), (1, 35.65), (2, 35.875)])     # 0.225 s pulse: bills on pulses 5, 9, 13
    # the abrupt turn: dark boom, a clean sine A1 and a high A/Bb string rub
    add('boom', sub_boom(70., 42., 1.0, 0.25, 0.35), STOP_T, gain=0.34)
    kick_at(STOP_T, gain=0.6, depth=0.5, f_hi=120., adec=0.22)
    nn = int(3.45 * SR)
    sub = sine_f(mtof(33), nn) * env_ad(nn, 0.3, 10, 1.0, gate=3.35, r=0.1)
    add('bass', sub, STOP_T, gain=0.10)
    rub = pad((57, 58, 69), 3.4, a=0.5, rel=0.1, c0=1800, c1=3200, voices=3, det=0.10, hpf=140., key=70)
    nn = rub.shape[0]
    tt = np.arange(nn) / SR
    rub = rub * (0.8 + 0.2 * np.sin(TAU * 4.44 * tt))[:, None]     # tremolo on the pulse (1/0.225 s)
    add('pad', rub, STOP_T, gain=0.12, send=0.2)
    hits = {5: ('Am_hit', 45, 0.78, (57, 60, 64, 69)), 9: ('Cm_hit', 48, 0.86, (60, 63, 67, 72)),
            13: ('Ebm_hit', 51, 1.0, (63, 66, 70, 75))}
    for k in range(15):
        t = P(k)
        if k in hits:
            continue
        add('bass', bass_note(45, 0.12, cut_hi=1100, cut_lo=300, fdec=0.04, s=0.4, sub=0.25), t, gain=0.30)
        add('drums', tick(1700. if k % 2 else 2300., 0.015, key=60 + k % 2), t, pan=0.2, gain=0.10)
        if k % 2 == 0 and k not in (0,):
            kick_at(t, gain=0.35, depth=0.35, f_hi=110., adec=0.12)
    # minor motif rising into the first bill: A C E F
    for k, m in zip((1, 2, 3, 4), (69, 72, 76, 77)):
        add('music', lead(m, 0.2 if k < 4 else 0.22, vib=0.0, cut=3500, det=0.12, bright=0.5), P(k), gain=0.22,
            send=0.2)
    # sighs between the bills (b6 -> 5)
    for k, m in ((7, 77), (8, 76), (11, 80), (12, 79)):
        add('music', lead(m, 0.2, vib=0.0, cut=3200, det=0.15, bright=0.4), P(k), gain=0.16, send=0.2)
    # three bill slams, rising A -> C -> Eb, the third the biggest
    for k, (ch, root, size, br) in hits.items():
        t = P(k)
        big_hit(t, V[ch], size=size, clap_g=0.46, noise_g=0.32, stab_g=0.40, crash_g=0.08 + 0.08 * (k == 13),
                stab_kw=dict(dur=0.5, fmin=500, fmax=7000, fdec=0.1, d=0.35, s=0.3, voices=6, det=0.3, key=k, Q=1.6),
                kick_kw=dict(f_hi=140., adec=0.22, drive=2.5))
        add('music', brass(br, 0.4, rel=0.15, f_lo=500, f_hi=4500, fdec=0.15, scoop=1.5, drive=2.4, key=80 + k),
            t, gain=0.26 * size, send=0.15)
        add('boom', sub_boom(70., 44., 0.7, 0.18, 0.22), t, gain=0.18 * size)
        add('bass', bass_note(root - 12 + (12 if root < 46 else 0), 0.4, cut_hi=1400, s=0.6, drive=2.0, sub=0.15),
            t, gain=0.34)
        add('fx', paper(k), t, pan=0.4, gain=0.22)
        add('drums', tom(80., 0.5, 0.16, 0.4), t, gain=0.34 * size)
    # 38.00 the screen turns red: alarm cluster swells into bill 3
    al = res_sweep((57, 58, 63), 0.342, 700., 4000., Q=3.0, curve=1.0, amp=lambda u: u ** 1.5, voices=3, det=0.2,
                   key=71, a=0.25)
    add('pad', al, 38.00, gain=0.16)
    rv = noise_sweep(0.34, 1000, 7000, lambda u: u ** 3, 'red')
    add('fx', fade(rv[:int(0.342 * SR)], 0.01, 0.006), 38.00, gain=0.16)


def sec_wallet():
    # 38.80 wallet appears
    add('fx', pop(), 38.80, gain=0.48)
    add('drums', tick(2600., 0.01, key=90), 38.80, gain=0.28)
    add('fx', impact(0.3, 0.05, 8000., 1500., key=5), 38.80, gain=0.12)
    add('music', stab((57, 60, 64), 0.12, fmin=800, fmax=5000, d=0.1, s=0.2, key=88), 38.80, gain=0.18, send=0.2)
    add('drums', tom(90., 0.3, 0.08, 0.3), 38.80, gain=0.32)
    # accelerating ticks + pulse 38.80 -> 41.0
    t, iv, k = 38.80 + 0.225, 0.225, 0
    while t < 40.96:
        add('drums', tick(1700. if k % 2 else 2300., 0.012, key=70 + k % 2), t, pan=0.2,
            gain=0.08 + 0.05 * (t - 38.8) / 2.2)
        if k % 2 == 0 and t < 40.3:
            add('bass', bass_note(45, 0.1, cut_hi=1000, cut_lo=300, fdec=0.04, s=0.4, sub=0.25), t, gain=0.28)
        iv = max(0.035, iv * 0.89)
        t += iv
        k += 1
    # tremolo string riser: diminished cluster rising, resonant filter opening, tremolo speeding up
    n = int(2.2 * SR)
    tt = np.arange(n) / SR
    u = tt / 2.2
    bend = 7.0 * u ** 1.6
    x = stack((57, 60, 63, 69), n, 4, 0.18, 0.8, key=77, bend=bend)
    x = hp(tvf(x, 600 * 8 ** u, 2.5), 150)
    rate = 5 + 25 * u ** 1.5
    trem = 0.55 + 0.45 * np.sin(TAU * np.cumsum(rate) / SR)
    amp = (0.32 + 0.68 * u ** 1.5) * trem
    add('pad', fade(x * amp[:, None], 0.2, 0.006), 38.80, gain=0.22, send=0.15)
    rub = pad((57, 58), 2.2, a=0.2, rel=0.01, c0=1500, c1=3000, voices=3, det=0.1, hpf=140., key=72)
    add('pad', fade(rub[:int(2.2 * SR)], 0.2, 0.004), 38.80, gain=0.09)
    # three gulps
    for i, t in enumerate((39.82, 40.02, 40.22)):
        add('fx', gulp(i), t, gain=0.62 + 0.12 * i)
        add('drums', tick(1500. + 200 * i, 0.006, key=95 + i), t, gain=0.25 + 0.1 * i)
        kick_at(t, gain=0.40 + 0.05 * i, depth=0.4, f_hi=120., adec=0.1)
    # 40.30 -> 41.00 tremble riser: noise + pitch + snare roll, ending on 41.0
    up = noise_sweep(0.70, 500, 9000, lambda u: 0.02 + u ** 2.2, 'tremble', curve=1.1)
    add('fx', up, 40.30, gain=0.34)
    n = int(0.70 * SR)
    tt = np.arange(n) / SR
    sweep = sine_f(mtof(60 + 30 * (tt / 0.7) ** 1.5), n) * (0.2 + 0.8 * (tt / 0.7) ** 2)
    add('music', fade(sweep, 0.05, 0.006), 40.30, gain=0.09)
    t, k = 40.30, 0
    while t < 40.94:
        add('drums', snare(0.08, key=k % 3), t, gain=0.05 + 0.12 * (t - 40.3) / 0.7)
        t += max(0.03, 0.08 - 0.05 * (t - 40.3) / 0.7)
        k += 1
    GAPS.append((41.00, 0.04, 0.05))          # 50 ms suck-out right before the explosion


def sec_explosion():
    t = 41.00
    kick_at(t, gain=1.0, depth=1.0, dur=0.6, f_hi=165., f_lo=46., pdec=0.045, adec=0.24, click=0.45, drive=3.0)
    add('boom', sub_boom(100., 42., 1.2, 0.30, 0.45), t, gain=0.46)
    add('boom', boom_body(0.9), t, gain=0.60)
    add('fx', impact(1.0, 0.28, 11000., 700., key=9), t, gain=0.62)
    add('drums', crash(2.2, 0.8, key=3), t, gain=0.34)
    add('drums', clap(), t, gain=0.5)
    add('music', brass((57, 58, 64, 69, 70), 0.45, rel=0.25, f_lo=600, f_hi=6500, fdec=0.2, scoop=-2.0, drive=3.0,
                       voices=4, det=0.3, key=91), t, gain=0.42, send=0.2)
    add('music', stab((45, 52, 57, 58, 64), 0.8, fmin=300, fmax=6000, fdec=0.2, d=0.5, s=0.3, voices=6, det=0.35,
                      bend=-3.0, bend_t=0.9, key=41, Q=1.5, hpf=90.), t, gain=0.40, send=0.25)
    # debris: coins and paper falling, thinning out
    rg = rng_for('debris')
    tt = 41.12
    k = 0
    while tt < 42.25:
        f = float(rg.uniform(1400, 3200) * (1 - 0.25 * (tt - 41.1)))
        add('fx', coin(f, 0.12 + 0.1 * rg.random()), tt, pan=float(rg.uniform(-0.8, 0.8)),
            gain=0.10 * (1.15 - 0.6 * (tt - 41.1)), send=0.3)
        if k % 2 == 0:
            add('fx', paper(k % 5), tt + 0.03, pan=float(rg.uniform(-0.8, 0.8)), gain=0.10)
        tt += float(rg.uniform(0.05, 0.11) * (1 + 0.8 * (tt - 41.1)))
        k += 1


def sec_end():
    t = 42.35
    big_hit(t, V['Am_hit'], size=0.85, clap_g=0.45, noise_g=0.22, stab_g=0.42, crash_g=0.10,
            stab_kw=dict(dur=0.22, fmin=600, fmax=7000, d=0.2, key=42))
    add('drums', snare(key=2), t, gain=0.32)
    add('bass', bass_note(33, 0.25, cut_hi=1500), t, gain=0.38)
    # the sad 'wah-wah-wah-waaah': D C# C, then a long B that droops, over an E pedal
    for tt, m, d, last in ((42.55, 62, 0.25, 0), (42.85, 61, 0.25, 0), (43.15, 60, 0.25, 0), (43.45, 59, 0.60, 1)):
        w = wah(m, d, droop=1.0 if last else 0.15, vib=0.35 if last else 0.0, wobble=700. if last else 0.0)
        add('music', w, tt, gain=0.40, send=0.15)
    add('bass', bass_note(40, 0.58, cut_hi=700, cut_lo=320, s=0.9, sub=0.3), 43.45, gain=0.24)
    # four tiles bounce in: dry woodblock plinks, the motif deflated (F E C A)
    for i, (tt, m) in enumerate(zip((43.15, 43.27, 43.39, 43.51), (77, 76, 72, 69))):
        add('music', woodblock(m), tt, pan=-0.5 + 0.33 * i, gain=0.42, send=0.05)
        add('drums', tick(3600. - 250 * i, 0.005, key=80 + i), tt, pan=-0.5 + 0.33 * i, gain=0.32)
    # final tag: 'dun ... dun' E -> A resolves the gag, short and dry
    add('bass', bass_note(40, 0.10, cut_hi=1200, s=0.5), 44.15, gain=0.34)
    add('music', pluck(64, 0.35, 0.12, 0.5), 44.15, gain=0.14, send=0.08)
    add('bass', bass_note(33, 0.22, cut_hi=1000, s=0.5), 44.33, gain=0.38)
    kick_at(44.33, gain=0.42, depth=0.3)
    for m in (57, 60, 64):
        add('music', pluck(m, 0.40, 0.14, 0.5), 44.33, gain=0.10, send=0.08)
    add('music', bell(81, 0.35, 0.8, 0.04, 0.10), 44.33, gain=0.05, send=0.1)


# ============================================================================
# mixing / mastering
# ============================================================================
def make_ir(rt60=1.3, length=1.9, pre=0.014, key='hall'):
    n = int(length * SR)
    t = np.arange(n) / SR
    rg = rng_for('ir', key)
    out = []
    for ch in range(2):
        a = lp(rg.standard_normal(n), 7000) * 10 ** (-3 * t / (rt60 * 0.6))
        b = lp(rg.standard_normal(n), 3000) * 10 ** (-3 * t / rt60)
        ir = 0.6 * a + b
        ir *= np.clip(t / 0.01, 0, 1)
        ir = np.concatenate([np.zeros(int(pre * SR)), ir])
        out.append(ir)
    m = max(np.sqrt(np.sum(o ** 2)) for o in out)
    return [o / m for o in out]


def reverb(send, ir):
    y = np.stack([ss.fftconvolve(send[:, 0], ir[0])[:N], ss.fftconvolve(send[:, 1], ir[1])[:N]], 1)
    return lp(hp(y, 250), 9000)


def duck_env(events, rel=0.14, att=0.004, scale=1.0):
    imp = np.zeros(N)
    for t, d in events:
        i = int(round(t * SR))
        if 0 <= i < N:
            imp[i] += d * scale
    k = int((att + 6 * rel) * SR)
    tk = np.arange(k) / SR
    kern = np.where(tk < att, tk / att, np.exp(-(tk - att) / rel))
    s = ss.fftconvolve(imp, kern)[:N]
    return np.clip(s, 0, 1)


def seg_curve(pts, smooth=0.03):
    """piecewise-constant automation [(t, value), ...], smoothed (no zipper)"""
    c = np.zeros(N)
    for i, (t0, v) in enumerate(pts):
        i0 = int(t0 * SR)
        i1 = int(pts[i + 1][0] * SR) if i + 1 < len(pts) else N
        c[i0:i1] = v
    k = np.hanning(int(smooth * SR))
    k /= k.sum()
    pad_ = len(k)
    cp = np.concatenate([np.full(pad_, c[0]), c, np.full(pad_, c[-1])])
    return ss.fftconvolve(cp, k, mode='same')[pad_:pad_ + N]


def gap_env(gaps, fall=0.006, rise=0.003):
    """tiny 'suck' right before big hits: everything dips so the hit cracks"""
    g = np.ones(N)
    for t, d, pre in gaps:
        i1 = int(round(t * SR))
        i0 = i1 - int(pre * SR)
        nf, nr = int(fall * SR), int(rise * SR)
        seg = np.full(i1 - i0, float(d))
        seg[:nf] = 0.5 * (1 + np.cos(np.pi * np.arange(nf) / nf)) * (1 - d) + d
        seg[-nr:] = d + (1 - d) * 0.5 * (1 - np.cos(np.pi * np.arange(nr) / nr))
        g[i0:i1] = np.minimum(g[i0:i1], seg)
    return g


def tape_stop(x, t0, t1):
    i0, i1 = int(round(t0 * SR)), int(round(t1 * SR))
    n = i1 - i0
    u = np.arange(n) / n
    rate = (1 - u) ** 1.7
    pos = np.cumsum(rate) - rate[0]
    seg = x[i0:i1].copy()
    idx = np.arange(n)
    out = np.stack([np.interp(pos, idx, seg[:, c]) for c in range(2)], 1)
    w = np.ones(n)
    nr = int(0.03 * SR)
    w[-nr:] = 0.5 + 0.5 * np.cos(np.pi * np.arange(nr) / nr)
    out = lp(out, 3000) * 0.5 + out * 0.5           # darken as it slows
    x[i0:i1] = out * w[:, None]


def limiter(x, ceil_db=-2.0, win_ms=3.0, hold_ms=4.0):
    c = 10 ** (ceil_db / 20)
    up = ss.resample_poly(x, 4, 1, axis=0)
    pk = np.abs(up).max(axis=1)[:len(x) * 4].reshape(len(x), 4).max(axis=1)
    g = np.minimum(1.0, c / np.maximum(pk, 1e-9))
    w = int(win_ms * SR / 1000)
    h = int(hold_ms * SR / 1000)
    gm = minimum_filter1d(g, size=2 * (w + h) + 1, mode='nearest')
    k = np.hanning(2 * w + 3)[1:-1]
    k /= k.sum()
    gs = np.minimum(ss.fftconvolve(gm, k, mode='same'), 1.0)
    return x * gs[:, None]


def _biquad(x, b, a):
    b = np.array(b) / a[0]
    a = np.array(a) / a[0]
    return ss.lfilter(b, a, x, axis=0)


def low_shelf(x, f0, gain_db, S=0.8):
    A = 10 ** (gain_db / 40)
    w0 = TAU * f0 / SR
    c, s = np.cos(w0), np.sin(w0)
    al = s / 2 * np.sqrt((A + 1 / A) * (1 / S - 1) + 2)
    b = [A * ((A + 1) - (A - 1) * c + 2 * np.sqrt(A) * al), 2 * A * ((A - 1) - (A + 1) * c),
         A * ((A + 1) - (A - 1) * c - 2 * np.sqrt(A) * al)]
    a = [(A + 1) + (A - 1) * c + 2 * np.sqrt(A) * al, -2 * ((A - 1) + (A + 1) * c),
         (A + 1) + (A - 1) * c - 2 * np.sqrt(A) * al]
    return _biquad(x, b, a)


def high_shelf(x, f0, gain_db, S=0.8):
    A = 10 ** (gain_db / 40)
    w0 = TAU * f0 / SR
    c, s = np.cos(w0), np.sin(w0)
    al = s / 2 * np.sqrt((A + 1 / A) * (1 / S - 1) + 2)
    b = [A * ((A + 1) + (A - 1) * c + 2 * np.sqrt(A) * al), -2 * A * ((A - 1) + (A + 1) * c),
         A * ((A + 1) + (A - 1) * c - 2 * np.sqrt(A) * al)]
    a = [(A + 1) - (A - 1) * c + 2 * np.sqrt(A) * al, 2 * ((A - 1) - (A + 1) * c),
         (A + 1) - (A - 1) * c - 2 * np.sqrt(A) * al]
    return _biquad(x, b, a)


def peaking(x, f0, gain_db, Q=0.8):
    A = 10 ** (gain_db / 40)
    w0 = TAU * f0 / SR
    c, s = np.cos(w0), np.sin(w0)
    al = s / (2 * Q)
    return _biquad(x, [1 + al * A, -2 * c, 1 - al * A], [1 + al / A, -2 * c, 1 - al / A])


def lufs(x):
    import pyloudnorm as pyln
    return pyln.Meter(SR).integrated_loudness(x)


def true_peak_db(x):
    up = ss.resample_poly(x, 4, 1, axis=0)
    return 20 * np.log10(np.max(np.abs(up)) + 1e-12)


# per-section side-chain depth (music / pads / bass) and section trims
MUSIC_DEPTH = [(0, 0.2), (5.35, 0.28), (11.75, 0.10), (21.65, 0.18), (27.75, 0.10), (32.15, 0.28), (35.40, 0.0)]
PAD_DEPTH = [(0, 0.3), (5.35, 0.42), (11.75, 0.12), (21.65, 0.25), (27.75, 0.15), (32.15, 0.42), (35.40, 0.0)]
BASS_DEPTH = [(0, 0.55), (11.75, 0.45), (21.65, 0.6), (35.40, 0.3), (41.6, 0.4)]
# section trims in dB (linear breakpoints): pull-back and bills sit under the explosion
TRIM_DB = [(0, 0.0), (32.30, 0.0), (32.48, -2.8), (33.20, -2.3), (34.40, -0.7), (35.38, -0.7), (35.40, -1.5), (38.70, -1.5),
           (38.80, 0.0),
           (40.98, 0.0), (40.995, 1.2), (41.55, 1.2), (42.2, 0.0)]


def trim_curve():
    tt = np.arange(N) / SR
    return 10 ** (np.interp(tt, [p[0] for p in TRIM_DB], [p[1] for p in TRIM_DB]) / 20)


def region_softclip(x, t0, t1, c, ramp=0.008):
    """tanh saturation of the whole mix inside [t0, t1] (the explosion), so it
    can be dense and loud without the limiter pumping it down"""
    i0, i1 = int(t0 * SR), int(t1 * SR)
    nr = int(ramp * SR)
    w = np.zeros(N)
    w[i0:i1] = 1.0
    w[i0 - nr:i0] = 0.5 - 0.5 * np.cos(np.pi * np.arange(nr) / nr)
    w[i1:i1 + nr] = 0.5 + 0.5 * np.cos(np.pi * np.arange(nr) / nr)
    seg = slice(i0 - nr, i1 + nr)
    y = x.copy()
    y[seg] = x[seg] + (c * np.tanh(x[seg] / c) - x[seg]) * w[seg, None]
    return y


def main():
    sec_intro()
    sec_apps()
    sec_clawd()
    sec_gemini()
    sec_codex()
    sec_openrouter()
    sec_pullback()
    sec_bills()
    sec_wallet()
    sec_explosion()
    sec_end()
    t_score = time.time() - T_START

    if os.environ.get('EL_DEBUG'):
        for k, b in BUS.items():
            m = b.mean(axis=1)
            spec = np.abs(np.fft.rfft(m)) ** 2
            fr = np.fft.rfftfreq(len(m), 1 / SR)
            tot = spec.sum() + 1e-12
            bands = [(0, 60), (60, 250), (250, 2000), (2000, 8000), (8000, 24000)]
            pct = [100 * spec[(fr >= a) & (fr < b_)].sum() / tot for a, b_ in bands]
            print(f'{k:6s} rms {20*np.log10(np.sqrt(np.mean(m**2))+1e-12):6.1f} dB  bands ' + ' '.join(f'{p:5.1f}' for p in pct))
    duck = duck_env(KICKS, rel=0.13)
    dm = seg_curve(MUSIC_DEPTH)
    dp = seg_curve(PAD_DEPTH)
    db = seg_curve(BASS_DEPTH)
    kickb = BUS['kick']
    drums = np.tanh(1.2 * 1.25 * BUS['drums']) / 1.2               # drum-bus glue / soft clip
    bass = 0.80 * BUS['bass'] * (1 - db * duck)[:, None]
    music = 1.40 * hp(BUS['music'], 120) * (1 - dm * duck)[:, None]
    pads = 1.75 * hp(BUS['pad'], 150) * (1 - dp * duck)[:, None]
    fx = hp(BUS['fx'], 60)
    boom = BUS['boom']
    ir = make_ir()
    revA = reverb(BUS['sendA'], ir)
    revB = reverb(BUS['sendB'], ir)
    revA[I_STOP:] = 0.0
    rev_duck = (1 - 0.5 * dm * duck)[:, None]
    mix = kickb + drums + bass + music + pads + fx + boom + revA * 0.55 * rev_duck
    mix *= gap_env(GAPS)[:, None]
    tape_stop(mix, 35.13, STOP_T)
    mix += revB * 0.55
    mix *= trim_curve()[:, None]
    mix = hp(mix, 30, 2)
    mix = low_shelf(mix, 80., -1.5)                               # tame the sub
    mix = peaking(mix, 3200., 2.0, 0.7)                           # presence for lead / claps / plucks
    mix = high_shelf(mix, 5000., 1.0)
    mix = lp(mix, 12500, 2)                                       # no fizz above ~12 kHz
    # master: loudness -> limiter, iterate so the post-limiter loudness is -14
    target = -14.0
    g_db = target - lufs(mix)
    for _ in range(5):
        z = mix * 10 ** (g_db / 20)
        y = limiter(region_softclip(z, 40.998, 41.75, 0.80))
        L = lufs(y)
        if abs(L - target) < 0.04:
            break
        g_db += target - L
    if os.environ.get('EL_DEBUG'):
        seg = z[int(41.0 * SR):int(41.75 * SR)]
        print(f'explosion pre-clip peak {np.abs(seg).max():.2f}, samples over 0.8: {np.mean(np.abs(seg) > 0.8) * 100:.1f}%')
        pre = mix * 10 ** (g_db / 20)
        gr = 20 * np.log10(np.maximum(np.abs(y).max(axis=1), 1e-9) / np.maximum(np.abs(pre).max(axis=1), 1e-9))
        for t0 in np.arange(0, DUR, 1.0):
            s = slice(int(t0 * SR), int(min(t0 + 1, DUR) * SR))
            print(f'  GR {t0:4.0f}s  max {gr[s].min():5.1f} dB', end='' if int(t0) % 5 != 4 else '\n')
        print()
    # fade to silence by the end (safety only: the tag has already decayed)
    nf = int(0.28 * SR)
    end_i = N - int(0.04 * SR)
    w = np.ones(N)
    w[end_i - nf:end_i] = 0.5 + 0.5 * np.cos(np.pi * np.arange(nf) / nf)
    w[end_i:] = 0.0
    w[:96] = np.linspace(0, 1, 96)
    y = y * w[:, None]
    y -= y.mean(axis=0)                                           # no DC
    y[end_i:] = 0.0
    tp = true_peak_db(y)
    if tp > -1.1:
        y *= 10 ** ((-1.1 - tp) / 20)
    # 16-bit with seeded TPDF dither
    rg = rng_for('dither')
    d = (rg.random(y.shape) - rg.random(y.shape))
    q = np.clip(np.round(y * 32767.0 + d), -32767, 32767).astype(np.int16)
    q[end_i:] = 0
    wavfile.write(OUT, SR, q)
    print(f'score {t_score:.1f}s  total {time.time() - T_START:.1f}s  LUFS {lufs(q / 32768.0):.2f}  '
          f'TP {true_peak_db(q / 32768.0):.2f}  -> {OUT}')


if __name__ == '__main__':
    main()
