"""Event-based QRS (beat) detector: integer-only, causal, portable to C (firmware/isoline_core.c mirrors it).

Input: per-slot up/down comparator event counts at 360 Hz. The staircase q[t] = cumsum(up - dn) is the signal
re-quantised to dV steps (an up/down counter in hardware).

  activity E[t] = events in the last WIN slots (ring buffer)
  arm when E >= thr and at least REFRACT slots since the previous beat
  the beat is located at the extreme deviation of the staircase from its level LOOKBACK slots before arming,
  searched over SEARCH slots after arming (so the decision is made SEARCH slots after arming)
  thr = max(THR_MIN, peak_ewma >> THR_SHIFT); peak_ewma tracks the activity peak of accepted beats (1/8 EWMA)
  search-back: if no beat for 1.66 x the RR average, re-arm at half threshold
"""
from __future__ import annotations

import numpy as np

WIN = 12          # 33 ms activity window
REFRACT = 72      # 200 ms
SEARCH = 36       # 100 ms after arming
LOOKBACK = 18     # 50 ms before arming = local baseline
THR_MIN = 3
THR_SHIFT = 2     # thr = peak/4  (chosen on DS1: Se 99.09 %, +P 97.65 %)
STEEP_WIN = 4     # 11 ms
STEEP_SHIFT = 1   # a candidate needs a steepness i.e. >= steep_ewma/2 (QRS edges are steep, T waves are not)


def detect(up: np.ndarray, dn: np.ndarray, win=WIN, refract=REFRACT, search=SEARCH, lookback=LOOKBACK,
           thr_min=THR_MIN, thr_shift=THR_SHIFT, steep_win=STEEP_WIN, steep_shift=STEEP_SHIFT) -> np.ndarray:
    a = (np.asarray(up, np.int64) + np.asarray(dn, np.int64))
    q = np.cumsum(np.asarray(up, np.int64) - np.asarray(dn, np.int64))
    n = len(a)
    E = np.convolve(a, np.ones(win, np.int64))[:n]
    S = np.convolve(a, np.ones(steep_win, np.int64))[:n]     # short-window event count = steepness
    steep = 8 * 6                                               # 8x EWMA of accepted-beat steepness
    beats = []
    peak = 8 * 16                     # 8x-scaled EWMA of activity peaks (initial guess: 16 events / 33 ms)
    rr_avg = 8 * 288                  # 8x-scaled RR average in slots (initial 0.8 s)
    last = -refract
    t = 0
    while t < n:
        thr = max(thr_min, (peak >> 3) >> thr_shift)
        if beats and t - last > (13 * (rr_avg >> 3)) // 8:   # search-back after 1.66 RR without a beat
            thr = max(thr_min, thr >> 1)
        if t - last >= refract and E[t] >= thr:
            s0 = max(t - lookback, 0); s1 = min(t + search, n)
            base = q[s0]
            seg = q[s0:s1] - base
            k = int(np.argmax(np.abs(seg)))
            r = s0 + k
            if r - last < refract:
                t += 1
                continue
            spk = int(S[t:s1].max())
            if spk < max(2, (steep >> 3) >> steep_shift):
                t += 1
                continue
            steep = steep - (steep >> 3) + spk
            epk = int(E[t:s1].max())
            peak = peak - (peak >> 3) + epk
            if beats:
                rr_avg = rr_avg - (rr_avg >> 3) + (r - last)
            beats.append(r); last = r
            t = max(r + refract, t + 1)
            continue
        t += 1
    return np.array(beats, np.int64)


def match(det: np.ndarray, ref: np.ndarray, tol: int = 54):
    """EC57-style matching within +/-150 ms. Returns (tp, fn, fp, index of matched det for each ref or -1)."""
    det = np.sort(det); used = np.zeros(len(det), bool); m = np.full(len(ref), -1)
    j = 0
    for i, r in enumerate(ref):
        while j < len(det) and det[j] < r - tol:
            j += 1
        best, bd = -1, tol + 1
        for k in range(j, min(j + 3, len(det))):
            if not used[k] and abs(det[k] - r) <= tol and abs(det[k] - r) < bd:
                best, bd = k, abs(det[k] - r)
        if best >= 0:
            used[best] = True; m[i] = best
    tp = int((m >= 0).sum())
    return tp, len(ref) - tp, int((~used).sum()), m
