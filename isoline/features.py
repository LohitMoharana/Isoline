"""Hardware-exact (integer-only) feature extraction.

Everything here maps 1:1 to RTL: shifts, adds, saturating counters. There is
no floating point and no data-dependent normalisation.

Deviations from the literal handoff formulas (deliberate, documented):

* Leaky integrator keeps a 12-bit internal state (4 fractional bits) and
  exposes the top 8 bits. With a plain 8-bit state, ``acc - (acc >> 4)``
  has a dead zone: any |acc| < 16 never decays, and the arithmetic shift is
  asymmetric for negative values. The decay here is sign-magnitude
  (symmetric), and the 4 extra bits shrink the dead zone to +/-1 output LSB.
* EWMAs keep an 8x-scaled accumulator: ``S = S - (S >> 3) + x``, ``ewma =
  S >> 3``. This is the same shift-only datapath but without the truncation
  bias of ``ewma - (ewma >> 3) + (x >> 3)`` (which settles up to 7 counts
  away from the true mean).
* The EWMA fed to the model is the value *before* the current beat is folded
  in, so the model can see how far the current beat deviates (causal).
* Causal gain adaptation: the three histogram channels are shifted right by
  ``k = max(0, bit_length(spike_ewma) - 6)`` (clipped to 6), i.e. a priority
  encoder on the EWMA, with sign-magnitude truncation. This removes most of
  the patient-to-patient amplitude dependence of the fixed-threshold spikes.
"""
from __future__ import annotations

import numpy as np

from .config import (ACC_FRAC, ACC_SHIFT, BIN, FS, HIST_CLIP, NBINS, PRE, RR_Q,
                     WIN)

SCAL_NAMES = ["pre_rr", "rr_ewma", "delta_rr", "beat_spikes", "spike_ewma",
              "rr_ratio", "spike_ratio",            # Q3.5 ratios (32 == 1.0)
              "prev_rr", "rr_ratio_prev",           # previous interval, cur/prev
              "d_up", "d_dn", "d_acc", "d_tot",     # L1 distance to patient template
              # --- NON-causal (one-beat decision delay); excluded unless requested ---
              "post_rr", "post_ratio", "post_pre_ratio",
              # --- extra CAUSAL context (columns 16..20) ---
              "irr_ewma", "prev_rr_ratio2", "d_prev_beat", "prev_d_tot", "prev_rr_ratio",
              # --- long-memory rhythm context (columns 21..25, causal) ---
              "rr_long", "pre_over_long", "ewma_over_long", "irr_long_ratio", "prem_count8",
              # --- very-long RR memory + baseline noise level (columns 26..31, causal) ---
              "rr_vlong", "pre_over_vlong", "long_over_vlong", "quiet_spikes", "noise_ewma", "noise_ratio",
              # --- online per-patient morphology clusters (columns 32..37, causal) ---
              "clu_dist", "clu_pop", "clu_rank", "clu_rr", "clu_active", "clu_dist_ratio", "clu_id"]
EXTRA_CAUSAL = [16, 17, 18, 19, 20]
LONG_CAUSAL = [21, 22, 23, 24, 25]
NOISE_CAUSAL = [26, 27, 28, 29, 30, 31]
CLUSTER_CAUSAL = [32, 33, 34, 35, 36, 37]
N_CAUSAL = 13                                       # first 13 columns are strictly causal
N_DELAYED = 16                                      # columns 13..15 are the next-RR columns

GAIN_T_LOG2 = 5   # histograms are shifted right so the spike-energy EWMA sits below 2**6


def _leaky_integrator(net, shift, frac, lo, hi):
    n = net.shape[0]
    out = np.zeros(n, np.int16)
    state = 0
    slo = lo << frac
    shi = ((hi + 1) << frac) - 1
    for i in range(n):
        if state >= 0:
            dec = state >> shift
        else:
            dec = -((-state) >> shift)
        state = state - dec + (int(net[i]) << frac)
        if state > shi:
            state = shi
        elif state < slo:
            state = slo
        if state >= 0:
            out[i] = state >> frac
        else:
            out[i] = -((-state) >> frac)
    return out


try:
    from numba import njit
    _leaky_fast = njit(cache=True)(_leaky_integrator)
except Exception:  # pragma: no cover
    _leaky_fast = _leaky_integrator


def leaky_integrator(up: np.ndarray, dn: np.ndarray) -> np.ndarray:
    """Signed 8-bit leaky integrator output (int16 container) per sample."""
    net = up.astype(np.int32) - dn.astype(np.int32)
    return _leaky_fast(net, ACC_SHIFT, ACC_FRAC, -128, 127)


def _sm_shift(a, k):
    """Sign-magnitude arithmetic right shift (symmetric decay)."""
    return np.sign(a) * (np.abs(a) >> k)


def extract_record(up, dn, acc, r, symbols, raw=None, filt=None,
                   jitter: int = 0, rng: np.random.Generator | None = None):
    """Per-beat features for one record (strictly causal).

    All beats (including AAMI class Q) update the RR / spike state exactly as
    the hardware would see them; filtering to N/S/V/F happens afterwards.
    ``jitter`` shifts the window start by +/- jitter samples (comparator
    trigger jitter augmentation); the RR/spike state is unaffected.

    Per-patient template: an EWMA (1/8) of the gain-normalised histograms of
    beats whose RR is within 0.75-1.25x of the running RR average. The model
    sees ``hist - template`` (computed before the current beat updates it) and
    the L1 distances, so each beat is judged against that patient's own recent
    normal morphology. The template is 96 x 12-bit registers in RTL.
    """
    n = up.shape[0]
    cu = np.concatenate([[0], np.cumsum(up, dtype=np.int64)])
    cd = np.concatenate([[0], np.cumsum(dn, dtype=np.int64)])
    hist, res, scal, sym, rr_idx, rawW, filtW = [], [], [], [], [], [], []
    rr_state = None
    prev_pre = 0
    prev_pre2 = 0                                  # RR two beats back
    irr_state = 0                                  # 8x EWMA of |dRR| (rhythm irregularity)
    prev_h3 = None                                 # previous beat's normalised histogram
    prev_dtot = 0                                  # previous beat's distance to the template
    prev_rrr = 32                                  # previous beat's RR ratio
    rrl_state = None                               # 64x EWMA of RR (~1 min memory)
    irl_state = 0                                  # 64x EWMA of |dRR|
    rrv_state = None                               # 256x EWMA of RR (~4 min memory)
    nz_state = 0                                   # 16x EWMA of off-QRS spike count (noise level)
    K = 4                                          # online morphology clusters per patient
    cen8 = None                                    # 8x-scaled centroids, (K, 3, NBINS)
    cnt = np.zeros(K, np.int64)                    # 16x EWMA of membership (popularity)
    crr = np.full(K, 8 * 32, np.int64)             # 8x EWMA of RR ratio of members
    dmin_state = 0                                 # 8x EWMA of nearest-cluster distance
    prem_bits = 0                                  # 8-beat shift register: premature (ratio < 28/32)
    sp_state = 0
    sp_primed = False
    tmpl8 = None                                   # 8x-scaled template accumulator
    for i in range(1, len(r)):
        pre = min(int(round((r[i] - r[i - 1]) * RR_Q / FS)), 65535)
        if rr_state is None:
            rr_state = 8 * pre
            rrl_state = 64 * pre
            rrv_state = 256 * pre
            prev_pre = pre
            prev_pre2 = pre
        rr_ewma_prev = rr_state >> 3
        rrl = rrl_state >> 6
        rrv = rrv_state >> 8
        delta = pre - prev_pre
        rr_ratio = min((pre * RR_Q) // max(rr_ewma_prev, 1), 255)
        rr_ratio_prev = min((pre * RR_Q) // max(prev_pre, 1), 255)
        post = (min(int(round((r[i + 1] - r[i]) * RR_Q / FS)), 65535)
                if i + 1 < len(r) else pre)
        post_ratio = min((post * RR_Q) // max(rr_ewma_prev, 1), 255)
        post_pre = min((post * RR_Q) // max(pre, 1), 255)

        j = int(rng.integers(-jitter, jitter + 1)) if (jitter and rng is not None) else 0
        s = int(r[i]) - PRE + j
        e = s + WIN
        if s >= 0 and e <= n:
            edges = s + BIN * np.arange(NBINS + 1)
            hu = np.minimum(cu[edges[1:]] - cu[edges[:-1]], HIST_CLIP)
            hd = np.minimum(cd[edges[1:]] - cd[edges[:-1]], HIST_CLIP)
            ha = acc[edges[1:] - 1]
            beat_spikes = int((cu[e] - cu[s]) + (cd[e] - cd[s]))
            if not sp_primed:
                sp_state = 8 * beat_spikes
                sp_primed = True
            sp_prev = sp_state >> 3
            sp_state = sp_state - (sp_state >> 3) + beat_spikes
            k = min(max(int(sp_prev).bit_length() - (GAIN_T_LOG2 + 1), 0), 6)
            h3 = _sm_shift(np.stack([hu, hd, ha]).astype(np.int32), k)
            sp_ratio = min((beat_spikes * 32) // max(sp_prev, 1), 255)
            quiet = int(hu[:4].sum() + hd[:4].sum() + hu[-6:].sum() + hd[-6:].sum())   # far from the R peak
            nz_prev = nz_state >> 4
            nz_state = nz_state - (nz_state >> 4) + quiet

            if tmpl8 is None:
                tmpl = h3.copy()
                tmpl8 = 8 * h3
            else:
                tmpl = _sm_shift(tmpl8, 3)
                if 24 <= rr_ratio <= 40:           # near-regular beat: update template
                    tmpl8 = tmpl8 - tmpl + h3
            r3 = h3 - tmpl
            # --- online clustering (leader algorithm, causal; features use pre-update state) ---
            if cen8 is None:
                cen8 = np.zeros((K, 3, NBINS), np.int64); cen8[0] = 8 * h3; cnt[0] = 16 * 4
            cen = cen8 >> 3
            dk = np.abs(cen - h3).sum(axis=(1, 2))
            dk = np.where(cnt > 0, dk, 1 << 30)
            jn = int(np.argmin(dk)); dn_ = int(min(dk[jn], 65535))
            tot = int(cnt.sum())
            pop = min((int(cnt[jn]) * 32) // max(tot, 1), 32)
            rank = int((cnt > cnt[jn]).sum())
            c_rr = int(crr[jn] >> 3)
            dthr = max(2 * (dmin_state >> 3), 16)
            clus_feats = [dn_, pop, rank, min(c_rr, 255), int((cnt > 0).sum()),
                          min((dn_ * 32) // max(dmin_state >> 3, 1), 255)]
            if dn_ > dthr and not (cnt > 0).all():
                jn = int(np.flatnonzero(cnt == 0)[0]); cen8[jn] = 8 * h3; crr[jn] = 8 * rr_ratio
            elif dn_ > dthr:
                jn = int(np.argmin(cnt)); cen8[jn] = 8 * h3; crr[jn] = 8 * rr_ratio; cnt[jn] = 0
            else:
                cen8[jn] = cen8[jn] - cen[jn] + h3
                crr[jn] = crr[jn] - (crr[jn] >> 3) + rr_ratio
            cnt = cnt - (cnt >> 4)
            cnt[jn] += 16
            clus_feats.append(jn)                  # assigned cluster slot (used by the 2nd stage)
            dmin_state = dmin_state - (dmin_state >> 3) + min(dn_, 4095)
            d = np.minimum(np.abs(r3).sum(axis=1), 65535)
            dprev = int(min(np.abs(h3 - prev_h3).sum(), 65535)) if prev_h3 is not None else 0
            irr = irr_state >> 3
            pre_pre2 = min((prev_pre * RR_Q) // max(prev_pre2, 1), 255)
            hist.append(h3.astype(np.int16))
            res.append(r3.astype(np.int16))
            scal.append([pre, rr_ewma_prev, delta, beat_spikes, sp_prev, rr_ratio,
                         sp_ratio, prev_pre, rr_ratio_prev,
                         int(d[0]), int(d[1]), int(d[2]), int(d.sum()),
                         post, post_ratio, post_pre,
                         irr, pre_pre2, dprev, prev_dtot, prev_rrr,
                         rrl, min((pre * RR_Q) // max(rrl, 1), 255),
                         min((rr_ewma_prev * RR_Q) // max(rrl, 1), 255),
                         min(((irl_state >> 6) * RR_Q) // max(rrl, 1), 255),
                         bin(prem_bits).count("1"),
                         rrv, min((pre * RR_Q) // max(rrv, 1), 255), min((rrl * RR_Q) // max(rrv, 1), 255),
                         min(quiet, 255), min(nz_prev, 255), min((nz_prev * 32) // max(sp_prev, 1), 255)]
                        + clus_feats)
            prev_h3 = h3
            prev_dtot = int(d.sum())
            sym.append(symbols[i])
            rr_idx.append(int(r[i]))
            if raw is not None:
                rawW.append(raw[s:e].astype(np.float32))
            if filt is not None:
                filtW.append(filt[s:e].astype(np.float32))
        rr_state = rr_state - (rr_state >> 3) + pre
        irr_state = irr_state - (irr_state >> 3) + abs(pre - prev_pre)
        rrl_state = rrl_state - (rrl_state >> 6) + pre
        irl_state = irl_state - (irl_state >> 6) + abs(pre - prev_pre)
        rrv_state = rrv_state - (rrv_state >> 8) + pre
        prem_bits = ((prem_bits << 1) | int(rr_ratio < 28)) & 0xFF
        prev_rrr = rr_ratio
        prev_pre2 = prev_pre
        prev_pre = pre

    out = dict(
        hist=np.asarray(hist, np.int16).reshape(-1, 3, NBINS),
        res=np.asarray(res, np.int16).reshape(-1, 3, NBINS),
        scal=np.asarray(scal, np.int32).reshape(-1, len(SCAL_NAMES)),
        sym=np.asarray(sym), r=np.asarray(rr_idx, np.int64),
    )
    if raw is not None:
        out["raw"] = np.asarray(rawW, np.float32).reshape(-1, WIN)
    if filt is not None:
        out["filt"] = np.asarray(filtW, np.float32).reshape(-1, WIN)
    return out
