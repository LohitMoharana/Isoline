"""Behavioural AFE emulation: HPF -> LPF -> dual level-crossing comparators.

Everything is causal (filters carry state, no filtfilt) because the hardware
cannot look ahead. The level-crossing stage is the delta modulator the two
TLV3691 comparators implement: whenever the conditioned signal moves
``delta`` away from the tracked reference an up/down event is emitted and the
reference steps by ``delta`` (a feedback DAC / charge step in hardware).
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .config import FS, AFEConfig


def _level_crossing(x, thr_up, thr_dn):
    n = x.shape[0]
    up = np.zeros(n, np.int16)
    dn = np.zeros(n, np.int16)
    ref = x[0]
    for i in range(n):
        d = x[i] - ref
        if d >= thr_up:
            k = int(d // thr_up)
            ref += k * thr_up
            up[i] = k
        elif -d >= thr_dn:
            k = int((-d) // thr_dn)
            ref -= k * thr_dn
            dn[i] = k
    return up, dn


try:  # optional speed-up
    from numba import njit
    _level_crossing_fast = njit(cache=True)(_level_crossing)
except Exception:  # pragma: no cover
    _level_crossing_fast = _level_crossing


def analog_chain(x_mv: np.ndarray, cfg: AFEConfig, rng: np.random.Generator,
                 fs: int = FS) -> np.ndarray:
    """Causal HPF + LPF (input-referred mV in, input-referred mV out)."""
    x = np.asarray(x_mv, dtype=np.float64)
    if cfg.noise_uv > 0:
        x = x + rng.normal(0.0, cfg.noise_uv * 1e-3, x.shape)
    if cfg.hum_mv > 0:
        t = np.arange(x.shape[0]) / fs
        x = x + cfg.hum_mv * np.sin(2 * np.pi * 50.0 * t)
    sos_hp = signal.butter(1, cfg.hp_hz, "highpass", fs=fs, output="sos")
    sos_lp = signal.butter(cfg.lp_order, cfg.lp_hz, "lowpass", fs=fs, output="sos")
    sos = np.vstack([sos_hp, sos_lp])
    zi = signal.sosfilt_zi(sos) * x[0]          # start settled (no power-up transient)
    y, _ = signal.sosfilt(sos, x, zi=zi)
    return y


def level_crossing(y_mv: np.ndarray, cfg: AFEConfig, rng: np.random.Generator):
    """Return per-sample (up_count, down_count) at the 360 Hz readout rate."""
    off = rng.normal(0.0, cfg.offset_sigma_mv, 2) if cfg.offset_sigma_mv > 0 else (0.0, 0.0)
    floor = 0.2 * cfg.delta_mv
    thr_up = max(cfg.delta_mv + off[0], floor)
    thr_dn = max(cfg.delta_mv + off[1], floor)
    return _level_crossing_fast(np.ascontiguousarray(y_mv), thr_up, thr_dn)


def run_afe(x_mv: np.ndarray, cfg: AFEConfig, rng: np.random.Generator):
    """Full chain. Returns (up, dn, conditioned_signal_mV)."""
    y = analog_chain(x_mv, cfg, rng)
    up, dn = level_crossing(y, cfg, rng)
    return up, dn, y
