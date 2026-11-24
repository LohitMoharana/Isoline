"""Unit tests for the hardware-exact parts. Run: python -m pytest -q tests"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isoline import afe, features  # noqa: E402
from isoline.config import DS1, DS2, DS1_VAL, PACED, AFEConfig, PRE, WIN  # noqa: E402


def test_splits_are_disjoint_and_sized():
    assert len(DS1) == 22 and len(DS2) == 22
    assert not set(DS1) & set(DS2)
    assert not (set(DS1) | set(DS2)) & set(PACED)
    assert set(DS1_VAL) <= set(DS1)


def test_leaky_integrator_is_symmetric_and_decays_to_zero():
    n = 400
    up = np.zeros(n, np.int16); dn = np.zeros(n, np.int16)
    up[10] = 16
    pos = features.leaky_integrator(up, dn)
    neg = features.leaky_integrator(dn, up)      # swapped: negative impulse
    assert pos[10] == 16
    assert np.array_equal(pos, -neg)             # symmetric (no shift asymmetry)
    assert pos[-1] == 0                          # no dead-zone residue
    assert np.all(np.diff(pos[10:]) <= 0)


def test_leaky_integrator_saturates_at_int8():
    up = np.full(500, 20, np.int16); dn = np.zeros(500, np.int16)
    out = features.leaky_integrator(up, dn)
    assert out.max() == 127


def test_level_crossing_tracks_signal_within_one_step():
    rng = np.random.default_rng(0)
    t = np.arange(3600) / 360.0
    x = 1.0 * np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 7 * t)
    cfg = AFEConfig(delta_mv=0.05)
    up, dn = afe.level_crossing(x, cfg, rng)
    ref = x[0] + 0.05 * np.cumsum(up.astype(int) - dn.astype(int))
    assert np.max(np.abs(x - ref)) < 0.05 + 1e-9


def test_flat_signal_is_fully_idle():
    rng = np.random.default_rng(0)
    up, dn = afe.level_crossing(np.zeros(1000), AFEConfig(), rng)
    assert up.sum() == 0 and dn.sum() == 0


def test_rr_fixed_point_and_ewma():
    n = 360 * 12
    up = np.zeros(n, np.int16); dn = np.zeros(n, np.int16)
    r = np.arange(200, n - 200, 360)             # exactly 1 s apart
    acc = features.leaky_integrator(up, dn)
    f = features.extract_record(up, dn, acc, r, np.array(["N"] * len(r)))
    assert np.all(f["scal"][:, 0] == 32)         # Q3.5: 1 s == 32 counts
    assert np.all(f["scal"][:, 1] == 32)         # ewma settled
    assert np.all(f["scal"][:, 2] == 0)          # no RR change


def test_premature_beat_shows_negative_delta_and_low_ratio():
    n = 360 * 14
    up = np.zeros(n, np.int16); dn = np.zeros(n, np.int16)
    r = np.array([200, 560, 920, 1280, 1640, 1820, 2540])   # 1.0 s ... 0.5 s, 2.0 s
    acc = features.leaky_integrator(up, dn)
    f = features.extract_record(up, dn, acc, r, np.array(["N"] * len(r)))
    pre, ewma, delta = f["scal"][:, 0], f["scal"][:, 1], f["scal"][:, 2]
    k = 4                                        # beat at 1820: RR = 0.5 s
    assert pre[k] == 16 and delta[k] == -16 and ewma[k] == 32
    assert pre[k + 1] == 64 and delta[k + 1] == 48


def test_gain_shift_and_ratios():
    n = 360 * 12
    up = np.full(n, 2, np.int16); dn = np.zeros(n, np.int16)   # 512 spikes per window
    r = np.arange(200, n - 200, 360)
    acc = features.leaky_integrator(up, dn)
    f = features.extract_record(up, dn, acc, r, np.array(["N"] * len(r)))
    assert np.all(f["scal"][:, 3] == 512) and np.all(f["scal"][:, 4] == 512)
    assert np.all(f["hist"][:, 0, :] == 1)       # 16 per bin >> k(=4)
    assert np.all(f["scal"][:, 5] == 32)         # pre_rr / rr_ewma == 1.0
    assert np.all(f["scal"][:, 6] == 32)         # spikes / spike_ewma == 1.0
    assert np.all(f["res"] == 0)                 # identical beats: zero residual vs template
    assert np.all(f["scal"][:, 9:13] == 0)       # and zero template distance
    assert np.all(f["scal"][:, 13:16] == np.array([32, 32, 32]))   # next RR == pre RR


def test_window_shapes_and_jitter_range():
    n = 360 * 20
    up = np.zeros(n, np.int16); dn = np.zeros(n, np.int16)
    raw = np.arange(n, dtype=np.float64)         # sample value == sample index
    r = np.arange(500, n - 500, 300)
    acc = features.leaky_integrator(up, dn)
    f = features.extract_record(up, dn, acc, r, np.array(["N"] * len(r)),
                                raw=raw, filt=raw, jitter=5,
                                rng=np.random.default_rng(1))
    assert f["hist"].shape[1:] == (3, 32)
    assert f["raw"].shape[1] == WIN
    starts = f["raw"][:, 0]                      # = window start sample
    offsets = starts - (f["r"] - PRE)
    assert np.all(np.abs(offsets) <= 5)
    assert len(np.unique(offsets)) > 1           # jitter actually varies


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
