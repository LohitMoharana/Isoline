"""MIT-BIH loading in true physical millivolts on a unified 360 Hz timebase.

No per-window normalisation is ever applied: physical comparators work on
fixed voltage thresholds, so amplitudes must stay in mV.
"""
from __future__ import annotations

import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal

from .config import BEAT_SYMBOLS, DS1, DS2, FS, PACED


def resample_to_fs(x: np.ndarray, fs_in: float, fs_out: int = FS) -> np.ndarray:
    """Zero-phase (linear-phase compensated) polyphase resampling."""
    if int(fs_in) == fs_out:
        return x
    frac = Fraction(fs_out, int(fs_in)).limit_denominator(1000)
    return signal.resample_poly(x, frac.numerator, frac.denominator)


def load_mitdb_record(data_dir: str | Path, rec: str, lead: str = "MLII"):
    """Return (signal_mV @360 Hz, beat sample indices @360 Hz, beat symbols)."""
    import wfdb  # imported lazily so tests of other modules do not need it

    path = str(Path(data_dir) / str(rec))
    r = wfdb.rdrecord(path, physical=True)
    if lead == "FIRST":                  # first channel, whatever it is called (EDB, mixed lead names)
        k = 0
    elif lead == "SECOND":                 # the channel that is not MLII (V1/V2/V4/V5)
        others = [i for i, n in enumerate(r.sig_name) if n != "MLII"]
        if "MLII" not in r.sig_name or len(others) != 1:
            raise ValueError(f"record {rec}: cannot pick a second lead from {r.sig_name}")
        k = others[0]
    elif lead not in r.sig_name:
        raise ValueError(f"record {rec}: lead {lead} not in {r.sig_name}")
    else:
        k = r.sig_name.index(lead)      # record 114 has the leads swapped
    if str(r.units[k]).lower() != "mv":
        raise ValueError(f"record {rec}: expected mV, got {r.units[k]}")
    x = np.asarray(r.p_signal[:, k], dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError(f"record {rec}: non-finite samples")

    ann = wfdb.rdann(path, "atr")
    samples = np.asarray(ann.sample, dtype=np.int64)
    symbols = np.asarray(ann.symbol)
    keep = np.isin(symbols, list(BEAT_SYMBOLS))
    samples, symbols = samples[keep], symbols[keep]

    x = resample_to_fs(x, r.fs)
    samples = np.round(samples * FS / r.fs).astype(np.int64)
    return x, samples, symbols


def available_records(data_dir: str | Path, wanted: list[str]):
    """Split `wanted` into (present, missing) according to files on disk."""
    d = Path(data_dir)
    present, missing = [], []
    for rec in wanted:
        ok = all((d / f"{rec}.{ext}").exists() for ext in ("hea", "dat", "atr"))
        (present if ok else missing).append(rec)
    return present, missing


def get_splits(data_dir: str | Path):
    """Return (ds1, ds2) restricted to records that exist on disk."""
    assert not set(DS1) & set(DS2), "DS1/DS2 overlap"
    assert not (set(DS1) | set(DS2)) & set(PACED), "paced record in split"
    ds1, m1 = available_records(data_dir, DS1)
    ds2, m2 = available_records(data_dir, DS2)
    if m1 or m2:
        warnings.warn(f"missing records  DS1: {m1}  DS2: {m2}")
    return ds1, ds2
