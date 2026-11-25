"""Dataset assembly: records -> AFE -> hardware-exact features (+ caching)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import zlib

import numpy as np

from . import afe, data, features
from .config import AAMI_MAP, CLASS_IDX, AFEConfig

VERSION = 9  # bump when feature definitions change (invalidates caches)


def _label(sym: str) -> int:
    c = AAMI_MAP.get(sym, "Q")
    return CLASS_IDX.get(c, -1)


def _rec_id(rec) -> int:
    """Numeric record id (MIT-BIH style) or a stable hash for names like 'I01'."""
    try:
        return int(rec)
    except ValueError:
        return int(zlib.crc32(str(rec).encode()) % 100000) + 1000


def build_dataset(records, data_dir, afe_cfg: AFEConfig, jitter: int = 0,
                  copies: int = 1, seed: int = 0, want_dense: bool = False,
                  cache_dir=None, verbose: bool = True, lead: str = "MLII") -> dict:
    """Build features for `records`.

    Returns dict with: hist (N,3,32) int16, scal (N,5) int32, y (N,) int8,
    rec (N,) int32, r (N,) int64, optional raw/filt (N,256) float32, and
    `stats` = idle sparsity numbers measured once per record (not per copy).
    """
    key = hashlib.sha1(json.dumps(dict(
        v=VERSION, afe=afe_cfg.key(), rec=list(records), jitter=jitter,
        copies=copies, seed=seed, dense=want_dense, dir=str(Path(data_dir).resolve()), lead=lead,
    ), sort_keys=True).encode()).hexdigest()[:16]
    cache_path = None
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_path = Path(cache_dir) / f"ds_{key}.npz"
        if cache_path.exists():
            z = np.load(cache_path, allow_pickle=False)
            ds = {k: z[k] for k in z.files if k != "stats"}
            ds["stats"] = json.loads(str(z["stats"]))
            if verbose:
                print(f"[cache] loaded {cache_path.name} ({len(ds['y'])} beats)")
            return ds

    parts = []
    stats = dict(samples=0, zero_slots=0, up=0, dn=0, seconds=0.0)
    for rec in records:
        x, r, sym = data.load_mitdb_record(data_dir, rec, lead=lead)
        rng = np.random.default_rng(seed * 100003 + _rec_id(rec))
        up, dn, filt = afe.run_afe(x, afe_cfg, rng)
        acc = features.leaky_integrator(up, dn)
        stats["samples"] += len(x)
        stats["zero_slots"] += int(np.sum((up == 0) & (dn == 0)))
        stats["up"] += int(up.sum())
        stats["dn"] += int(dn.sum())
        stats["seconds"] += len(x) / 360.0
        for _ in range(copies):
            f = features.extract_record(
                up, dn, acc, r, sym,
                raw=x if want_dense else None, filt=filt if want_dense else None,
                jitter=jitter, rng=rng)
            f["rec"] = np.full(len(f["sym"]), _rec_id(rec), np.int32)
            parts.append(f)
        if verbose:
            print(f"  rec {rec}: {len(r)} beats, "
                  f"idle sparsity {np.mean((up == 0) & (dn == 0)):.3f}")

    ds: dict = {}
    for k in parts[0]:
        ds[k] = np.concatenate([p[k] for p in parts], axis=0)
    ds["y"] = np.array([_label(s) for s in ds.pop("sym")], np.int8)
    keep = ds["y"] >= 0                                  # drop AAMI class Q
    ds = {k: v[keep] for k, v in ds.items()}
    stats["idle_sparsity"] = stats["zero_slots"] / max(stats["samples"], 1)
    stats["spikes_per_s"] = (stats["up"] + stats["dn"]) / max(stats["seconds"], 1e-9)
    ds["stats"] = stats
    if cache_path is not None:
        np.savez_compressed(cache_path, stats=json.dumps(stats),
                            **{k: v for k, v in ds.items() if k != "stats"})
    return ds
