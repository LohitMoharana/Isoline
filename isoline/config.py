"""Constants and configuration for the Isoline PoC (Phase 1 / Phase 2)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

# ----------------------------------------------------------------- timebase
FS = 360                # unified timebase (Hz)
WIN = 256               # beat window (samples, ~711 ms)
PRE = 96                # samples before the R-peak annotation
POST = WIN - PRE        # samples from R-peak onwards (decision latency ~ POST/FS)
NBINS = 32              # temporal histogram bins
BIN = WIN // NBINS      # 8 samples per bin
RR_Q = 32               # Q3.5 fixed point: 1 s == 32 counts
HIST_CLIP = 255         # 8-bit spike counters
ACC_FRAC = 4            # extra fractional bits inside the leaky integrator
ACC_SHIFT = 4           # leak: acc -= acc >> 4

# ------------------------------------------------------------------- labels
CLASSES = ["N", "S", "V", "F"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

# AAMI EC57 mapping of MIT-BIH beat annotations. "Q" is never classified.
AAMI_MAP: dict[str, str] = {}
for _s in "NLRej":
    AAMI_MAP[_s] = "N"
for _s in "AaJS":
    AAMI_MAP[_s] = "S"
for _s in "VE":
    AAMI_MAP[_s] = "V"
AAMI_MAP["F"] = "F"
for _s in "/fQ?":
    AAMI_MAP[_s] = "Q"
BEAT_SYMBOLS = set(AAMI_MAP)          # everything else is a non-beat annotation

# ------------------------------------------------- MIT-BIH inter-patient split
PACED = ["102", "104", "107", "217"]  # excluded (AAMI recommendation)
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
       "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
       "223", "230"]                  # De Chazal et al. train set
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
       "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
       "233", "234"]                  # De Chazal et al. test set
# Record-wise validation hold-out taken from DS1 (never from DS2).
DS1_VAL = DS1[2::5]


@dataclass(frozen=True)
class AFEConfig:
    """Behavioural model of INA333 -> DC servo HPF -> LPF -> level-crossing.

    All voltages are *input-referred* millivolts (physical ECG units). The
    hardware threshold at the comparators is ``gain * delta_mv``.
    """
    gain: float = 100.0
    hp_hz: float = 0.5          # 1st-order HPF (use 0.25 for ST-segment test)
    lp_hz: float = 40.0         # 2nd-order Butterworth LPF
    lp_order: int = 2
    delta_mv: float = 0.05      # level-crossing step, input referred
    noise_uv: float = 0.0       # extra white input-referred noise (uV rms)
    hum_mv: float = 0.0         # 50 Hz common-mode residue (mV, input referred)
    offset_sigma_mv: float = 0.0  # comparator threshold mismatch (per record)

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)
