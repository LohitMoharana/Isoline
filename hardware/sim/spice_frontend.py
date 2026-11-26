"""Drop-in replacement for isoline.afe.run_afe that runs the ngspice circuit (hardware/sim/make_afe_sim.py netlist,
values from hardware/design.py) on a whole record and applies the comparator level crossing to the circuit output.
Comparator offset and hysteresis are not modelled here (the threshold step equals dV x total gain)."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import design as DZ  # noqa: E402
from make_afe_sim import level_cross, netlist  # noqa: E402

FS = 360


def spice_afe(x_mv: np.ndarray, delta_mv: float = 0.1, offset_mv: float = 100.0):
    x = np.asarray(x_mv, float) - np.median(x_mv[: FS * 10])
    t = np.arange(len(x)) / FS
    with tempfile.TemporaryDirectory() as d:
        cir = netlist(t, x, offset_mv).replace("tran 0.5m", f"tran {1000 / FS:.4f}m").replace(
            f" 0 0.5m uic", f" 0 {1000 / FS:.4f}m uic")
        (Path(d) / "afe.cir").write_text(cir)
        r = subprocess.run(["ngspice", "-b", "afe.cir"], capture_output=True, text=True, cwd=d)
        if r.returncode != 0 or not (Path(d) / "afe_out.txt").exists():
            raise RuntimeError(r.stdout[-1500:] + r.stderr[-1500:])
        out = np.loadtxt(Path(d) / "afe_out.txt")
    vo = np.interp(t, out[:, 0], out[:, 1])
    y_mv = (vo - DZ.VREF) / DZ.TOTAL_GAIN * 1e3
    up, dn = level_cross(y_mv, delta_mv)
    return up.astype(np.int32), dn.astype(np.int32), y_mv


# ---------------------------------------------------------------- fast path: ngspice AC response + rail clipping
_H = None


def circuit_response(fmin=1e-4, fmax=180.0, pts=400):
    """Small-signal transfer function V(out)/V(ecg) of the same netlist, computed by ngspice .ac."""
    t = np.arange(10) / FS
    cir = netlist(t, np.zeros(10), 0.0)                      # small-signal response is independent of the offset
    cir = cir.replace("VECG ecg 0 PWL(", "VECG ecg 0 DC 0 AC 1 PWL(")
    # linear versions of the rail clamps for the small-signal solve (the transient path keeps the clamps)
    import re as _re
    cir = cir.replace("Bout out 0 V = max(min(V(n2), V(vcc)-0.012), V(vee)+0.012)", "Bout out 0 V = V(n2)")
    cir = _re.sub(r"BINA ina_o 0 V = max\(min\((.*?), 2\.95\), 0\.05\)", r"BINA ina_o 0 V = \1", cir)
    cir = cir.replace(".options", f".nodeset v(srv)={DZ.VREF} v(s_in)={DZ.VREF} v(ina_o)={DZ.VREF} v(g_out)={DZ.VREF} v(out)={DZ.VREF}\n.options")
    ctl = cir[cir.index(".control"):cir.index(".endc")]
    cir = cir.replace(ctl, f".control\nop\nac dec {pts} {fmin} {fmax}\nwrdata h.txt v(out)\nquit\n")
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "h.cir").write_text(cir)
        subprocess.run(["ngspice", "-b", "h.cir"], capture_output=True, text=True, cwd=d, check=True)
        a = np.loadtxt(Path(d) / "h.txt")
    f = a[:, 0]; H = a[:, 1] + 1j * a[:, 2]
    return f, H


def spice_afe_fast(x_mv: np.ndarray, delta_mv: float = 0.1, offset_mv: float = 100.0):
    """Apply the ngspice-computed response to a whole record (FFT, zero-padded), then clip to the rails as the
    output stage would. Exact for the linear operating range; clipping is checked explicitly."""
    global _H
    if _H is None:
        _H = circuit_response()
    f_h, H = _H
    x = np.asarray(x_mv, float) * 1e-3
    x = x - np.median(x[: FS * 10])
    n = len(x); pad = int(2 ** np.ceil(np.log2(n + FS * 120)))
    X = np.fft.rfft(x, pad); fr = np.fft.rfftfreq(pad, 1 / FS)
    Hi = np.interp(fr, f_h, H.real, left=0, right=0) + 1j * np.interp(fr, f_h, H.imag, left=0, right=0)
    y = np.fft.irfft(X * Hi, pad)[:n]
    vo = np.clip(DZ.VREF + y, 0.012, DZ.VSUPPLY - 0.012)
    y_mv = (vo - DZ.VREF) / DZ.TOTAL_GAIN * 1e3
    up, dn = level_cross(y_mv, delta_mv)
    return up.astype(np.int32), dn.astype(np.int32), y_mv, float(np.mean((vo <= 0.013) | (vo >= DZ.VSUPPLY - 0.013)))
