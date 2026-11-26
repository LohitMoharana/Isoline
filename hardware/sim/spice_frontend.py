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
