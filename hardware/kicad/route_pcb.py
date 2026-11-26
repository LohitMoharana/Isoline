"""Autoroute isoline_afe.kicad_pcb with Freerouting (Specctra DSN/SES round trip), fill the GND zones, save.

usage: python3 route_pcb.py <freerouting.jar> [max_passes]
"""
import subprocess
import sys
from pathlib import Path

import pcbnew

HERE = Path(__file__).resolve().parent
PCB = HERE / "isoline_afe.kicad_pcb"


def main():
    jar = sys.argv[1]; passes = sys.argv[2] if len(sys.argv) > 2 else "40"
    b = pcbnew.LoadBoard(str(PCB))
    for t in list(b.GetTracks()):                       # start from an unrouted board
        b.Remove(t)
    dsn, ses = HERE / "isoline_afe.dsn", HERE / "isoline_afe.ses"
    if not pcbnew.ExportSpecctraDSN(b, str(dsn)):
        raise SystemExit("DSN export failed")
    r = subprocess.run(["java", "-Djava.awt.headless=true", "-jar", jar, "-de", str(dsn), "-do", str(ses), "-mp", passes],
                       capture_output=True, text=True, timeout=1800)
    print("\n".join((r.stdout + r.stderr).splitlines()[-8:]))
    if not ses.exists():
        raise SystemExit("Freerouting produced no SES")
    if not pcbnew.ImportSpecctraSES(b, str(ses)):
        raise SystemExit("SES import failed")
    pcbnew.ZONE_FILLER(b).Fill(b.Zones())
    b.Save(str(PCB))
    n_tr = sum(1 for t in b.GetTracks() if t.Type() == pcbnew.PCB_TRACE_T)
    n_via = sum(1 for t in b.GetTracks() if t.Type() == pcbnew.PCB_VIA_T)
    print(f"routed: {n_tr} track segments, {n_via} vias -> {PCB.name}")


if __name__ == "__main__":
    main()
