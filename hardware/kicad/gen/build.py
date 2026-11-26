"""Rebuild the whole board from hardware/design.py, in the required order:
netlist + BOM -> schematic (checked against the netlist) -> placement -> routing -> renders -> fabrication files.
(The schematic step rewrites the project file; the placement step then stores the fab design rules in it.)

    python hardware/kicad/gen/build.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for step in ("make_netlist.py", "make_schematic.py", "make_pcb.py", "route_grid.py", "render_pcb.py", "export_fab.py"):
    print(f"== {step}", flush=True)
    r = subprocess.run([sys.executable, str(HERE / step)], capture_output=True, text=True, cwd=HERE)  # SKiDL scratch stays in gen/
    print("\n".join(l for l in r.stdout.splitlines() if not l.startswith("WARNING"))[-1500:])
    if r.returncode:
        print(r.stderr[-2000:]); sys.exit(r.returncode)
