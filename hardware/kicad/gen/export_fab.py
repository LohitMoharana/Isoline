"""Fabrication outputs for the routed board: KiCad DRC report, Gerbers + Excellon drill (zipped for upload),
pick-and-place, all under hardware/fab/."""
import subprocess
import zipfile
from pathlib import Path

import pcbnew

HERE = Path(__file__).resolve().parent
PROJ, HW = HERE.parent, HERE.parents[1]
PCB, FAB = PROJ / "isoline_afe.kicad_pcb", HW / "fab"
LAYERS = "F.Cu,B.Cu,F.Paste,F.SilkS,B.SilkS,F.Mask,B.Mask,Edge.Cuts"


def main():
    ger = FAB / "gerbers"; ger.mkdir(parents=True, exist_ok=True)
    for f in ger.iterdir():
        f.unlink()
    run = lambda *a: subprocess.run(["kicad-cli", "pcb", "export", *a, str(PCB)], check=True, capture_output=True)
    run("gerbers", "--layers", LAYERS, "--subtract-soldermask", "-o", f"{ger}/")
    run("drill", "--format", "excellon", "--excellon-separate-th", "-o", f"{ger}/")
    run("pos", "--format", "csv", "--units", "mm", "--side", "front", "-o", str(FAB / "isoline_afe-pos.csv"))
    with zipfile.ZipFile(FAB / "isoline_afe_gerbers.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(ger.iterdir()):
            if f.suffix != ".gbrjob":
                z.write(f, f.name)
    b = pcbnew.LoadBoard(str(PCB))
    pcbnew.WriteDRCReport(b, str(FAB / "drc_report.txt"), pcbnew.EDA_UNITS_MILLIMETRES, True)
    summary = [l for l in (FAB / "drc_report.txt").read_text().splitlines() if l.startswith("** Found")]
    print("\n".join(summary)); print(f"fabrication files -> {FAB}")


if __name__ == "__main__":
    main()
