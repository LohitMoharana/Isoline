"""Photo-style 2D renders of the board (KiCad 7 has no CLI 3D render): composite of per-layer SVG exports."""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pcbnew
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes

HERE = Path(__file__).resolve().parent
PCB = HERE / "isoline_afe.kicad_pcb"
WPX = 2200


def layer_mask(layer, tmp, mirror=False):
    svg, png = Path(tmp) / f"{layer}.svg", Path(tmp) / f"{layer}.png"
    cmd = ["kicad-cli", "pcb", "export", "svg", "--layers", f"{layer},Edge.Cuts", "--page-size-mode", "2",
           "--exclude-drawing-sheet", "--black-and-white", "-o", str(svg), str(PCB)] + (["--mirror"] if mirror else [])
    subprocess.run(cmd, check=True, capture_output=True)
    subprocess.run(["rsvg-convert", "-w", str(WPX), "-b", "white", str(svg), "-o", str(png)], check=True)
    a = np.asarray(Image.open(png).convert("L")) < 128
    return a


def render(side, out):
    mir = side == "B"
    with tempfile.TemporaryDirectory() as tmp:
        edge = layer_mask("Edge.Cuts", tmp, mir)
        cu = layer_mask(f"{side}.Cu", tmp, mir) & ~edge
        mask = layer_mask(f"{side}.Mask", tmp, mir) & ~edge
        silk = layer_mask(f"{side}.SilkS", tmp, mir) & ~edge
    board = binary_fill_holes(edge)
    img = np.zeros(edge.shape + (3,), np.uint8); img[:] = (245, 245, 242)
    img[board] = (14, 74, 40)                      # substrate under solder mask
    img[board & cu] = (34, 118, 62)                # copper under mask
    img[board & mask] = (196, 160, 74)             # exposed copper (ENIG/HASL)
    img[board & silk & ~mask] = (250, 250, 250)
    img[edge] = (30, 30, 30)
    im = Image.fromarray(img)
    # drill holes from the board itself
    b = pcbnew.LoadBoard(str(PCB)); bb = b.GetBoardEdgesBoundingBox()
    ys, xs = np.nonzero(edge)                       # calibrate on the rendered outline (line centres)
    lw = 0.1                                        # Edge.Cuts line width, mm
    bw, bh = bb.GetWidth() / 1e6 - lw, bb.GetHeight() / 1e6 - lw
    sx = (xs.max() - xs.min()) / (bw + lw)
    x0 = bb.GetX() / 1e6 + lw / 2 - (xs.min() / sx + lw / 2)
    y0 = bb.GetY() / 1e6 + lw / 2 - (ys.min() / sx + lw / 2)
    d = ImageDraw.Draw(im)
    holes = [(p.GetPosition(), p.GetDrillSize().x) for f in b.GetFootprints() for p in f.Pads() if p.GetDrillSize().x > 0]
    holes += [(v.GetPosition(), v.GetDrillValue()) for v in b.GetTracks() if v.Type() == pcbnew.PCB_VIA_T]
    for pos, dr in holes:
        x = (pos.x / 1e6 - x0) * sx; y = (pos.y / 1e6 - y0) * sx; r = dr / 2e6 * sx
        if mir:
            x = xs.min() + xs.max() - x
        d.ellipse([x - r, y - r, x + r, y + r], fill=(20, 20, 20))
    im.save(out)
    return out


if __name__ == "__main__":
    for s in ("F", "B"):
        print(render(s, HERE / f"pcb_{'top' if s == 'F' else 'bottom'}.png"))
