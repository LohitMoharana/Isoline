"""Photo-style 2D renders of the board (KiCad 7 has no CLI 3D render), composited from per-layer SVG exports.

Every layer is exported on the full sheet (same coordinates for all layers), rasterised at a fixed scale, then cropped
around the Edge.Cuts outline with a margin, so nothing is clipped. Outside the board the PNG is transparent, so the
render sits naturally on light and dark pages alike.
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pcbnew
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import binary_fill_holes

HERE = Path(__file__).resolve().parent
PROJ, HW = HERE.parent, HERE.parents[1]          # kicad project dir, hardware/
PCB = PROJ / "isoline_afe.kicad_pcb"
PX_MM = 26                      # raster scale while compositing
OUT_W = 2200                    # final width (px)
MARGIN_MM = 1.5


def sheet_layer(layer, tmp, page_w_mm=None):
    import re
    svg, png = Path(tmp) / f"{layer}.svg", Path(tmp) / f"{layer}.png"
    subprocess.run(["kicad-cli", "pcb", "export", "svg", "--layers", layer, "--page-size-mode", "0",
                    "--exclude-drawing-sheet", "--black-and-white", "-o", str(svg), str(PCB)], check=True, capture_output=True)
    head = svg.read_text()[:2000]
    page_w_mm = float(re.search(r'width="([0-9.]+)mm"', head).group(1))
    subprocess.run(["rsvg-convert", "-w", str(int(round(page_w_mm * PX_MM))), "-b", "white", str(svg), "-o", str(png)], check=True)
    return np.asarray(Image.open(png).convert("L")) < 128


def render(side, out):
    b = pcbnew.LoadBoard(str(PCB))
    page_w = None
    with tempfile.TemporaryDirectory() as tmp:
        edge = sheet_layer("Edge.Cuts", tmp, page_w)
        cu = sheet_layer(f"{side}.Cu", tmp, page_w)
        mask = sheet_layer(f"{side}.Mask", tmp, page_w)
        silk = sheet_layer(f"{side}.SilkS", tmp, page_w)
    ys, xs = np.nonzero(edge)
    m = int(MARGIN_MM * PX_MM)
    y0, y1, x0, x1 = ys.min() - m, ys.max() + m + 1, xs.min() - m, xs.max() + m + 1
    edge, cu, mask, silk = (a[y0:y1, x0:x1] for a in (edge, cu, mask, silk))
    board = binary_fill_holes(edge)
    rgba = np.zeros(edge.shape + (4,), np.uint8)
    rgba[board] = (14, 74, 40, 255)                         # substrate under solder mask
    rgba[board & cu] = (36, 120, 64, 255)                   # copper under mask
    rgba[board & mask] = (201, 164, 78, 255)                # exposed copper (HASL / ENIG)
    rgba[board & silk & ~mask] = (246, 246, 242, 255)       # silkscreen
    rgba[edge & board] = (9, 46, 25, 255)                   # board edge
    im = Image.fromarray(rgba, "RGBA")
    # drill holes, placed from the board geometry with the same sheet scale (1 mm = PX_MM px; sheet origin at 0, 0)
    d = ImageDraw.Draw(im)
    holes = [(p.GetPosition(), p.GetDrillSize().x) for f in b.GetFootprints() for p in f.Pads() if p.GetDrillSize().x > 0]
    holes += [(v.GetPosition(), v.GetDrillValue()) for v in b.GetTracks() if v.Type() == pcbnew.PCB_VIA_T]
    for pos, dr in holes:
        x = pos.x / 1e6 * PX_MM - x0; y = pos.y / 1e6 * PX_MM - y0; r = dr / 2e6 * PX_MM
        d.ellipse([x - r, y - r, x + r, y + r], fill=(0, 0, 0, 0))   # see-through, like a real hole
    if side == "B":
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im = im.resize((OUT_W, round(im.height * OUT_W / im.width)), Image.LANCZOS)
    # soft shadow under the board so it lifts off either page colour
    a = np.asarray(im)[..., 3]
    sh = Image.fromarray((a > 0).astype(np.uint8) * 90).filter(ImageFilter.GaussianBlur(10))
    canvas = Image.new("RGBA", (im.width + 40, im.height + 40), (0, 0, 0, 0))
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0)); shadow.putalpha(0)
    sh_full = Image.new("L", canvas.size, 0); sh_full.paste(sh, (22, 26))
    shadow = Image.merge("RGBA", (Image.new("L", canvas.size, 0),) * 3 + (sh_full,))
    canvas = Image.alpha_composite(canvas, shadow)
    canvas.alpha_composite(im, (20, 20))
    canvas.save(out)
    return out


if __name__ == "__main__":
    for s in ("F", "B"):
        print(render(s, HW / "img" / f"pcb_{'top' if s == 'F' else 'bottom'}.png"))
