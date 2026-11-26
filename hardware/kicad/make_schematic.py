"""Write a native KiCad schematic (isoline_afe.kicad_sch) from the same circuit description as make_netlist.py.

Every pin gets a short wire stub and a net label, so connectivity is explicit and editable in Eeschema.
Symbols are embedded (library "Isoline"), footprints are standard KiCad library names.
Validation: kicad-cli exports a netlist from this schematic and validate_schematic() compares it with the SKiDL
netlist, net by net.
"""
from __future__ import annotations

import re
import subprocess
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_netlist as MN  # noqa: E402  (building the circuit happens at import)
import builtins  # noqa: E402
default_circuit = builtins.default_circuit  # SKiDL keeps the default circuit in builtins

P = 2.54
U = lambda: str(uuid.uuid4())
F = '(effects (font (size 1.27 1.27)))'
FH = '(effects (font (size 1.27 1.27)) hide)'

GROUPS = [  # (title, predicate on part) -> rows on the sheet
    ("Power + reference", lambda p: p.name in ("Battery_2xAAA", "Jumper_3") or (p.ref.startswith(("R", "C")) and any(
        n in ("VREF_DIV",) for n in nets_of(p))) or p.ref == "U3" and False),
]


def nets_of(p):
    return [pin.net.name for pin in p.pins if pin.net is not None]


def symbol_def(p):
    """Return (lib symbol name, sexpr, {pin_num: (x, y, angle)}) for a part, KiCad symbol coordinates (y up)."""
    pins = sorted(p.pins, key=lambda q: int(q.num))
    n = len(pins)
    name = f"{p.name}_{n}"
    if n == 2 and p.name in ("R", "C", "Battery_2xAAA"):
        loc = {pins[0].num: (0, 3.81, 270), pins[1].num: (0, -3.81, 90)}
        body = (f'(rectangle (start -1.016 -2.54) (end 1.016 2.54) (stroke (width 0.254) (type default)) (fill (type none)))'
                if p.name != "C" else
                '(polyline (pts (xy -2.032 -0.762) (xy 2.032 -0.762)) (stroke (width 0.508) (type default)) (fill (type none))) '
                '(polyline (pts (xy -2.032 0.762) (xy 2.032 0.762)) (stroke (width 0.508) (type default)) (fill (type none)))')
        if p.name == "C":
            loc = {pins[0].num: (0, 3.81, 270), pins[1].num: (0, -3.81, 90)}
        w = 0
    else:
        left = pins[: (n + 1) // 2]; right = pins[(n + 1) // 2:][::-1]
        rows = max(len(left), len(right))
        h = (rows + 1) * P
        w = (20.32 if n > 10 else 15.24) if n > 3 else 7.62
        top = h / 2
        loc = {}
        for i, q in enumerate(left):
            loc[q.num] = (-w / 2 - P, round(top - (i + 1) * P, 3), 0)
        for i, q in enumerate(right):
            loc[q.num] = (w / 2 + P, round(top - (i + 1) * P, 3), 180)
        body = (f'(rectangle (start {-w / 2} {top}) (end {w / 2} {-top}) (stroke (width 0.254) (type default)) '
                f'(fill (type background)))')
    pin_s = []
    for q in pins:
        x, y, a = loc[q.num]
        pin_s.append(f'(pin passive line (at {x} {y} {a}) (length {P if w else 1.27}) '
                     f'(name "{q.name}" {F}) (number "{q.num}" {F}))')
    sexpr = (f'(symbol "Isoline:{name}" (pin_names (offset 0.508)) (in_bom yes) (on_board yes)\n'
             f'  (property "Reference" "{p.ref_prefix}" (at 0 {6 + len(pins)} 0) {F})\n'
             f'  (property "Value" "{p.name}" (at 0 {-6 - len(pins)} 0) {F})\n'
             f'  (property "Footprint" "" (at 0 0 0) {FH})\n  (property "Datasheet" "" (at 0 0 0) {FH})\n'
             f'  (symbol "{name}_0_1" {body})\n  (symbol "{name}_1_1" {" ".join(pin_s)})\n)')
    return name, sexpr, loc


ORDER = ["power", "input", "inamp", "gain", "comp", "nucleo", "tp"]


def place_order(parts):
    global place_order_key
    place_order_key = lambda p: ORDER.index(getattr(p, "isoline_group", "tp"))
    return sorted(parts, key=lambda p: (place_order_key(p), 0 if p.ref.startswith(("U", "BT", "J")) else 1, p.ref))


def build():
    parts = list(default_circuit.parts)
    root = U()
    libs, defs = {}, {}
    for p in parts:
        nm, sx, loc = symbol_def(p)
        libs[nm] = sx; defs[p.ref] = (nm, loc)
    out = [f'(kicad_sch (version 20230121) (generator eeschema)\n(uuid {root})\n(paper "A2")\n'
           f'(title_block (title "Isoline AFE carrier (Architecture A)") (comment 1 "generated from hardware/design.py"))\n'
           f'(lib_symbols\n' + "\n".join(libs.values()) + ")\n"]
    order = place_order(parts)
    titles = {0: "1 POWER + VREF", 1: "2 ELECTRODES + HIL", 2: "3 IN-AMP + SERVO", 3: "4 GAIN + LPF",
              4: "5 COMPARATORS", 5: "6 NUCLEO-L432KC", 6: "TEST POINTS"}
    positions, texts = {}, []
    x, colx = 20.32, {}
    from itertools import groupby
    for g, items in groupby(order, key=lambda p: place_order_key(p)):
        items = list(items)
        y = 40.64; col_start = x
        texts.append((titles.get(g, ""), x, 27.94))
        for p in items:
            npins = len(p.pins)
            h = max(25.4, (npins // 2 + 3) * P + 7.62)
            if y + h > 400 * 0 + 395:
                y = 40.64; x += 50.8
            positions[p.ref] = (round(x, 2), round(y + h / 2 - 7.62, 2))
            y += h
        x += 71.12
    for t, tx, ty in texts:
        out.append(f'(text "{t}" (at {tx - 7.62} {ty} 0) (effects (font (size 2.0 2.0) bold) (justify left bottom)) (uuid {U()}))')
    for p in order:
        nm, loc = defs[p.ref]
        X, Y = positions[p.ref]
        pin_uuids = " ".join(f'(pin "{q.num}" (uuid {U()}))' for q in p.pins)
        out.append(
            f'(symbol (lib_id "Isoline:{nm}") (at {X} {Y} 0) (unit 1) (in_bom yes) (on_board yes) (dnp no) (uuid {U()})\n'
            f'  (property "Reference" "{p.ref}" (at {X + 3.81} {Y - 6.35} 0) {F})\n'
            f'  (property "Value" "{p.value}" (at {X + 3.81} {Y + 6.35} 0) {F})\n'
            f'  (property "Footprint" "{p.footprint}" (at {X} {Y} 0) {FH})\n'
            f'  (property "Datasheet" "" (at {X} {Y} 0) {FH})\n'
            f'  {pin_uuids}\n'
            f'  (instances (project "isoline_afe" (path "/{root}" (reference "{p.ref}") (unit 1)))))')
        for q in p.pins:
            if q.net is None:
                continue
            px, py, a = loc[q.num]
            cx, cy = round(X + px, 3), round(Y - py, 3)               # connection point (schematic y down)
            dx = {0: -1, 180: 1, 90: 0, 270: 0}[a]; dy = {0: 0, 180: 0, 90: 1, 270: -1}[a]
            ex, ey = round(cx + dx * P, 3), round(cy + dy * P, 3)    # stub end, away from the body
            out.append(f'(wire (pts (xy {cx} {cy}) (xy {ex} {ey})) (stroke (width 0) (type default)) (uuid {U()}))')
            ang = {0: 180, 180: 0, 90: 270, 270: 90}[a]
            just = "right" if ang == 180 else "left"
            out.append(f'(label "{q.net.name}" (at {ex} {ey} {ang if ang in (0, 180) else 90}) '
                       f'(effects (font (size 1.016 1.016)) (justify {just} bottom)) (uuid {U()}))')
    out.append('(sheet_instances (path "/" (page "1")))\n)')
    (HERE / "isoline_afe.kicad_sch").write_text("\n".join(out))
    (HERE / "isoline_afe.kicad_pro").write_text('{"meta": {"filename": "isoline_afe.kicad_pro", "version": 1}}\n')


def nets_from(netfile):
    s = Path(netfile).read_text(); blk = s[s.index("(nets"):]
    res = {}
    for part in re.split(r"\(net\s", blk)[1:]:
        nm = re.search(r'\(name "([^"]*)"\)', part).group(1).lstrip("/")
        nodes = frozenset(re.findall(r'\(node \(ref "?([^")\s]+)"?\) \(pin "?([^")\s]+)"?\)', part) or
                          re.findall(r'\(ref "([^"]+)"\)\s*\(pin "([^"]+)"\)', part))
        if len(nodes) > 0:
            res[nm] = nodes
    return res


def validate():
    exp = HERE / "_from_sch.net"
    subprocess.run(["kicad-cli", "sch", "export", "netlist", "--format", "kicadsexpr", "-o", str(exp),
                    str(HERE / "isoline_afe.kicad_sch")], check=True, capture_output=True)
    a = {v for v in nets_from(HERE / "isoline_afe.net").values() if len(v) > 1}
    b = {v for v in nets_from(exp).values() if len(v) > 1}
    exp.unlink()
    print(f"SKiDL netlist: {len(a)} nets | schematic (via kicad-cli): {len(b)} nets | identical: {a == b}")
    if a != b:
        print("only in SKiDL:", [sorted(x)[:4] for x in a - b][:5]); print("only in schematic:", [sorted(x)[:4] for x in b - a][:5])
    return a == b


if __name__ == "__main__":
    MN.generate_netlist(file_=str(HERE / "isoline_afe.net"))
    build()
    ok = validate()
    subprocess.run(["kicad-cli", "sch", "export", "svg", "-o", str(HERE), str(HERE / "isoline_afe.kicad_sch")],
                   capture_output=True)
    sys.exit(0 if ok else 1)
