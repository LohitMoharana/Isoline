"""Place the Isoline Architecture-A carrier board (Nucleo-L432KC carrier) from isoline_afe.net.

Writes isoline_afe.kicad_pcb (unrouted, with GND zones). route_pcb.py then autoroutes it (Freerouting), fills the
zones, and runs DRC. Design rules target a standard JLCPCB/PCBWay 2-layer 1.6 mm board.

Layout (top view, USB of the Nucleo at the top edge):
  left  : electrode header J1 + HIL input/jumpers on the board edge, input protection/bias network
  middle: INA333 (U1) -> servo + x30 gain (U2) -> Sallen-Key + VREF buffer (U4), fast-restore switch (U3)
  right : Nucleo-32 sockets J4 (A-side, analog pins) and J3 (D-side); comparators U5/U6 in a strip just left of J4
          (nothing under the Nucleo); 2xAAA pack off-board on the JST-PH connector BT1
Nucleo-32 socket orientation (UM1956, Arduino Nano layout): pin 1 of both CN3 and CN4 at the end away from USB;
viewed from the top with USB up, CN4 (VIN..A0..D13) is the left column and CN3 (D1..D12) the right one.
CHECK this against your board before ordering (see hardware/README.md, PCB checklist).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pcbnew

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import netparse  # noqa: E402

LIB = "/usr/share/kicad/footprints"
# Fab "flat offer" profile (<= 100 x 100 mm, 2 layers, 1.6 mm FR-4): trace/space >= 0.25 mm, holes >= 0.5 mm
TRACK, CLEAR, VIA_PAD, VIA_HOLE = 0.25, 0.25, 0.9, 0.5
FINE_PITCH = ("U1", "U3", "U5", "U6")     # 0.65 mm pitch: pads trimmed to 0.39 mm wide -> 0.26 mm gaps
FINE_PAD_W = 0.39
W, H = 84.0, 52.0
OX, OY = 100.0, 100.0          # board origin on the KiCad sheet
mm = pcbnew.FromMM

# ref: (x, y, rotation deg, side)   coordinates in mm from the board's top-left corner
J4X, J3X, PTOP = 60.0, 75.24, 7.0           # socket columns; pin 15 (USB end) at y = PTOP
PIN1 = PTOP + 14 * 2.54
PLACE = {
    # ---- connectors / jumpers on the left edge
    "J1": (3.2, 15.0, 0, "F"),      # electrodes: 1 RA(E_NEG), 2 LL(E_POS), 3 RL(E_REF)
    "J2": (3.2, 32.0, 0, "F"),      # HIL input
    "JP2": (8.0, 32.0, 0, "F"), "JP3": (8.0, 40.0, 0, "F"),
    "C5": (12.5, 31.5, 90, "F"), "R8": (12.5, 36.5, 90, "F"), "R9": (12.5, 41.0, 90, "F"),
    # ---- input network
    "R3": (9.0, 16.4, 0, "F"), "R4": (9.0, 13.0, 0, "F"), "R7": (9.0, 22.8, 0, "F"),
    "R5": (13.5, 19.5, 90, "F"), "R6": (13.5, 11.0, 90, "F"),
    # ---- INA333
    "U1": (19.0, 15.0, 0, "F"), "R10": (19.0, 10.0, 0, "F"), "C6": (19.0, 20.0, 180, "F"),
    # ---- servo + gain (U2) and fast restore (U3)
    "U2": (28.5, 15.0, 0, "F"), "C7": (28.5, 20.0, 180, "F"),
    "R11": (24.0, 21.5, 90, "F"), "C8": (24.0, 9.5, 90, "F"),
    "R12": (28.5, 25.0, 0, "F"), "U3": (24.0, 26.5, 0, "F"), "R13": (24.0, 31.0, 0, "F"), "C9": (20.0, 26.5, 90, "F"),
    "R14": (33.0, 21.0, 90, "F"), "R15": (33.0, 9.5, 90, "F"),
    # ---- Sallen-Key + VREF buffer (U4), divider
    "U4": (39.5, 15.0, 0, "F"), "C10": (39.5, 20.0, 180, "F"),
    "R16": (36.5, 25.0, 0, "F"), "R17": (36.5, 28.4, 0, "F"), "C11": (41.0, 25.0, 0, "F"), "C12": (41.0, 28.4, 0, "F"),
    "R1": (36.5, 31.8, 0, "F"), "R2": (36.5, 35.2, 0, "F"), "C3": (41.0, 31.8, 0, "F"), "C4": (41.0, 35.2, 0, "F"),
    # ---- comparators in their own strip left of J4 (nothing under the Nucleo: probe-able, no height constraint)
    "U5": (51.0, 25.5, 0, "F"), "C13": (55.6, 25.5, 90, "F"),
    "U6": (51.0, 32.5, 0, "F"), "C14": (55.6, 32.5, 90, "F"),
    "C15": (51.0, 21.0, 0, "F"), "C16": (51.0, 37.0, 0, "F"), "R18": (55.0, 14.6, 0, "F"),
    # ---- Nucleo-32 sockets (rotated 180: pin 1 away from USB)
    "J4": (J4X, PIN1, 180, "F"), "J3": (J3X, PIN1, 180, "F"),
    # ---- power
    "BT1": (14.0, 5.4, 90, "F"),     # JST-PH to the external 2xAAA pack
    "JP1": (23.0, 3.6, 90, "F"), "C1": (33.5, 3.5, 0, "F"), "C2": (37.5, 3.5, 0, "F"),
    # ---- test points: row along the bottom-left
    "TP1": (17.0, 47.0, 0, "F"), "TP2": (21.0, 47.0, 0, "F"), "TP3": (25.0, 47.0, 0, "F"), "TP4": (29.0, 47.0, 0, "F"),
    "TP5": (33.0, 47.0, 0, "F"), "TP6": (37.0, 47.0, 0, "F"), "TP7": (41.0, 47.0, 0, "F"), "TP8": (45.0, 47.0, 0, "F"),
}
# supply current is < 1 mA: 0.25 mm tracks everywhere, GND as planes


def pt(x, y):
    return pcbnew.VECTOR2I(mm(OX + x), mm(OY + y))


def load_fp(fpid):
    lib, name = fpid.split(":")
    f = pcbnew.FootprintLoad(f"{LIB}/{lib}.pretty", name)
    if f is None:
        raise SystemExit(f"footprint not found: {fpid}")
    f.SetFPID(pcbnew.LIB_ID(lib, name))
    return f


def add_line(board, a, b, layer=pcbnew.Edge_Cuts, w=0.1):
    s = pcbnew.PCB_SHAPE(board); s.SetShape(pcbnew.SHAPE_T_SEGMENT)
    s.SetStart(pt(*a)); s.SetEnd(pt(*b)); s.SetLayer(layer); s.SetWidth(mm(w)); board.Add(s)


def add_text(board, txt, x, y, size=1.0, layer=pcbnew.F_SilkS, mirror=False):
    t = pcbnew.PCB_TEXT(board); t.SetText(txt); t.SetPosition(pt(x, y)); t.SetLayer(layer)
    t.SetTextSize(pcbnew.VECTOR2I(mm(size), mm(size))); t.SetTextThickness(mm(size * 0.15))
    if mirror:
        t.SetMirrored(True)
    board.Add(t)


def zone(board, net, layer, prio):
    z = pcbnew.ZONE(board); z.SetLayer(layer); z.SetNet(net); z.SetAssignedPriority(prio)
    z.SetLocalClearance(mm(0.3)); z.SetMinThickness(mm(0.25))
    z.SetPadConnection(pcbnew.ZONE_CONNECTION_THT_THERMAL)   # SMD solid, PTH thermal
    z.SetThermalReliefGap(mm(0.3)); z.SetThermalReliefSpokeWidth(mm(0.4))
    o = z.Outline(); o.NewOutline()
    for x, y in ((0.5, 0.5), (W - 0.5, 0.5), (W - 0.5, H - 0.5), (0.5, H - 0.5)):
        o.Append(mm(OX + x), mm(OY + y))
    board.Add(z)


def check_nucleo_orientation(board):
    """The Nucleo-32 follows the Arduino Nano layout. Overlay KiCad's own Module:Arduino_Nano footprint (pin k =
    Nano pin k; CN3 pin p = Nano pin p, CN4 pin p = Nano pin 31 - p) rotated 180 deg with pin 1 on J3 pin 1, and
    require every socket pin to coincide with the matching Nano pin and the Nano's 'USB' marking to be at the top."""
    nano = load_fp("Module:Arduino_Nano")
    nano.SetPosition(pt(J3X, PIN1)); nano.SetOrientationDegrees(180)
    npos = {int(p.GetNumber()): p.GetPosition() for p in nano.Pads()}
    fps = {f.GetReference(): f for f in board.GetFootprints()}
    for ref, mapf in (("J3", lambda p: p), ("J4", lambda p: 31 - p)):
        for pad in fps[ref].Pads():
            a, b = pad.GetPosition(), npos[mapf(int(pad.GetNumber()))]
            assert abs(a.x - b.x) < 1000 and abs(a.y - b.y) < 1000, f"{ref}.{pad.GetNumber()} does not match the Nano pinout"
    usb = [t for t in nano.GraphicalItems() if t.GetClass() in ("FP_TEXT", "PCB_TEXT") and t.GetText() == "USB"]
    assert usb and usb[0].GetPosition().y < pt(0, PTOP).y + 1000, "USB end is not at the top edge"
    print("Nucleo-32 sockets match KiCad's Arduino_Nano footprint (USB at the top edge)")


def main():
    comps, nets = netparse.read(HERE / "isoline_afe.net")
    board = pcbnew.BOARD()
    ds = board.GetDesignSettings()
    ds.SetCopperLayerCount(2)
    ds.m_TrackMinWidth = mm(TRACK); ds.m_MinClearance = mm(CLEAR); ds.m_ViasMinSize = mm(VIA_PAD)
    ds.m_MinThroughDrill = mm(VIA_HOLE); ds.m_HoleClearance = mm(CLEAR); ds.m_HoleToHoleMin = mm(CLEAR)
    ds.SetBoardThickness(mm(1.6))
    nc = ds.m_NetSettings.m_DefaultNetClass
    nc.SetTrackWidth(mm(TRACK)); nc.SetClearance(mm(CLEAR)); nc.SetViaDiameter(mm(VIA_PAD)); nc.SetViaDrill(mm(VIA_HOLE))
    netinfo = {}
    for name in nets:
        ni = pcbnew.NETINFO_ITEM(board, name); board.Add(ni); netinfo[name] = ni
    pad_net = {}
    for name, nodes in nets.items():
        for ref, pin in nodes:
            pad_net[(ref, pin)] = name
    missing = set(comps) - set(PLACE)
    if missing:
        raise SystemExit(f"no placement for {sorted(missing)}")
    for ref, (value, fpid, _desc) in comps.items():
        f = load_fp(fpid)
        f.SetReference(ref); f.SetValue(value)
        if ref in FINE_PITCH:
            for pd in f.Pads():
                sz = pd.GetSize()
                if sz.x < sz.y:
                    pd.SetSize(pcbnew.VECTOR2I(mm(FINE_PAD_W), sz.y))
                else:
                    pd.SetSize(pcbnew.VECTOR2I(sz.x, mm(FINE_PAD_W)))
        board.Add(f)                                                   # flip needs a parent board
        x, y, rot, side = PLACE[ref]
        f.SetPosition(pt(x, y)); f.SetOrientationDegrees(rot)
        if side == "B":
            f.Flip(pt(x, y), True)
        for p in f.Pads():
            n = pad_net.get((ref, p.GetNumber()))
            if n:
                p.SetNet(netinfo[n])
                if n == "GND" and ref in ("J3", "J4"):
                    p.SetZoneConnection(pcbnew.ZONE_CONNECTION_FULL)     # socket GND pins: solid to the planes
    # outline + mounting holes
    for a, b in (((0, 0), (W, 0)), ((W, 0), (W, H)), ((W, H), (0, H)), ((0, H), (0, 0))):
        add_line(board, a, b)
    for i, (x, y) in enumerate(((3.5, 3.5), (W - 3.5, 3.5), (3.5, H - 3.5), (W - 3.5, H - 3.5)), 1):
        f = load_fp("MountingHole:MountingHole_3.2mm_M3"); f.SetReference(f"H{i}"); f.SetValue("M3")
        f.SetPosition(pt(x, y)); board.Add(f)
    # GND planes: bottom (solid reference plane) and top (pour)
    zone(board, netinfo["GND"], pcbnew.B_Cu, 0)
    zone(board, netinfo["GND"], pcbnew.F_Cu, 0)
    # silkscreen
    add_text(board, "ISOLINE AFE  rev A", 30.0, 51.0 - 0.2, 1.0)
    add_text(board, "^ USB ^", (J4X + J3X) / 2, 1.9, 1.0)
    add_text(board, "RA", 6.3, 15.0, 0.8); add_text(board, "LL", 6.3, 17.54, 0.8); add_text(board, "RL", 6.3, 20.08, 0.8)
    add_text(board, "HIL IN", 4.5, 36.9, 0.8)
    add_text(board, "CN4", J4X, PTOP - 3.2, 0.8); add_text(board, "CN3", J3X, PTOP - 3.2, 0.8)
    add_text(board, "BAT", 8.0, 5.4, 0.8)
    add_text(board, "github.com/LohitMoharana/Isoline", W / 2, H - 3.0, 0.8, pcbnew.B_SilkS, True)
    check_nucleo_orientation(board)
    out = HERE / "isoline_afe.kicad_pcb"
    board.Save(str(out))
    print(f"placed {len(comps)} parts + 4 holes, {len(nets)} nets -> {out.name}")


if __name__ == "__main__":
    main()
