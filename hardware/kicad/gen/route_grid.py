"""Small grid router for the Isoline carrier board (instead of an external autorouter).

  grid 0.1 mm, two layers (F.Cu preferred; B.Cu costs more because it is the GND reference plane)
  clearance: a node is usable by net n only if its distance to any other net's copper (pads, earlier tracks/vias),
             to holes and to the board edge is >= clearance + track_width/2 (+ rasterisation margin); vias likewise
             with the via radius, on both layers
  search   : A* (8-connected, 45-degree moves, small turn penalty), multi-terminal: grow a tree pad by pad
  GND      : not routed as tracks; every SMD GND pad gets a short stub to a stitching via into the B.Cu plane,
             plus a lattice of stitching vias for the top pour; zones are filled at the end
  repair   : nets that fail are moved to the front of the order and everything is rerouted (a few passes)
KiCad's DRC (kicad-cli pcb drc) is the judge afterwards; this script only aims to make that pass.
"""
from __future__ import annotations

import heapq
import math
import sys
import time
from pathlib import Path

import numpy as np
import pcbnew

HERE = Path(__file__).resolve().parent
PROJ, HW = HERE.parent, HERE.parents[1]          # kicad project dir, hardware/
PCB = PROJ / "isoline_afe.kicad_pcb"
P = 0.1                    # grid pitch, mm
OX, OY, W, H = 100.0, 100.0, 84.0, 52.0
TRACK_W, CLR, VIA_D, VIA_DRILL = 0.25, 0.25, 0.9, 0.5      # fab flat-offer profile
MARGIN = 0.012          # exact geometry; only covers mid-step approach between grid nodes
EDGE = 0.5
VIA_COST, BCU_FACTOR, TURN = 25.0, 2.5, 0.4
NX, NY = int(round(W / P)) + 1, int(round(H / P)) + 1
DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
mm = pcbnew.FromMM


def cells_rect(x0, y0, x1, y1):
    i0, i1 = math.ceil((x0 - OX) / P - 1e-9), math.floor((x1 - OX) / P + 1e-9)
    j0, j1 = math.ceil((y0 - OY) / P - 1e-9), math.floor((y1 - OY) / P + 1e-9)
    return max(i0, 0), min(i1, NX - 1), max(j0, 0), min(j1, NY - 1)


def paint_disk(arr, cx, cy, r, val):
    i0, i1, j0, j1 = cells_rect(cx - r, cy - r, cx + r, cy + r)
    if i0 > i1 or j0 > j1:
        return
    ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="xy")
    m = (OX + ii * P - cx) ** 2 + (OY + jj * P - cy) ** 2 <= r * r + 1e-9
    sub = arr[j0:j1 + 1, i0:i1 + 1]
    sub[m] = val


def paint_pad(arr, shape, x0, y0, x1, y1, val, shrink):
    """Rasterise a pad (circle, oval/stadium, rect/roundrect as rect) shrunk by `shrink` mm on every side."""
    x0, y0, x1, y1 = x0 + shrink, y0 + shrink, x1 - shrink, y1 - shrink
    if x1 <= x0 or y1 <= y0:
        return
    if shape in (pcbnew.PAD_SHAPE_CIRCLE, pcbnew.PAD_SHAPE_OVAL):
        w, h = x1 - x0, y1 - y0; r = min(w, h) / 2
        if w >= h:
            ax, ay, bx, by = x0 + r, (y0 + y1) / 2, x1 - r, (y0 + y1) / 2
        else:
            ax, ay, bx, by = (x0 + x1) / 2, y0 + r, (x0 + x1) / 2, y1 - r
        i0, i1, j0, j1 = cells_rect(x0, y0, x1, y1)
        if i0 > i1 or j0 > j1:
            return
        ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="xy")
        px, py = OX + ii * P, OY + jj * P
        dx, dy = bx - ax, by - ay; L2 = dx * dx + dy * dy
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0, 1) if L2 > 0 else 0.0
        m = (px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2 <= r * r + 1e-9
        arr[j0:j1 + 1, i0:i1 + 1][m] = val
    else:
        i0, i1, j0, j1 = cells_rect(x0, y0, x1, y1)
        arr[j0:j1 + 1, i0:i1 + 1] = val


def window(x0, y0, x1, y1, pad):
    i0, i1, j0, j1 = cells_rect(x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="xy")
    return i0, i1, j0, j1, OX + ii * P, OY + jj * P


def seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay; L2 = dx * dx + dy * dy
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0, 1) if L2 > 0 else 0.0
    return np.hypot(px - ax - t * dx, py - ay - t * dy)


def shape_dist(kind, prm, reach):
    """Exact distance from grid nodes (within `reach` mm) to a copper shape. Returns (i0, i1, j0, j1, d)."""
    if kind == "rect":
        x0, y0, x1, y1 = prm
        i0, i1, j0, j1, px, py = window(x0, y0, x1, y1, reach)
        d = np.hypot(np.maximum(np.maximum(x0 - px, 0), px - x1), np.maximum(np.maximum(y0 - py, 0), py - y1))
    else:                                   # capsule: segment (ax, ay)-(bx, by) with radius r (circle when a == b)
        ax, ay, bx, by, r = prm
        i0, i1, j0, j1, px, py = window(min(ax, bx) - r, min(ay, by) - r, max(ax, bx) + r, max(ay, by) + r, reach)
        d = np.maximum(seg_dist(px, py, ax, ay, bx, by) - r, 0)
    return i0, i1, j0, j1, d


def pad_shape(shape, x0, y0, x1, y1):
    if shape in (pcbnew.PAD_SHAPE_CIRCLE, pcbnew.PAD_SHAPE_OVAL):
        w, h = x1 - x0, y1 - y0; r = min(w, h) / 2
        if w >= h:
            return "cap", (x0 + r, (y0 + y1) / 2, x1 - r, (y0 + y1) / 2, r)
        return "cap", ((x0 + x1) / 2, y0 + r, (x0 + x1) / 2, y1 - r, r)
    return "rect", (x0, y0, x1, y1)


class Router:
    def __init__(self, board):
        self.b = board
        self.own = np.zeros((2, NY, NX), np.int32)          # copper owner: net code, 0 empty, -1 keep-out
        self.padid = np.full((2, NY, NX), -1, np.int32)
        self.pads = []                                      # (net, layers, bbox mm, centre)
        self.padref = []
        self.shapes = []                                    # (net, layers, kind, params): all copper, exact
        self.holes = []                                     # (x, y, drill radius) of every drilled hole, any net
        self.netname = {n.GetNetCode(): n.GetNetname() for n in board.GetNetInfo().NetsByNetcode().values()}
        self.gnd = board.FindNet("GND").GetNetCode()
        for l in (0, 1):                                    # board-edge keep-out
            self.own[l, :, :int(EDGE / P)] = -1; self.own[l, :, NX - int(EDGE / P):] = -1
            self.own[l, :int(EDGE / P), :] = -1; self.own[l, NY - int(EDGE / P):, :] = -1
        for f in board.GetFootprints():
            for pad in f.Pads():
                bb = pad.GetBoundingBox()
                x0, y0 = bb.GetX() / 1e6, bb.GetY() / 1e6
                x1, y1 = x0 + bb.GetWidth() / 1e6, y0 + bb.GetHeight() / 1e6
                cx, cy = pad.GetPosition().x / 1e6, pad.GetPosition().y / 1e6
                layers = [l for l, L in ((0, pcbnew.F_Cu), (1, pcbnew.B_Cu)) if pad.IsOnLayer(L)]
                if pad.GetDrillSize().x > 0:
                    self.holes.append((cx, cy, pad.GetDrillSize().x / 2e6))
                if pad.GetAttribute() == pcbnew.PAD_ATTRIB_NPTH:
                    self.shapes.append((-1, (0, 1), "cap", (cx, cy, cx, cy, (x1 - x0) / 2)))
                    for l in (0, 1):
                        paint_disk(self.own[l], cx, cy, (x1 - x0) / 2, -1)
                    continue
                net = pad.GetNetCode()
                if net <= 0:
                    net = -1
                k = len(self.pads); self.pads.append((net, layers, (x0, y0, x1, y1), (cx, cy), pad))
                self.padref.append(f"{f.GetReference()}.{pad.GetNumber()}")
                kind, prm = pad_shape(pad.GetShape(), x0, y0, x1, y1)
                self.shapes.append((net, tuple(layers), kind, prm))
                for l in layers:
                    paint_pad(self.own[l], pad.GetShape(), x0, y0, x1, y1, net, 0.0)
                    paint_pad(self.padid[l], pad.GetShape(), x0, y0, x1, y1, k, 0.12)   # target: 0.12 mm inside
        self.tracks = []        # (net, layer, [(i, j), ...]) polylines
        self.vias = []          # (net, i, j)

    # ------------------------------------------------------------------ maps
    def free_maps(self, net):
        r_tr = CLR + TRACK_W / 2 + MARGIN
        r_via = CLR + VIA_D / 2 + MARGIN
        ii, jj = np.meshgrid(np.arange(NX), np.arange(NY), indexing="xy")
        px, py = ii * P, jj * P
        edge = np.minimum(np.minimum(px, W - px), np.minimum(py, H - py)) - (EDGE - CLR)   # copper >= EDGE from edge
        dists = [edge.copy(), edge.copy()]
        for n, layers, kind, prm in self.shapes:
            if n == net and n != -1:
                continue
            i0, i1, j0, j1, d = shape_dist(kind, prm, r_via + 0.2)
            if i0 > i1 or j0 > j1:
                continue
            for l in layers:
                sub = dists[l][j0:j1 + 1, i0:i1 + 1]
                np.minimum(sub, d, out=sub)
        free = [dists[0] >= r_tr, dists[1] >= r_tr]
        via = (dists[0] >= r_via) & (dists[1] >= r_via)
        for hx, hy, hr in self.holes:                       # hole-to-hole spacing applies to every net
            rr = hr + VIA_DRILL / 2 + 0.3
            i0, i1, j0, j1 = cells_rect(hx - rr, hy - rr, hx + rr, hy + rr)
            sub = via[j0:j1 + 1, i0:i1 + 1]
            jj, ii = np.mgrid[j0:j1 + 1, i0:i1 + 1]
            sub[(OX + ii * P - hx) ** 2 + (OY + jj * P - hy) ** 2 < rr * rr] = False
        return free, via

    def paint_path(self, net, path):
        r = TRACK_W / 2
        for (i, j, l) in path:
            paint_disk(self.own[l], OX + i * P, OY + j * P, r, net)
        for a, b in zip(path, path[1:]):
            if a[2] != b[2]:
                for l in (0, 1):
                    paint_disk(self.own[l], OX + a[0] * P, OY + a[1] * P, VIA_D / 2, net)

    # ------------------------------------------------------------------ search
    def astar(self, net, sources, target_pads, free, via, goal_fn=None):
        fF, fB = free[0].ravel(), free[1].ravel()
        fl = (fF, fB); vf = via.ravel()
        padid = (self.padid[0].ravel(), self.padid[1].ravel())
        tb = [self.pads[k][2] for k in target_pads]
        tset = set(target_pads)

        def h(i, j):
            best = 1e18
            for (x0, y0, x1, y1) in tb:
                dx = max((x0 - OX) / P - i, 0, i - (x1 - OX) / P); dy = max((y0 - OY) / P - j, 0, j - (y1 - OY) / P)
                d = max(dx, dy) + 0.4142 * min(dx, dy)
                best = min(best, d)
            return best if tb else 0.0

        g, parent, pdir = {}, {}, {}
        heap = []
        for s in sources:
            g[s] = 0.0; parent[s] = None; pdir[s] = -1
            i, j, l = s
            heapq.heappush(heap, (h(i, j), 0.0, s))
        expanded = 0
        while heap:
            f, gc, s = heapq.heappop(heap)
            if gc > g.get(s, 1e18):
                continue
            i, j, l = s
            idx = j * NX + i
            if goal_fn is None:
                k = padid[l][idx]
                if k in tset:
                    return self.backtrack(parent, s), k
            elif goal_fn(s):
                return self.backtrack(parent, s), None
            expanded += 1
            if expanded > 400000:
                return None, None
            lf = BCU_FACTOR if l == 1 else 1.0
            d0 = pdir[s]
            for di, (dx, dy) in enumerate(DIRS):
                ni, nj = i + dx, j + dy
                if not (0 <= ni < NX and 0 <= nj < NY):
                    continue
                nidx = nj * NX + ni
                if not fl[l][nidx]:
                    continue
                if dx and dy and not (fl[l][j * NX + ni] or fl[l][nj * NX + i]):
                    continue
                step = (1.4142 if dx and dy else 1.0) * lf + (TURN if (d0 >= 0 and d0 != di) else 0.0)
                t = (ni, nj, l); ng = gc + step
                if ng < g.get(t, 1e18):
                    g[t] = ng; parent[t] = s; pdir[t] = di
                    heapq.heappush(heap, (ng + h(ni, nj), ng, t))
            if vf[idx]:
                t = (i, j, 1 - l); ng = gc + VIA_COST
                if fl[1 - l][idx] and ng < g.get(t, 1e18):
                    g[t] = ng; parent[t] = s; pdir[t] = -1
                    heapq.heappush(heap, (ng + h(i, j), ng, t))
        return None, None

    @staticmethod
    def backtrack(parent, s):
        path = []
        while s is not None:
            path.append(s); s = parent[s]
        return path[::-1]

    def pad_cells(self, k):
        net, layers, (x0, y0, x1, y1), (cx, cy), _ = self.pads[k]
        out = []
        for l in layers:
            m = np.argwhere(self.padid[l] == k)
            out += [(int(i), int(j), l) for j, i in m]
        return out

    def route_net(self, net):
        ks = [k for k, p in enumerate(self.pads) if p[0] == net]
        if len(ks) < 2:
            return True
        free, via = self.free_maps(net)
        # start from the pad nearest the centroid of the net
        cx = np.mean([self.pads[k][3][0] for k in ks]); cy = np.mean([self.pads[k][3][1] for k in ks])
        k0 = min(ks, key=lambda k: (self.pads[k][3][0] - cx) ** 2 + (self.pads[k][3][1] - cy) ** 2)
        tree = set(self.pad_cells(k0)); remaining = [k for k in ks if k != k0]
        paths = []
        while remaining:
            path, k = self.astar(net, tree, remaining, free, via)
            if path is None:
                return False
            paths.append(path); tree.update(path); tree.update(self.pad_cells(k)); remaining.remove(k)
        for pth in paths:
            self.commit(net, pth)
        return True

    def commit(self, net, path):
        self.paint_path(net, path)
        for a, b in zip(path, path[1:]):
            if a[2] == b[2]:
                self.shapes.append((net, (a[2],), "cap", (OX + a[0] * P, OY + a[1] * P, OX + b[0] * P, OY + b[1] * P, TRACK_W / 2)))
            else:
                self.shapes.append((net, (0, 1), "cap", (OX + a[0] * P, OY + a[1] * P, OX + a[0] * P, OY + a[1] * P, VIA_D / 2)))
                self.holes.append((OX + a[0] * P, OY + a[1] * P, VIA_DRILL / 2))
        seg = [path[0]]
        for a, b in zip(path, path[1:]):
            if a[2] != b[2]:
                self.tracks.append((net, a[2], seg)); self.vias.append((net, a[0], a[1])); seg = [b]
            else:
                seg.append(b)
        self.tracks.append((net, seg[-1][2], seg))

    # ------------------------------------------------------------------ GND
    def stitch_gnd(self, pads_only=False, lattice_only=False):
        free, via = self.free_maps(self.gnd)
        vflat = via.ravel()
        n_ok = 0
        for k, (net, layers, bb, (cx, cy), pad) in enumerate(self.pads):
            if lattice_only or net != self.gnd or 1 in layers:
                continue
            src = [c for c in self.pad_cells(k)]
            goal = lambda s: s[2] == 0 and vflat[s[1] * NX + s[0]] and \
                math.hypot(OX + s[0] * P - cx, OY + s[1] * P - cy) >= 0.9
            free_f = [free[0], np.zeros_like(free[1])]
            path, _ = self.astar(self.gnd, src, [], free_f, np.zeros_like(via), goal_fn=goal)
            if path is None:
                print(f"  GND stitch failed for pad {self.padref[k]}")
                continue
            self.commit(self.gnd, path)
            e = path[-1]; self.vias.append((self.gnd, e[0], e[1]))
            self.shapes.append((self.gnd, (0, 1), "cap", (OX + e[0] * P, OY + e[1] * P, OX + e[0] * P, OY + e[1] * P, VIA_D / 2)))
            self.holes.append((OX + e[0] * P, OY + e[1] * P, VIA_DRILL / 2))
            for l in (0, 1):
                paint_disk(self.own[l], OX + e[0] * P, OY + e[1] * P, VIA_D / 2, self.gnd)
            free, via = self.free_maps(self.gnd); vflat = via.ravel()
            n_ok += 1
        lattice = 0                                     # stitching vias for the top pour, every 5 mm where free
        if pads_only:
            return n_ok, 0
        for j in range(int(4 / P), NY - int(4 / P), int(5 / P)):
            for i in range(int(4 / P), NX - int(4 / P), int(5 / P)):
                if via[j, i]:
                    self.vias.append((self.gnd, i, j)); lattice += 1
                    self.shapes.append((self.gnd, (0, 1), "cap", (OX + i * P, OY + j * P, OX + i * P, OY + j * P, VIA_D / 2 + CLR)))
                    self.holes.append((OX + i * P, OY + j * P, VIA_DRILL / 2))
                    for l in (0, 1):
                        paint_disk(self.own[l], OX + i * P, OY + j * P, VIA_D / 2 + CLR, self.gnd)
                    free, via = self.free_maps(self.gnd)
        return n_ok, lattice

    # ------------------------------------------------------------------ output
    def to_board(self):
        b = self.b
        for net, l, pts in self.tracks:
            # merge collinear steps
            simp = [pts[0]]
            for a, c in zip(pts[1:], pts[2:]):
                p0 = simp[-1]
                if (a[0] - p0[0]) * (c[1] - a[1]) != (a[1] - p0[1]) * (c[0] - a[0]):
                    simp.append(a)
            if len(pts) > 1:
                simp.append(pts[-1])
            for a, c in zip(simp, simp[1:]):
                t = pcbnew.PCB_TRACK(b)
                t.SetStart(pcbnew.VECTOR2I(mm(OX + a[0] * P), mm(OY + a[1] * P)))
                t.SetEnd(pcbnew.VECTOR2I(mm(OX + c[0] * P), mm(OY + c[1] * P)))
                t.SetWidth(mm(TRACK_W)); t.SetLayer(pcbnew.F_Cu if l == 0 else pcbnew.B_Cu)
                t.SetNet(b.FindNet(self.netname[net])); b.Add(t)
        for net, i, j in self.vias:
            v = pcbnew.PCB_VIA(b)
            v.SetPosition(pcbnew.VECTOR2I(mm(OX + i * P), mm(OY + j * P)))
            v.SetWidth(mm(VIA_D)); v.SetDrill(mm(VIA_DRILL)); v.SetNet(b.FindNet(self.netname[net])); b.Add(v)


def main():
    order_bias = {}
    for attempt in range(6):
        b = pcbnew.LoadBoard(str(PCB))
        for t in list(b.GetTracks()):
            b.Remove(t)
        r = Router(b)
        nets = sorted({p[0] for p in r.pads if p[0] > 0 and p[0] != r.gnd})

        def span(n):
            xs = [p[3][0] for p in r.pads if p[0] == n]; ys = [p[3][1] for p in r.pads if p[0] == n]
            return (max(xs) - min(xs)) + (max(ys) - min(ys))
        nets.sort(key=lambda n: (-order_bias.get(n, 0), span(n)))
        failed = []
        t0 = time.time()
        ok, _ = r.stitch_gnd(pads_only=True)          # GND fan-out first, so signals route around the vias
        for n in nets:
            if not r.route_net(n):
                failed.append(n)
        print(f"pass {attempt}: routed {len(nets) - len(failed)}/{len(nets)} nets in {time.time() - t0:.0f} s; "
              f"failed: {[r.netname[n] for n in failed]}", flush=True)
        if not failed:
            break
        for n in failed:
            order_bias[n] = order_bias.get(n, 0) + 10
    _, lat = r.stitch_gnd(lattice_only=True)
    print(f"GND: {ok} pad stitches, {lat} lattice vias")
    r.to_board()
    pcbnew.ZONE_FILLER(b).Fill(b.Zones())
    b.Save(str(PCB))
    print(f"saved {PCB.name}: {len(r.tracks)} polylines, {len(r.vias)} vias; unrouted nets: {[r.netname[n] for n in failed]}")


if __name__ == "__main__":
    main()
