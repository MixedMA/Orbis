"""Write map/rivers.bmp from the thin water channels found in build_base.

Usage: python write_rivers.py <orbis_dir> <mod_dir> <hoi4_dir>

HOI4 rivers are 1-pixel, 4-connected lines in an indexed bitmap using the
vanilla palette: 255 land, 254 water, 0 source (green), 1 tributary joining
(red), 3..11 river width from narrow to wide.
"""
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

LAND, WATER, SOURCE, JOIN = 255, 254, 0, 1
MIN_BRANCH = 14       # shorter side branches are dropped
MIN_RIVER = 25        # whole rivers shorter than this are dropped
N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def zhang_suen(img):
    """Thin a binary image to an 8-connected 1-pixel skeleton."""
    img = img.astype(np.uint8).copy()
    while True:
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            P2, P3, P4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]
            P5, P6, P7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
            P8, P9 = p[1:-1, :-2], p[:-2, :-2]
            nb = [P2, P3, P4, P5, P6, P7, P8, P9]
            B = sum(n.astype(int) for n in nb)
            A = sum(((nb[i] == 0) & (nb[(i + 1) % 8] == 1)).astype(int) for i in range(8))
            if step == 0:
                c = (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                c = (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
            rm = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if rm.any():
                img[rm] = 0
                changed = True
        if not changed:
            return img.astype(bool)


def width_index(upstream):
    for limit, idx in ((40, 3), (90, 4), (160, 5), (260, 6), (400, 7), (600, 8), (900, 9)):
        if upstream < limit:
            return idx
    return 10


def main():
    root, mod, game = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    b = root / "build"
    land = np.load(b / "land.npy")
    river = np.load(b / "river.npy") & land
    H, W = land.shape

    # Do not let rivers hug the coast: keep them 2 px inland except at the mouth.
    sk = zhang_suen(river)
    water_near = ndimage.binary_dilation(~land, iterations=1)
    out = np.where(land, LAND, WATER).astype(np.uint8)

    lab, n = ndimage.label(sk, structure=np.ones((3, 3)))
    kept = 0
    for comp, sl in enumerate(ndimage.find_objects(lab), 1):
        ys, xs = np.nonzero(lab[sl] == comp)
        ys, xs = ys + sl[0].start, xs + sl[1].start
        if len(ys) < MIN_RIVER:
            continue
        pix = set(zip(ys.tolist(), xs.tolist()))
        # mouth: a pixel touching water, else any endpoint
        mouths = [q for q in pix if water_near[q]]
        if mouths:
            root_px = mouths[0]
        else:
            ends = [q for q in pix if sum((q[0] + dy, q[1] + dx) in pix for dy, dx in N8) == 1]
            root_px = ends[0] if ends else next(iter(pix))
        # BFS tree from the mouth
        parent = {root_px: None}
        order = [root_px]
        dq = deque([root_px])
        while dq:
            q = dq.popleft()
            for dy, dx in N8:
                r = (q[0] + dy, q[1] + dx)
                if r in pix and r not in parent:
                    parent[r] = q
                    order.append(r)
                    dq.append(r)
        children = {q: [] for q in order}
        for q in order[1:]:
            children[parent[q]].append(q)
        # Only the main stem: mouth -> farthest pixel. Tributaries made HOI4
        # report circular rivers (and those can crash the river mesh builder).
        depth = {root_px: 0}
        for q in order[1:]:
            depth[q] = depth[parent[q]] + 1
        far = max(order, key=lambda q: depth[q])
        stem = []
        q = far
        while q is not None:
            stem.append(q)
            q = parent[q]
        stem.reverse()                       # mouth first
        # expand diagonal steps to 4-connected steps
        path = [stem[0]]
        for q in stem[1:]:
            p = path[-1]
            if q[0] != p[0] and q[1] != p[1]:
                a_, c_ = (q[0], p[1]), (p[0], q[1])
                path.append(a_ if land[a_] else c_)
            path.append(q)
        # walk upstream, stop where the river would touch itself or another river
        drawn = []
        for i, q in enumerate(path):
            if not land[q] or out[q] != LAND:
                break
            allowed = set(drawn[-2:])
            touch = False
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    r = (q[0] + dy, q[1] + dx)
                    if (dy or dx) and r not in allowed and out[r] <= 11:
                        touch = True
            if touch:
                break
            drawn.append(q)
            out[q] = 3                        # placeholder, width set below
        if len(drawn) < MIN_RIVER:
            for q in drawn:
                out[q] = LAND
            continue
        kept += 1
        n = len(drawn)
        for i, q in enumerate(drawn):        # wider towards the mouth
            out[q] = width_index((n - i) * 6)
        out[drawn[-1]] = SOURCE

    van = Image.open(game / "map/rivers.bmp")
    img = Image.fromarray(out, "P")
    img.putpalette(van.getpalette())
    img.save(mod / "map/rivers.bmp")
    rp = np.isin(out, list(range(0, 12)))
    print(f"{kept} rivers, {rp.sum()} river pixels")
    prev = np.where(land[..., None], np.array([200, 200, 190], np.uint8), np.array([40, 70, 120], np.uint8))
    prev[rp] = (20, 90, 230)
    prev[out == SOURCE] = (0, 220, 0)
    prev[out == JOIN] = (230, 0, 0)
    Image.fromarray(prev).save(b / "rivers_preview.png")


if __name__ == "__main__":
    main()
