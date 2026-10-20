"""Find the small white town dots on the scenario's political map.

Usage: python extract_towns.py <orbis_dir>
Writes the scenario's town list (tools/scenario.py TOWNS): [{name, fx, fy}], fx/fy as world fractions
like the city list. Named city markers (rings) and fort icons (towers) are skipped.
Names start empty; the user can fill them in by hand (re-running keeps names of dots that did not move).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from scenario import POLITICAL_MAP, ROOT, TOWNS, cities_path  # noqa: E402

FRAME = 1500          # out/alignment_1850.json and out/cities_1850.json use the 1500 px map frame
DOT_D = (20, 30)      # town dots are ~25 px wide on the 4000 px map; tower icons ~34, city rings larger


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT
    a = np.array(Image.open(root / POLITICAL_MAP).convert("RGB")).astype(int)
    S = a.shape[0]
    white = (a.min(2) > 215).astype(np.uint8)
    # Opening with a disc removes roads, outlines and letters that touch a dot; solid discs survive.
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    core = cv2.morphologyEx(white, cv2.MORPH_OPEN, disc).astype(bool)
    lab, n = ndimage.label(core)
    al = json.load(open(root / "out/alignment_1850.json"))["m1850_in_terrain_frac"]
    named = [((c["fx"] - al[0]) / (al[2] - al[0]) * S, (c["fy"] - al[1]) / (al[3] - al[1]) * S)
             for c in json.load(open(cities_path(), encoding="utf-8"))]
    towns = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        m = lab[sl] == i
        # A dot leaves a round core of dot size; tower icons leave smaller, non-round pieces.
        if not (DOT_D[0] <= h <= DOT_D[1] and DOT_D[0] <= w <= DOT_D[1]) or m.sum() / (np.pi / 4 * h * w) < 0.85:
            continue
        cy, cx = ndimage.center_of_mass(m)
        cy += sl[0].start; cx += sl[1].start
        # Named city markers are a dot inside a white ring; a town dot has only a dark shadow around it.
        ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        ring = max(white[np.clip((cy + r * np.sin(ang)).astype(int), 0, S - 1),
                         np.clip((cx + r * np.cos(ang)).astype(int), 0, S - 1)].mean() for r in range(16, 27))
        if ring > 0.5 or min(np.hypot(cx - x, cy - y) for x, y in named) < 85:   # capital markers inside emblems
            continue
        towns.append((cx, cy))
    old = {}
    path = root / TOWNS
    if path.exists():
        old = {(round(t["fx"], 3), round(t["fy"], 3)): t["name"] for t in json.load(open(path, encoding="utf-8"))}
    out = []
    for cx, cy in sorted(towns, key=lambda t: (t[1], t[0])):
        fx = al[0] + cx / S * (al[2] - al[0])
        fy = al[1] + cy / S * (al[3] - al[1])
        out.append({"name": old.get((round(fx, 3), round(fy, 3)), ""), "fx": round(fx, 6), "fy": round(fy, 6)})
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(out)} towns -> {path}")


if __name__ == "__main__":
    main()
