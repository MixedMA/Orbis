"""Strait crossings from red lines drawn on in-game screenshots (user, 2026-10-18).

Usage: python -I tools/make_crossings.py <orbis_dir> <screenshot> <x0> <y0> <x1> <y1>

<x0..y1> is a rough map area (3072 px map coordinates) that contains the screenshot. The screenshot is placed
on the map by matching its coastlines (water = blue-grey) against source/land.png at many zoom levels; each
red line becomes a crossing {a, b} in data/crossings.json (entries from the same screenshot are replaced).
write_game_files turns them into "sea" adjacencies (straits) through the water under each line.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


def main():
    root, shot = Path(sys.argv[1]), Path(sys.argv[2])
    X0, Y0, X1, Y1 = map(int, sys.argv[3:7])
    s = np.array(Image.open(shot).convert("RGB")).astype(int)
    r, g, b = s[..., 0], s[..., 1], s[..., 2]
    water = (b > r + 8) & (b > g + 4) & (b < 170)
    water = cv2.morphologyEx(water.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    L = np.array(Image.open(root / "source/land.png").convert("RGB")).astype(int)
    # lake channels are drawn as land in game (rivers), so they count as land here
    land = ((np.abs(L - (225, 205, 160)).max(2) < 7) | (np.abs(L - (80, 160, 230)).max(2) < 7)).astype(np.float32)
    sub = land[Y0:Y1, X0:X1]
    T = cv2.resize((~water).astype(np.float32), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    best = (-1,)
    for k in np.exp(np.linspace(np.log(0.2), np.log(3.0), 140)):       # map px per screen px
        f = 0.25 / k
        M = cv2.resize(sub, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        if T.shape[0] > M.shape[0] or T.shape[1] > M.shape[1]:
            continue
        res = cv2.matchTemplate(M, T, cv2.TM_CCOEFF_NORMED)
        th, tw = T.shape                                           # patches that are (almost) all sea or all
        mean = cv2.boxFilter(M, -1, (tw, th), normalize=True, anchor=(0, 0))   # land give bogus scores
        mean = mean[:res.shape[0], :res.shape[1]]
        res[~np.isfinite(res) | (mean < 0.1) | (mean > 0.9)] = -1
        _, v, _, loc = cv2.minMaxLoc(res)
        if v > best[0]:
            best = (v, k, X0 + loc[0] / f, Y0 + loc[1] / f)
    v, k, ox, oy = best
    print(f"{shot.name}: map = {k:.3f} * screen + ({ox:.0f}, {oy:.0f}), match {v:.2f}"
          + ("  (weak match, check the result)" if v < 0.75 else ""))

    red = (r > 180) & (g < 70) & (b < 70)
    lab, n = ndimage.label(red, np.ones((3, 3)))
    new = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < 15:
            continue
        P = np.c_[xs, ys].astype(float)
        c = P.mean(0)
        u = np.linalg.svd(P - c)[2][0]
        t = (P - c) @ u
        a, bb = c + u * t.min(), c + u * t.max()
        new.append({"a": [round(a[0] * k + ox, 1), round(a[1] * k + oy, 1)],
                    "b": [round(bb[0] * k + ox, 1), round(bb[1] * k + oy, 1)], "source": shot.name})
    path = root / "data/crossings.json"
    old = json.load(open(path, encoding="utf-8")) if path.exists() else []
    old = [c for c in old if c.get("source") != shot.name]
    json.dump(old + new, open(path, "w", encoding="utf-8"), indent=1)
    print(f"{len(new)} crossings from {shot.name}, {len(old) + len(new)} in data/crossings.json")


if __name__ == "__main__":
    main()
