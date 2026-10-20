"""Turn the user's marked-up screenshots into map-space edit masks.

Usage: python make_edit_masks.py <orbis_dir>
Reads source/edits/<name>.png and data/edit_alignment.json (where each
screenshot sits on the 3072px map). Strokes: red = enclosed area becomes sea
(claimed land inside stays land), yellow = land, blue = river line.
Writes data/edits/{sea,land,river}.png (white = edit).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

root = Path(sys.argv[1])
align = json.load(open(root / "data/edit_alignment.json"))
H = W = 3072
sea = np.zeros((H, W), bool)
land = np.zeros((H, W), bool)
river = np.zeros((H, W), np.uint8)

for name, al in align.items():
    a = np.array(Image.open(root / f"source/edits/{name}.png").convert("RGB")).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    red = (r > 170) & (g < 90) & (b < 90)
    yel = (r > 190) & (g > 190) & (b < 90)
    blu = (b > 200) & (r < 40) & (g > 40) & (g < 120)
    sx, sy = al["w"] / al["img_w"], al["h"] / al["img_h"]
    M = np.float32([[sx, 0, al["x0"]], [0, sy, al["y0"]]])

    def to_map(mask):
        return cv2.warpAffine(mask.astype(np.uint8) * 255, M, (W, H), flags=cv2.INTER_LINEAR) > 64

    if red.sum() > 50:
        if al.get("close_k"):   # strokes that touch each other: close in screen space
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (al["close_k"],) * 2)
            filled = ndimage.binary_fill_holes(cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_CLOSE, ker))
            if al.get("band_k"):    # open double lines: the band between them is sea
                kb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (al["band_k"],) * 2)
                filled |= cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_CLOSE, kb).astype(bool)
            if al.get("hull_above"):   # loops open at the top: use the outline of the strokes above this line
                ys_, xs_ = np.nonzero(red[: int(al["hull_above"] * red.shape[0])])
                hull = cv2.convexHull(np.column_stack([xs_, ys_]).astype(np.int32))
                hm = np.zeros(red.shape, np.uint8)
                cv2.fillPoly(hm, [hull], 1)
                filled |= hm.astype(bool)
            sea |= to_map(filled)
            print(f"{name}: sea {to_map(filled).sum()} px (closing {al['close_k']} in screen space)")
            red = np.zeros_like(red)
        red_all = to_map(ndimage.binary_dilation(red, iterations=2))
        # each stroke group separately: close gaps until it encloses an area
        groups, n = ndimage.label(ndimage.binary_dilation(red_all, iterations=4))
        total = 0
        for gi in range(1, n + 1):
            red_m = red_all & (groups == gi)
            if red_m.sum() < 30:
                continue
            for k in (5, 9, 15, 25, 41, 61):
                ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                closed = cv2.morphologyEx(red_m.astype(np.uint8), cv2.MORPH_CLOSE, ker).astype(bool)
                filled = ndimage.binary_fill_holes(closed)
                if filled.sum() > closed.sum() * 1.6:
                    break
            sea |= filled
            total += filled.sum()
        print(f"{name}: sea {total} px in {n} stroke groups")
    if yel.sum() > 50:
        y_m = ndimage.binary_fill_holes(to_map(ndimage.binary_dilation(yel, iterations=3)))
        land |= y_m
        print(f"{name}: land {y_m.sum()} px")
    if name == "river" and blu.sum() > 200:   # elsewhere blue is just map labels
        sk = cv2.warpAffine(blu.astype(np.uint8) * 255, M, (W, H), flags=cv2.INTER_LINEAR) > 40
        river |= sk
        print(f"{name}: river {sk.sum()} px")

sea &= ~land
out = root / "data/edits"
out.mkdir(parents=True, exist_ok=True)
for n, m in (("sea", sea), ("land", land), ("river", river.astype(bool))):
    Image.fromarray((m * 255).astype(np.uint8)).save(out / f"{n}.png")
