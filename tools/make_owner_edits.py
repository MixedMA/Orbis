"""Turn the user's border drawings on in-game screenshots into an ownership edit map.

Usage: python make_owner_edits.py <orbis_dir>
Config: data/owner_edits.json
  screens: {name: {scale, tx, ty}}   source/edits/<name>.png, map = scale * screen + (tx, ty)
  close:   [[x0, y0, x1, y1], ...]    extra barrier segments in map px (to close open line ends)
  regions: [{tag, seed: [x, y], take: [tags]}]
           the land area around seed, bounded by the red lines, sea and the world border, goes to tag;
           only land currently owned by a tag in take (or unclaimed, "WLD") changes.
Needs build/{land,border,owner}.npy from a previous build_base run (current ownership).
Writes data/edits/owner.png (country index per map px, 0 = no change), applied by build_base.py,
and out/owner_edits_check.png (preview).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

root = Path(sys.argv[1])
cfg = json.load(open(root / "data/owner_edits.json", encoding="utf-8"))
countries = json.load(open(root / "data/countries.json", encoding="utf-8"))
idx = {c["tag"]: c["index"] for c in countries}
H = W = 3072

stroke = np.zeros((H, W), np.uint8)
for name, t in cfg["screens"].items():
    a = np.array(Image.open(root / f"source/edits/{name}.png").convert("RGB")).astype(int)
    red = (a[..., 0] > 190) & (a[..., 1] < 70) & (a[..., 2] < 70)
    lab, n = ndimage.label(ndimage.binary_dilation(red, iterations=2))
    size = ndimage.sum(red, lab, range(1, n + 1))
    red &= np.isin(lab, 1 + np.nonzero(size > 150)[0])     # drop red map markers, keep drawn lines
    M = np.float32([[t["scale"], 0, t["tx"]], [0, t["scale"], t["ty"]]])
    stroke |= (cv2.warpAffine(red.astype(np.uint8) * 255, M, (W, H)) > 40).astype(np.uint8)
land = np.load(root / "build/land.npy") & ~np.load(root / "build/border.npy")
# Bridge small gaps where hand-drawn lines meet, then seal open line ends: from the stroke nearest
# to each "seal" point, through the point, to the nearest water.
k = cfg.get("close_k", 31)
stroke = cv2.morphologyEx(stroke, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
_, (sy, sx) = ndimage.distance_transform_edt(stroke == 0, return_indices=True)
_, (wy, wx) = ndimage.distance_transform_edt(land, return_indices=True)
seal = np.zeros_like(stroke)
for x, y in cfg.get("seal", []):
    cv2.line(seal, (int(sx[y, x]), int(sy[y, x])), (x, y), 1, 9)
    cv2.line(seal, (x, y), (int(wx[y, x]), int(wy[y, x])), 1, 9)
for x0, y0, x1, y1 in cfg.get("close", []):
    cv2.line(seal, (x0, y0), (x1, y1), 1, 9)
stroke = cv2.dilate(stroke | seal, np.ones((5, 5), np.uint8)).astype(bool)
owner = np.load(root / "build/owner.npy")
edit = np.zeros((H, W), np.uint8)
for r in cfg["regions"]:
    # A region spreads only through land it may take: other countries are walls like the lines.
    take = np.isin(owner, [idx[t] for t in r["take"]])
    area, _ = ndimage.label(land & ~stroke & take)
    _, (ay, ax) = ndimage.distance_transform_edt(area == 0, return_indices=True)
    x, y = r["seed"]
    a = area[ay[y, x], ax[y, x]]          # seeds snap to the nearest such land between the lines
    m = (area == a) & (edit == 0)
    edit[m] = idx[r["tag"]]
    print(f"{r['tag']}: {m.sum()} px from area of {(area == a).sum()} px")
# the strokes themselves: give each stroke pixel the edit of its nearest edited neighbour on the same side
d, (iy, ix) = ndimage.distance_transform_edt(edit == 0, return_indices=True)
near = edit[iy, ix]
fill = stroke & land & (d <= 6) & np.isin(owner, sorted({idx[t] for r in cfg["regions"] for t in r["take"]}))
edit[fill] = near[fill]
(root / "data/edits").mkdir(parents=True, exist_ok=True)
Image.fromarray(edit).save(root / "data/edits/owner.png")

# preview: current owners, edits on top, lines in red
pal = np.zeros((256, 3), np.uint8)
for c in countries:
    if c.get("map_color"):
        pal[c["index"]] = c["map_color"]
pal[0] = (110, 110, 110)
new = np.where(edit > 0, edit, owner)
img = np.where(land[..., None], pal[new], (20, 25, 45)).astype(np.uint8)
ch = np.zeros(new.shape, bool)
ch[:, 1:] |= new[:, 1:] != new[:, :-1]
ch[1:] |= new[1:] != new[:-1]
img[ch & land] = (255, 255, 255)                                  # borders after the edit
img[stroke & ~land] = (120, 0, 0)
img[stroke & land] = (255, 0, 0)
Image.fromarray(img).resize((1536, 1536)).save(root / "out/owner_edits_check.png")
