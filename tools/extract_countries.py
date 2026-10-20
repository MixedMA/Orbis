"""Extract claimed areas per country colour from the political map and place
them on the full-world grid using out/alignment.json.

Usage: python extract_countries.py <political_map.png> <out_dir>
Colours and names come from data/countries.json ("map_color"), see tools/scenario.py.
Writes countries_political.npy (country index per political-map pixel, 0 = none)
and countries_preview.png (terrain preview with country overlay + labels).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from scenario import countries as _countries  # noqa: E402

# (name, colour on the scenario's political map), in owner-index order 1..N.
COUNTRIES = [(c["name"], tuple(c["map_color"])) for c in sorted(_countries(), key=lambda c: c["index"]) if c["index"]]
BLACK = (0, 12, 12)            # unclaimed land on the political map
HOLE_FILL = 30                 # emblems, labels and roads inside a claim take the nearest claim


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    a = np.array(Image.open(src).convert("RGB")).astype(int)
    owner = np.zeros(a.shape[:2], np.uint8)
    for i, (name, col) in enumerate(COUNTRIES, 1):
        m = np.abs(a - col).max(2) <= 6
        m = ndimage.binary_opening(m, iterations=1)
        owner[m & (owner == 0)] = i
    # Everything that is neither a fill colour nor unclaimed land/sea (emblems, labels,
    # roads, borders) takes the nearest claim if one is close.
    dark = a.max(2) < 70
    bluish = (a[..., 2] > a[..., 0] + 25) & (a.max(2) < 130)
    known = (owner > 0) | (np.abs(a - BLACK).max(2) <= 10) | (dark & ~ndimage.binary_dilation(owner > 0, iterations=2)) | bluish
    dist, idx = ndimage.distance_transform_edt(owner == 0, return_indices=True)
    fill = ~known & (dist <= HOLE_FILL)
    owner[fill] = owner[idx[0], idx[1]][fill]
    for i in range(1, len(COUNTRIES) + 1):
        m = ndimage.binary_closing(owner == i, iterations=4)
        owner[m & (owner == 0)] = i
    np.save(out / "countries_political.npy", owner)

    # Preview on the terrain map.
    al = json.load(open(out / "alignment.json"))["pol_in_terrain_frac"]
    tp = Image.open(out / "terrain_preview.png").convert("RGB")
    W, H = tp.size
    x0, y0, x1, y1 = al[0] * W, al[1] * H, al[2] * W, al[3] * H
    size = (round(x1 - x0), round(y1 - y0))
    own = cv2.resize(owner, size, interpolation=cv2.INTER_NEAREST)
    t = np.array(tp).astype(float)
    ox, oy = round(x0), round(y0)
    sub = t[oy:oy + size[1], ox:ox + size[0]]
    for i, (_, col) in enumerate(COUNTRIES, 1):
        m = own[:sub.shape[0], :sub.shape[1]] == i
        sub[m] = sub[m] * 0.35 + np.clip(col, 0, 255) * 0.65
    img = Image.fromarray(t.astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.rectangle([x0, y0, x1, y1], outline=(255, 255, 255))
    try:
        font = ImageFont.truetype("segoeuib.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for i, (name, _) in enumerate(COUNTRIES, 1):
        m = own == i
        lab, n = ndimage.label(m)
        if not n:
            continue
        big = np.argmax(ndimage.sum(m, lab, range(1, n + 1))) + 1
        cy, cx = ndimage.center_of_mass(lab == big)
        d.text((ox + cx, oy + cy), name, fill=(255, 255, 255), font=font,
               anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0))
    img.save(out / "countries_preview.png")
    for i, (name, _) in enumerate(COUNTRIES, 1):
        print(f"{name:16s} {(owner == i).mean():6.2%} of political map")


if __name__ == "__main__":
    main()
