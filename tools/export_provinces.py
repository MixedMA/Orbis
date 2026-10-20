"""Write the current provinces as a paintable map, the user's source of provinces from then on.

Usage: python export_provinces.py <orbis_dir> [--force]
Reads build/{province_ids.npy,provinces.json,owner.npy} from the last build. Writes
  source/provinces.png        one unique colour per province (land varied, sea dark blue, lakes light blue,
                              world border grey), read by build_provinces.PAINTED
  source/provinces_guide.png  the same with country borders in black and coasts in white, for reference only
Will not overwrite an existing provinces.png without --force (it holds the user's painting).
"""
import colorsys
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(sys.argv[1])
dst = root / "source/provinces.png"
if dst.exists() and "--force" not in sys.argv:
    sys.exit(f"{dst} exists (the user's painting); pass --force to overwrite it")
prov = np.load(root / "build/province_ids.npy")
meta = json.load(open(root / "build/provinces.json"))
owner = np.load(root / "build/owner.npy")

rng = np.random.default_rng(3)
used = set()
pal = np.zeros((len(meta) + 1, 3), np.uint8)
for p in meta:
    while True:
        if p["border"]:
            h, s, v = rng.uniform(0, 1), rng.uniform(0, 0.15), rng.uniform(0.35, 0.6)   # greyish
        elif p["type"] == "sea":
            h, s, v = rng.uniform(0.58, 0.68), rng.uniform(0.5, 0.9), rng.uniform(0.25, 0.5)
        elif p["type"] == "lake":
            h, s, v = rng.uniform(0.5, 0.56), rng.uniform(0.3, 0.6), rng.uniform(0.75, 0.95)
        else:   # land: anything but the water hues
            h = rng.uniform(0.0, 0.48) if rng.random() < 0.8 else rng.uniform(0.72, 1.0)
            s, v = rng.uniform(0.35, 0.85), rng.uniform(0.55, 0.95)
        c = tuple(int(round(x * 255)) for x in colorsys.hsv_to_rgb(h, s, v))
        if c not in used:
            used.add(c)
            pal[p["id"]] = c
            break
img = pal[prov]
Image.fromarray(img).save(dst)
# What each colour was at export time, so build_provinces can tell repainted coasts apart.
kinds = {"%02x%02x%02x" % tuple(int(v) for v in pal[p["id"]]): ("border" if p["border"] else p["type"]) for p in meta}
json.dump(kinds, open(root / "source/provinces_kinds.json", "w"), indent=0)

is_water = np.array([False] + [p["type"] in ("sea", "lake") and not p["border"] for p in meta])
water = is_water[prov]
guide = img.copy()
for a, b in ((np.s_[:, 1:], np.s_[:, :-1]), (np.s_[1:], np.s_[:-1])):
    coast = water[a] != water[b]
    border = (owner[a] != owner[b]) & ~water[a] & ~water[b]
    guide[a][coast] = (255, 255, 255)
    guide[a][border] = (0, 0, 0)
Image.fromarray(guide).save(root / "source/provinces_guide.png")
print(f"wrote {dst} ({len(meta)} provinces) and source/provinces_guide.png")
