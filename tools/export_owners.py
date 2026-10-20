"""Write the current ownership as a paintable map, the user's source of borders from then on.

Usage: python export_owners.py <orbis_dir> [--force]
Reads build/{owner,land,border}.npy (+ data/edits/owner.png, the screenshot edits) from the last build.
Writes source/owners.png (MAP x MAP, one flat colour per country, see build_base.PAINTED) and
source/owners_legend.png. Will not overwrite an existing owners.png without --force (it holds the user's painting).
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from build_base import PAINTED, SEA_RGB, UNCLAIMED_RGB, WALL_RGB  # noqa: E402
from extract_countries import COUNTRIES  # noqa: E402

root = Path(sys.argv[1])
dst = root / PAINTED
if dst.exists() and "--force" not in sys.argv:
    sys.exit(f"{dst} exists (the user's painting); pass --force to overwrite it")
owner = np.load(root / "build/owner.npy")
land = np.load(root / "build/land.npy")
border = np.load(root / "build/border.npy")
if (root / "data/edits/owner.png").exists():
    e = np.array(Image.open(root / "data/edits/owner.png"))
    owner = np.where((e > 0) & land & ~border, e, owner)
pal = np.zeros((256, 3), np.uint8)
pal[0] = UNCLAIMED_RGB
for i, (_, c) in enumerate(COUNTRIES, 1):
    pal[i] = c
img = pal[owner]
img[~land] = SEA_RGB
img[border] = WALL_RGB
Image.fromarray(img).save(dst)

# Legend: one swatch per colour with name and RGB, so the colour can be picked with the eyedropper.
tags = {c["name"]: c["tag"] for c in json.load(open(root / "data/countries.json", encoding="utf-8"))}
rows = [(f"{n} ({tags.get(n, '')})", c) for n, c in COUNTRIES] + [
    ("Unclaimed land", UNCLAIMED_RGB), ("Sea / lakes (ignored)", SEA_RGB), ("World border (ignored)", WALL_RGB)]
try:
    font = ImageFont.truetype("segoeui.ttf", 22)
except OSError:
    font = ImageFont.load_default()
leg = Image.new("RGB", (560, 20 + 44 * len(rows)), (245, 245, 245))
d = ImageDraw.Draw(leg)
for k, (name, c) in enumerate(rows):
    y = 10 + 44 * k
    d.rectangle([10, y, 110, y + 36], fill=tuple(c), outline=(0, 0, 0))
    d.text((125, y + 6), f"{name}   {tuple(int(v) for v in c)}", fill=(0, 0, 0), font=font)
leg.save(root / "source/owners_legend.png")
print(f"wrote {dst} and source/owners_legend.png")
