"""Render the province map with country colours and names (out/orbis_provinces.png).
Usage: python render_overview.py <orbis_dir>"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).parent))
from extract_countries import COUNTRIES

root = Path(sys.argv[1])
prov = np.load(root / "build/province_ids.npy")
meta = json.load(open(root / "build/provinces.json"))
N = len(meta)
rng = np.random.default_rng(3)
col = np.zeros((N + 1, 3), np.uint8)
own = np.zeros(N + 1, int)
for p in meta:
    own[p["id"]] = p["owner"]
    if p["type"] == "sea":
        c = np.array([30, 55, 105]) + rng.integers(-8, 8, 3)
    elif p["type"] == "lake":
        c = np.array([60, 110, 170])
    elif p["owner"]:
        c = np.clip(COUNTRIES[p["owner"] - 1][1], 0, 255) * 0.85 + rng.integers(0, 30, 3)
    else:
        c = np.array([150, 145, 125]) + rng.integers(-15, 15, 3)
    col[p["id"]] = np.clip(c, 0, 255)
img = col[prov]
edge = np.zeros(prov.shape, bool)
edge[:, 1:] |= prov[:, 1:] != prov[:, :-1]
edge[1:, :] |= prov[1:, :] != prov[:-1, :]
img[edge] = (25, 25, 25)
S = 1536 / prov.shape[1]
im = Image.fromarray(img).resize((1536, 1536), Image.LANCZOS)
d = ImageDraw.Draw(im)
f = ImageFont.truetype("segoeuib.ttf", 22)
om = own[prov]
for i, (name, _) in enumerate(COUNTRIES, 1):
    lab, n = ndimage.label(om == i)
    if not n:
        continue
    big = np.argmax(ndimage.sum(om == i, lab, range(1, n + 1))) + 1
    cy, cx = ndimage.center_of_mass(lab == big)
    d.text((cx * S, cy * S), name, fill="white", font=f, anchor="mm", stroke_width=3, stroke_fill="black")
im.save(root / "out/orbis_provinces.png")

# scenario cities (capitals as larger squares)
from scenario import cities_path
cities_f = cities_path()
if cities_f.exists():
    fs = ImageFont.truetype("segoeui.ttf", 15)
    for c in json.load(open(cities_f, encoding="utf-8")):
        x, y = c["x"] * S, c["y"] * S
        r = 6 if c["capital"] else 4
        d.rectangle([x - r, y - r, x + r, y + r], fill="white" if c["capital"] else (230, 230, 230), outline="black")
        d.text((x + 8, y), c["name"], fill="white", font=fs, anchor="lm", stroke_width=2, stroke_fill="black")
    im.save(root / "out/orbis_provinces.png")
