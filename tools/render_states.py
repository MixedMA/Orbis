"""Preview of states (fill) and strategic regions (thick borders): out/orbis_states.png"""
import json, sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
root = Path(sys.argv[1])
prov = np.load(root / "build/province_ids.npy")
meta = {p["id"]: p for p in json.load(open(root / "build/provinces.json"))}
states = json.load(open(root / "build/states.json"))
regions = json.load(open(root / "build/regions.json"))
countries = {c["index"]: c for c in json.load(open(root / "data/countries.json", encoding="utf-8"))}
N = len(meta)
st = np.zeros(N + 1, np.int32); rg = np.zeros(N + 1, np.int32)
for s in states:
    st[s["provinces"]] = s["id"]
for r in regions:
    rg[r["provinces"]] = r["id"]
rng = np.random.default_rng(1)
col = np.zeros((N + 1, 3), np.float32)
for s in states:
    base = np.array(countries[s["owner"]]["color"], np.float32)
    col[s["provinces"]] = np.clip(base * 0.8 + rng.integers(-25, 26, 3), 0, 255)
for i, p in meta.items():
    if p["type"] == "sea": col[i] = (30, 55, 100)
    if p["type"] == "lake": col[i] = (60, 110, 170)
img = col[prov].astype(np.uint8)
S, R = st[prov], rg[prov]
def edges(a):
    e = np.zeros(a.shape, bool); e[:, 1:] |= a[:, 1:] != a[:, :-1]; e[1:, :] |= a[1:, :] != a[:-1, :]; return e
img[edges(S) & (S > 0)] = (40, 40, 40)
re = cv2.dilate(edges(R).astype(np.uint8), np.ones((3, 3))) > 0
img[re] = (255, 230, 80)
im = Image.fromarray(img).resize((1536, 1536), Image.LANCZOS)
d = ImageDraw.Draw(im); f = ImageFont.truetype("segoeui.ttf", 13)
for r in regions:
    d.text((r["x"] / 2, r["y"] / 2), str(r["id"]), fill="white", font=f, anchor="mm", stroke_width=2, stroke_fill="black")
im.save(root / "out/orbis_states.png")
