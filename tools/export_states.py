"""Write the current states as a paintable map, the user's source of states from then on.

Usage: python export_states.py <orbis_dir> <mod_dir> [--force]
Needs a full build (build/states.json, build/province_ids.npy, build/provinces.json and the mod's
state names in localisation). Writes
  source/states.png         one unique colour per land state; sea, lakes and the world border in the
                            no-state colours of build_states (NO_STATE_RGB)
  source/state_names.json   colour (hex) -> state name, editable
  source/states_guide.png   the same with state borders, country borders and names, for reference only
Will not overwrite an existing states.png without --force (it holds the user's painting).
"""
import colorsys
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

root, mod = Path(sys.argv[1]), Path(sys.argv[2])
dst = root / "source/states.png"
if dst.exists() and "--force" not in sys.argv:
    sys.exit(f"{dst} exists (the user's painting); pass --force to overwrite it")
states = json.load(open(root / "build/states.json", encoding="utf-8"))
prov = np.load(root / "build/province_ids.npy")
meta = json.load(open(root / "build/provinces.json"))
loc = open(next((mod / "localisation/english/replace").glob("*.yml")), encoding="utf-8-sig").read()
names = dict(re.findall(r'^\s*STATE_(\d+):\d* "(.*)"', loc, re.M))

SEA, LAKE, WALL = (10, 20, 70), (80, 160, 230), (0, 0, 0)
pal = np.zeros((len(meta) + 1, 3), np.uint8)
for p in meta:
    pal[p["id"]] = WALL if p["border"] else SEA if p["type"] == "sea" else LAKE if p["type"] == "lake" else (0, 0, 0)
# Each state is a shade of its country's colour (user, 2026-10-18); neighbouring states of one country get
# different shades (greedy graph colouring), and a small jitter makes every state colour unique.
base = {c["index"]: tuple(c["map_color"]) for c in json.load(open(root / "data/countries.json", encoding="utf-8"))
        if c.get("map_color")}
base[0] = (190, 180, 150)                       # unclaimed land, as in owners.png
SHADES = [0, -40, 40, -20, 20, -60, 60, -30, 30, 50]
state_of = {p: s["id"] for s in states for p in s["provinces"]}
sadj = {s["id"]: set() for s in states}
for s in states:
    for p in s["provinces"]:
        for r in meta[p - 1]["neighbors"]:
            if r in state_of and state_of[r] != s["id"]:
                sadj[s["id"]].add(state_of[r])
rng = np.random.default_rng(5)
used = {SEA, LAKE, WALL}
shade_of, colour_of = {}, {}
out_names = {}
owner_of = np.zeros(len(meta) + 1, np.int32)
by_id = {s["id"]: s for s in states}
for s in sorted(states, key=lambda s: -len(sadj[s["id"]])):
    for p in s["provinces"]:
        owner_of[p] = s["owner"] + 1
    if s.get("impassable"):
        continue
    taken = {shade_of.get(t) for t in sadj[s["id"]] if by_id[t]["owner"] == s["owner"]}
    sh = next((v for v in SHADES if v not in taken), SHADES[0])
    shade_of[s["id"]] = sh
    bc = base.get(s["owner"], base[0])
    if min(bc) + sh < 20 or max(bc) + sh > 235:  # a shade leaving the usable range goes the other way
        sh = -sh
    while True:
        j = rng.integers(-6, 7, 3)
        c = tuple(int(np.clip(v + sh + d, 0, 255)) for v, d in zip(bc, j))
        if c not in used and min(sum((a - b) ** 2 for a, b in zip(c, n)) for n in (SEA, LAKE, WALL)) > 15 ** 2:
            break
    used.add(c)
    for p in s["provinces"]:
        pal[p] = c
    out_names["%02x%02x%02x" % c] = names.get(str(s["id"]), "")
img = pal[prov]
Image.fromarray(img).save(dst)
json.dump(out_names, open(root / "source/state_names.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# guide: state borders grey, country borders black, names at state centres
sid = np.zeros(len(meta) + 1, np.int32)
for s in states:
    for p in s["provinces"]:
        sid[p] = s["id"]
S, O = sid[prov], owner_of[prov]
g = img.copy()
for a, b in ((np.s_[:, 1:], np.s_[:, :-1]), (np.s_[1:], np.s_[:-1])):
    g[a][(S[a] != S[b]) & (S[a] > 0) & (S[b] > 0)] = (90, 90, 90)
    g[a][(O[a] != O[b]) & (S[a] > 0) & (S[b] > 0)] = (0, 0, 0)
gi = Image.fromarray(g)
d = ImageDraw.Draw(gi)
try:
    font = ImageFont.truetype("segoeuib.ttf", 13)
except OSError:
    font = ImageFont.load_default()
for s in states:
    if not s.get("impassable"):
        d.text((s["x"], s["y"]), names.get(str(s["id"]), ""), fill=(255, 255, 255), font=font, anchor="mm",
               stroke_width=2, stroke_fill=(0, 0, 0))
gi.save(root / "source/states_guide.png")
print(f"wrote {dst} ({len(out_names)} states), source/state_names.json, source/states_guide.png")
