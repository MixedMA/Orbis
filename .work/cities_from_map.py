"""One-off: city list from the user's marked political map source/cities_map.png (2026-10-18).

Marker pixel positions were detected in the image (out/cities_map_markers.json + out/_extra.json, the latter
partly placed by hand); names were read off the image by eye. The image is aligned to the world by fitting
an affine transform on cities that already exist in data/cities_1850.json (RANSAC, capitals on emblems drop out).
Writes data/cities_1850.json (merged: new map wins, old cities not on it are kept) and out/cities_check.png.
"""
import json
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

old_px = json.load(open("out/cities_map_markers.json"))
extra = json.load(open("out/_extra.json"))

NAMES = ["Forodwaith", "Yaskul", "Atlantis", "Bootchest*", "Yergen", "Vyra", "Dun Tyene", "Dawei", "Zverski", "Alzir",
         "Izmir", "Cosmodrome", "Sansretour", "Jirsi", "Tuonela", "Assalka", "Drumm", "Anthropia*", "Baltimore",
         "Alanta", "Ard Velen", "Kofinneia", "Tomiyuu", "Anchorage", "Yakutsk", "Elbaf", "Blumenau", "Kaisagrad",
         "Uelen", "Anadyr", "Carroll-Town", "Orbokraj", "Gvareny", "Neu Katzland", "Omolon", "Farigharnon",
         "Anadune", "Troy", "Guam", "Tari'ara Ur'atar", "Vostaniye", "Mito", "Tijuana", "Tamatarkha*", "Freiburg",
         "Valimar*", "York", "Gary", "Baccah*", "Aurtiarnon", "Port-au-Prince", "Kokomo", "Limatigrad", "Targoviste",
         "Folsna", "Paloga", "Strasbourg", "Humaita", "Nova Sempiscatus", "Tucson", "St. Benisburg", "Flint",
         "Sauble", "Navajo", "Kuhland*", "Flagstaff", "Plovdiv*", "Chrysanthemum", "Portland", "Peten K'uhul",
         "Salem", "Petrograd", "Lagash*", "Sivas", "Sarajevo", "Plymouth", "Tuguegarao", "Uruk", "Jubileum",
         "Villa*", "Site-13", "Fort Pesh", "Shroomtown", "Jacksonville", "Chesapeake*", "Istanbul", "Phoenix",
         "Dinocity", "Site Anvil", "Carcamano's Complex", "Providence", "Albuquerque", "Lynchburg",
         "German's Complex*", "Sondlitz", "Arcadia*", "Elysium", "Beograd*"]
EXTRA = ["Jerusalem*", "Samartha*", "Nazareth*", "Beringskygrad*", "Worcester", "Riverwood", "Site-60", "Ezellohar",
         None, "Cressenatoru", "Lakeside", "Reno", "Detroit", "Mosca", "Byzantium", "Dauper", "El Paso", "Azuria",
         "Columbia", "Boston", "Tokat", None, None, None]
# names in the old list that are the same city under another spelling / the user's renames
SAME = {"Strasbourg": "Strasburg", "Site-60": "Site 60", "Anadune": "Anadûnê"}

assert len(NAMES) == len(old_px) and len(EXTRA) == len(extra)
new = []
for (x, y, *_), n in list(zip(old_px, NAMES)) + list(zip(extra, EXTRA)):
    if n is None:
        continue
    new.append({"name": n.rstrip("*"), "capital": n.endswith("*"), "px": (x, y)})
seen = {}
for c in new:                      # duplicates found twice: keep one
    seen.setdefault(c["name"], c)
new = list(seen.values())

old = json.load(open("data/cities_1850.json", encoding="utf-8"))
old_by = {c["name"]: c for c in old}
pairs = [(c["px"], (old_by[k]["fx"], old_by[k]["fy"])) for c in new
         for k in [SAME.get(c["name"], c["name"])] if k in old_by]
src = np.float32([p for p, _ in pairs]); dst = np.float32([q for _, q in pairs])
M, inl = cv2.estimateAffine2D(src, dst, ransacReprojThreshold=0.006)
err = np.hypot(*((src @ M[:, :2].T + M[:, 2]) - dst).T) * 3008
print(f"{len(pairs)} shared cities, {int(inl.sum())} inliers, median error {np.median(err[inl.ravel() == 1]):.1f} px")
for (p, q), e, i in zip(pairs, err, inl.ravel()):
    if not i:
        n = next(c["name"] for c in new if c["px"] == p)
        print(f"  outlier {n}: {e:.0f} px off (old position probably on an emblem)")

out = []
for c in new:
    fx, fy = (np.float32(c["px"]) @ M[:, :2].T + M[:, 2]).tolist()
    out.append({"name": c["name"], "country": None, "capital": c["capital"], "fx": round(fx, 6), "fy": round(fy, 6),
                "x": int(round(fx * 3008 + 32)), "y": int(round(fy * 3008 + 32))})
names_new = {c["name"] for c in out} | {SAME.get(c["name"], c["name"]) for c in out}
kept = [c for c in old if c["name"] not in names_new]
print(f"{len(out)} cities from the map, {len(kept)} old cities kept: {[c['name'] for c in kept]}")
if "--write" in sys.argv:
    json.dump(out + kept, open("data/cities_1850.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

own = Image.open("source/owners.png").convert("RGB")
d = ImageDraw.Draw(own)
for c in out + kept:
    r = 9 if c["capital"] else 6
    col = (255, 255, 255) if c in out else (255, 0, 255)
    d.ellipse([c["x"] - r, c["y"] - r, c["x"] + r, c["y"] + r], fill=col, outline=(0, 0, 0), width=2)
    d.text((c["x"] + 11, c["y"] - 7), c["name"], fill=col, stroke_width=2, stroke_fill=(0, 0, 0))
own.crop((380, 380, 2700, 2700)).resize((1600, 1600), Image.LANCZOS).save("out/cities_check.png")
