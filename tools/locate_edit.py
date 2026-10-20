"""Locate an edit screenshot on the map by matching country-colour masks.
Usage: python locate_edit.py <name> <country>[,<country2>] <window_px>
Updates out/edit_alignment.json."""
import json, sys
import cv2, numpy as np
from PIL import Image
sys.path.insert(0, "tools")
from extract_countries import COUNTRIES

name, countries, win = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3])
names = [n for n, _ in COUNTRIES]
prov = np.load("build/province_ids.npy")
meta = json.load(open("build/provinces.json"))
own = np.zeros(len(meta) + 1, np.int32)
for p in meta:
    own[p["id"]] = p["owner"]
O = own[prov]
a = cv2.medianBlur(np.array(Image.open(f"source/edits/{name}.png").convert("RGB")), 7).astype(int)
H, W = a.shape[:2]
scr, ref = [], []
for c in countries:
    col = np.array(COUNTRIES[names.index(c)][1]).clip(0, 255)
    # in-game political colours are darker/tinted: match by hue direction, not exact rgb
    hsv = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(int)
    target = cv2.cvtColor(np.uint8([[col]]), cv2.COLOR_RGB2HSV)[0, 0].astype(int)
    dh = np.minimum(abs(hsv[..., 0] - target[0]), 180 - abs(hsv[..., 0] - target[0]))
    scr.append((dh < 10) & (hsv[..., 1] > 90) & (hsv[..., 2] > 90))
    ref.append(O == names.index(c) + 1)
cy, cx = np.argwhere(ref[0]).mean(0).astype(int)
y0, x0 = max(cy - win, 0), max(cx - win, 0)
refw = [r[y0:cy + win, x0:cx + win].astype(np.float32) * 2 - 1 for r in ref]
best = None
for tw in range(60, 2 * win, 4):
    th = int(round(tw * H / W))
    if th >= refw[0].shape[0] or tw >= refw[0].shape[1]:
        break
    tot = 0
    for s, r in zip(scr, refw):
        t = cv2.resize(s.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA) * 2 - 1
        tot = tot + cv2.matchTemplate(r, t, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(tot / len(scr))
    if best is None or mx > best[0]:
        best = (mx, tw, th, loc)
mx, tw, th, (lx, ly) = best
res = json.load(open("out/edit_alignment.json"))
res[name] = {"score": round(float(mx), 3), "x0": int(x0 + lx), "y0": int(y0 + ly), "w": int(tw), "h": int(th), "img_w": int(W), "img_h": int(H)}
json.dump(res, open("out/edit_alignment.json", "w"), indent=1)
print(name, res[name], "screen px matched:", [int(s.sum()) for s in scr])
