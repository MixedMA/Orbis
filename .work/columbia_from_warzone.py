"""One-off (user, 2026-10-18): Columbia's states follow the yellow territories of a Warzone screenshot,
and everything yellow becomes Columbia. Edits the user's painted maps in place (backups in build/):
source/owners.png (yellow -> Columbia) and source/states.png (one new colour per territory), names in
source/state_names.json carried over from the state that covered most of each territory before.
Screenshot placed on the map by matching coastlines: map = 2.10 * screen + (796, 1624).
"""
import json
import shutil

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

K, X0, Y0 = 2.10, 796, 1624
CLB = (80, 134, 134)
import os
for f in ("owners.png", "states.png", "state_names.json"):     # always start from the user's own versions
    if not os.path.exists(f"build/_before_warzone_{f}"):
        shutil.copy(f"source/{f}", f"build/_before_warzone_{f}")
    shutil.copy(f"build/_before_warzone_{f}", f"source/{f}")

a = np.array(Image.open("source/edits/columbia_states_warzone.png").convert("RGB")).astype(int)
yel = (a[..., 0] > 150) & (a[..., 1] > 150) & (a[..., 2] < 80)
lab, n = ndimage.label(yel)
size = ndimage.sum(yel, lab, range(1, n + 1))
lab[np.isin(lab, 1 + np.nonzero(size < 40)[0])] = 0           # specks / letters
# the whole yellow area incl. the border lines between territories; lines take the nearest territory
area = cv2.morphologyEx((lab > 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
_, (iy, ix) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
terr = np.where(area, lab[iy, ix], 0).astype(np.int32)

# Group territories into Warzone bonuses: a territory's outline is drawn in its bonus colour, so touching
# territories whose outlines share a colour form one state (user: states should follow these borders).
orange = (a[..., 0] > 180) & (a[..., 1] < 150) & (a[..., 2] < 80)
line = area & ~yel & ~orange
col_of = {}
for t in range(1, n + 1):
    m = lab == t
    if not m.any():
        continue
    ring = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~m & line
    if ring.sum() >= 5:
        col_of[t] = np.median(a[ring], 0)
parent = {t: t for t in col_of}
def find(t):
    while parent[t] != t:
        parent[t] = parent[parent[t]]
        t = parent[t]
    return t
for t in col_of:                                   # neighbours across the border lines
    near = cv2.dilate((terr == t).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    for u in np.unique(terr[near]):
        if u in col_of and u != t and np.abs(col_of[t] - col_of[u]).max() < 35:
            parent[find(t)] = find(u)
grp = np.zeros(n + 1, np.int32)
for t in col_of:
    grp[t] = find(t)
print(f"{len(col_of)} territories -> {len(set(grp[list(col_of)]))} colour groups")

# Neighbouring bonuses often share a colour, so split each colour group at its bonus number boxes: every
# bonus has one box (or the big round one). Each territory goes to the nearest box of its group (BFS).
m_box = (~yel & ~orange & ~(np.abs(a - [13, 19, 56]).max(2) < 25) & ~(a.max(2) < 40)) | (a.min(2) > 220)
core = cv2.morphologyEx(m_box.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
bl, bn = ndimage.label(core)
seeds = {}                      # territory -> box id
for i, sl in enumerate(ndimage.find_objects(bl), 1):
    h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
    bm = bl[sl] == i
    px = a[sl][bm]
    col = np.median(px[px.min(1) < 200], 0) if (px.min(1) < 200).sum() > 5 else None
    if col is None or not (10 <= h <= 34 and 10 <= w <= 34):
        continue                                  # white city circles, lines
    ring = cv2.dilate((bl == i).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    ts, cs = np.unique(terr[ring & (terr > 0)], return_counts=True)
    if len(ts):
        seeds.setdefault(int(ts[cs.argmax()]), i)
adjt = {t: set() for t in col_of}
for t in col_of:
    near = cv2.dilate((terr == t).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    adjt[t] = {int(u) for u in np.unique(terr[near]) if u in col_of and u != t and grp[u] == grp[t]}
owner_box = {t: b for t, b in seeds.items() if t in col_of}
frontier = list(owner_box)
while frontier:
    nxt = []
    for t in frontier:
        for u in adjt[t]:
            if u not in owner_box:
                owner_box[u] = owner_box[t]
                nxt.append(u)
    frontier = nxt
final = np.zeros(n + 1, np.int32)
for t in col_of:
    final[t] = 1000 + owner_box[t] if t in owner_box else grp[t]
terr = final[terr]
print(f"{len(seeds)} bonus boxes -> {len(set(final[list(col_of)]))} states")
M = np.float32([[K, 0, X0], [0, K, Y0]])
T = cv2.warpAffine(terr.astype(np.float32), M, (3072, 3072), flags=cv2.INTER_NEAREST).astype(np.int32)

land = np.array(Image.open("source/land.png").convert("RGB"))
is_land = (np.abs(land.astype(int) - (225, 205, 160)).max(2) < 10)
T[~is_land] = 0
own = np.array(Image.open("source/owners.png").convert("RGB"))
# Caelum lies inside the yellow area but stays its own country (it was swallowed by mistake, 2026-10-18)
T[np.abs(own.astype(int) - (0, 109, 58)).max(2) == 0] = 0
print(f"{len(np.unique(T)) - 1} territories on land, {(T > 0).sum()} px")

own[T > 0] = CLB
Image.fromarray(own).save("source/owners.png")
# Columbia land inside the screenshot that Warzone shows as no territory (black / orange pockets) joins the
# nearest new state, so no old state patches are left in between.
frame = cv2.warpAffine(np.ones(a.shape[:2], np.uint8), M, (3072, 3072), flags=cv2.INTER_NEAREST) > 0
pocket = frame & is_land & (np.abs(own.astype(int) - CLB).max(2) == 0) & (T == 0)
d, (ny, nx) = ndimage.distance_transform_edt(T == 0, return_indices=True)
fill = pocket & (d <= 120)
T[fill] = T[ny, nx][fill]
print(f"{fill.sum()} px of Columbia pockets joined the nearest new state")

st = np.array(Image.open("source/states.png").convert("RGB"))
key = (st[..., 0].astype(np.int64) << 16) | (st[..., 1].astype(np.int64) << 8) | st[..., 2]
names = json.load(open("source/state_names.json", encoding="utf-8"))
used_cols = set(np.unique(key).tolist())
used_names = set()
rng = np.random.default_rng(11)
new_names = {}
for t in np.unique(T[T > 0]):
    m = T == t
    ks, cs = np.unique(key[m], return_counts=True)
    prev = "%06x" % ks[cs.argmax()]
    name = names.get(prev, "")
    if name in used_names:
        name = ""
    used_names.add(name)
    while True:                                   # a fresh shade of Columbia
        c = tuple(int(np.clip(v + d, 20, 235)) for v, d in zip(CLB, rng.integers(-55, 56, 3)))
        ck = (c[0] << 16) | (c[1] << 8) | c[2]
        if ck not in used_cols:
            break
    used_cols.add(ck)
    st[m] = c
    new_names["%02x%02x%02x" % c] = name
Image.fromarray(st).save("source/states.png")
names.update(new_names)
json.dump(names, open("source/state_names.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{len(new_names)} Columbia states painted, {sum(1 for v in new_names.values() if v)} kept an old name")
