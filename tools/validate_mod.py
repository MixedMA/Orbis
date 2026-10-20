"""Cross-check the generated mod files for the usual HOI4 map/history crashes.

Usage: python validate_mod.py <mod_dir>
"""
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    mod = Path(sys.argv[1])
    errors, warns = [], []
    m = mod / "map"

    # definition.csv vs provinces.bmp
    defs = {}
    for line in (m / "definition.csv").read_text(encoding="utf-8").splitlines()[1:]:
        f = line.split(";")
        defs[int(f[0])] = {"rgb": (int(f[1]), int(f[2]), int(f[3])), "type": f[4], "coastal": f[5], "terrain": f[6]}
    ids = sorted(defs)
    if ids != list(range(1, len(ids) + 1)):
        errors.append("definition.csv ids are not contiguous 1..N")
    rgbs = Counter(d["rgb"] for d in defs.values())
    if any(c > 1 for c in rgbs.values()):
        errors.append("duplicate province colours")
    img = np.array(Image.open(m / "provinces.bmp").convert("RGB")).astype(np.int64)
    H, W = img.shape[:2]
    if W % 256 or H % 256:
        errors.append(f"map size {W}x{H} not a multiple of 256")
    keys = np.unique((img[..., 0] << 16) | (img[..., 1] << 8) | img[..., 2])
    defkeys = {(r << 16) | (g << 8) | b for r, g, b in rgbs}
    if set(keys.tolist()) - defkeys:
        errors.append(f"{len(set(keys.tolist()) - defkeys)} colours in provinces.bmp missing from definition.csv")
    if defkeys - set(keys.tolist()):
        errors.append(f"{len(defkeys - set(keys.tolist()))} provinces defined but not on the map")
    # Provinces touching only at a pixel corner: HOI4 counts that as adjacency (a land province
    # touching the sea at a corner becomes coastal without a port sea zone and crashes game start).
    pk = (img[..., 0] << 16) | (img[..., 1] << 8) | img[..., 2]
    a, b, c, d = pk[:-1, :-1], pk[:-1, 1:], pk[1:, :-1], pk[1:, 1:]
    corners = ((a != d) & (b != a) & (b != d) & (c != a) & (c != d)) | \
              ((b != c) & (a != b) & (a != c) & (d != b) & (d != c))
    if corners.any():
        y, x = np.argwhere(corners)[0]
        errors.append(f"{int(corners.sum())} pixel-corner province contacts in provinces.bmp (first at x={x}, y={y})")
    for f in ("heightmap.bmp", "terrain.bmp", "rivers.bmp", "cities.bmp"):
        if Image.open(m / f).size != (W, H):
            errors.append(f"{f} size differs from provinces.bmp")
    if Image.open(m / "world_normal.bmp").size != (W // 2, H // 2):
        errors.append("world_normal.bmp must be half size")

    land = {i for i, d in defs.items() if d["type"] == "land"}

    # states
    state_of, state_owner = {}, {}
    tags = set(re.findall(r"^([A-Z][A-Z0-9]{2})\s*=", (mod / "common/country_tags/00_countries.txt").read_text(), re.M))
    for f in (mod / "history/states").glob("*.txt"):
        t = f.read_text(encoding="utf-8")
        sid = int(re.search(r"\bid\s*=\s*(\d+)", t).group(1))
        provs = [int(x) for x in re.search(r"provinces\s*=\s*\{([^}]*)\}", t).group(1).split()]
        owner = re.search(r"owner\s*=\s*([A-Z0-9]{3})", t).group(1)
        if owner not in tags:
            errors.append(f"state {sid}: owner {owner} is not a defined tag")
        state_owner[sid] = owner
        for p in provs:
            if p not in land:
                errors.append(f"state {sid}: province {p} is not land")
            if p in state_of:
                errors.append(f"province {p} in states {state_of[p]} and {sid}")
            state_of[p] = sid
        for vp in re.findall(r"victory_points\s*=\s*\{\s*(\d+)", t):
            if int(vp) not in provs:
                errors.append(f"state {sid}: VP province {vp} not in state")
        for nb in re.findall(r"(\d+)\s*=\s*\{\s*naval_base", t):
            if defs[int(nb)]["coastal"] != "true":
                errors.append(f"state {sid}: naval base in non-coastal province {nb}")
    sids = sorted(state_owner)
    if sids != list(range(1, len(sids) + 1)):
        errors.append("state ids not contiguous")
    missing = land - set(state_of)
    if missing:
        errors.append(f"{len(missing)} land provinces in no state")

    # strategic regions
    reg_of = {}
    for f in (m / "strategicregions").glob("*.txt"):
        t = f.read_text(encoding="utf-8")
        rid = int(re.search(r"\bid\s*=\s*(\d+)", t).group(1))
        for p in map(int, re.search(r"provinces\s*=\s*\{([^}]*)\}", t).group(1).split()):
            if p in reg_of:
                errors.append(f"province {p} in two strategic regions")
            reg_of[p] = rid
        if "weather=" not in t:
            errors.append(f"region {rid} has no weather")
    if set(reg_of) != set(defs):
        errors.append(f"{len(set(defs) - set(reg_of))} provinces in no strategic region")

    # countries
    for tag in tags:
        hist = list((mod / "history/countries").glob(f"{tag} - *.txt"))
        if not hist:
            errors.append(f"{tag}: no history file")
            continue
        cap = int(re.search(r"capital\s*=\s*(\d+)", hist[0].read_text(encoding="utf-8")).group(1))
        if state_owner.get(cap) != tag:
            errors.append(f"{tag}: capital state {cap} is owned by {state_owner.get(cap)}")
        if not (mod / f"gfx/flags/{tag}.tga").exists():
            warns.append(f"{tag}: no flag")
    owned = set(state_owner.values())
    for tag in tags - owned:
        warns.append(f"{tag} owns no state")

    # positions
    us = Counter(int(l.split(";")[0]) for l in (m / "unitstacks.txt").read_text().splitlines() if l)
    if set(us) != set(defs):
        errors.append("unitstacks.txt does not cover every province")
    for l in (m / "buildings.txt").read_text().splitlines():
        f = l.split(";")
        if int(f[0]) not in state_owner:
            errors.append(f"buildings.txt: unknown state {f[0]}")
            break
        if not (0 <= float(f[2]) < W and 0 <= float(f[4]) <= H):
            errors.append(f"buildings.txt: position out of map: {l}")
            break
    # Each building position must lie clearly inside one province of its state: the game finds the province
    # from the position, and a naval base on a border pixel went to the neighbour, leaving a coastal province
    # without a port (game-start crash in CControllerArea, 2026-10-18).
    key2id = {(r << 16) | (g << 8) | b: i for i, (r, g, b) in ((i, d["rgb"]) for i, d in defs.items())}
    uk, inv = np.unique(pk, return_inverse=True)
    pid = np.array([key2id.get(int(k), 0) for k in uk])[inv].reshape(pk.shape)
    sea_adj = {}
    for A, B in ((pid[:, :-1], pid[:, 1:]), (pid[:-1], pid[1:])):
        mm = A != B
        for x_, y_ in set(zip(A[mm].tolist(), B[mm].tolist())):
            sea_adj.setdefault(x_, set()).add(y_)
            sea_adj.setdefault(y_, set()).add(x_)
    bad = []
    for l in (m / "buildings.txt").read_text().splitlines():
        f = l.split(";")
        x, y = int(float(f[2])), int(H - float(f[4]))
        around = {int(pid[min(max(y + dy, 0), H - 1), min(max(x + dx, 0), W - 1)]) for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
        p = int(pid[min(max(y, 0), H - 1), x])
        naval = f[6] != "0"
        if len(around) > 1 or state_of.get(p) != int(f[0]) or (naval and int(f[6]) not in sea_adj.get(p, ())):
            bad.append(f"{f[1]} of state {f[0]} at province {p}")
    if bad:
        errors.append(f"{len(bad)} building positions on a province border or outside their state, e.g. {bad[:3]}")
    for l in (m / "supply_nodes.txt").read_text().splitlines():
        p = int(l.split()[1])
        if p not in land:
            errors.append(f"supply node on non-land province {p}")
    for l in (m / "railways.txt").read_text().splitlines():
        f = list(map(int, l.split()))
        if f[1] != len(f) - 2:
            errors.append(f"railway count mismatch: {l[:40]}")
    for l in (m / "adjacencies.csv").read_text().splitlines()[1:]:
        f = l.split(";")
        if f[0] == "-1":
            continue
        frm, to, thr = int(f[0]), int(f[1]), int(f[3])
        if defs.get(frm, {}).get("type") != "land" or defs.get(to, {}).get("type") != "land" or \
                defs.get(thr, {}).get("type") not in ("sea", "lake"):
            errors.append(f"adjacencies.csv: crossing {frm}-{to} through {thr} must link land through water")
        elif not ({frm, to} <= sea_adj.get(thr, set())):
            warns.append(f"adjacencies.csv: water province {thr} does not touch both {frm} and {to}")
    wp = {int(l.split(";")[0]) for l in (m / "weatherpositions.txt").read_text().splitlines() if l}
    if wp != set(reg_of.values()):
        errors.append("weatherpositions.txt does not cover every strategic region")

    # localisation has BOM
    for f in (mod / "localisation").rglob("*.yml"):
        if not f.read_bytes().startswith(b"\xef\xbb\xbf"):
            errors.append(f"{f.name}: missing UTF-8 BOM")

    print(f"{len(defs)} provinces, {len(state_owner)} states, {len(set(reg_of.values()))} regions, {len(tags)} tags")
    for wmsg in warns:
        print("WARN ", wmsg)
    for e in errors[:40]:
        print("ERROR", e)
    print("OK - no errors" if not errors else f"{len(errors)} errors")


if __name__ == "__main__":
    main()
