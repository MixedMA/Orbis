"""Combine the terrain screenshot and the political map into base rasters at
final HOI4 map resolution.

Usage: python build_base.py <orbis_dir>

Map layout: MAP x MAP pixels; the 4000x4000-block world fills the centre
WORLD x WORLD, surrounded by a MARGIN of open sea (HOI4 wraps the map
horizontally, the sea keeps east and west apart).

Writes to <orbis_dir>/build/:
  land.npy      bool   land mask
  water.npy     uint8  0 land, 1 sea, 2 lake
  river.npy     bool   thin water channels turned into land (river candidates)
  terrain.npy   uint8  terrain class ids (classify_biomes.CLASSES)
  owner.npy     uint8  country index (extract_countries.COUNTRIES, 0 = none)
  base_preview.png
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from classify_biomes import CLASSES, WATER  # noqa: E402
from extract_countries import COUNTRIES  # noqa: E402
from scenario import cities_path  # noqa: E402

MAP, WORLD = 3072, 3008
MARGIN = (MAP - WORLD) // 2
RNG = np.random.default_rng(1941)

# The user's hand-painted ownership map (tools/export_owners.py writes the first version).
# When it exists it replaces the political map + screenshot edits as the source of ownership.
PAINTED = "source/owners.png"
UNCLAIMED_RGB, SEA_RGB, WALL_RGB = (190, 180, 150), (10, 20, 70), (0, 0, 0)
# The user's hand-painted coastlines (tools/export_land.py writes the first version): land / sea / lake.
PAINTED_LAND = "source/land.png"
LAND_RGB, LAKE_RGB = (225, 205, 160), (80, 160, 230)


def read_palette_map(path, pal, what):
    """Value per world pixel from a painted palette map: each pixel takes the value of the nearest palette
    colour; anti-aliased pixels (off-palette) take the nearest clean pixel's value. pal: [(value, rgb)]."""
    a = np.array(Image.open(path).convert("RGB"))[MARGIN:MARGIN + WORLD, MARGIN:MARGIN + WORLD].astype(np.int64)
    cols = np.array([c for _, c in pal]); vals = np.array([v for v, _ in pal])
    key = (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]
    uk, inv = np.unique(key, return_inverse=True)
    rgb = np.stack([uk >> 16, (uk >> 8) & 255, uk & 255], 1)
    d = ((rgb[:, None, :] - cols[None]) ** 2).sum(2)
    clean = (d.min(1) <= 12 ** 2)[inv].reshape(key.shape)
    out = vals[d.argmin(1)][inv].reshape(key.shape).astype(np.uint8)
    _, (iy, ix) = ndimage.distance_transform_edt(~clean, return_indices=True)
    print(f"painted {what}: {(~clean).sum()} px with off-palette colour, took the nearest clean pixel")
    return out[iy, ix]


def painted_owner(path):
    """Country index per world pixel from the painted ownership map."""
    # Sea/lake/wall colours carry no owner: land under them (narrow lake channels that become rivers, coast
    # drawn differently in land.png) takes the nearest painted owner instead of becoming unclaimed.
    NONE = 255
    pal = [(0, UNCLAIMED_RGB), (NONE, SEA_RGB), (NONE, LAKE_RGB), (NONE, WALL_RGB)] +           [(i, c) for i, (_, c) in enumerate(COUNTRIES, 1)]
    o = read_palette_map(path, pal, "owners")
    _, (iy, ix) = ndimage.distance_transform_edt(o == NONE, return_indices=True)
    return o[iy, ix]


def painted_land(path):
    """0 land, 1 sea, 2 lake per world pixel from the painted coastline map (specks cleaned up)."""
    k = read_palette_map(path, [(0, LAND_RGB), (1, SEA_RGB), (2, LAKE_RGB), (1, WALL_RGB)], "coastlines")
    for val, min_px in ((0, 200), (1, 60), (2, 60)):      # tiny islands / ponds join their surroundings
        lab, n = ndimage.label(k == val)
        if n:
            small = np.isin(lab, 1 + np.nonzero(ndimage.sum(k == val, lab, range(1, n + 1)) < min_px)[0])
            _, (iy, ix) = ndimage.distance_transform_edt(small | (k == val), return_indices=True)
            k = np.where(small, k[iy, ix], k)
    return k


def fractal_noise(shape, scales=(256, 96, 32, 12), amps=(1, .5, .25, .12)):
    out = np.zeros(shape, np.float32)
    for s, a in zip(scales, amps):
        small = RNG.standard_normal((shape[0] // s + 2, shape[1] // s + 2)).astype(np.float32)
        out += a * cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    return out / np.abs(out).max()


def warp(img, strength, interp):
    """Domain-warp a raster so upscaled shapes get organic edges."""
    h, w = img.shape[:2]
    dx = fractal_noise((h, w)) * strength
    dy = fractal_noise((h, w)) * strength
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + dx, gy + dy, interp, borderMode=cv2.BORDER_REFLECT)


def political_layers(pol_rgb, owner):
    """Land probability from the political map: 1 land, 0 water, nan unknown."""
    a = pol_rgb.astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    water = (r < 16) & (g >= 10) & (g < 40) & (b >= 10) & (b < 48) & (g > r + 6)
    land = a.max(2) < 12
    claim = ndimage.binary_dilation(owner > 0, iterations=3)
    # Labels and emblems have dark anti-aliased fringes that look like water.
    colored = a.max(2) > 60
    fringe = ndimage.binary_dilation(colored, iterations=2)
    known = (water | land) & ~claim & ~fringe
    # Thin unknowns (coast outlines, labels) take the nearest known value.
    idx = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
    filled = land[idx[0], idx[1]].astype(np.float32)
    filled[claim] = np.nan
    return filled


def extend_cut_claims(owner, rect, reach=110):
    """Claims touching the edge of the political map continue a little beyond
    it (the map was cropped there), fading out with a noisy edge."""
    px0, py0, px1, py1 = rect
    inset = 10  # claims stop a few pixels short of the crop edge
    inside = np.zeros(owner.shape, bool)
    inside[py0 + inset:py1 - inset, px0 + inset:px1 - inset] = True
    dist, idx = ndimage.distance_transform_edt(~inside, return_indices=True)
    nearest = owner[idx[0], idx[1]]
    limit = reach * (0.75 + 0.5 * (fractal_noise(owner.shape, (128, 40, 12), (1, .5, .3)) * 0.5 + 0.5))
    grow = ~inside & (owner == 0) & (nearest > 0) & (dist < limit)
    out = owner.copy()
    noise = fractal_noise(owner.shape, (96, 32), (1, .5)) * 0.22
    # Round off the straight extrusion: blur each country's extension.
    for i in np.unique(nearest[grow]):
        ext = (grow & (nearest == i)) | ((owner == i) & ~inside)
        soft = cv2.GaussianBlur(ext.astype(np.float32), (0, 0), 22) + noise
        keep = (soft > 0.5) & (owner == 0) & ~inside
        keep |= grow & (nearest == i) & (dist < 6)  # stay attached to the claim
        out[keep] = i
    return out


def smooth_owner(owner, land, min_px=900):
    """Smooth claim edges and fold small claim fragments into their surroundings."""
    ids = np.unique(owner)
    best = np.full(owner.shape, -1.0, np.float32)
    out = np.zeros_like(owner)
    for i in ids:
        p = cv2.GaussianBlur((owner == i).astype(np.float32), (0, 0), 2.5)
        better = p > best
        best[better] = p[better]
        out[better] = i
    out[~land] = 0
    for _ in range(2):
        for i in ids:
            if i == 0:
                continue
            lab, n = ndimage.label((out == i) & land)
            if not n:
                continue
            sizes = ndimage.sum(lab > 0, lab, range(1, n + 1))
            largest = int(np.argmax(sizes)) + 1  # small countries keep their core
            for k in np.nonzero(sizes < min_px)[0] + 1:
                if k == largest:
                    continue
                part = lab == k
                ring = ndimage.binary_dilation(part, iterations=2) & ~part & land
                cand = out[ring]
                out[part] = np.bincount(cand).argmax() if cand.size else 0
        # unclaimed specks inside claims
        lab, n = ndimage.label((out == 0) & land)
        sizes = ndimage.sum(lab > 0, lab, range(1, n + 1))
        for k in np.nonzero(sizes < min_px)[0] + 1:
            part = lab == k
            ring = ndimage.binary_dilation(part, iterations=2) & ~part & land
            cand = out[ring]
            if cand.size:
                out[part] = np.bincount(cand).argmax()
    return out


def main():
    root = Path(sys.argv[1])
    out = root / "build"
    out.mkdir(exist_ok=True)
    cls_small = np.load(root / "out/terrain_classes.npy")
    pol_rgb = np.array(Image.open(root / "source/political_map.png").convert("RGB"))
    land_claims = np.load(root / "out/land_claims_1941.npy")       # geography: claims of the 1941 map
    owner_pol = np.load(root / "out/countries_political.npy")      # ownership: active scenario
    fx0, fy0, fx1, fy1 = json.load(open(root / "out/alignment.json"))["pol_in_terrain_frac"]

    # --- terrain-derived land probability (whole world) ---
    land_small = (~np.isin(cls_small, list(WATER))).astype(np.float32)
    land_t = cv2.resize(land_small, (WORLD, WORLD), interpolation=cv2.INTER_CUBIC)
    land_t = cv2.GaussianBlur(land_t, (0, 0), 5)
    land_t = warp(land_t, 10, cv2.INTER_LINEAR)
    land_t = np.clip(land_t + fractal_noise((WORLD, WORLD), (64, 24, 8), (1, .6, .3)) * 0.12, 0, 1)

    # --- political land probability inside its rectangle ---
    px0, py0 = round(fx0 * WORLD), round(fy0 * WORLD)
    px1, py1 = round(fx1 * WORLD), round(fy1 * WORLD)
    pw, ph = px1 - px0, py1 - py0
    pol = political_layers(pol_rgb, land_claims)
    pol_unknown = np.isnan(pol)
    pol_r = cv2.resize(np.nan_to_num(pol), (pw, ph), interpolation=cv2.INTER_AREA)
    unk_r = cv2.resize(pol_unknown.astype(np.float32), (pw, ph), interpolation=cv2.INTER_AREA) > 0.5

    feather = 48
    yy, xx = np.mgrid[0:ph, 0:pw]
    edge = np.minimum.reduce([xx, yy, pw - 1 - xx, ph - 1 - yy]).astype(np.float32)
    wgt = np.clip(edge / feather, 0, 1)
    wgt[unk_r] = 0
    land_soft = land_t.copy()
    sub = land_soft[py0:py1, px0:px1]
    land_soft[py0:py1, px0:px1] = wgt * pol_r + (1 - wgt) * sub
    land = land_soft > 0.5
    # Claims sit on land: under a claim only wide terrain water stays water.
    big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71))
    wide_t = cv2.morphologyEx((land_t <= 0.5).astype(np.uint8), cv2.MORPH_OPEN, big).astype(bool)
    claim_zone = np.zeros_like(land)
    claim_zone[py0:py1, px0:px1] = unk_r
    land[claim_zone] = ~wide_t[claim_zone]
    # Cities always stand on land (small islands and coastal towns).
    cities = json.load(open(cities_path(), encoding="utf-8"))
    cyy, cxx = np.mgrid[0:WORLD, 0:WORLD]
    city_disks = []
    for c in cities:
        cx, cy = c["fx"] * WORLD, c["fy"] * WORLD
        c["x"], c["y"] = int(round(cx + MARGIN)), int(round(cy + MARGIN))
        d = (cxx - cx) ** 2 + (cyy - cy) ** 2
        land |= d <= 16 ** 2
        city_disks.append((c["country"], d <= 15 ** 2))

    # --- clean up: specks, thin channels -> rivers, lakes vs seas ---
    def drop_small(mask, min_px):
        lab, n = ndimage.label(mask)
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        return mask & ~np.isin(lab, 1 + np.nonzero(sizes < min_px)[0])

    land = drop_small(land, 200)
    water = ~land
    water = drop_small(water, 60)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    wide = cv2.morphologyEx(water.astype(np.uint8), cv2.MORPH_OPEN, disk).astype(bool)
    river = water & ~wide
    water = drop_small(wide, 60)
    land = ~water
    river &= land

    lab, n = ndimage.label(water)
    sizes = ndimage.sum(water, lab, range(1, n + 1))
    edge_ids = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    water_kind = np.zeros(water.shape, np.uint8)
    for i, s in enumerate(sizes, 1):
        if s < 700 and i not in edge_ids:
            continue  # ponds and river bits become land (rivers come later)
        kind = 1 if (i in edge_ids or s > 20000) else 2
        water_kind[lab == i] = kind
    pond = water & (water_kind == 0)
    river |= pond & ~ndimage.binary_erosion(pond, iterations=2)
    water = water_kind > 0
    land = ~water

    # --- terrain classes at full resolution ---
    # Fill water cells with the nearest land class first, then upscale each
    # class as a blurred probability layer so boundaries come out smooth.
    small_land = np.where(np.isin(cls_small, list(WATER)), 255, cls_small).astype(np.uint8)
    idx = ndimage.distance_transform_edt(small_land == 255, return_distances=False, return_indices=True)
    small_land = small_land[idx[0], idx[1]]
    land_ids = [c for c, _, _ in CLASSES if c not in WATER]
    noise = fractal_noise((WORLD, WORLD), (128, 48, 16), (1, .5, .25)) * 0.18
    best_p = np.full((WORLD, WORLD), -1.0, np.float32)
    cls_land = np.zeros((WORLD, WORLD), np.uint8)
    for k, c in enumerate(land_ids):
        p = cv2.resize((small_land == c).astype(np.float32), (WORLD, WORLD), interpolation=cv2.INTER_LINEAR)
        p = cv2.GaussianBlur(p, (0, 0), 9)
        p = warp(p, 12, cv2.INTER_LINEAR) + np.roll(noise, k * 211, axis=(0, 1))
        better = p > best_p
        best_p[better] = p[better]
        cls_land[better] = c
    terrain = np.where(land, cls_land, np.where(water_kind == 2, 1, 0)).astype(np.uint8)
    deep = cv2.resize((cls_small == 0).astype(np.uint8), (WORLD, WORLD), interpolation=cv2.INTER_NEAREST)
    terrain[~land & (deep == 0)] = 1
    terrain[~land & (deep == 1)] = 0

    # --- ownership ---
    owner = np.zeros((WORLD, WORLD), np.uint8)
    owner[py0:py1, px0:px1] = cv2.resize(owner_pol, (pw, ph), interpolation=cv2.INTER_NEAREST)
    owner = extend_cut_claims(owner, (px0, py0, px1, py1))
    owner = smooth_owner(owner, land)
    names = [n for n, _ in COUNTRIES]
    for country, disk_m in city_disks:
        if country in names:
            owner[disk_m & land] = names.index(country) + 1
    # Border drawings from the user's in-game screenshots (tools/make_owner_edits.py): country index, 0 = keep.
    if (root / "data/edits/owner.png").exists() and not (root / PAINTED).exists():
        e_own = np.array(Image.open(root / "data/edits/owner.png"))[MARGIN:MARGIN + WORLD, MARGIN:MARGIN + WORLD]
        owner = np.where((e_own > 0) & land, e_own, owner).astype(owner.dtype)
        print(f"owner edits: {((e_own > 0) & land).sum()} px")

    # --- hand edits from the user's marked screenshots (data/edits/*.png) ---
    ed = root / "data/edits"
    if (ed / "sea.png").exists():
        crop = (slice(MARGIN, MARGIN + WORLD),) * 2
        e_sea = (np.array(Image.open(ed / "sea.png")) > 0)[crop]
        e_land = (np.array(Image.open(ed / "land.png")) > 0)[crop]
        e_river = (np.array(Image.open(ed / "river.png")) > 0)[crop]
        city_any = np.zeros_like(land)
        for _, disk_m in city_disks:
            city_any |= disk_m
        new_sea = e_sea & (owner == 0) & ~city_any & land
        land = (land & ~new_sea) | e_land
        lab, n = ndimage.label(~land)
        kinds = np.zeros(n + 1, np.uint8)
        for i, sl in enumerate(ndimage.find_objects(lab), 1):
            part = lab[sl] == i
            old = water_kind[sl][part]
            kinds[i] = 1 if (old == 1).any() or part.sum() > 20000 else 2
        water_kind = np.where(land, 0, kinds[lab]).astype(np.uint8)
        terrain = np.where(land, np.where(terrain <= 2, cls_land, terrain), np.where(water_kind == 2, 1, terrain))
        terrain[(~land) & (terrain > 2)] = 1
        owner[~land] = 0
        river = (river | e_river) & land
        print(f"edits: +{new_sea.sum()} px sea, +{e_land.sum()} px land, +{e_river.sum()} px river")

    # --- the painted coastline map wins over the land/sea from the screenshots and edits above ---
    if (root / PAINTED_LAND).exists():
        pk = painted_land(root / PAINTED_LAND)
        # A city in painted water moves to the nearest coast (old city positions sat on emblems, with a round
        # island forced under them, which the user painted away); far from land it keeps a small island.
        dist, (ly, lx) = ndimage.distance_transform_edt(pk != 0, return_indices=True)
        for c in cities:
            y, x = min(int(c["fy"] * WORLD), WORLD - 1), min(int(c["fx"] * WORLD), WORLD - 1)
            if pk[y, x] == 0:
                continue
            if dist[y, x] <= 80:
                ny, nx = int(ly[y, x]), int(lx[y, x])
                c["fx"], c["fy"] = (nx + 0.5) / WORLD, (ny + 0.5) / WORLD
                c["x"], c["y"] = nx + MARGIN, ny + MARGIN
                print(f"  city {c['name']} was in painted water; moved {dist[y, x]:.0f} px to the coast")
            else:
                pk[(cyy - y) ** 2 + (cxx - x) ** 2 <= 6 ** 2] = 0
                print(f"  city {c['name']} is {dist[y, x]:.0f} px out at sea; kept a small island under it")
        # Narrow lake-coloured channels are the user's rivers: land with a HoI4 river (user, 2026-10-18).
        # They replace the automatic rivers, which the painted map does not show.
        lake = (pk == 2).astype(np.uint8)
        wide = cv2.morphologyEx(lake, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        narrow = lake.astype(bool) & ~wide.astype(bool)
        if narrow.any():
            river = narrow.copy()
            pk = np.where(narrow, 0, pk).astype(np.uint8)
            print(f"  painted rivers: {narrow.sum()} px of narrow lake channels became rivers")
        land = pk == 0
        water_kind = pk
        terrain = np.where(land, np.where(terrain <= 2, cls_land, terrain),
                           np.where(water_kind == 2, 1, np.where(terrain > 2, 1, terrain))).astype(np.uint8)
        river &= land
        owner[~land] = 0

    # --- the painted ownership map wins over everything above; cities follow it ---
    if (root / PAINTED).exists():
        owner = np.where(land, painted_owner(root / PAINTED), 0).astype(np.uint8)
        for c in cities:
            o = owner[min(int(c["fy"] * WORLD), WORLD - 1), min(int(c["fx"] * WORLD), WORLD - 1)]
            c["country"] = names[o - 1] if o else None

    # --- the world border: an impassable mountain wall around the world ---
    def pad(a, val):
        return np.pad(a, MARGIN, constant_values=val)
    land, river, owner = pad(land, True), pad(river, False), pad(owner, 0)
    water_kind = pad(np.where(land[MARGIN:-MARGIN, MARGIN:-MARGIN], 0, water_kind), 0)
    terrain = pad(terrain, 11)
    border = pad(np.zeros((WORLD, WORLD), bool), True)
    np.save(out / "border.npy", border)
    json.dump(cities, open(cities_path(), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    np.save(out / "land.npy", land)
    np.save(out / "water.npy", water_kind)
    np.save(out / "river.npy", river)
    np.save(out / "terrain.npy", terrain)
    np.save(out / "owner.npy", owner)

    # --- preview ---
    pal = np.zeros((256, 3), np.uint8)
    for c, _, col in CLASSES:
        pal[c] = col
    img = pal[terrain].astype(np.float32)
    img[water_kind == 2] = (60, 120, 190)
    img[river] = (70, 130, 220)
    for i, (_, col) in enumerate(COUNTRIES, 1):
        m = owner == i
        img[m] = img[m] * 0.4 + np.clip(col, 0, 255) * 0.6
    Image.fromarray(img.astype(np.uint8)).save(out / "base_preview.png")
    Image.fromarray(img.astype(np.uint8)).resize((1024, 1024), Image.LANCZOS).save(out / "base_preview_small.png")
    print(f"land {land.mean():.1%}  sea {(water_kind == 1).mean():.1%}  "
          f"lakes {(water_kind == 2).mean():.2%}  river px {river.sum()}")


if __name__ == "__main__":
    main()
