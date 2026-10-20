"""Split the base rasters into HOI4 provinces.

Usage: python build_provinces.py <orbis_dir>

Reads build/{land,water,terrain,owner}.npy and writes
  build/province_ids.npy   int32 province id per pixel (1..N)
  build/provinces.json     per-province metadata
  build/provinces_preview.png

Provinces never cross an owner border or a land/sea/lake boundary. Inside each
such region, seeds are spread by a few Lloyd iterations and pixels are
assigned to the nearest seed in noise-warped coordinates, which gives evenly
sized provinces with organic borders.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from build_base import fractal_noise  # noqa: E402

TARGET = {"land": 1800, "sea": 9000, "lake": 6000, "border": 5000}
MIN_PX = {"land": 350, "sea": 1500, "lake": 150, "border": 1000}
# The user's hand-painted provinces (tools/export_provinces.py writes the first version): one colour per
# province. Smaller provinces are allowed there, the user may draw them on purpose.
PAINTED = "source/provinces.png"
MIN_PX_PAINTED = {"land": 100, "sea": 300, "lake": 60, "border": 300}
WARP = 11.0
LLOYD_ITERS = 4
RNG = np.random.default_rng(1940)


def assign(coords, region_of, seeds, seed_region):
    """Nearest seed within the same region (region id as a far-apart 3rd axis)."""
    big = 1e7
    tree = cKDTree(np.column_stack([seeds, seed_region * big]))
    _, nearest = tree.query(np.column_stack([coords, region_of * big]), workers=-1)
    return nearest


def main():
    root = Path(sys.argv[1])
    b = root / "build"
    land = np.load(b / "land.npy")
    water = np.load(b / "water.npy")
    owner = np.load(b / "owner.npy")
    terrain = np.load(b / "terrain.npy")
    H, W = land.shape

    # Regions: connected areas of the same kind (land/sea/lake) and owner.
    kind = np.where(land, 0, np.where(water == 2, 2, 1)).astype(np.uint8)
    border_f = b / "border.npy"
    if border_f.exists():
        kind[np.load(border_f)] = 3   # impassable world border wall
    key = kind.astype(np.int32) * 256 + owner
    region = np.zeros((H, W), np.int32)
    next_id = 1
    for k in np.unique(key):
        lab, n = ndimage.label(key == k)
        m = lab > 0
        region[m] = lab[m] + next_id - 1
        next_id += n
    n_regions = next_id - 1
    reg_sizes = np.bincount(region.ravel(), minlength=n_regions + 1)
    reg_kind = np.zeros(n_regions + 1, np.uint8)
    reg_kind[region.ravel()] = kind.ravel()
    kinds = ["land", "sea", "lake", "border"]
    print(f"{n_regions} regions")

    if (root / PAINTED).exists():
        MIN_PX.update(MIN_PX_PAINTED)
        prov = painted_provinces(root / PAINTED, region, reg_kind, kinds)
    else:
        prov = generated_provinces(region, n_regions, reg_sizes, reg_kind, kinds)

    # Every province one connected piece; strays and tiny ones join a neighbour.
    prov = fix_fragments(prov, region, reg_kind)
    # No two provinces may touch only at a pixel corner: HOI4 counts corner contacts as adjacency,
    # we do not, and a land province touching the sea only at a corner crashed the game at start
    # (it became "coastal" in game but had no port sea zone).
    for _ in range(6):
        if not fix_corner_contacts(prov, region):
            break
        prov = fix_fragments(prov, region, reg_kind)

    # Renumber 1..N
    ids = np.unique(prov)
    ids = ids[ids > 0]
    remap = np.zeros(prov.max() + 1, np.int32)
    remap[ids] = np.arange(1, len(ids) + 1)
    prov = remap[prov]
    N = len(ids)

    # Terrain overrides (user): the province under each point gets that terrain class (11 = mountain), in the
    # terrain raster too, so the map picture and heightmap follow (province ids change, so points identify them).
    ov = root / "data/terrain_overrides.json"
    if ov.exists():
        for o in json.load(open(ov, encoding="utf-8")):
            pid = prov[o["y"], o["x"]]
            terrain[(prov == pid) & land] = o["terrain_class"]
        np.save(b / "terrain.npy", terrain)

    # Metadata
    meta = describe(prov, N, kind, owner, terrain, region)
    # A province holding a claimed city always belongs to that city's country.
    from extract_countries import COUNTRIES
    from scenario import cities_path
    names = [n for n, _ in COUNTRIES]
    for c in json.load(open(cities_path(), encoding="utf-8")):
        if c["country"] in names:
            p = meta[prov[c["y"], c["x"]] - 1]
            p["owner"] = names.index(c["country"]) + 1
    np.save(b / "province_ids.npy", prov)
    json.dump(meta, open(b / "provinces.json", "w"))
    counts = {k: sum(1 for p in meta if p["type"] == k) for k in kinds[:3]}
    counts["border"] = sum(1 for p in meta if p["border"])
    print(f"{N} provinces: {counts}")
    preview(prov, kind, b)


def painted_provinces(path, region, reg_kind, kinds):
    """Province ids from the painted map: each connected piece of one colour inside one region
    (land/sea/lake/wall of one owner) becomes a province, so painting across a coast or a country
    border splits the province there. Pieces whose colour mostly lies on the other kind (a sea province's
    colour on newly painted land, or the reverse) are re-divided by the generator instead."""
    a = np.array(Image.open(path).convert("RGB")).astype(np.int64)
    if a.shape[:2] != region.shape:
        sys.exit(f"{path} is {a.shape[1]}x{a.shape[0]}, must be {region.shape[1]}x{region.shape[0]}")
    uk, cid = np.unique((a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2], return_inverse=True)
    cid = cid.reshape(region.shape).astype(np.int32) + 1
    ncol = int(cid.max())
    # The kind each colour was exported as (source/provinces_kinds.json); new colours count as what they lie on.
    major = np.bincount(cid.ravel() * 4 + reg_kind[region].ravel(), minlength=(ncol + 1) * 4)
    major = major.reshape(ncol + 1, 4).argmax(1)
    kinds_f = path.parent / "provinces_kinds.json"
    if kinds_f.exists():
        known = json.load(open(kinds_f))
        for c, k in enumerate(uk, 1):
            if "%06x" % k in known:
                major[c] = kinds.index(known["%06x" % k])
    prov = np.zeros(region.shape, np.int32)
    auto = np.zeros(region.shape, np.int32)
    auto_kind = [0]
    nid = 1
    for c, sl in enumerate(ndimage.find_objects(cid), 1):
        if sl is None:
            continue
        m = cid[sl] == c
        reg = region[sl]
        out, aut = prov[sl], auto[sl]
        for r in np.unique(reg[m]):
            lab, n = ndimage.label(m & (reg == r))
            part = lab > 0
            if reg_kind[r] != major[c]:
                aut[part] = lab[part] + len(auto_kind) - 1
                auto_kind += [reg_kind[r]] * n
            else:
                out[part] = lab[part] + nid - 1
                nid += n
    print(f"painted provinces: {ncol} colours -> {nid - 1} pieces before clean-up")
    if len(auto_kind) > 1:
        # Repainted slivers along a coast join the nearest province of the same region (within 40 px);
        # only what is left (new seas, lakes, landmasses) is divided into new provinces.
        dist, (iy, ix) = ndimage.distance_transform_edt(prov == 0, return_indices=True)
        join = (auto > 0) & (dist <= 40) & (region[iy, ix] == region)
        prov[join] = prov[iy, ix][join]
        auto[join] = 0
        print(f"  repainted coast: {join.sum()} px joined neighbouring provinces")
        sizes = np.bincount(auto.ravel(), minlength=len(auto_kind))
        gen = generated_provinces(auto, len(auto_kind) - 1, sizes, np.array(auto_kind, np.uint8), kinds)
        prov[auto > 0] = gen[auto > 0] + nid
        print(f"  repainted coast: {sizes[1:].sum()} px re-divided into {len(np.unique(gen[auto > 0]))} provinces")
    return prov


def generated_provinces(region, n_regions, reg_sizes, reg_kind, kinds):
    H, W = region.shape
    # Seeds per region, proportional to area (at least one).
    seeds, seed_region = [], []
    ys, xs = np.nonzero(region)
    order = np.argsort(region[ys, xs], kind="stable")
    ys, xs = ys[order], xs[order]
    starts = np.searchsorted(region[ys, xs], np.arange(1, n_regions + 2))
    for r in range(1, n_regions + 1):
        a, z = starts[r - 1], starts[r]
        n = max(1, round(reg_sizes[r] / TARGET[kinds[reg_kind[r]]]))
        pick = RNG.choice(z - a, size=min(n, z - a), replace=False) + a
        seeds.append(np.column_stack([xs[pick], ys[pick]]).astype(np.float64))
        seed_region.append(np.full(len(pick), r))
    seeds = np.concatenate(seeds)
    seed_region = np.concatenate(seed_region)
    print(f"{len(seeds)} seeds")

    # Lloyd relaxation in plain coordinates.
    coords = np.column_stack([xs, ys]).astype(np.float64)
    reg_of = region[ys, xs].astype(np.float64)
    for it in range(LLOYD_ITERS):
        nearest = assign(coords, reg_of, seeds, seed_region)
        cnt = np.bincount(nearest, minlength=len(seeds))
        sx = np.bincount(nearest, coords[:, 0], minlength=len(seeds))
        sy = np.bincount(nearest, coords[:, 1], minlength=len(seeds))
        ok = cnt > 0
        seeds[ok, 0], seeds[ok, 1] = sx[ok] / cnt[ok], sy[ok] / cnt[ok]

    # Final assignment in warped coordinates for organic borders.
    dx = fractal_noise((H, W), (96, 32, 10), (1, .5, .3)) * WARP
    dy = fractal_noise((H, W), (96, 32, 10), (1, .5, .3)) * WARP
    wcoords = coords + np.column_stack([dx[ys, xs], dy[ys, xs]])
    nearest = assign(wcoords, reg_of, seeds, seed_region)
    prov = np.zeros((H, W), np.int32)
    prov[ys, xs] = nearest + 1
    return prov


def corner_contacts(prov):
    """Top-left corners (y, x) of 2x2 blocks where a diagonal pair of different provinces
    is not joined through the other two pixels."""
    a, b = prov[:-1, :-1], prov[:-1, 1:]
    c, d = prov[1:, :-1], prov[1:, 1:]
    main = (a != d) & (b != a) & (b != d) & (c != a) & (c != d)
    anti = (b != c) & (a != b) & (a != c) & (d != b) & (d != c)
    return np.argwhere(main | anti)


def fix_corner_contacts(prov, region):
    """Give the top-left pixel of each crossing block to its right or lower neighbour (preferring
    one of the same region, so land stays land), which joins both diagonals edge to edge.
    Returns the number of blocks fixed (0 = clean)."""
    blocks = corner_contacts(prov)
    for y, x in blocks:
        right, down = prov[y, x + 1], prov[y + 1, x]
        same_down = region[y + 1, x] == region[y, x] and region[y, x + 1] != region[y, x]
        prov[y, x] = down if same_down else right
    print(f"  corner contacts: {len(blocks)} fixed")
    return len(blocks)


def fix_fragments(prov, region, reg_kind):
    H, W = prov.shape
    for _ in range(3):
        changed = 0
        objs = ndimage.find_objects(prov)
        for pid, sl in enumerate(objs, 1):
            if sl is None:
                continue
            sl = (slice(max(sl[0].start - 1, 0), min(sl[0].stop + 1, H)),
                  slice(max(sl[1].start - 1, 0), min(sl[1].stop + 1, W)))
            sub = prov[sl]
            m = sub == pid
            lab, n = ndimage.label(m)
            r = region[sl][m][0]
            kmin = MIN_PX[["land", "sea", "lake", "border"][reg_kind[r]]]
            sizes = ndimage.sum(m, lab, range(1, n + 1))
            keep = np.argmax(sizes) + 1
            for i in range(1, n + 1):
                if i == keep and sizes[i - 1] >= kmin:
                    continue
                part = lab == i
                ring = ndimage.binary_dilation(part) & ~part
                cand = sub[ring & (region[sl] == r) & (sub != pid)]
                if cand.size == 0 and sizes[i - 1] < kmin:
                    # tiny claim speck: join a neighbour of the same kind (other owner)
                    same_kind = reg_kind[region[sl]] == reg_kind[r]
                    cand = sub[ring & same_kind & (sub != pid)]
                    if cand.size == 0 and sizes[i - 1] < 60:
                        cand = sub[ring & (sub != pid)]  # speck island: swallowed
                if cand.size == 0:
                    continue  # real island of its own: keep as is
                sub[part] = np.bincount(cand).argmax()
                changed += 1
        print(f"  fragment pass: {changed} merges")
        if not changed:
            break
    return prov


def describe(prov, N, kind, owner, terrain, region):
    flat = prov.ravel()
    cnt = np.bincount(flat, minlength=N + 1)
    ys, xs = np.indices(prov.shape)
    cx = np.bincount(flat, xs.ravel(), minlength=N + 1) / np.maximum(cnt, 1)
    cy = np.bincount(flat, ys.ravel(), minlength=N + 1) / np.maximum(cnt, 1)
    pkind = np.zeros(N + 1, np.uint8)
    pkind[flat] = kind.ravel()
    ocount = np.zeros((N + 1, int(owner.max()) + 1), np.int64)
    np.add.at(ocount, (flat, owner.ravel()), 1)
    powner = ocount.argmax(1).astype(np.uint8)  # majority owner of the province
    # majority terrain class per province
    tcount = np.zeros((N + 1, 16), np.int64)
    np.add.at(tcount, (flat, terrain.ravel()), 1)
    pterr = tcount.argmax(1)
    # adjacency (4-neighbour) for coastal flags
    adj = set()
    for a, c in ((prov[:, :-1], prov[:, 1:]), (prov[:-1, :], prov[1:, :])):
        d = a != c
        pairs = np.unique(np.column_stack([a[d], c[d]]), axis=0)
        adj.update(map(tuple, pairs))
    neigh = [set() for _ in range(N + 1)]
    for a, c in adj:
        neigh[a].add(int(c))
        neigh[c].add(int(a))
    names = ["land", "sea", "lake", "land"]
    meta = []
    for p in range(1, N + 1):
        k = names[pkind[p]]
        coastal = any(pkind[q] == 1 for q in neigh[p]) if k == "land" else False
        if k == "sea":
            coastal = any(pkind[q] in (0, 3) for q in neigh[p])
        meta.append({"id": p, "type": k, "border": bool(pkind[p] == 3), "owner": int(powner[p]), "terrain_class": int(pterr[p]),
                     "px": int(cnt[p]), "x": round(float(cx[p]), 1), "y": round(float(cy[p]), 1),
                     "coastal": bool(coastal), "neighbors": sorted(neigh[p])})
    return meta


def preview(prov, kind, b):
    rng = np.random.default_rng(7)
    N = prov.max()
    col = rng.integers(60, 230, (N + 1, 3)).astype(np.uint8)
    img = col[prov]
    img[kind == 1] = img[kind == 1] // 3 + np.array([20, 40, 90], np.uint8)
    img[kind == 2] = img[kind == 2] // 3 + np.array([40, 90, 150], np.uint8)
    edge = np.zeros(prov.shape, bool)
    edge[:, 1:] |= prov[:, 1:] != prov[:, :-1]
    edge[1:, :] |= prov[1:, :] != prov[:-1, :]
    img[edge] = (20, 20, 20)
    Image.fromarray(img).save(b / "provinces_preview.png")
    Image.fromarray(img).resize((1024, 1024), Image.LANCZOS).save(b / "provinces_preview_small.png")


if __name__ == "__main__":
    main()
