"""Write heightmap.bmp, terrain.bmp, trees.bmp and world_normal.bmp into the mod.

Usage: python write_terrain.py <orbis_dir> <mod_dir> <hoi4_dir>

Heights follow vanilla conventions: sea floor 89 (deeper water a bit lower),
land starts at ~96 and mountains rise to ~215. Palettes for the indexed BMPs
are copied from the vanilla files so the indices mean the same thing.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from build_base import fractal_noise  # noqa: E402

# terrain class -> (terrain.bmp index, base height offset, relief amplitude)
CLASS_STYLE = {
    3: (0, 4, 4),     # plains
    4: (5, 5, 4),     # savanna
    5: (1, 7, 6),     # forest
    6: (21, 4, 5),    # jungle
    7: (3, 4, 5),     # desert
    8: (2, 14, 18),   # badlands / mesa -> hills
    9: (19, 3, 3),    # snow / tundra
    10: (17, 14, 16), # rocky -> hills
    11: (6, 45, 70),  # mountain
}
WATER_IDX = 15
SNOW_PEAK_IDX = 16
# terrain class -> list of (tree index, density)
TREES = {5: [(6, .55), (5, .25)], 6: [(28, .6), (29, .3)], 4: [(11, .12)], 9: [(5, .08)]}


def ridged(shape, scales, amps):
    n = fractal_noise(shape, scales, amps)
    return 1 - np.abs(n)


def main():
    root, mod, game = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    b = root / "build"
    land = np.load(b / "land.npy")
    water = np.load(b / "water.npy")
    terrain = np.load(b / "terrain.npy")
    H, W = land.shape
    out = mod / "map"

    # ---- heightmap ----
    coast_dist = ndimage.distance_transform_edt(land)
    sea_dist = ndimage.distance_transform_edt(~land)
    base = np.zeros((H, W), np.float32)
    amp = np.zeros((H, W), np.float32)
    for c, (_, off, a) in CLASS_STYLE.items():
        m = terrain == c
        base[m], amp[m] = off, a
    base = cv2.GaussianBlur(base, (0, 0), 14)   # gentle transitions between classes
    amp = cv2.GaussianBlur(amp, (0, 0), 10)
    hills = fractal_noise((H, W), (160, 60, 20, 7), (1, .6, .35, .15)) * 0.5 + 0.5
    ridges = ridged((H, W), (90, 34, 12), (1, .5, .25))
    relief = np.where(amp > 30, ridges ** 2, hills)
    inland = np.clip(coast_dist / 60, 0, 1)      # land rises away from the coast
    h_land = 96 + inland * (base + amp * relief) + np.clip(coast_dist / 3, 0, 2)
    h_sea = 92 - np.clip(sea_dist / 90, 0, 1) ** 0.7 * 8
    h = np.where(land, h_land, h_sea)
    border_f = b / "border.npy"
    if border_f.exists():   # the world border: a snowy mountain wall
        border = np.load(border_f)
        ramp = np.clip(1 - ndimage.distance_transform_edt(~border) / 18, 0, 1)
        wall = 180 + ridged((H, W), (40, 16, 6), (1, .5, .25)) * 50
        h = np.where(land, np.maximum(h, ramp * wall + (1 - ramp) * h), h)
    h = cv2.GaussianBlur(h.astype(np.float32), (0, 0), 2.0)
    h = np.where(land, np.maximum(h, 96), np.minimum(h, 93))
    height = np.clip(h, 0, 255).astype(np.uint8)
    Image.fromarray(height, "L").save(out / "heightmap.bmp")

    # ---- terrain.bmp (indexed, vanilla palette) ----
    van_t = Image.open(game / "map/terrain.bmp")
    idx = np.full((H, W), WATER_IDX, np.uint8)
    for c, (ti, _, _) in CLASS_STYLE.items():
        idx[land & (terrain == c)] = ti
    idx[land & (terrain == 11) & (height > 175)] = SNOW_PEAK_IDX
    t_img = Image.fromarray(idx, "P")
    t_img.putpalette(van_t.getpalette())
    t_img.save(out / "terrain.bmp")

    # ---- trees.bmp (lower resolution, vanilla palette) ----
    van_tr = Image.open(game / "map/trees.bmp")
    scale = van_tr.size[0] / 5632
    tw, th = round(W * scale), round(H * scale)
    t_small = cv2.resize(terrain, (tw, th), interpolation=cv2.INTER_NEAREST)
    l_small = cv2.resize(land.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST).astype(bool)
    rng = np.random.default_rng(5)
    roll = rng.random((th, tw))
    trees = np.zeros((th, tw), np.uint8)
    for c, kinds in TREES.items():
        m = l_small & (t_small == c)
        acc = 0.0
        for ti, dens in kinds:
            sel = m & (roll >= acc) & (roll < acc + dens)
            trees[sel] = ti
            acc += dens
    tr_img = Image.fromarray(trees, "P")
    tr_img.putpalette(van_tr.getpalette())
    tr_img.save(out / "trees.bmp")

    # ---- world_normal.bmp (half resolution) ----
    # Water is drawn flat, so only land relief goes into the normal map.
    hs = cv2.resize(np.maximum(h, 95).astype(np.float32), (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    gy, gx = np.gradient(hs)
    k = 0.6
    n = np.dstack([-gx * k, gy * k, np.ones_like(hs)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    normal = np.clip(n * 127.5 + 127.5, 0, 255).astype(np.uint8)
    Image.fromarray(normal, "RGB").save(out / "world_normal.bmp")

    # quick shaded preview
    shade = np.clip(normal[..., 0].astype(int) * 0.6 + normal[..., 1] * 0.4, 0, 255)
    Image.fromarray(shade.astype(np.uint8)).save(root / "build/relief_preview.png")
    print(f"heightmap land {np.percentile(height[land], [5, 50, 99])}  sea {np.percentile(height[~land], [1, 50, 99])}")
    print(f"trees {tw}x{th}, normal {W // 2}x{H // 2}")


if __name__ == "__main__":
    main()
