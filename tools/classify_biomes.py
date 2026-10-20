"""Group the cleaned map colours into a small set of Orbis terrain classes.

Usage: python classify_biomes.py <world_clean.png> <out_dir>

Writes terrain_classes.npy (uint8 class ids), terrain_preview.png and
landmask.png. The k-means cluster -> class table below was assigned by eye
from out/kmeans_vis.png; re-check it if the source screenshot changes.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

CLASSES = [  # id, name, preview colour
    (0, "deep_ocean", (20, 32, 80)),
    (1, "ocean", (35, 70, 130)),
    (2, "frozen_sea", (150, 175, 215)),
    (3, "plains", (125, 160, 80)),
    (4, "savanna", (175, 165, 85)),
    (5, "forest", (45, 105, 45)),
    (6, "jungle", (20, 130, 30)),
    (7, "desert", (225, 205, 140)),
    (8, "badlands", (185, 95, 55)),
    (9, "snow", (240, 245, 250)),
    (10, "rocky", (130, 130, 125)),
    (11, "mountain", (95, 80, 70)),
]
WATER = {0, 1, 2}
MIXED = 255

# Cluster centres (median RGB) -> class. Order matches the seeded k-means run.
PROTOTYPES = [
    ((29, 55, 106), 1), ((43, 91, 25), 6), ((68, 87, 63), 5), ((63, 83, 89), MIXED),
    ((187, 184, 152), 7), ((223, 230, 230), 9), ((141, 88, 60), 8), ((33, 79, 111), 1),
    ((28, 41, 100), 0), ((100, 97, 46), 4), ((113, 115, 105), 10), ((74, 88, 139), 2),
    ((72, 99, 49), 3), ((38, 102, 114), 1), ((80, 101, 60), MIXED), ((148, 155, 167), 9),
]
# Texture (local std) separates the speckled frozen sea from plain water.
PROTO_STD = [6, 8, 8, 11, 36, 35, 14, 6, 5, 6, 22, 34, 7, 7, 48, 41]


def classify(rgb: np.ndarray) -> np.ndarray:
    med = cv2.medianBlur(rgb, 3).astype(np.float32)
    lab = cv2.cvtColor(med / 255, cv2.COLOR_RGB2Lab)
    std = ndimage.generic_filter(rgb.astype(float).mean(2), np.std, size=5)
    proto_rgb = np.array([p[0] for p in PROTOTYPES], np.float32)[None] / 255
    proto_lab = cv2.cvtColor(proto_rgb, cv2.COLOR_RGB2Lab)[0]
    feat = np.concatenate([lab, std[..., None] * 0.5], -1)
    proto = np.concatenate([proto_lab, np.array(PROTO_STD)[:, None] * 0.5], -1)
    dist = ((feat[:, :, None, :] - proto[None, None]) ** 2).sum(-1)
    cls = np.array([p[1] for p in PROTOTYPES], np.uint8)[dist.argmin(-1)]
    return cls


def fix_claims(rgb: np.ndarray, cls: np.ndarray) -> None:
    """Player claims are drawn as translucent tinted areas; undo their effect.
    Boxes are fractions of the cropped world image."""
    h, w = cls.shape
    def box(xa, ya, xb, yb):
        return (slice(int(ya * h), int(yb * h)), slice(int(xa * w), int(xb * w)))
    a = rgb.astype(int)
    # Inland-sea archipelago under a blue tint: land is far less blue than sea.
    # Claim outline traced from the screenshot (fractions of the crop).
    poly = np.array([(0.358, 0.282), (0.430, 0.300), (0.511, 0.345), (0.538, 0.395),
                     (0.543, 0.484), (0.520, 0.507), (0.385, 0.507), (0.376, 0.422),
                     (0.354, 0.358)]) * (w, h)
    inside = np.zeros((h, w), np.uint8)
    cv2.fillPoly(inside, [poly.astype(np.int32)], 1)
    inside = ndimage.binary_erosion(inside.astype(bool), iterations=2)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    tinted_land = inside & (b < 1.45 * g)
    cls[tinted_land & np.isin(cls, list(WATER) + [MIXED])] = 5
    cls[tinted_land & (r + g + b > 330)] = 9
    # Badlands claim (purple fill + white outline) and a small square claim.
    sub = cls[box(0.21, 0.585, 0.31, 0.68)]
    sub[np.isin(sub, [0, 1, 2, 9, MIXED])] = 8
    sub = cls[box(0.34, 0.74, 0.37, 0.775)]
    sub[np.isin(sub, [0, 1, 2, 9, MIXED])] = 6


def majority(cls: np.ndarray, size: int, ignore=MIXED) -> np.ndarray:
    """Per-pixel most common class in a size x size window, ignoring `ignore`."""
    ids = [c for c, _, _ in CLASSES]
    counts = np.stack([ndimage.uniform_filter((cls == c).astype(np.float32), size)
                       for c in ids])
    best = np.array(ids, np.uint8)[counts.argmax(0)]
    return np.where(counts.max(0) > 0, best, ignore)


def main():
    src, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    rgb = np.array(Image.open(src).convert("RGB"))
    cls = classify(rgb)
    fix_claims(rgb, cls)

    # Mixed shoreline/town pixels take the class of their neighbourhood.
    for size in (3, 5, 9):
        cls = np.where(cls == MIXED, majority(cls, size), cls)

    # Land/sea mask: smooth, drop specks, fill tiny lakes.
    water = np.isin(cls, list(WATER))
    water = ndimage.binary_opening(water, iterations=1) | (
        ndimage.uniform_filter(water.astype(float), 3) > 0.6)
    lab_w, n = ndimage.label(water)
    small = np.nonzero(ndimage.sum(water, lab_w, range(1, n + 1)) < 30)[0] + 1
    water[np.isin(lab_w, small)] = False
    lab_l, n = ndimage.label(~water)
    small = np.nonzero(ndimage.sum(~water, lab_l, range(1, n + 1)) < 6)[0] + 1
    water[np.isin(lab_l, small)] = True

    # Land classes: majority filter so terrain comes in coherent patches.
    land_cls = np.where(np.isin(cls, list(WATER)), MIXED, cls)
    smooth = majority(land_cls, 5)
    smooth = np.where(smooth == MIXED, 3, smooth)
    sea_cls = majority(np.where(np.isin(cls, list(WATER)), cls, MIXED), 5)
    sea_cls = np.where(sea_cls == MIXED, 1, sea_cls)
    final = np.where(water, sea_cls, smooth).astype(np.uint8)

    # Small snow patches inside temperate land are snowy peaks, not tundra.
    snow = final == 9
    lab_s, n = ndimage.label(snow)
    near_ice = ndimage.binary_dilation(final == 2, iterations=4)
    for i, sl in enumerate(ndimage.find_objects(lab_s), 1):
        comp = lab_s[sl] == i
        if comp.sum() < 150 and not (near_ice[sl] & comp).any():
            final[sl][comp] = 11

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "terrain_classes.npy", final)
    pal = np.zeros((256, 3), np.uint8)
    for c, _, col in CLASSES:
        pal[c] = col
    prev = Image.fromarray(pal[final])
    prev.resize((prev.width * 3, prev.height * 3), Image.NEAREST).save(out_dir / "terrain_preview.png")
    Image.fromarray(np.where(water, 0, 255).astype(np.uint8)).save(out_dir / "landmask.png")
    for c, name, _ in CLASSES:
        print(f"{name:11s} {(final == c).mean():6.1%}")
    print(f"land total  {(~water).mean():6.1%}")


if __name__ == "__main__":
    main()
