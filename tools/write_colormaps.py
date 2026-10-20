"""Write Orbis's own map colour textures, replacing vanilla's (which showed Earth's continents, e.g. Mexico
inside Occidentalis).

Usage: python write_colormaps.py <orbis_dir> <mod_dir>
Reads the mod's map/terrain.bmp + map/heightmap.bmp, build/{land,water}.npy, the city and town lists.
Writes to <mod>/map/terrain/, at half the map resolution like vanilla:
  colormap_rgb_cityemissivemask_a.dds  land colour per terrain type (vanilla's median colour for that terrain
                                       index, measured from vanilla's own colormap), relief shading and noise;
                                       alpha = city lights at night (capitals, cities, towns)
  colormap_water_0/1/2.dds             water colour: light at the coast, dark in deep water (vanilla's values by
                                       distance to land)
  fow_rgb_waterspec_a.dds              fog of war: grey land with relief, darker cloudy sea
"""
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from build_base import fractal_noise  # noqa: E402
from scenario import TOWNS, cities_path  # noqa: E402

# Vanilla's median colormap colour per terrain.bmp index (measured from vanilla 1.19 colormap + terrain.bmp).
TERRAIN_RGB = {0: (60, 65, 35), 1: (43, 53, 21), 2: (57, 62, 33), 3: (120, 101, 66), 4: (49, 56, 24),
               5: (74, 73, 41), 7: (126, 105, 63), 8: (93, 81, 57), 9: (49, 55, 29), 10: (99, 85, 57),
               11: (57, 59, 33), 12: (90, 71, 46), 13: (60, 68, 33), 16: (52, 54, 32), 17: (57, 64, 33),
               19: (54, 59, 33), 20: (54, 59, 30), 21: (22, 35, 7), 22: (33, 48, 13)}
DEFAULT_RGB = (60, 65, 35)
SEA_RGB = (79, 102, 142)                 # flat sea in the land colormap, as vanilla
# vanilla water colour by distance to land (half-res px): coast -> shallow -> deep
WATER_KNOTS = [(0, (60, 105, 145)), (3, (66, 112, 156)), (8, (74, 121, 167)), (22, (43, 74, 101)), (50, (33, 56, 74))]
WATER_ON_LAND = (57, 85, 101)


def save_dds_argb(img, path):
    """Uncompressed 32-bit A8R8G8B8 DDS without mipmaps, the format of vanilla's colormap."""
    h, w = img.shape[:2]
    hdr = struct.pack("<4sIIIIIII44x", b"DDS ", 124, 0x1 | 0x2 | 0x4 | 0x8 | 0x1000, h, w, w * 4, 0, 1)
    pf = struct.pack("<II4sIIIII", 32, 0x41, b"\0\0\0\0", 32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
    caps = struct.pack("<IIIII", 0x1000, 0, 0, 0, 0)
    bgra = img[..., [2, 1, 0, 3]].astype(np.uint8)
    with open(path, "wb") as f:
        f.write(hdr + pf + caps + bgra.tobytes())


def main():
    root, mod = Path(sys.argv[1]), Path(sys.argv[2])
    out = mod / "map/terrain"
    out.mkdir(parents=True, exist_ok=True)
    terr = np.array(Image.open(mod / "map/terrain.bmp"))
    height = np.array(Image.open(mod / "map/heightmap.bmp")).astype(np.float32)
    land = np.load(root / "build/land.npy")
    H, W = terr.shape
    h2, w2 = H // 2, W // 2

    # --- land colour ---
    lut = np.array([TERRAIN_RGB.get(i, DEFAULT_RGB) for i in range(256)], np.float32)
    rgb = cv2.GaussianBlur(lut[terr], (0, 0), 2.5)
    hs = cv2.GaussianBlur(height, (0, 0), 2)
    gx, gy = np.gradient(hs)
    shade = np.clip(1 - 0.045 * (gx + gy), 0.7, 1.3)                    # light from the north-west
    noise = 1 + 0.10 * fractal_noise((H, W), (128, 32, 8), (1, .6, .4))
    rgb = rgb * (shade * noise)[..., None]
    lm = cv2.GaussianBlur(land.astype(np.float32), (0, 0), 1.5)[..., None]   # soft coastline
    rgb = rgb * lm + np.array(SEA_RGB, np.float32) * (1 - lm)
    rgb = cv2.resize(np.clip(rgb, 0, 255), (w2, h2), interpolation=cv2.INTER_AREA)

    # --- city lights (alpha): capitals, cities, towns as soft glows on land ---
    lights = np.zeros((h2, w2), np.float32)
    cities = json.load(open(cities_path(), encoding="utf-8"))
    towns = json.load(open(root / TOWNS, encoding="utf-8")) if (root / TOWNS).exists() else []
    caps = {c.get("capital_city") for c in json.load(open(root / "data/countries.json", encoding="utf-8"))}
    spots = [(c["x"], c["y"], (4.0, 255) if c["name"] in caps else (2.6, 220)) for c in cities]
    margin = (W - 3008) // 2
    spots += [(t["fx"] * 3008 + margin, t["fy"] * 3008 + margin, (1.6, 160)) for t in towns]
    yy, xx = np.mgrid[-16:17, -16:17]
    for x, y, (sig, peak) in spots:
        cx, cy = int(round(x / 2)), int(round(y / 2))
        blob = peak * np.exp(-(xx ** 2 + yy ** 2) / (2 * sig ** 2))
        y0, y1, x0, x1 = max(cy - 16, 0), min(cy + 17, h2), max(cx - 16, 0), min(cx + 17, w2)
        lights[y0:y1, x0:x1] = np.maximum(lights[y0:y1, x0:x1], blob[y0 - cy + 16:y1 - cy + 16, x0 - cx + 16:x1 - cx + 16])
    land2 = cv2.resize(land.astype(np.uint8), (w2, h2), interpolation=cv2.INTER_NEAREST).astype(bool)
    lights[~land2] = 0
    save_dds_argb(np.dstack([rgb, lights]).astype(np.uint8), out / "colormap_rgb_cityemissivemask_a.dds")

    # --- water colour by distance to land ---
    d = ndimage.distance_transform_edt(~land2)
    ks = np.array([k for k, _ in WATER_KNOTS], np.float32)
    wc = np.stack([np.interp(d, ks, [c[i] for _, c in WATER_KNOTS]) for i in range(3)], -1)
    wn = 1 + 0.08 * fractal_noise((h2, w2), (96, 24, 8), (1, .6, .3))
    wc = wc * wn[..., None]
    wc[land2] = WATER_ON_LAND
    wimg = Image.fromarray(np.dstack([np.clip(wc, 0, 255), np.full((h2, w2), 255)]).astype(np.uint8), "RGBA")
    for i, s in enumerate((1, 2, 4)):
        wimg.resize((w2 // s, h2 // s), Image.LANCZOS).save(out / f"colormap_water_{i}.dds", pixel_format="DXT5")

    # --- fog of war: grey land with relief, darker cloudy sea; alpha = water specular ---
    relief = cv2.resize(shade, (w2, h2), interpolation=cv2.INTER_AREA)
    fog_land = 148 + 110 * (relief - 1) + 10 * fractal_noise((h2, w2), (64, 16, 6), (1, .6, .4))
    fog_sea = 68 + 14 * fractal_noise((h2, w2), (256, 96, 32), (1, .6, .3))
    g = np.clip(np.where(land2, fog_land, fog_sea), 0, 255)
    fa = np.where(land2, 21, 38)
    Image.fromarray(np.dstack([g, g, g, fa]).astype(np.uint8), "RGBA").save(out / "fow_rgb_waterspec_a.dds",
                                                                            pixel_format="DXT5")
    print(f"colormaps {w2}x{h2}: {len(spots)} city lights")
    Image.fromarray(rgb.astype(np.uint8)).resize((1024, 1024), Image.LANCZOS).save(root / "out/colormap_preview.png")


if __name__ == "__main__":
    main()
