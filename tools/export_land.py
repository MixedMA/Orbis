"""Write the current coastlines as a paintable map, the user's source of land and water from then on.

Usage: python export_land.py <orbis_dir> [--force]
Reads build/{land,water,border}.npy from the last build. Writes source/land.png (MAP x MAP):
land, sea and lake in the colours of build_base (LAND_RGB, SEA_RGB, LAKE_RGB), the world border black (ignored).
Will not overwrite an existing land.png without --force (it holds the user's painting).
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from build_base import LAKE_RGB, LAND_RGB, PAINTED_LAND, SEA_RGB, WALL_RGB  # noqa: E402

root = Path(sys.argv[1])
dst = root / PAINTED_LAND
if dst.exists() and "--force" not in sys.argv:
    sys.exit(f"{dst} exists (the user's painting); pass --force to overwrite it")
land = np.load(root / "build/land.npy")
water = np.load(root / "build/water.npy")
border = np.load(root / "build/border.npy")
img = np.zeros(land.shape + (3,), np.uint8)
img[land] = LAND_RGB
img[~land & (water != 2)] = SEA_RGB
img[~land & (water == 2)] = LAKE_RGB
img[border] = WALL_RGB
Image.fromarray(img).save(dst)
print(f"wrote {dst}")
