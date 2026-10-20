"""Crop the world border area out of the map screenshot and paint out
waypoint icons, player labels and the claim outline.

Usage: python clean_screenshot.py <screenshot.png> <out_dir>
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

src, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
img = np.array(Image.open(src).convert("RGB"))

# Red world-border rectangle: crop to its interior.
r, g, b = (img[..., i].astype(int) for i in range(3))
red = (r > 180) & (g < 60) & (b < 60)
cols = np.nonzero(red.sum(0) > 100)[0]
rows = np.nonzero(red.sum(1) > 100)[0]
x0, x1, y0, y1 = cols.min() + 2, cols.max() - 1, rows.min() + 2, rows.max() - 1
img = img[y0:y1, x0:x1].copy()
a = img.astype(int)
lo, hi = a.min(2), a.max(2)

# Waypoint icons: bright white rings with a dark rim and a small interior.
white = lo > 225
ring_fill = ndimage.binary_fill_holes(white)
labels, n = ndimage.label(ring_fill)
idx = range(1, n + 1)
sizes = ndimage.sum(ring_fill, labels, idx)
dark = hi < 70
# Icons have a dark rim right outside the white ring; snow patches do not.
rim = ndimage.grey_dilation(labels, size=5) * ~ring_fill
rim_dark = ndimage.mean(dark, rim, idx)
rim_dark = np.nan_to_num(rim_dark)
# ...or they are round, roughly 8-14 px wide rings with a hole in the middle.
white_px = ndimage.sum(white, labels, idx)
hole = 1 - white_px / np.maximum(sizes, 1)
round_ = np.zeros(n, bool)
for i, sl in enumerate(ndimage.find_objects(labels)):
    bh, bw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
    extent = sizes[i] / (bh * bw)
    round_[i] = 7 <= bw <= 15 and 7 <= bh <= 15 and abs(bw - bh) <= 3 \
        and 0.55 <= extent <= 0.95 and hole[i] > 0.08
keep = (sizes >= 12) & (sizes < 160) & ((rim_dark > 0.25) | round_)
icon = np.isin(labels, 1 + np.nonzero(keep)[0])

# Ring template match catches rings that are not closed at the 225 cutoff.
yy, xx = np.mgrid[-7:8, -7:8]
rr = np.hypot(yy, xx)
ring = np.where((rr >= 3.2) & (rr <= 5.0), 1.0, 0.0).astype(np.float32)
ring[(rr < 2.2) | ((rr > 5.6) & (rr < 7))] = -1.0
score = cv2.matchTemplate(lo.astype(np.float32), ring, cv2.TM_CCOEFF_NORMED)
peaks = (score > 0.40) & (score == ndimage.maximum_filter(score, size=5))
centers = np.zeros_like(icon)
py, px = np.nonzero(peaks)
centers[py + 7, px + 7] = True
icon |= ndimage.binary_dilation(centers, structure=rr <= 5.5)
mask = icon | (dark & ndimage.binary_dilation(icon, iterations=3))
mask = ndimage.binary_dilation(mask, iterations=2)

# Player name tags and the claim outline in the inland sea (fixed areas).
h, w = mask.shape
def box(xa, ya, xb, yb):
    return (slice(int(ya * h), int(yb * h)), slice(int(xa * w), int(xb * w)))
mask[box(0.28, 0.22, 0.47, 0.32)] = True               # name tags
claim = box(0.34, 0.27, 0.55, 0.58)
light = (lo > 150) & (a[..., 2] > a[..., 0])
mask[claim] |= ndimage.binary_dilation(light[claim], iterations=1)
mask[box(0.455, 0.465, 0.485, 0.495)] = True            # globe marker
mask[box(0.58, 0.07, 0.80, 0.095)] |= dark[box(0.58, 0.07, 0.80, 0.095)]  # black bar

clean = cv2.inpaint(img, mask.astype(np.uint8) * 255, 4, cv2.INPAINT_TELEA)

out_dir.mkdir(parents=True, exist_ok=True)
Image.fromarray(clean).save(out_dir / "world_clean.png")
Image.fromarray((mask * 255).astype(np.uint8)).save(out_dir / "icon_mask.png")
print(f"crop {x0},{y0} -> {x1},{y1} ({clean.shape[1]}x{clean.shape[0]} px), "
      f"masked {mask.mean():.1%}")
