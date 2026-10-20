"""Side-by-side: cleaned screenshot | terrain classes, with a legend.
Usage: python render_compare.py <out_dir>"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).parent))
from classify_biomes import CLASSES

LABELS = {"deep_ocean": "Tiefsee", "ocean": "Meer", "frozen_sea": "Packeis", "plains": "Ebene",
          "savanna": "Savanne", "forest": "Wald", "jungle": "Dschungel", "desert": "Wüste",
          "badlands": "Badlands", "snow": "Schnee/Tundra", "rocky": "Felsig/Hügel",
          "mountain": "Gebirge"}
out = Path(sys.argv[1])
a = Image.open(out / "world_clean.png").convert("RGB")
a = a.resize((a.width * 2, a.height * 2), Image.LANCZOS)
b = Image.open(out / "terrain_preview.png").convert("RGB").resize(a.size, Image.NEAREST)
pad, leg_h = 16, 70
img = Image.new("RGB", (a.width * 2 + pad * 3, a.height + pad * 2 + leg_h), (24, 24, 28))
img.paste(a, (pad, pad)); img.paste(b, (a.width + pad * 2, pad))
d = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("segoeui.ttf", 15)
except OSError:
    font = ImageFont.load_default()
x, y = pad, a.height + pad * 2
for _, name, col in CLASSES:
    d.rectangle([x, y, x + 16, y + 16], fill=col, outline=(200, 200, 200))
    d.text((x + 22, y - 1), LABELS[name], fill=(230, 230, 230), font=font)
    x += 22 + d.textlength(LABELS[name], font=font) + 22
    if x > img.width - 160:
        x, y = pad, y + 28
img.save(out / "orbis_terrain_v1.png")
print(img.size)
