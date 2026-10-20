"""Write map/provinces.bmp, map/definition.csv and map/continent.txt into the mod.

Usage: python write_provinces.py <orbis_dir> <mod_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# terrain class id (classify_biomes.CLASSES) -> HOI4 province terrain
CLASS_TO_TERRAIN = {
    3: "plains", 4: "plains", 5: "forest", 6: "jungle", 7: "desert",
    8: "hills", 9: "plains", 10: "hills", 11: "mountain",
}
CONTINENTS = [  # name, anchor (fraction of map)
    ("wozgoria", (0.30, 0.45)),
    ("daukia", (0.72, 0.72)),
    ("afontova", (0.72, 0.20)),
    ("eos", (0.14, 0.86)),
]


def main():
    root, mod = Path(sys.argv[1]), Path(sys.argv[2])
    prov = np.load(root / "build/province_ids.npy")
    meta = json.load(open(root / "build/provinces.json"))
    H, W = prov.shape
    (mod / "map").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2000)
    used = {(0, 0, 0)}
    colors = np.zeros((len(meta) + 1, 3), np.uint8)
    for p in meta:
        while True:
            c = tuple(int(v) for v in rng.integers(1, 256, 3))
            if c not in used:
                break
        used.add(c)
        colors[p["id"]] = c
        p["color"] = list(c)

    img = colors[prov]
    Image.fromarray(img, "RGB").save(mod / "map/provinces.bmp")

    anchors = np.array([a for _, a in CONTINENTS]) * (W, H)
    lines = ["0;0;0;0;land;false;unknown;0"]
    for p in meta:
        r, g, b = p["color"]
        if p["type"] == "land":
            terr = CLASS_TO_TERRAIN.get(p["terrain_class"], "plains")
            cont = int(np.argmin(((anchors - (p["x"], p["y"])) ** 2).sum(1))) + 1
            p["continent"] = cont
        elif p["type"] == "sea":
            terr, cont = "ocean", 0
        else:
            terr, cont = "lakes", 0
        p["terrain"] = terr
        lines.append(f"{p['id']};{r};{g};{b};{p['type']};{str(p['coastal']).lower()};{terr};{cont}")
    (mod / "map/definition.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (mod / "map/continent.txt").write_text(
        "continents = {\n" + "".join(f"\t{n}\n" for n, _ in CONTINENTS) + "}\n", encoding="utf-8")
    json.dump(meta, open(root / "build/provinces.json", "w"))
    print(f"wrote {len(meta)} provinces, {W}x{H}")


if __name__ == "__main__":
    main()
