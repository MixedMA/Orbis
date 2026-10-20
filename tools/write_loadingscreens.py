"""Loading screens / menu backgrounds from the user's pictures.

Usage: python write_loadingscreens.py <orbis_dir> <mod_dir>

Every picture (png/jpg) in source/loadingscreens/ becomes one background (sorted by file name): HoI4 shows them
on the start-up menu and while a game loads. Pictures are scaled and centre-cropped to 1920x1440 (vanilla's
size; widescreen monitors crop the top and bottom). Writes
  gfx/loadingscreens/orbis_<n>.dds and orbis_<n>_small.dds   (the small one is the picker icon)
  interface/orbis_backgrounds.gfx                              GFX_orbis_<n>_small
  common/frontend/backgrounds/orbis_backgrounds.txt            the background list
and adds replace_path="common/frontend/backgrounds" to the descriptor files, so the vanilla pictures are not
mixed in. Without pictures nothing is written and the vanilla backgrounds stay.
"""
import sys
from pathlib import Path

from PIL import Image, ImageOps

W, H = 1920, 1440


def main():
    root, mod = Path(sys.argv[1]), Path(sys.argv[2])
    src = root / "source/loadingscreens"
    src.mkdir(parents=True, exist_ok=True)
    pics = sorted(p for p in src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    out = mod / "gfx/loadingscreens"
    for d in (out, mod / "common/frontend/backgrounds"):
        if d.exists():
            for f in d.glob("orbis_*"):
                f.unlink()
    if not pics:
        print("no pictures in source/loadingscreens: vanilla backgrounds stay")
        return
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for i, p in enumerate(pics, 1):
        img = ImageOps.fit(Image.open(p).convert("RGB"), (W, H), Image.LANCZOS)
        img.save(out / f"orbis_{i}.dds", pixel_format="DXT1")
        img.resize((W // 10, H // 10), Image.LANCZOS).save(out / f"orbis_{i}_small.dds", pixel_format="DXT1")
        names.append(f"orbis_{i}")
        print(f"{p.name} -> orbis_{i}.dds")
    (mod / "interface").mkdir(exist_ok=True)
    gfx = "spriteTypes = {\n" + "".join(
        f'\tspriteType = {{\n\t\tname = "GFX_{n}_small"\n\t\ttexturefile = "gfx/loadingscreens/{n}_small.dds"\n\t}}\n\n'
        for n in names) + "}\n"
    (mod / "interface/orbis_backgrounds.gfx").write_text(gfx, encoding="utf-8-sig")
    (mod / "common/frontend/backgrounds").mkdir(parents=True, exist_ok=True)
    (mod / "common/frontend/backgrounds/orbis_backgrounds.txt").write_text(
        "".join(f"{n} = {{\n}}\n\n" for n in names), encoding="utf-8")
    line = 'replace_path="common/frontend/backgrounds"'
    docs = Path.home() / "Documents/Paradox Interactive/Hearts of Iron IV/mod/Orbis_mod.mod"
    for f in (mod / "descriptor.mod", docs):
        if f.exists() and line not in f.read_text(encoding="utf-8"):
            f.write_text(f.read_text(encoding="utf-8").rstrip("\n") + "\n" + line + "\n", encoding="utf-8")
            print("added replace_path to", f)
    print(f"{len(names)} loading screens")


if __name__ == "__main__":
    main()
