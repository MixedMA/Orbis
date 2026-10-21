# Orbis: a Hearts of Iron IV total conversion

A total conversion mod for **Hearts of Iron IV** (1.19) based on the world of the Minecraft server *Orbis* (BMC
Season 1, a 4000x4000 block map). It starts in 1850 BCS, two years before the Orbis WW2. The map, provinces,
states, cities and borders are drawn by hand as pictures and turned into game files by the scripts in `tools/`.

Not affiliated with Paradox Interactive. You need your own copy of Hearts of Iron IV. Apart from two small game files
the mod needs to start (the generic focus tree and the dynamic country tags), the repository contains no game files.

## Playing
The built mod is in the repository (`mod/Orbis_mod`, including the generated map), so you can play without building:
clone or download the repository and copy both `mod/Orbis_mod.mod` and the folder `mod/Orbis_mod` into
`Documents/Paradox Interactive/Hearts of Iron IV/mod/`. Then enable only the Orbis mod in the launcher.
(If you keep the folder somewhere else, change the `path=` line in `Orbis_mod.mod` to that folder.)

## How it works
You edit the **painted maps** in `source/` and run the pipeline, which writes the mod to `mod/Orbis_mod`.

| What you want to change | Edit this |
|---|---|
| Coastlines, lakes, narrow rivers | `source/land.png` (sea / land / lake colours) |
| Which country owns land | `source/owners.png` (flat country colours, legend in `source/owners_legend.png`) |
| Provinces | `source/provinces.png` |
| States | `source/states.png` (one colour = one state per owner) |
| State names | `source/state_names.json` (state colour in hex -> name) |
| Cities, countries, scenario | `data/cities_1850.json`, `data/countries.json`, `tools/scenario.py` |
| Terrain of single provinces | `data/terrain_overrides.json` |
| Strait crossings | `data/crossings.json` (see `tools/make_crossings.py`) |
| Loading screens | pictures in `source/loadingscreens/` |
| Flags | `source/flags/<TAG>.png` (any size; scaled and cropped to the game's flag shape; countries without one get a placeholder) |
| Start economy (factories, population) | `ECONOMY` in `tools/scenario.py` |

Use exact colours with no anti-aliasing when painting. `CLAUDE.md` has the details: decisions, the file formats and
the problems found so far (read it before changing the pipeline).

## Building
1. Python 3.11+ and `pip install -r requirements.txt`.
2. Adjust the two paths (`MOD`, `GAME`) at the top of `tools/run_all.sh` to your folders.
3. `sh tools/run_all.sh` (about 6 minutes). It ends with `validate_mod.py`; it must print `OK - no errors`.
4. Make a launcher file `Orbis_mod.mod` in `Documents/Paradox Interactive/Hearts of Iron IV/mod/` whose `path=`
   points to `mod/Orbis_mod`, then start the game with only the Orbis mod enabled.

## Rules for contributors
- **Generated folders: never edit by hand.** Every build deletes and rewrites `mod/Orbis_mod/history/states`,
  `history/countries`, `map/strategicregions`, `common/bookmarks`, the map files and the flags/localisation it writes.
  Change their sources instead (painted maps, `data/`, `tools/scenario.py`). Hand-made content (focus trees, events,
  decisions, ideas) goes in its own files, e.g. `common/national_focus/<tag>.txt` or `events/<topic>.txt`, which the
  build does not touch.
- Never overwrite the painted maps with exports (`tools/export_*.py` refuse without `--force`); change them in a
  paint program and commit the picture.
- One change per commit, and run the pipeline before pushing.

## Licence
MIT (see `LICENSE`). It covers the scripts and the hand-made maps and data in this repository; Hearts of Iron IV, its
name and its game files belong to Paradox Interactive, and the lore texts to their BMC wiki authors.

## Credits
Lore from the BMC wiki (bmcwikia.fandom.com, CC BY-SA). Orbis map data and the world by the BMC server community.
