# Orbis – HOI4 Total Conversion

The user speaks German but wants answers in English (said so 2026-10-18). Mod for Hearts of Iron IV 1.19 based on the
BMC Minecraft server "Orbis" (Season 1, 4000×4000 blocks, seed -109313319010159853).

## Paths
- Tools/pipeline (this folder): `F:\Claude\Orbis` – `tools/`, `data/`, `source/`, `build/` (generated), `out/` (previews)
- Mod: `F:\Claude\Orbis\mod\Orbis_mod` (moved to F on 2026-10-18, the user wants files on F). The launcher still needs
  `C:\Users\<you>\Documents\Paradox Interactive\Hearts of Iron IV\mod\Orbis_mod.mod`, whose `path=` points to F.
- Scratch work: `F:\Claude\Orbis\.work\` (helper scripts, pylib with pefile/capstone/minidump, ProcDump);
  full crash dumps go to `build/crashdump/`.
  The game reads `replace_path` from `mod/Orbis_mod.mod`, not from `descriptor.mod`: keep both in sync, and the repo's
  launcher file `F:\Claude\Orbis\mod\Orbis_mod.mod` (relative path="mod/Orbis_mod", for friends) as well.
  Vanilla `common/on_actions` and `events` are replaced: their on_startup scripts hard-code vanilla states/tags and crashed game start.
  `common/national_focus` is replaced too, so the mod must ship a focus tree with `default = yes`
  (copy of vanilla `generic.txt`); without one the game crashes while loading.
- Game: `F:\SteamLibrary\steamapps\common\Hearts of Iron IV` (1.19.3)
- Logs: `...\Documents\Paradox Interactive\Hearts of Iron IV\logs\error.log`, crashes in `...\crashes\`
- Lore summary: `docs/lore.md` (from bmcwikia.fandom.com/wiki/Orbis; WebFetch is blocked with 402, read the wiki in a browser via `fetch('/wiki/<Page>?action=raw')`)

## Decisions
- Scenario: start 1850 BCS, two years before Orbis WW2 (23 Dec 1848 – 1846 BCS). BCS counts down:
  engine year = 4000 − BCS year (start 2150.1.1.12, second bookmark 2152.12.23 = outbreak of WW2;
  a second bookmark is needed so the bookmark screen shows). Settings live in `tools/scenario.py`.
  The old WW1 scenario (1941 BCS) is kept in `data/scenario_1941/`.
- WW2 sides (wiki): Anthro-Columbian Alliance = Anthropia + Columbia, with the Pindorama Pact (Caelum, Beringia,
  Ultimaborea; Ultimaborea stays neutral). Against them Occidentalis, later SCP Foundation and Paloga ("Paloscka").
- Playable: Occidentalis (OCD), Columbia (CLB), SCP Foundation (SCP), Diveterra (tag RHO, successor of Rhodoa, SCP puppet),
  Anthropia (ATH). Names/leaders follow the wiki for 1850 BCS (table in `docs/lore.md`); faction at start:
  Pindorama Pact (ATH + CAE + BRG), set in `tools/scenario.py`. Others are minor war parties. Almaren (ALM) is not playable. English localisation only.
- Capitals are where the national emblems sit on `source/political_map_1850.png` (user).
- Unclaimed land = owner 0 / WLD in `data/countries.json`, meant to be passable and annexable by focuses/war.
  `write_game_files.split_wilds` hands it to WILD_TAGS wild tags (now 1 = W01, user wish 2026-10-20; it was 8 after a crash
  that was blamed on one big tag, but the real cause was the building-position bug, so watch the first test).
  The wild states are one state per continent ("Daukian Wilds" etc., painted as one colour each in states.png).
  The wild countries have empty names in the localisation (user: unclaimed land has no name).
- Every impassable (wall) state needs `force_link_ownership_to` a passable neighbour state (as all vanilla
  impassable states have); `write_game_files.link_impassable` sets it and gives the wall state that owner.
- Localisation goes in `localisation/english/replace/`: STATE_<id> and VICTORY_POINTS_<id> keys also exist in
  vanilla, and outside replace/ the vanilla text wins (state names showed as "Bihar" etc.).
- Do not use vanilla tags (OCC/RHD/KHA/CHA/ROM/PAL/BER... collide). Tags: see `data/countries.json`.
- World edge = impassable mountain wall (no sea margin). Map 3072×3072, world 3008 px centred, 32 px wall.

## Git
The repository is public: https://github.com/MixedMA/Orbis (branch main). The user wants every change committed and pushed,
always: after each finished piece of work `git add -A && git commit && git push` (identity MixedMA, repo-local config).
build/, out/ and most of .work/ are ignored. The built mod incl. the generated map is committed (user, 2026-10-21, so
friends can play from the repo); unchanged vanilla map copies are left out, generic.txt and zz_dynamic_countries.txt are in.
No personal data in the repo.

## Pipeline (run `sh tools/run_all.sh`, ~6 min, ends with validate_mod.py)
build_base → build_provinces → write_provinces → write_terrain → write_rivers → build_states →
write_game_files → render_overview → validate_mod.
- Geography comes from `source/worldmap_screenshot.png` (terrain), `source/political_map.png` (1941 map; its claim
  mask `out/land_claims_1941.npy` decides land) and `data/edits/{sea,land,river}.png` (user hand edits, made by
  `tools/make_edit_masks.py` from marked screenshots in `source/edits/` + `data/edit_alignment.json`).
- **Painted maps (user's sources since 2026-10-18), all 3072×3072, never overwrite them** (each export tool
  refuses without --force). Order of use: coastlines → owners → provinces.
  - `source/land.png` (tools/export_land.py): land/sea/lake colours in build_base (LAND_RGB etc.); replaces the
    land/sea from screenshots + edits. New land gets the nearest terrain class; a city keeps a 6 px island.
  - `source/provinces.png` (tools/export_provinces.py, + `provinces_kinds.json` = colour → kind at export, and
    `provinces_guide.png` with country borders/coasts for reference): build_provinces uses it instead of generating.
    Each (colour, region) piece becomes a province (split at coasts and country borders), fix_fragments /
    corner-contact fixes still run (painted minimum sizes are smaller, MIN_PX_PAINTED). Pieces whose colour was
    exported as another kind (sea colour on newly painted land, or reverse) are re-divided by the generator.
  - `source/states.png` (tools/export_states.py <orbis> <mod>, needs a full build; + `state_names.json` colour → name,
    `states_guide.png` with names): build_states gives each land province the colour covering most of it; one colour =
    one state per owner, also where land separates the parts (the user paints them that way on purpose; the crash once blamed on
    this was the building-position bug); painted states are never
    merged as "tiny". A colour spilling 1-2 provinces over a country border: an enclave piece (no own-country
    neighbour) joins the colour's main state and owner, other remnants merge into their own country's neighbour. Land under no-state colours
    (sea/lake/wall colours) gets automatic states; the world-border wall is always automatic. Names from
    state_names.json override the automatic names (write_game_files). The export colours each state as a shade of
    its country's map_color (unclaimed: beige), neighbours within a country in different shades (user's wish).
- **Ownership source since 2026-10-18: `source/owners.png`, painted by the user** (3072×3072, one flat colour per
  country, legend `source/owners_legend.png`; unclaimed (190,180,150), sea/wall colours are ignored, so land and sea
  don't change through it). build_base reads it last (`painted_owner`, off-palette anti-aliasing pixels take the
  nearest clean pixel) and sets each city's country from it. Never overwrite it: `tools/export_owners.py` made the first
  version from the political map + screenshot edits and refuses to overwrite without --force.
  The political map and the screenshot edits below are only the history of how it was made.
  Its sea/lake pixels were synced to land.png on 2026-10-18 (user could not see the new coastline; backup
  build/_before_coastsync_*); after a coastline change, sync owners.png and states.png again the same way.
- Border edits (user, 2026-10-18): red lines drawn on in-game screenshots `source/edits/owners_*.png`.
  `python -I tools/make_owner_edits.py .` (not part of run_all; needs build/ from a previous run) turns them into
  `data/edits/owner.png`, which build_base applies over the political map. Config `data/owner_edits.json`:
  per-screenshot transform (in-game political view ≈ top-down: map = 1.72 × screen + offset, found from small
  countries such as Tuareg and Madagascar), seals for open line ends, and regions (seed + which owners may be taken;
  a region only spreads through land it may take, so other countries act as walls).
  In-game colours mislead: the vanilla colormap is blended in, so countries look bigger than they are.
- Ownership comes from the scenario map: `python -I tools/extract_countries.py source/political_map_1850.png out`
  (not part of run_all) writes `out/countries_political.npy` from the `map_color` of each country in `data/countries.json`.
  Both political maps share the same frame (`out/alignment.json`).
- Cities: `data/cities_1850.json` (positions as fractions fx/fy; `country` null = unclaimed). Since 2026-10-18 the
  list is the user's city map `source/cities_map.png` (118 cities, names read by eye, user renames applied; see
  `.work/cities_from_map.py`) plus the old capitals of countries without one on that map. Capitals: countries.json
  `capital_city` (OCD Kuhland, UBR Samartha, BRG Beringskygrad, KHZ Tamatarkha per user).
- Columbia's states follow the Warzone bonuses of `source/edits/columbia_states_warzone.png`
  (`.work/columbia_from_warzone.py`, edits owners.png/states.png; backups build/_before_warzone_*).
  Victory points (user): country capital 10, other named city 5, town 1.
- Towns: the small white dots on the political map, found by `python -I tools/extract_towns.py .` (not part of
  run_all) into `data/towns_1850.json`. Names start empty and the user may fill them in; a re-run keeps them.
  White tower icons on the map are not used yet. Towns only add victory points; they don't change borders or states.
- In owners.png, sea/lake/wall colours carry no owner: land under them (lake channels that become rivers) takes the
  nearest painted owner; in states.png lake colour takes the nearest state colour (sea colour = no state).
- Strait crossings (user): red lines on in-game screenshots `source/edits/crossings_*.png` ->
  `python -I tools/make_crossings.py . <shot> x0 y0 x1 y1` (rough map area; fits the screenshot by coastline,
  check the printed match >= 0.75; naval/sea-zone screenshots match badly: pass a window of the map and, if that fails,
  `k ox oy` by hand as extra arguments, then check the fit by eye; env KMIN/KMAX limit the zoom search) -> `data/crossings.json` -> write_game_files writes them to adjacencies.csv.
- Loading screens / menu backgrounds (user): put pictures (png/jpg) in `source/loadingscreens/`;
  tools/write_loadingscreens.py (in run_all) makes 1920x1440 DXT1 dds files, the background list and
  replace_path "common/frontend/backgrounds" in both descriptors. Not tested in game yet.
- Anthropian states with a city are named after it (Riverwood, Gvareny; user, 2026-10-19); states without a city keep their names.
- Flags: `source/flags/<TAG>.png` (user, 2026-10-20: OCD ATH BRG CLB VBN RME SCP RHO PLG CAE; the 5th picture was for "Alanta" and went to
  Valanta VBN), write_game_files fits them to 82x52/41x26/10x7 tga; other countries keep the placeholder.
- Start economy per country in `scenario.ECONOMY` (civil + military factories, population), spread over the states by write_game_files.
- Rivers must stay single-strand (no loops/branches).
- No pixel-corner contacts between provinces (`build_provinces.fix_corner_contacts`, checked by validate_mod):
  HOI4 counts them as adjacency. Province 291 touching the sea only at a corner was the game-start crash
  (coastal in game, but no port sea zone -> null province in CControllerArea).
  The vanilla tutorial (`tutorial/tutorial.txt`, state 550 etc.) crashed the game ~4 s after start
  (CInGameIdler::InitData). An empty tutorial file also crashes (the idler reads the last screen of an empty list,
  rva 0xdd82d2), so the mod ships only vanilla screen 1, which references no states or provinces.
  Game-start crash in CControllerArea (rva 0xcf2c40, null at +0xc8), seen again 2026-10-18: the area is a
  country's list of coastal provinces, each with its port sea province (CProvince+0xb8). A province gets no port
  when its naval_base_spawn position in buildings.txt sits on a border pixel: the game assigns the position to
  the neighbouring province. write_game_files now places every building >= 3 px inside its province
  (`inside()`), and validate_mod rejects positions on a border or outside their state.
  Debug recipe for engine crashes: ProcDump full dump (`.work/procdump_dl/procdump64.exe -ma -e hoi4.exe <dir>`),
  then read registers/objects with `.work/dump.py` and `.work/full.py`.

## Status (2026-10-21)
The game starts and runs (user confirmed after the building-position fix, the single Wilds nation, flags and the
start economy). Own colormap textures: tools/write_colormaps.py (in run_all;
land colour per vanilla terrain index measured from vanilla, water by distance to land, city lights, fog of war).
Not done yet: armies (history/units),
focus trees, events, more factions, real flags (placeholders now), BCS date display in the top bar.
