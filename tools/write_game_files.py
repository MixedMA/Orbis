"""Write everything besides the map rasters that HOI4 needs to start Orbis.

Usage: python write_game_files.py <orbis_dir> <mod_dir> <hoi4_dir>

Inputs: build/{provinces,states,regions}.json, build/province_ids.npy,
data/countries.json + the scenario's city list (tools/scenario.py), mod map/heightmap.bmp.
"""
import heapq
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from scenario import BOOKMARKS, DEFAULT_COUNTRY, ECONOMY, END, FACTIONS, MAJORS, START, TOWNS  # noqa: E402

LAST_ELECTION = f"{int(START.split('.')[0]) - 4}.1.1"
WILD_TAGS = 1        # all unclaimed land is one wild country (user, 2026-10-20)
WILD_BASE = 100      # owner index of the first wild country
LEADER_TYPE = {"neutrality": "despotism", "democratic": "conservatism",
               "communism": "marxism", "fascism": "nazism"}
CULTURE_GFX = {
    "western_european": ("western_european_gfx", "western_european_2d"),
    "eastern_european": ("eastern_european_gfx", "eastern_european_2d"),
    "middle_eastern": ("middle_eastern_gfx", "middle_eastern_2d"),
    "asian": ("asian_gfx", "asian_2d"),
    "south_american": ("southamerican_gfx", "southamerican_2d"),
}
WEATHER_TEMPLATES = {        # vanilla strategic region to borrow weather from
    "temperate": "1-Southern England.txt", "cold": "147-Siberian Region.txt",
    "desert": "127-Saharadesert.txt", "tropical": "163-Amazonian Brazil.txt",
    "sea": "16-North Sea.txt", "sea_cold": "46-Barents Sea.txt",
    "sea_warm": "29-Central Mediterranean Sea.txt",
}
UNIT_TYPES_LAND = list(range(39))
UNIT_TYPES_SEA = list(range(11)) + list(range(21, 29)) + [38]


def w(path, text, bom=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8-sig" if bom else "utf-8")


def compass(dx, dy):
    if abs(dx) < 0.25 * abs(dy) + 1e-9:
        return "Northern" if dy < 0 else "Southern"
    if abs(dy) < 0.25 * abs(dx):
        return "Western" if dx < 0 else "Eastern"
    return ("North" if dy < 0 else "South") + ("west" if dx < 0 else "east") + "ern"


def split_wilds(countries, states, W, H):
    """Hand the unclaimed land (owner 0) to WILD_TAGS regional wild countries.

    As one tag it formed a single area of ~1900 provinces that rings the whole world
    (with the border wall), and the game crashed in CControllerArea when a game started.
    Passable wild states are clustered by position; wall states join the nearest cluster.
    Every wild country is a copy of the WLD entry with "wild": true.
    """
    wld = next(c for c in countries if c["tag"] == "WLD")
    wild = [s for s in states if s["owner"] == 0 and not s.get("impassable")]
    wall = [s for s in states if s["owner"] == 0 and s.get("impassable")]
    pts = np.array([(s["x"], s["y"]) for s in wild])
    wts = np.array([len(s["provinces"]) for s in wild], float)
    k = min(WILD_TAGS, len(wild))
    cen = [pts[int(np.argmax(wts))]]                       # farthest-point start, then weighted k-means
    while len(cen) < k:
        d = np.min([((pts - c) ** 2).sum(1) for c in cen], axis=0)
        cen.append(pts[int(np.argmax(d))])
    cen = np.array(cen, float)
    for _ in range(30):
        lab = np.argmin(((pts[:, None] - cen[None]) ** 2).sum(2), 1)
        cen = np.array([np.average(pts[lab == j], axis=0, weights=wts[lab == j]) if (lab == j).any() else cen[j]
                        for j in range(k)])
    out = [c for c in countries if c["tag"] != "WLD"]
    names = [f"{compass(cx - W / 2, cy - H / 2)} Wilds" for cx, cy in cen]
    for j in range(k):
        name = names[j] if names.count(names[j]) == 1 else f"{names[j]} {'I' * (names[:j + 1].count(names[j]))}"
        out.append(dict(wld, index=WILD_BASE + j, tag=f"W{j + 1:02d}", name=name,
                        long=f"Unclaimed Lands ({name})", wild=True))
    for s, j in zip(wild, lab):
        s["owner"] = WILD_BASE + int(j)
    for s in wall:
        s["owner"] = WILD_BASE + int(np.argmin(((cen - (s["x"], s["y"])) ** 2).sum(1)))
    return out


def link_impassable(states, meta):
    """Tie every impassable (wall) state to its passable neighbour with the longest shared border,
    via force_link_ownership_to, and give it that neighbour's owner. All vanilla impassable states
    have such a link; without it the game crashed at game start (CControllerArea, null province)."""
    state_of = {p: s for s in states for p in s["provinces"]}
    for s in states:
        if not s.get("impassable"):
            continue
        shared = {}
        for p in s["provinces"]:
            for r in meta[p]["neighbors"]:
                t = state_of.get(r)
                if t is not None and not t.get("impassable"):
                    shared[t["id"]] = shared.get(t["id"], 0) + 1
        if shared:
            link = next(t for t in states if t["id"] == max(shared, key=shared.get))
        else:   # the wall piece only touches water (painted coastline): link to the nearest passable state
            link = min((t for t in states if not t.get("impassable")),
                       key=lambda t: (t["x"] - s["x"]) ** 2 + (t["y"] - s["y"]) ** 2)
        s["link"], s["owner"] = link["id"], link["owner"]


def main():
    root, mod, game = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    meta = {p["id"]: p for p in json.load(open(root / "build/provinces.json"))}
    states = json.load(open(root / "build/states.json", encoding="utf-8"))
    regions = json.load(open(root / "build/regions.json"))
    countries = json.load(open(root / "data/countries.json", encoding="utf-8"))
    prov = np.load(root / "build/province_ids.npy")
    height = np.array(Image.open(mod / "map/heightmap.bmp")).astype(np.float32)
    H, W = prov.shape
    countries = split_wilds(countries, states, W, H)
    link_impassable(states, meta)
    by_index = {c["index"]: c for c in countries}
    by_tag = {c["tag"]: c for c in countries}
    loc = []   # (key, text)

    # ---------- per-province interior points ----------
    edge = np.zeros(prov.shape, bool)
    edge[:, 1:] |= prov[:, 1:] != prov[:, :-1]
    edge[:, :-1] |= prov[:, 1:] != prov[:, :-1]
    edge[1:, :] |= prov[1:, :] != prov[:-1, :]
    edge[:-1, :] |= prov[1:, :] != prov[:-1, :]
    dist = ndimage.distance_transform_edt(~edge)
    ids = list(range(1, len(meta) + 1))
    inner = ndimage.maximum_position(dist, prov, ids)
    centre = {i: (int(p[1]), int(p[0])) for i, p in zip(ids, inner)}   # (x, y)

    # ---------- towns: small dots on the political map, victory point 1 ----------
    # Each goes to the passable land province under it (or the nearest one), at most one victory point per province.
    state_of = {p: s for s in states for p in s["provinces"]}
    ok = np.isin(prov, [p for p, s in state_of.items() if meta[p]["type"] == "land" and not s.get("impassable")])
    _, (ny, nx) = ndimage.distance_transform_edt(~ok, return_indices=True)
    taken = {ct["province"] for s in states for ct in s["cities"]}
    for s in states:
        s["towns"] = []
    margin = (W - 3008) // 2     # build_base: 3008 px world centred in the map
    for t in json.load(open(root / TOWNS, encoding="utf-8")) if (root / TOWNS).exists() else []:
        x, y = int(round(t["fx"] * 3008 + margin)), int(round(t["fy"] * 3008 + margin))
        p = int(prov[ny[y, x], nx[y, x]])
        if p not in taken:
            taken.add(p)
            state_of[p]["towns"].append({"province": p, "name": t["name"]})

    def xyz(x, y):
        x, y = min(max(x, 0.5), W - 1.5), min(max(y, 0.5), H - 1.5)
        return f"{x:.2f};{height[int(y), int(x)] / 10:.2f};{H - y:.2f}"

    # ---------- countries: owner tags ----------
    tag_of = {c["index"]: c["tag"] for c in countries}
    state_tag = {s["id"]: tag_of[s["owner"]] for s in states}
    capital_state = {}
    for c in countries:
        own = [s for s in states if s["owner"] == c["index"] and not s.get("impassable")]
        cap = None
        if c["capital_city"]:
            cap = next((s for s in own if any(ct["name"] == c["capital_city"] for ct in s["cities"])), None)
        if cap is None:
            cap = max(own, key=lambda s: len(s["provinces"]))
        capital_state[c["tag"]] = cap["id"]

    # ---------- state names ----------
    used = {}
    for s in states:
        c = by_index[s["owner"]]
        if s.get("impassable"):
            name = f"{compass(s['x'] - W / 2, s['y'] - H / 2)} World Border"
        elif s.get("name"):           # the user's name from source/state_names.json
            name = s["name"]
        elif s["cities"]:
            name = next((ct["name"] for ct in s["cities"] if ct["capital"]), s["cities"][0]["name"])
        else:
            own = [t for t in states if t["owner"] == s["owner"]]
            cx = np.mean([t["x"] for t in own]); cy = np.mean([t["y"] for t in own])
            base = c["name"] if not c.get("wild") else {1: "Wozgorian", 2: "Daukian", 3: "Afontovan", 4: "Eosan"}[s["continent"]]
            if c.get("wild"):
                name = f"{compass(s['x'] - W / 2, s['y'] - H / 2)} {base} Wilds"
            elif len(own) == 1:
                name = c["name"]
            else:
                name = f"{compass(s['x'] - cx, s['y'] - cy)} {base}"
        used[name] = used.get(name, 0) + 1
        s["name"] = name if used[name] == 1 else f"{name} {['', '', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X'][min(used[name], 10)] if used[name] <= 10 else used[name]}"

    # ---------- history/states ----------
    sdir = mod / "history/states"
    if sdir.exists():
        shutil.rmtree(sdir)
    # Spread each country's start economy (scenario.ECONOMY) over its states by largest remainder.
    def allocate(total, weight):
        tot = sum(weight.values())
        raw = {k: total * w / tot for k, w in weight.items()}
        out = {k: int(v) for k, v in raw.items()}
        for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[:total - sum(out.values())]:
            out[k] += 1
        return out
    econ = {}
    for tag, (n_civ, n_mil, pop) in ECONOMY.items():
        mine = [s for s in states if state_tag[s["id"]] == tag and not s.get("impassable")]
        if not mine:
            continue
        cap = capital_state[tag]
        wf = {s["id"]: (3.0 if s["id"] == cap else 1.0) * (2.0 if s["cities"] else 0.4) * (1 + 0.1 * len(s["provinces"]))
              for s in mine}
        wp = {s["id"]: len(s["provinces"]) * (1 + 1.5 * len(s["cities"])) * (3 if s["id"] == cap else 1) for s in mine}
        civ_, mil_, pop_ = allocate(n_civ, wf), allocate(n_mil, wf), allocate(pop // 1000, wp)
        for s in mine:
            econ[s["id"]] = (civ_[s["id"]], mil_[s["id"]], pop_[s["id"]] * 1000)
        print(f"{tag}: {n_civ} civilian + {n_mil} military factories, population {pop:,} over {len(mine)} states")
    SLOTS = [(4, "town"), (5, "large_town"), (6, "city"), (8, "large_city"), (10, "metropolis"), (12, "megalopolis")]
    for s in states:
        tag = state_tag[s["id"]]
        major, wild = tag in MAJORS, by_tag[tag].get("wild", False)
        is_cap = capital_state[tag] == s["id"]
        n = len(s["provinces"])
        manpower = n * (2500 if wild else 45000 if major else 20000) * (3 if is_cap and not wild else 1)
        if s.get("impassable"):
            manpower, cat, infra, civ, mil = 1, "wasteland", 0, 0, 0
        elif wild:
            cat, infra, civ, mil = "pastoral", 1, 0, 0
        elif is_cap:
            cat, infra, civ, mil = ("large_city", 4, 6, 4) if major else ("city", 3, 3, 1)
        else:
            cat = "town" if s["cities"] else "rural"
            infra, civ, mil = (3, 2 if s["cities"] else 1, 1) if major else (2, 1 if s["cities"] else 0, 0)
        if s["id"] in econ:
            civ, mil, manpower = econ[s["id"]]
            need = civ + mil
            if need > {"rural": 2, "town": 4, "large_town": 5, "city": 6, "large_city": 8, "metropolis": 10, "megalopolis": 12}.get(cat, 2):
                cat = next((c for sl, c in SLOTS if sl >= need), "megalopolis")
        tm = {} if s.get("impassable") else s["terrain_mix"]
        res = {}
        if tm.get("mountain", 0) + tm.get("hills", 0) >= 3:
            res.update(steel=4 + tm.get("mountain", 0) * 2, tungsten=2, chromium=2)
        if tm.get("desert", 0) >= 3:
            res["oil"] = 3 + tm["desert"]
        if tm.get("jungle", 0) >= 3:
            res["rubber"] = 2 + tm["jungle"]
        if tm.get("forest", 0) >= 5:
            res["aluminium"] = 2
        vps = []
        for ct in s["cities"]:
            vps.append((ct["province"], 10 if ct["capital"] and is_cap else 5))
        vps += [(t["province"], 1) for t in s["towns"]]
        prov_buildings = ""
        naval = next((ct["province"] for ct in s["cities"] if meta[ct["province"]]["coastal"]), None)
        if naval and not wild:
            prov_buildings = f"\t\t\t{naval} = {{ naval_base = {3 if major else 1} }}\n"
        cores = f"\t\tadd_core_of = {tag}\n"
        lines = [
            "state={", f"\tid={s['id']}", f"\tname=\"STATE_{s['id']}\"",
            f"\tmanpower = {manpower}", f"\tstate_category = {cat}",
        ]
        if s.get("impassable"):
            lines += ["\timpassable = yes", f"\tforce_link_ownership_to = {s['link']}"]
        if res:
            lines.append("\tresources={")
            lines += [f"\t\t{k}={v:.3f}" for k, v in res.items()]
            lines.append("\t}")
        lines += ["", "\thistory={", f"\t\towner = {tag}"]
        for p, v in vps:
            lines.append(f"\t\tvictory_points = {{ {p} {v} }}")
        lines += ["\t\tbuildings = {", f"\t\t\tinfrastructure = {infra}"]
        if civ:
            lines.append(f"\t\t\tindustrial_complex = {civ}")
        if mil:
            lines.append(f"\t\t\tarms_factory = {mil}")
        if is_cap and not wild:
            lines.append(f"\t\t\tair_base = {3 if major else 1}")
        if prov_buildings:
            lines.append(prov_buildings.rstrip("\n"))
        lines += ["\t\t}", cores.rstrip("\n"), "\t}", "",
                  "\tprovinces={", "\t\t" + " ".join(map(str, s["provinces"])) + " ", "\t}",
                  "\tlocal_supplies=0.0", "}", ""]
        w(sdir / f"{s['id']}-{s['name'].replace(chr(39), '')}.txt", "\n".join(lines))
        loc.append((f"STATE_{s['id']}", s["name"]))
        for ct in s["cities"] + s["towns"]:
            loc.append((f"VICTORY_POINTS_{ct['province']}", ct["name"]))

    # ---------- strategic regions ----------
    rdir = mod / "map/strategicregions"
    if rdir.exists():
        shutil.rmtree(rdir)
    templates = {}
    for k, f in WEATHER_TEMPLATES.items():
        txt = (game / "map/strategicregions" / f).read_text(encoding="utf-8-sig")
        templates[k] = txt[txt.index("weather={"):txt.rindex("}")].rstrip()
    sea_count = land_count = 0
    for r in regions:
        north = r["y"] < H * 0.33
        tm = r["terrain_mix"]
        if r["kind"] == "sea":
            kind = "sea_cold" if r["y"] < H * 0.25 else "sea_warm" if r["y"] > H * 0.7 else "sea"
            sea_count += 1
            name = f"Sea Zone {sea_count}"
        else:
            total = sum(tm.values()) or 1
            if tm.get("desert", 0) / total > 0.3:
                kind = "desert"
            elif tm.get("jungle", 0) / total > 0.3:
                kind = "tropical"
            elif north and r["x"] > W * 0.45:
                kind = "cold"
            else:
                kind = "temperate"
            land_count += 1
            cont = max(set(meta[p].get("continent", 1) for p in r["provinces"] if meta[p]["type"] == "land"),
                       key=lambda c: sum(meta[p].get("continent") == c for p in r["provinces"]))
            cname = {1: "Wozgoria", 2: "Daukia", 3: "Afontova", 4: "Eos"}[cont]
            name = f"{compass(r['x'] - W / 2, r['y'] - H / 2)} {cname} ({r['id']})"
        r["name"] = name
        body = (f"strategic_region={{\n\tid={r['id']}\n\tname=\"STRATEGICREGION_{r['id']}\"\n"
                f"\tprovinces={{\n\t\t{' '.join(map(str, r['provinces']))} \n\t}}\n\t{templates[kind]}\n}}\n")
        w(rdir / f"{r['id']}-{name.replace(' ', '_').replace('(', '').replace(')', '')}.txt", body)
        loc.append((f"STRATEGICREGION_{r['id']}", name))

    # ---------- countries ----------
    tag_lines = []
    colors = ["#Orbis country colours\n"]
    for c in countries:
        fname = c["name"].replace("'", "")
        tag_lines.append(f"{c['tag']}\t= \"countries/{fname}.txt\"")
        gfx, gfx2d = CULTURE_GFX[c["culture"]]
        r, g, b = c["color"]
        w(mod / f"common/countries/{fname}.txt",
          f"graphical_culture = {gfx}\ngraphical_culture_2d = {gfx2d}\n\ncolor = {{ {r} {g} {b} }}\n")
        ui = [min(255, int(v * 1.25) + 20) for v in c["color"]]
        colors.append(f"{c['tag']} = {{\n\tcolor = rgb {{ {r} {g} {b} }}\n\tcolor_ui = rgb {{ {ui[0]} {ui[1]} {ui[2]} }}\n}}\n")
    w(mod / "common/country_tags/00_countries.txt", "\n".join(tag_lines) + "\n")
    shutil.copy(game / "common/country_tags/zz_dynamic_countries.txt", mod / "common/country_tags/zz_dynamic_countries.txt")
    # Vanilla's generic focus tree (the mod replaces common/national_focus and needs one default tree); copied from
    # the player's own game so the repository holds no Paradox files.
    (mod / "common/national_focus").mkdir(parents=True, exist_ok=True)
    shutil.copy(game / "common/national_focus/generic.txt", mod / "common/national_focus/generic.txt")
    # keep vanilla colours for dynamic tags so civil wars still get a colour
    van_colors = (game / "common/countries/colors.txt").read_text(encoding="utf-8-sig")
    dyn = ""
    if "D01 = {" in van_colors and "D50 = {" in van_colors:
        a = van_colors.index("D01 = {")
        z = van_colors.index("}", van_colors.index("color_ui", van_colors.index("D50 = {")))
        z = van_colors.index("}", z + 1) + 1
        dyn = van_colors[a:z] + "\n"
    w(mod / "common/countries/colors.txt", "".join(colors) + "\n" + dyn)

    hdir = mod / "history/countries"
    if hdir.exists():
        shutil.rmtree(hdir)
    chars = []
    for c in countries:
        tag, ide = c["tag"], c["ideology"]
        pops = {"democratic": 10, "fascism": 5, "communism": 10, "neutrality": 75}
        if ide != "neutrality":
            pops = {k: 10 for k in pops}
            pops[ide] = 70
            pops["fascism"] = 5
            pops["neutrality"] = 100 - sum(v for k, v in pops.items() if k != "neutrality")
        tech = "" if c.get("wild") else (
            "set_technology = {\n\tinfantry_weapons = 1\n\tinfantry_weapons1 = 1\n\ttech_support = 1\n"
            "\ttech_engineers = 1\n\ttech_recon = 1\n\tgw_artillery = 1\n\tbasic_train = 1\n}\n")
        cid = f"{tag}_leader"
        hist = (f"capital = {capital_state[tag]}\n\nset_research_slots = {3 if tag in MAJORS else 2}\n"
                f"set_stability = 0.6\nset_war_support = {0.4 if tag in MAJORS else 0.2}\n\n{tech}\n"
                f"set_politics = {{\n\truling_party = {ide}\n\tlast_election = \"{LAST_ELECTION}\"\n"
                f"\telection_frequency = 48\n\telections_allowed = {'yes' if ide == 'democratic' else 'no'}\n}}\n"
                f"set_popularities = {{\n" + "".join(f"\t{k} = {v}\n" for k, v in pops.items()) + "}\n\n"
                f"recruit_character = {cid}\n")
        if c.get("overlord"):
            hist += ""  # subject relation is set from the overlord's file
        for sub in countries:
            if sub.get("overlord") == tag:
                hist += f"\nset_autonomy = {{\n\ttarget = {sub['tag']}\n\tautonomous_state = autonomy_puppet\n}}\n"
        for key, _, leader, members in FACTIONS:
            if leader == tag:
                hist += f"\ncreate_faction = {key}\n" + "".join(f"add_to_faction = {m}\n" for m in members)
        w(hdir / f"{tag} - {c['name'].replace(chr(39), '')}.txt", hist)
        chars.append(f"\t{cid} = {{\n\t\tname = \"{c['leader']}\"\n\t\tcountry_leader = {{\n"
                     f"\t\t\tideology = {LEADER_TYPE[ide]}\n\t\t\texpire = \"2999.1.1.1\"\n\t\t}}\n\t}}\n")
        # The wild countries show no name at all (user); their internal names only name files.
        name, long, adj = ("", "", "") if c.get("wild") else (c["name"], c["long"], c["adj"])
        loc += [(tag, name), (f"{tag}_DEF", long), (f"{tag}_ADJ", adj)]
        for i in ("democratic", "fascism", "communism", "neutrality"):
            loc += [(f"{tag}_{i}", name), (f"{tag}_{i}_DEF", long), (f"{tag}_{i}_ADJ", adj)]
    w(mod / "common/characters/orbis_leaders.txt", "characters = {\n" + "".join(chars) + "}\n")
    w(mod / "history/units/.keep", "")

    # ---------- flags: the user's pictures in source/flags/<TAG>.png, a coloured placeholder otherwise ----------
    for c in countries:
        for sub, size in (("", (82, 52)), ("medium/", (41, 26)), ("small/", (10, 7))):
            own = root / f"source/flags/{c['tag']}.png"
            if own.exists():                     # scaled and centre-cropped to the flag shape
                img = ImageOps.fit(Image.open(own).convert("RGBA"), size, Image.LANCZOS)
                p = mod / f"gfx/flags/{sub}{c['tag']}.tga"
                p.parent.mkdir(parents=True, exist_ok=True)
                img.save(p)
                continue
            img = Image.new("RGBA", size, tuple(c["color"]) + (255,))
            d = ImageDraw.Draw(img)
            light = tuple(min(255, v + 70) for v in c["color"]) + (255,)
            d.rectangle([0, size[1] * 2 // 3, size[0], size[1]], fill=light)
            if size[0] >= 41:
                try:
                    f = ImageFont.truetype("segoeuib.ttf", size[1] // 2)
                    d.text((size[0] / 2, size[1] / 2.6), c["tag"], fill=(255, 255, 255, 255), font=f, anchor="mm")
                except OSError:
                    pass
            p = mod / f"gfx/flags/{sub}{c['tag']}.tga"
            p.parent.mkdir(parents=True, exist_ok=True)
            img.save(p)

    # ---------- map support files ----------
    m = mod / "map"
    shutil.copy(game / "map/default.map", m / "default.map")
    for f in ("seasons.txt", "colors.txt", "cities.txt"):
        shutil.copy(game / "map" / f, m / f)
    # Strait crossings drawn by the user (data/crossings.json, tools/make_crossings.py): along each line, the
    # water run nearest its middle; the land provinces on both sides are linked through the water province
    # that covers most of that run.
    rows = []
    cpath = root / "data/crossings.json"
    for c in json.load(open(cpath, encoding="utf-8")) if cpath.exists() else []:
        a, b = np.array(c["a"]), np.array(c["b"])
        pts = [a + (b - a) * t for t in np.linspace(-1.5, 2.5, 400)]
        seq = [int(prov[int(round(min(max(y, 0), H - 1))), int(round(min(max(x, 0), W - 1)))]) for x, y in pts]
        is_land = [meta[q]["type"] == "land" and not meta[q]["border"] for q in seq]
        runs, i = [], 0
        while i < len(seq):                           # water runs with land on both ends
            if is_land[i]:
                i += 1
                continue
            j = i
            while j < len(seq) and not is_land[j]:
                j += 1
            if i > 0 and j < len(seq):
                runs.append((abs((i + j) / 2 - 200), i, j))
            i = j
        if not runs:
            print(f"crossing {c['a']} -> {c['b']}: no water between land, skipped")
            continue
        _, i, j = min(runs)
        frm, to = seq[i - 1], seq[j]
        water = [q for q in seq[i:j] if meta[q]["type"] in ("sea", "lake")]
        if frm == to or not water:
            print(f"crossing {c['a']} -> {c['b']}: same province on both sides, skipped")
            continue
        thr = max(set(water), key=water.count)
        rows.append(f"{frm};{to};sea;{thr};-1;-1;-1;-1;;{c.get('source', '')}")
    w(m / "adjacencies.csv", "From;To;Type;Through;start_x;start_y;stop_x;stop_y;adjacency_rule_name;Comment\n"
      + "".join(r + "\n" for r in rows) + "-1;-1;;-1;-1;-1;-1;-1;-1\n")
    print(f"{len(rows)} strait crossings")
    w(m / "adjacency_rules.txt", "")
    w(m / "positions.txt", "")
    # Ambient objects: map frame and HOI4 banner sit just above/below our map;
    # vanilla's water objects and the Chinese wall are dropped.
    def amb(t, name, scale, pos):
        return (f"type={{\n\ttype=\"{t}\"\n\tuse_animation=no\n\tscale={scale:.6f}\n\talways_visible=yes\n"
                f"\tobject={{\n\t\tname=\"{name}\"\n\t\tposition={{\n\t\t\t{pos} \n\t\t}}\n"
                f"\t\trotation={{\n\t\t\t0 0 0 \n\t\t}}\n\t}}\n}}\n")
    w(m / "ambient_object.txt",
      "type={\n\ttype=\"ambient_wind_entity\"\n\tuse_animation=no\n\talways_visible=yes\n\tobject={\n"
      "\t\tname=\"ambient_wind\"\n\t\tposition={\n\t\t\t0 0 0 \n\t\t}\n\t\trotation={\n\t\t\t0 0 0 \n\t\t}\n\t}\n}\n"
      + amb("frame_border_entity", "frame_border_entity_top", 100, f"0 0 {H + 142}")
      + amb("frame_border_bottom_entity", "frame_border_bottom_entity_bottom", 100, "0 0 -140")
      + amb("frame_border_logo_entity", "frame_border_logo_entity_top", 300, f"{W // 2} -1 {H + 82}"))
    cities_bmp = Image.new("P", (W, H), 15)
    cities_bmp.putpalette(Image.open(game / "map/cities.bmp").getpalette())
    cities_bmp.save(m / "cities.bmp")

    # unitstacks
    us = []
    for i, p in meta.items():
        x, y = centre[i]
        types = UNIT_TYPES_SEA if p["type"] != "land" else UNIT_TYPES_LAND
        us += [f"{i};{t};{xyz(x, y)};0.00;0.00" for t in types]
    w(m / "unitstacks.txt", "\n".join(us) + "\n")

    # buildings: positions spread over each state's provinces
    rng = np.random.default_rng(42)
    bl = []
    slices = ndimage.find_objects(prov)

    def inside(q, tx, ty, margin=3):
        """The pixel of province q nearest to (tx, ty) that is at least `margin` px from any other province.
        The game finds a building's province from its position: a point on a border pixel went to the
        neighbour, which left a coastal province without its port and crashed the game at start
        (CControllerArea, 2026-10-18)."""
        sl = slices[q - 1]
        sub = prov[sl] == q
        ok = sub & (dist[sl] >= margin)
        if not ok.any():
            ok = sub & (dist[sl] >= dist[sl][sub].max())
        ys, xs = np.nonzero(ok)
        k = np.argmin((xs + sl[1].start - tx) ** 2 + (ys + sl[0].start - ty) ** 2)
        return xs[k] + sl[1].start + 0.5, ys[k] + sl[0].start + 0.5

    for s in states:
        provs = s["provinces"]
        def spot(pp=None):
            q = pp if pp is not None else provs[rng.integers(len(provs))]
            x, y = centre[q]
            j = min(3.0, max(0.0, dist[y, x] - 3))      # jitter that stays clear of the province border
            return x + 0.5 + rng.uniform(-j, j), y + 0.5 + rng.uniform(-j, j)
        for btype, count in (("arms_factory", 6), ("industrial_complex", 6), ("air_base", 1),
                             ("anti_air_building", 3), ("fuel_silo", 1), ("nuclear_reactor_spawn", 1),
                             ("radar_station", 1), ("rocket_site_spawn", 1), ("stronghold_network", 1),
                             ("synthetic_refinery", 1)):
            for _ in range(count):
                x, y = spot()
                bl.append(f"{s['id']};{btype};{xyz(x, y)};{rng.uniform(0, 6.28):.2f};0")
        for q in provs:
            for btype in ("bunker", "supply_node", "special_project_facility_spawn"):
                x, y = spot(q)
                bl.append(f"{s['id']};{btype};{xyz(x, y)};{rng.uniform(0, 6.28):.2f};0")
            if meta[q]["coastal"]:
                seas = [r for r in meta[q]["neighbors"] if r and meta[r]["type"] == "sea"]
                if not seas:
                    continue
                sx, sy = centre[seas[0]]
                x, y = centre[q]
                # place harbour structures on the coast-facing side of the province
                hx, hy = inside(q, x + (sx - x) * 0.35, y + (sy - y) * 0.35)
                for btype in ("coastal_bunker", "naval_base_spawn", "naval_headquarters", "naval_supply_hub", "floating_harbor"):
                    bl.append(f"{s['id']};{btype};{xyz(hx, hy)};0.00;{seas[0]}")
                if s["cities"] and q == provs[0]:
                    bl.append(f"{s['id']};dockyard;{xyz(hx, hy)};0.00;{seas[0]}")
    w(m / "buildings.txt", "\n".join(bl) + "\n")

    # weather positions
    wp = []
    for r in regions:
        for q in r["provinces"][:: max(1, len(r["provinces"]) // 3)][:3]:
            x, y = centre[q]
            wp.append(f"{r['id']};{xyz(x, y)};small")
    w(m / "weatherpositions.txt", "\n".join(wp) + "\n")

    # supply nodes on state capitals of real countries, railways between them
    land_adj = {i: [r for r in p["neighbors"] if r and meta[r]["type"] == "land"] for i, p in meta.items() if p["type"] == "land"}
    owner_tag = {p: state_tag[s["id"]] for s in states for p in s["provinces"]}
    nodes, rails = [], []
    for c in countries:
        if c.get("wild"):
            continue
        caps = [s["capital"] for s in states if state_tag[s["id"]] == c["tag"]]
        nodes += caps
        connected = {caps[0]}
        for target in caps[1:]:
            # shortest path inside own territory from the connected network
            dist_, prev = {q: 0 for q in connected}, {}
            pq = [(0, q) for q in connected]
            found = None
            while pq:
                d, q = heapq.heappop(pq)
                if q == target:
                    found = q
                    break
                for r in land_adj.get(q, []):
                    if owner_tag.get(r) != c["tag"]:
                        continue
                    if r not in dist_ or d + 1 < dist_[r]:
                        dist_[r] = d + 1
                        prev[r] = q
                        heapq.heappush(pq, (d + 1, r))
            if found is None:
                continue
            path = [found]
            while path[-1] not in connected:
                path.append(prev[path[-1]])
            connected.update(path)
            if len(path) >= 2:
                rails.append(f"1 {len(path)} " + " ".join(map(str, reversed(path))))
    w(m / "supply_nodes.txt", "".join(f"1 {n}\n" for n in nodes))
    w(m / "railways.txt", "\n".join(rails) + ("\n" if rails else ""))

    # ---------- bookmark, defines, ai areas ----------
    entries = ""
    for c in countries:
        if c["playable"]:
            entries += (f"\t\t\"{c['tag']}\" = {{\n\t\t\thistory = \"{c['tag']}_BOOKMARK_DESC\"\n"
                        f"\t\t\tideology = {c['ideology']}\n\t\t\tideas = {{ }}\n\t\t\tfocuses = {{ }}\n\t\t}}\n")
            loc.append((f"{c['tag']}_BOOKMARK_DESC", f"{c['long']}, led by {c['leader']}."))
    # Two bookmarks: with only one, HOI4 skips the bookmark selection screen.
    def bookmark(key, date, pic, default):
        return (f"\tbookmark = {{\n\t\tname = \"{key}_NAME\"\n\t\tdesc = \"{key}_DESC\"\n"
                f"\t\tdate = {date}\n\t\tpicture = \"{pic}\"\n\t\tdefault_country = \"{DEFAULT_COUNTRY}\"\n"
                + ("\t\tdefault = yes\n" if default else "")
                + f"{entries}\t\t\"---\" = {{\n\t\t\thistory = \"OTHER_GATHERING_STORM_DESC\"\n\t\t}}\n"
                f"\t\teffect = {{\n\t\t\trandomize_weather = 22345\n\t\t}}\n\t}}\n")
    bm_dir = mod / "common/bookmarks"
    if bm_dir.exists():
        shutil.rmtree(bm_dir)
    w(bm_dir / "orbis.txt", "bookmarks = {\n" + "".join(
        bookmark(key, date, pic, k == 0) for k, (key, date, pic, _, _) in enumerate(BOOKMARKS)) + "}\n")
    for key, _, _, name, desc in BOOKMARKS:
        loc += [(f"{key}_NAME", name), (f"{key}_DESC", desc)]
    loc += [(key, name) for key, name, _, _ in FACTIONS]
    # The vanilla tutorial points at vanilla states/provinces (state 550 Eritrea, ...) and crashed the
    # game a few seconds after start, when the in-game idler loads it. An empty file crashes too: the idler
    # flags the last tutorial screen without checking the list is empty. So ship only vanilla screen 1,
    # which lists objectives and references no state or province.
    w(mod / "tutorial/tutorial.txt",
      "# Orbis: only vanilla tutorial screen 1 (the others use vanilla state and province ids).\n"
      "tutorial = {\n"
      "\twindow = \"tutorial_screen_1\"\n"
      "\tuse_mil_fac = { textbox = \"obj_1\" }\n"
      "\tuse_civ_fac = { textbox = \"obj_2\" }\n"
      "\tuse_nav_dock = { textbox = \"obj_3\" }\n"
      "\thas_set_nat_focus = { textbox = \"obj_4\" }\n"
      "\tis_queue_div = { textbox = \"obj_5\" }\n"
      "\tis_researching_tech = { textbox = \"obj_6\" }\n"
      "}\n")
    w(mod / "common/defines/zz_orbis_defines.lua",
      f"NDefines.NGame.START_DATE = \"{START}\"\nNDefines.NGame.END_DATE = \"{END}\"\n"
      f"NDefines.NDiplomacy.TENSION_TIME_SCALE_START_DATE = \"{START}\"\n")
    areas = "areas = {\n"
    for cid, cname in ((1, "wozgoria"), (2, "daukia"), (3, "afontova"), (4, "eos")):
        areas += f"\t{cname} = {{\n\t\tcontinents = {{ {cname} }}\n\t}}\n"
    w(mod / "common/ai_areas/default.txt", areas + "}\n")

    # ---------- localisation ----------
    # In replace/: STATE_<id> and VICTORY_POINTS_<id> also exist in vanilla, and outside replace/ the vanilla
    # text wins (state 435 showed as "Bihar").
    text = "l_english:\n" + "".join(f" {k}:0 \"{v}\"\n" for k, v in loc)
    (mod / "localisation/english/orbis_map_l_english.yml").unlink(missing_ok=True)
    w(mod / "localisation/english/replace/orbis_map_l_english.yml", text, bom=True)
    print(f"{len(states)} states, {len(regions)} regions, {len(countries)} countries, "
          f"{len(us)} unitstacks, {len(bl)} building positions, {len(nodes)} supply nodes, {len(rails)} railways")


if __name__ == "__main__":
    main()
