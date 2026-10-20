"""Group land provinces into states and states/seas into strategic regions.

Usage: python build_states.py <orbis_dir>
Writes build/states.json and build/regions.json.

States never cross an owner border, are contiguous where possible and grow
outwards from seeds (scenario cities first, then farthest-point samples) so they
end up roughly equal in size. Strategic regions are grown the same way on the
state adjacency graph (land) and on the sea province graph (sea).
"""
import heapq
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from scenario import cities_path  # noqa: E402

TARGET_STATE = {"country": 7, "wild": 12}
# The user's hand-painted states (tools/export_states.py writes the first version): one colour per state,
# names per colour in state_names.json. Colours of water / the world border are no state.
PAINTED = "source/states.png"
NO_STATE_RGB = [(10, 20, 70), (80, 160, 230), (0, 0, 0)]
TARGET_LAND_REGION = 7      # states per land strategic region
TARGET_SEA_REGION = 18      # sea provinces per sea strategic region
CONTINENT_NAMES = {1: "Wozgoria", 2: "Daukia", 3: "Afontova", 4: "Eos"}


def components(nodes, adj):
    seen, comps = set(), []
    for n in nodes:
        if n in seen:
            continue
        comp, stack = [], [n]
        seen.add(n)
        while stack:
            q = stack.pop()
            comp.append(q)
            for r in adj[q]:
                if r in nodes and r not in seen:
                    seen.add(r)
                    stack.append(r)
        comps.append(comp)
    return comps


def grow(nodes, adj, seeds, weight):
    """Balanced multi-source growth: always extend the currently smallest group."""
    nodes = set(nodes)
    group = {s: i for i, s in enumerate(seeds)}
    size = [weight(s) for s in seeds]
    frontier = []
    for i, s in enumerate(seeds):
        for r in adj[s]:
            if r in nodes and r not in group:
                heapq.heappush(frontier, (size[i], i, r))
    while frontier:
        sz, i, r = heapq.heappop(frontier)
        if r in group:
            continue
        if sz != size[i]:                      # stale entry: re-queue with current size
            heapq.heappush(frontier, (size[i], i, r))
            continue
        group[r] = i
        size[i] += weight(r)
        for q in adj[r]:
            if q in nodes and q not in group:
                heapq.heappush(frontier, (size[i], i, q))
    out = [[] for _ in seeds]
    for n, i in group.items():
        out[i].append(n)
    return out


def farthest_seeds(comp, pos, k, fixed):
    seeds = list(fixed)
    pts = np.array([pos[c] for c in comp])
    if not seeds:
        centre = pts.mean(0)
        seeds.append(comp[int(np.argmin(((pts - centre) ** 2).sum(1)))])
    while len(seeds) < k:
        sp = np.array([pos[s] for s in seeds])
        d = ((pts[:, None, :] - sp[None, :, :]) ** 2).sum(-1).min(1)
        seeds.append(comp[int(np.argmax(d))])
    return seeds


def water_linked(provs, adj, water):
    """Pieces of provs that are connected through each other or through water provinces (sea/lake)."""
    provs = set(provs)
    seen, out = set(), []
    for p in provs:
        if p in seen:
            continue
        comp, stack, visited = [], [p], {p}
        while stack:
            q = stack.pop()
            if q in provs:
                comp.append(q)
            for r in adj[q]:
                if r not in visited and (r in provs or r in water):
                    visited.add(r)
                    stack.append(r)
        seen |= set(comp)
        out.append(comp)
    return out


def painted_states(root, prov_ids, meta, adj, by_owner):
    """States from the painted map: each land province goes to the colour covering most of it; one colour
    becomes one state per owner and connected piece. Provinces under no-state colours are left out."""
    from PIL import Image
    a = np.array(Image.open(root / PAINTED).convert("RGB")).astype(np.int64)
    key = (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]
    # Lake colour is drawn as in land.png, but narrow lake channels become land (rivers): they take the
    # nearest painted colour so they don't turn into automatic states. Sea colour still means "no state".
    lake = key == (80 << 16 | 160 << 8 | 230)
    _, (iy, ix) = ndimage.distance_transform_edt(lake, return_indices=True)
    key = key[iy, ix].ravel()
    no_state = {(r << 16) | (g << 8) | b for r, g, b in NO_STATE_RGB}
    names = {}
    if (root / "source/state_names.json").exists():
        names = {int(k, 16): v for k, v in json.load(open(root / "source/state_names.json", encoding="utf-8")).items()}
    flat = prov_ids.ravel()
    ok = ~np.isin(key, list(no_state))
    pk, cnt = np.unique(flat[ok].astype(np.int64) << 24 | key[ok], return_counts=True)
    best = {}
    for p, k, n in zip((pk >> 24).tolist(), (pk & 0xFFFFFF).tolist(), cnt.tolist()):
        if n > best.get(p, (0, 0))[1]:
            best[p] = (k, n)
    groups = {}
    for owner, provs in by_owner.items():
        for p in provs:
            if p in best:
                groups.setdefault((owner, best[p][0]), set()).add(p)
    # One colour = one state per owner, even in separate patches (HoI4 states may be non-contiguous, e.g. a coast
    # with its islands). Where a colour spills over a country border by 1-2 provinces, that remnant is not the
    # user's state: it is left unprotected so the tiny-state rule merges it into its own country's neighbour.
    total = {}
    for (owner, k), provs in groups.items():
        total[k] = total.get(k, 0) + len(provs)
    # An enclave remnant (no province of its own country next to it) goes with the colour's main state and owner:
    # it is a speck in owners.png that the user painted as part of the surrounding state.
    owner_of = {p: o for o, ps in by_owner.items() for p in ps}
    for (owner, k), provs in list(groups.items()):
        if len(provs) <= 2 and total[k] > len(provs) and \
                not any(owner_of.get(r) == owner for p in provs for r in adj[p] if r not in provs):
            main = max((key for key in groups if key[1] == k and key != (owner, k)), key=lambda key: len(groups[key]))
            groups[main] |= groups.pop((owner, k))
    # One colour = one state per owner, also where land separates the parts: the user painted them that way on
    # purpose (2026-10-18). The game-start crash blamed on this was really building positions on province
    # borders (CControllerArea), fixed in write_game_files.
    water = {i for i, p in meta.items() if p["type"] in ("sea", "lake")}
    states = []
    for (owner, k), provs in groups.items():
        remnant = len(provs) <= 2 and total[k] > len(provs)
        states.append({"owner": owner, "provinces": sorted(provs), "painted": not remnant,
                       "colour": "%06x" % k, "name": names.get(k) if not remnant else None})
    print(f"painted states: {len(groups) - sum(not s['painted'] for s in states)} states, "
          f"{sum(not s['painted'] for s in states)} border remnants to merge")
    return states


def main():
    root = Path(sys.argv[1])
    meta = {p["id"]: p for p in json.load(open(root / "build/provinces.json"))}
    cities = json.load(open(cities_path(), encoding="utf-8"))
    prov_ids = np.load(root / "build/province_ids.npy")
    adj = {i: set(p["neighbors"]) - {0} for i, p in meta.items()}
    pos = {i: (p["x"], p["y"]) for i, p in meta.items()}
    city_prov = {}
    for c in cities:
        city_prov[int(prov_ids[c["y"], c["x"]])] = c

    land = [i for i, p in meta.items() if p["type"] == "land"]
    border = {i for i in land if meta[i].get("border")}
    by_owner = {}
    for i in land:
        if i not in border:
            by_owner.setdefault(meta[i]["owner"], set()).add(i)

    states = []
    painted = set()
    if (root / PAINTED).exists():
        states = painted_states(root, prov_ids, meta, adj, by_owner)
        painted = {p for s in states for p in s["provinces"]}
        by_owner = {o: ps - painted for o, ps in by_owner.items()}   # the rest is grouped automatically
    for owner, provs in sorted(by_owner.items()):
        target = TARGET_STATE["wild" if owner == 0 else "country"]
        comps = components(provs, adj)
        comps.sort(key=len, reverse=True)
        for comp in comps:
            fixed = [c for c in comp if c in city_prov]
            k = max(len(fixed), round(len(comp) / target), 1)
            seeds = farthest_seeds(comp, pos, k, fixed)
            for grp in grow(comp, adj, seeds, lambda n: 1):
                if grp:
                    states.append({"owner": owner, "provinces": sorted(grp)})

    # The world border wall: impassable states of about five provinces each.
    for comp in components(border, adj):
        k = max(1, round(len(comp) / 5))
        for grp in grow(comp, adj, farthest_seeds(comp, pos, k, []), lambda n: 1):
            if grp:
                states.append({"owner": 0, "provinces": sorted(grp), "impassable": True})

    # Tiny states (1-2 provinces, no city) merge into a same-owner neighbour state.
    def state_of():
        m = {}
        for si, s in enumerate(states):
            for p in s["provinces"]:
                m[p] = si
        return m
    changed = True
    while changed:
        changed = False
        m = state_of()
        for si, s in enumerate(states):
            if not s["provinces"] or len(s["provinces"]) > 2 or any(p in city_prov for p in s["provinces"]) \
                    or s.get("painted"):     # the user's own small states stay
                continue
            neigh = {m[r] for p in s["provinces"] for r in adj[p] if r in m and m[r] != si
                     and states[m[r]]["owner"] == s["owner"]
                     and states[m[r]].get("impassable", False) == s.get("impassable", False)}
            if neigh:
                tgt = min(neigh, key=lambda j: len(states[j]["provinces"]))
                states[tgt]["provinces"] += s["provinces"]
                s["provinces"] = []
                changed = True
        states = [s for s in states if s["provinces"]]

    # Capitals, VPs, ids
    for sid, s in enumerate(states, 1):
        s["id"] = sid
        s["provinces"].sort()
        cs = [city_prov[p] | {"province": p} for p in s["provinces"] if p in city_prov]
        s["cities"] = cs
        pts = np.array([pos[p] for p in s["provinces"]])
        centre = pts.mean(0)
        if cs:
            s["capital"] = next((c["province"] for c in cs if c["capital"]), cs[0]["province"])
        else:
            s["capital"] = s["provinces"][int(np.argmin(((pts - centre) ** 2).sum(1)))]
        s["x"], s["y"] = float(centre[0]), float(centre[1])
        conts = [meta[p].get("continent", 1) for p in s["provinces"]]
        s["continent"] = max(set(conts), key=conts.count)
        terr = [meta[p]["terrain"] for p in s["provinces"]]
        s["terrain_mix"] = {t: terr.count(t) for t in set(terr)}
        s["coastal_provinces"] = [p for p in s["provinces"] if meta[p]["coastal"]]

    # ---- strategic regions ----
    m = {p: s["id"] for s in states for p in s["provinces"]}
    sadj = {s["id"]: set() for s in states}
    for s in states:
        for p in s["provinces"]:
            for r in adj[p]:
                if r in m and m[r] != s["id"]:
                    sadj[s["id"]].add(m[r])
    spos = {s["id"]: (s["x"], s["y"]) for s in states}
    regions = []
    for comp in components(set(sadj), sadj):
        k = max(1, round(len(comp) / TARGET_LAND_REGION))
        seeds = farthest_seeds(comp, spos, k, [])
        for grp in grow(comp, sadj, seeds, lambda n: 1):
            if grp:
                provs = [p for sid in grp for p in states[sid - 1]["provinces"]]
                regions.append({"kind": "land", "states": sorted(grp), "provinces": provs})
    # tiny land regions (single island states) join the nearest land region
    big = [r for r in regions if len(r["states"]) >= 2]
    for r in [r for r in regions if len(r["states"]) < 2]:
        c = np.mean([spos[s] for s in r["states"]], 0)
        tgt = min(big, key=lambda b: ((np.mean([spos[s] for s in b["states"]], 0) - c) ** 2).sum())
        tgt["states"] += r["states"]
        tgt["provinces"] += r["provinces"]
    regions = big

    sea = {i for i, p in meta.items() if p["type"] == "sea"}
    for comp in components(sea, adj):
        k = max(1, round(len(comp) / TARGET_SEA_REGION))
        seeds = farthest_seeds(comp, pos, k, [])
        for grp in grow(comp, adj, seeds, lambda n: 1):
            if grp:
                regions.append({"kind": "sea", "provinces": sorted(grp)})

    # lakes go to the land region they touch most
    reg_of = {p: ri for ri, r in enumerate(regions) for p in r["provinces"]}
    for i, p in meta.items():
        if p["type"] != "lake":
            continue
        votes = [reg_of[r] for r in adj[i] if r in reg_of and regions[reg_of[r]]["kind"] == "land"]
        if not votes:
            votes = [reg_of[r] for r in adj[i] if r in reg_of]
        if not votes:  # lake inside lakes only: nearest region by centre
            ri = min(range(len(regions)), key=lambda j: min(
                (pos[q][0] - pos[i][0]) ** 2 + (pos[q][1] - pos[i][1]) ** 2 for q in regions[j]["provinces"]))
        else:
            ri = max(set(votes), key=votes.count)
        regions[ri]["provinces"].append(i)
        reg_of[i] = ri

    for rid, r in enumerate(regions, 1):
        r["id"] = rid
        r["provinces"].sort()
        pts = np.array([pos[p] for p in r["provinces"]])
        r["x"], r["y"] = map(float, pts.mean(0))
        terr = [meta[p]["terrain"] for p in r["provinces"] if meta[p]["type"] == "land"]
        r["terrain_mix"] = {t: terr.count(t) for t in set(terr)}

    json.dump(states, open(root / "build/states.json", "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(regions, open(root / "build/regions.json", "w"))
    sizes = [len(s["provinces"]) for s in states]
    print(f"{len(states)} states (provinces/state min {min(sizes)} median {int(np.median(sizes))} max {max(sizes)})")
    print(f"{sum(r['kind'] == 'land' for r in regions)} land regions, {sum(r['kind'] == 'sea' for r in regions)} sea regions")
    covered = sorted(p for r in regions for p in r["provinces"])
    assert covered == sorted(meta), "every province must be in exactly one strategic region"
    in_states = sorted(p for s in states for p in s["provinces"])
    assert in_states == sorted(land), "every land province must be in exactly one state"
    print("coverage checks passed")


if __name__ == "__main__":
    main()
