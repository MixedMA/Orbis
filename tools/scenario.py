"""The active start scenario: which political map, city list and dates the pipeline uses.

Geography (land, coasts, rivers) always comes from source/political_map.png (the 1941 BCS map)
and the user's hand edits; a scenario only changes who owns what.
Old scenarios are kept in data/scenario_<year>/.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

YEAR_BCS = 1850
POLITICAL_MAP = "source/political_map_1850.png"   # 4000x4000, same frame as political_map.png
CITIES = "data/cities_1850.json"
TOWNS = "data/towns_1850.json"          # small white dots on the political map (tools/extract_towns.py)
COUNTRIES_FILE = "data/countries.json"

# BCS counts down: engine year = 4000 - BCS year.
START = "2150.1.1.12"            # 1 January 1850 BCS
SECOND = "2152.12.23.12"         # 23 December 1848 BCS: Anthropia declares war on Occidentalis
END = "2350.1.1.1"
MAJORS = {"OCD", "CLB", "SCP", "ATH", "RHO"}
DEFAULT_COUNTRY = "OCD"
# (localisation key, name, leader tag, member tags) of factions at game start
FACTIONS = [
    ("PINDORAMA_PACT", "Pindorama Pact", "ATH", ["CAE", "BRG"]),   # wiki: 1854 – 1847 BCS
]

BOOKMARKS = [
    ("ORBIS_1850", START, "GFX_select_date_1936",
     "The Gathering Storm",
     "1850 BCS. The Dauk Wars have broken Rhodoa, now a puppet of the SCP Foundation. Anthropia and "
     "Columbia stand together against Occidentalis, the Pindorama Pact of Caelum, Ultimaborea and Beringia "
     "arms in the north, and the Cold War is about to turn hot."),
    ("ORBIS_1848", SECOND, "GFX_select_date_1939",
     "The Second World War",
     "23 December 1848 BCS. Occidentalis refuses Anthropia's ultimatum over Kanata. The Anthro-Columbian "
     "Alliance marches west, and the largest war in the history of Orbis begins."),
]


def countries():
    return json.load(open(ROOT / COUNTRIES_FILE, encoding="utf-8"))


def cities_path():
    return ROOT / CITIES

# Economy at the start (user, 2026-10-20): tag -> (civilian factories, military factories, population).
# write_game_files spreads these over the country's states (capital and cities weigh more);
# every other country keeps the generic per-state rules.
ECONOMY = {
    "SCP": (60, 50, 8_000_000),
    "ATH": (30, 20, 25_000_000),
    "CLB": (80, 30, 35_000_000),
    "OCD": (90, 25, 40_000_000),
    "CAE": (40, 10, 3_000_000),
}
