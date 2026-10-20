#!/bin/sh
# Rebuild the whole Orbis map + game files and validate.  Usage: sh tools/run_all.sh
set -e
cd "$(dirname "$0")/.."
MOD="/f/Claude/Orbis/mod/Orbis_mod"
GAME="/f/SteamLibrary/steamapps/common/Hearts of Iron IV"
python -I tools/build_base.py .
python -I tools/build_provinces.py .
python -I tools/write_provinces.py . "$MOD"
python -I tools/write_terrain.py . "$MOD" "$GAME"
python -I tools/write_rivers.py . "$MOD" "$GAME"
python -I tools/build_states.py .
python -I tools/write_game_files.py . "$MOD" "$GAME"
python -I tools/write_colormaps.py . "$MOD"
python -I tools/write_loadingscreens.py . "$MOD"
python -I tools/render_overview.py .
python -I tools/validate_mod.py "$MOD"
