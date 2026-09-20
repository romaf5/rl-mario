#!/usr/bin/env bash
# The whole game, searched: 1-1 -> 8-4 (axe) along the warp route, then replayed
# in stable-retro (every frame must match) and rendered as a video.
#
#   search/full_game.sh [OUT_DIR] [THREADS]
#
# OUT_DIR (default search/out/e2e) gets route.npz, stats.json, solve.log,
# demo.mp4 (50 fps: the game is PAL; 4x, HUD with level splits) and demo.gif. ~30-60 min on 64 threads.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-search/out/e2e}
THREADS=${2:-$(nproc)}
PY=${PYTHON:-venv_retro/bin/python}

[ -f search/build/libsmbsearch.so ] || search/build.sh
mkdir -p "$OUT"
PYTHONUNBUFFERED=1 "$PY" search/tools/solve.py --start FullGame --segments 8 --out "$OUT" \
    --threads "$THREADS" --explore-budget 900 --explore-settle 60 --beam 4000 2>&1 | tee "$OUT/solve.log"
"$PY" search/tools/verify_retro.py "$OUT/route.npz"
"$PY" search/tools/render_demo.py "$OUT/route.npz"
echo "[full_game] video: $OUT/demo.mp4"
