#!/usr/bin/env bash
# One-shot setup (Linux or Apple Silicon Mac): venv_retro + Python deps, the SMB ROM
# (from the gym-super-mario-bros wheel, SHA-1 checked against rom.sha), the native
# core (native/build.sh) and the search engine (search/build.sh).
#   PYTHON=/path/to/python3.12 ./setup.sh     # to pick the interpreter
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-$(command -v python3.11 || command -v python3.12 || command -v python3)}
if [ ! -d venv_retro ]; then
  echo "[setup] creating venv_retro with $PY"
  "$PY" -m venv venv_retro
fi
venv_retro/bin/pip install -q -U pip wheel setuptools
venv_retro/bin/pip install -r requirements.txt

ROM=retro_integration/SuperMarioBros-Nes-v0/rom.nes
if [ ! -f "$ROM" ]; then
  TMP=$(mktemp -d)
  venv_retro/bin/pip download -q gym-super-mario-bros==7.4.0 --no-deps -d "$TMP"
  venv_retro/bin/python - "$TMP" "$ROM" <<'EOF'
import glob, hashlib, sys, zipfile
tmp, dst = sys.argv[1], sys.argv[2]
whl = glob.glob(tmp + '/gym_super_mario_bros-*.whl')[0]
data = zipfile.ZipFile(whl).read('gym_super_mario_bros/_roms/super-mario-bros.nes')
want = open('retro_integration/SuperMarioBros-Nes-v0/rom.sha').read().strip()
got = hashlib.sha1(data[16:]).hexdigest()      # retro hashes the ROM without its iNES header
if got != want:
    sys.exit('[setup] ROM hash mismatch: %s != %s' % (got, want))
open(dst, 'wb').write(data)
print('[setup] ROM installed:', dst)
EOF
  rm -rf "$TMP"
fi

native/build.sh
search/build.sh
echo "[setup] done: source venv_retro/bin/activate"
