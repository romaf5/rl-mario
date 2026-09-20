#!/usr/bin/env bash
# Build search/build/libsmbsearch.so (Python) and search/build/smbsearch (CLI).
#   search/build.sh          -O3 -march=native -flto
#   search/build.sh --pgo    + profile-guided optimisation (profiles bench + explore on 1-1 / 4-2)
set -euo pipefail
cd "$(dirname "$0")"
CXX=${CXX:-g++}
PY=${PYTHON:-../venv_retro/bin/python}
SRCS="emu/emu route/route explore/explore optimize/beam api"
FLAGS="-std=c++17 -O3 -march=native -mtune=native -flto=auto -fno-plt -fomit-frame-pointer \
  -fvisibility=hidden -fPIC -DNDEBUG -Wall -Wno-invalid-offsetof -Wno-unused-function -pthread"
OBJ=build/obj

compile() {  # $1: extra flags
  mkdir -p $OBJ
  for s in $SRCS cli; do
    mkdir -p "$OBJ/$(dirname $s)"
    $CXX $FLAGS $1 -c src/$s.cpp -o $OBJ/$s.o
  done
  local objs=""
  for s in $SRCS; do objs="$objs $OBJ/$s.o"; done
  $CXX $FLAGS $1 -shared -o build/libsmbsearch.so $objs
  $CXX $FLAGS $1 -o build/smbsearch $objs $OBJ/cli.o
}

if [ "${1:-}" = "--pgo" ]; then
  rm -rf build/pgo $OBJ && mkdir -p build/pgo
  compile "-fprofile-generate -fprofile-update=atomic -fprofile-dir=$(pwd)/build/pgo"
  ROM=../retro_integration/SuperMarioBros-Nes-v0/rom.nes
  for L in 1-1 4-2; do
    $PY -c "import gzip,sys; open('build/L$L.state','wb').write(gzip.open('../native/states/Level$L.state').read())"
  done
  ./build/smbsearch bench $ROM build/L4-2.state 0 3000000
  ./build/smbsearch explore $ROM build/L1-1.state 1-1,1-2,4-1,4-2,8-1,8-2,8-3,8-4 20 build/pgo_ref.bin || true
  ./build/smbsearch optimize $ROM build/L1-1.state 1-1,1-2,4-1,4-2,8-1,8-2,8-3,8-4 build/pgo_ref.bin build/pgo_opt.bin 2000 || true
  rm -rf $OBJ
  compile "-fprofile-use -fprofile-partial-training -fprofile-dir=$(pwd)/build/pgo -Wno-missing-profile"
else
  rm -rf $OBJ
  compile ""
fi
echo "[build] ok: $(pwd)/build/libsmbsearch.so $(pwd)/build/smbsearch"
