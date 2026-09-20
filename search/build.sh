#!/usr/bin/env bash
# Build search/build/libsmbsearch.so (Python) and search/build/smbsearch (CLI).
#   search/build.sh          -O3 -march=native -flto (PGO was measured slower: 205k vs 384k frames/s)
set -euo pipefail
cd "$(dirname "$0")"
CXX=${CXX:-g++}
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

rm -rf $OBJ
compile ""
echo "[build] ok: $(pwd)/build/libsmbsearch.so $(pwd)/build/smbsearch"
