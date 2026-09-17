#!/usr/bin/env bash
# Build the native SMB core libraries next to this script.
#   libbatchenv.so -- batched threadpool env used by training (mario_native_vecenv.py)
#   libsmbcore.so  -- bare core used by difftest.py / deep_difftest.py / gen_states_native.py
# Linux (GPU box): g++ -march=native, exactly the documented build.
# macOS (Apple Silicon): clang++ -mcpu=native; the Mach-O dylib keeps the .so
# name so ctypes loads it from the same path on both platforms.
set -euo pipefail
cd "$(dirname "$0")"

case "$(uname -s)" in
  Darwin)
    CXX=${CXX:-/usr/bin/clang++}
    ARCH_FLAGS=${ARCH_FLAGS:--mcpu=native}
    STD="-std=c++17"
    ;;
  *)
    CXX=${CXX:-g++}
    ARCH_FLAGS=${ARCH_FLAGS:--march=native}
    STD=""
    ;;
esac

echo "[build] $CXX $ARCH_FLAGS"
$CXX $STD -O3 $ARCH_FLAGS -fPIC -shared -o libbatchenv.so.tmp batchenv.cpp -lpthread
mv -f libbatchenv.so.tmp libbatchenv.so
$CXX $STD -O3 $ARCH_FLAGS -fPIC -shared -o libsmbcore.so.tmp smbcore.cpp
mv -f libsmbcore.so.tmp libsmbcore.so
echo "[build] ok: $(pwd)/libbatchenv.so $(pwd)/libsmbcore.so"
