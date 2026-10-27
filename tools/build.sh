#!/bin/sh
# Build build/libomega_native.so + bindings from om/dnn.om with the pinned compilers and omega-bridge.
# Usage: tools/build.sh            (fetches the pins into .deps/ when OMEGA_MV / OMEGA_BRIDGE are unset)
#        OMEGA_MV=<omega-mv clone> OMEGA_BRIDGE=<omega-bridge clone> tools/build.sh
# The pins (parent omega-mv, bridge ABI v2) are in PINS; a compiler whose sha256 differs is refused.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/PINS"
DEPS=$ROOT/.deps
fetch() { # repo sha dir
  [ -d "$3/.git" ] || git clone -q --filter=blob:none --no-checkout "https://github.com/aien-dev/$1" "$3"
  git -C "$3" fetch -q origin "$2" 2>/dev/null || git -C "$3" fetch -q origin
  git -C "$3" checkout -q "$2"
}
if [ -z "${OMEGA_MV:-}" ]; then OMEGA_MV=$DEPS/omega-mv; fetch omega-mv "$OMEGA_MV_SHA" "$OMEGA_MV"; fi
if [ -z "${OMEGA_BRIDGE:-}" ]; then OMEGA_BRIDGE=$DEPS/omega-bridge; fetch omega-bridge "$OMEGA_BRIDGE_SHA" "$OMEGA_BRIDGE"; fi
OMC1=$OMEGA_MV/evidence/SEED-0/omc/omc.elf
OMC2=$OMEGA_MV/seed/out/omc
chmod +x "$OMC1" "$OMC2"
check() { got=$(sha256sum < "$1" | cut -d' ' -f1); [ "$got" = "$2" ] || { echo "build.sh: $1 sha256 $got, pinned $2" >&2; exit 1; }; }
check "$OMC1" "$OMC1_SHA256"
check "$OMC2" "$OMC2_SHA256"
SRC=${OMEGA_DNN_SRC:-$ROOT/om/dnn.om}
OUT=${OMEGA_DNN_OUT:-$ROOT/build}
"$OMC2" < "$SRC" > /dev/null
OMEGA_NATIVE_OMC=$OMC1 OMEGA_NATIVE_OMC2=$OMC2 "$OMEGA_BRIDGE/tools/native" "$OUT" "$SRC"
