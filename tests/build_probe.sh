#!/usr/bin/env bash
# ==========================================================================
# End-to-end checks of build.rs's MPI resolution.
#
# Builds hello_world against fake wrappers, pkg-config files, headers and
# Cray-style directories that point at the MPI whose `mpicc` is on PATH.
# Every build starts from a fresh build-script run (`cargo clean -p`).
#
# Prints "ok <case>" or "not ok <case>" per case and exits 1 if any failed.
# ==========================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

FX=$(mktemp -d)
trap 'rm -rf "$FX"' EXIT
export CARGO_TARGET_DIR="$FX/target"
unset MPI_PKG_CONFIG MPICC CRAY_MPICH_DIR

# The real wrapper's flags without the compiler name.
REAL_FLAGS=$(mpicc -show 2>/dev/null || mpicc --showme)
REAL_FLAGS=${REAL_FLAGS#* }
REAL_INCLUDE=$(tr ' ' '\n' <<<"$REAL_FLAGS" | sed -n 's/^-I//p' | head -1)
REAL_LIBDIR=$(tr ' ' '\n' <<<"$REAL_FLAGS" | sed -n 's/^-L//p' | head -1)
REAL_LIB=$(tr ' ' '\n' <<<"$REAL_FLAGS" | sed -n 's/^-l//p' | head -1)
REAL_MPICC=$(command -v mpicc)
FAILED=0

# build [VAR=VALUE]...: a fresh build of hello_world with these variables set.
build() {
  cargo clean -q -p ferrompi
  env "$@" cargo build -q --example hello_world >"$FX/log" 2>&1
}

emitted() {
  grep -qxF "$1" "$CARGO_TARGET_DIR"/debug/build/ferrompi-*/output
}

logged() {
  grep -qF "$1" "$FX/log"
}

pass() {
  echo "ok $1"
}

fail() {
  echo "not ok $1"
  sed 's/^/    /' "$FX/log"
  FAILED=1
}

# wrapper NAME OPTION FLAGS: a fake mpicc that prints FLAGS for OPTION only.
wrapper() {
  mkdir -p "$FX/bin"
  cat >"$FX/bin/$1" <<EOF
#!/bin/sh
[ "\$1" = $2 ] || exit 1
echo 'cc $3'
EOF
  chmod +x "$FX/bin/$1"
}

# header DIR LINE...: a fixture mpi.h found before the real one.
header() {
  mkdir -p "$FX/$1"
  printf '%s\n' "${@:2}" >"$FX/$1/mpi.h"
}

# pcfile PATH LIBS: a pkg-config file for package `mpich`.
pcfile() {
  mkdir -p "$(dirname "$1")"
  printf 'Name: mpich\nDescription: fixture\nVersion: 0\nLibs: %s\n' "$2" >"$1"
}

# --- explicit selection variables -------------------------------------------

name="an explicit MPICC beats a decoy mpich.pc; an empty variable counts as unset"
mkdir -p "$FX/decoy/include" "$FX/decoy/lib"
pcfile "$FX/decoy/mpich.pc" "-I$FX/decoy/include -L$FX/decoy/lib -lmpich_decoy"
if build PKG_CONFIG_PATH="$FX/decoy" MPI_PKG_CONFIG= MPICC="$REAL_MPICC"; then
  pass "$name"
else fail "$name"; fi

name="a failing MPI_PKG_CONFIG stops the build before MPICC is tried"
if ! build MPI_PKG_CONFIG=nonexistent MPICC="$REAL_MPICC" &&
  logged "MPI_PKG_CONFIG=nonexistent"; then
  pass "$name"
else fail "$name"; fi

name="a failing MPICC stops the build before CRAY_MPICH_DIR is tried"
if ! build MPICC="$FX/missing-mpicc" CRAY_MPICH_DIR="$FX" &&
  logged "MPICC=$FX/missing-mpicc"; then
  pass "$name"
else fail "$name"; fi

name="a CRAY_MPICH_DIR without MPI stops the build"
mkdir -p "$FX/cray-empty"
if ! build CRAY_MPICH_DIR="$FX/cray-empty" &&
  logged "CRAY_MPICH_DIR=$FX/cray-empty"; then
  pass "$name"
else fail "$name"; fi

name="CRAY_MPICH_DIR resolves lib/pkgconfig/mpich.pc"
mkdir -p "$FX/cray-pc/lib"
pcfile "$FX/cray-pc/lib/pkgconfig/mpich.pc" "-L$FX/cray-pc/lib $REAL_FLAGS"
if build CRAY_MPICH_DIR="$FX/cray-pc" &&
  emitted "cargo:rustc-link-search=native=$FX/cray-pc/lib"; then
  pass "$name"
else fail "$name"; fi

name="CRAY_MPICH_DIR resolves include/mpi.h and lib64/libmpich"
mkdir -p "$FX/cray-fs/lib64"
ln -s "$REAL_INCLUDE" "$FX/cray-fs/include"
ln -s "$REAL_LIBDIR/lib$REAL_LIB.so" "$FX/cray-fs/lib64/libmpich.so"
if build CRAY_MPICH_DIR="$FX/cray-fs" &&
  emitted "cargo:rustc-link-search=native=$FX/cray-fs/lib64" &&
  emitted "cargo:rustc-link-lib=mpich"; then
  pass "$name"
else fail "$name"; fi

# --- defines, quotes and --showme -------------------------------------------

header abi-define '#ifndef MPI_ABI' '#error "MPI_ABI was not passed to the compiler"' '#endif' \
  '#undef MPI_ABI' '#include_next <mpi.h>'

name="a wrapper's -D reaches the shim compile and its quotes are stripped"
mkdir -p "$FX/quoted/lib"
wrapper abi-mpicc -show "-I$FX/abi-define -DMPI_ABI -L\"$FX/quoted/lib\" $REAL_FLAGS"
if build MPICC="$FX/bin/abi-mpicc" &&
  emitted "cargo:rustc-link-search=native=$FX/quoted/lib"; then
  pass "$name"
else fail "$name"; fi

name="a pkg-config -D reaches the shim compile"
pcfile "$FX/pc/abi-fixture.pc" "-I$FX/abi-define -DMPI_ABI $REAL_FLAGS"
if build PKG_CONFIG_PATH="$FX/pc" MPI_PKG_CONFIG=abi-fixture; then
  pass "$name"
else fail "$name"; fi

name="a wrapper that answers only --showme"
mkdir -p "$FX/showme/lib"
wrapper showme-mpicc --showme "-L$FX/showme/lib $REAL_FLAGS"
if build MPICC="$FX/bin/showme-mpicc" &&
  emitted "cargo:rustc-link-search=native=$FX/showme/lib"; then
  pass "$name"
else fail "$name"; fi

# --- draft MPI ABI ----------------------------------------------------------

name="the draft MPI ABI stops the build"
header draft-abi '#include_next <mpi.h>' '#define MPI_ABI_VERSION 1'
wrapper draft-mpicc -show "-I$FX/draft-abi $REAL_FLAGS"
if ! build MPICC="$FX/bin/draft-mpicc" && logged "draft MPI ABI"; then
  pass "$name"
else fail "$name"; fi

# --- rebuild tracking --------------------------------------------------------

name="a changed selection variable reruns the build script"
if build && ! MPI_PKG_CONFIG=nonexistent cargo build -q --example hello_world >"$FX/log" 2>&1 &&
  logged "MPI_PKG_CONFIG=nonexistent" &&
  emitted "cargo:rerun-if-env-changed=MPI_PKG_CONFIG" &&
  emitted "cargo:rerun-if-env-changed=MPICC" &&
  emitted "cargo:rerun-if-env-changed=CRAY_MPICH_DIR" &&
  emitted "cargo:rerun-if-env-changed=PATH"; then
  pass "$name"
else fail "$name"; fi

exit "$FAILED"
