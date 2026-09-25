#!/usr/bin/env bash
# Installs MPICH 4.2.1-5 + UCX 1.18.1+ds-2 from a pinned Ubuntu snapshot,
# working around the broken MPICH 4.2.0 package in Ubuntu 24.04 (Noble).
# See: https://bugs.launchpad.net/ubuntu/+source/mpich/+bug/2072338
#
# The two .debs and the SHA256 sums below are copied verbatim from
# dists/plucky/universe/binary-amd64/Packages.xz of this snapshot, whose
# hash was checked against the gpgv-verified dists/plucky/InRelease -- never
# computed from a fresh download. See the PR description for the
# verification commands and their output.
set -euo pipefail

readonly SNAPSHOT_TS="20250601T000000Z"
readonly SNAPSHOT_BASE="https://snapshot.ubuntu.com/ubuntu/${SNAPSHOT_TS}"
readonly LIBUCX_DEB="libucx0_1.18.1+ds-2_amd64.deb"
readonly LIBMPICH_DEB="libmpich12_4.2.1-5_amd64.deb"
readonly LIBUCX_SHA256="268c7a04b2fcaa1fa1c65846f326124103da502e7f11522d6a005fe647751717"
readonly LIBMPICH_SHA256="87a1622661bf4875af6fabfdb163f5e8bcbdb43c22524f7882330090a074ca34"

# download <dir>
# Fetches the two pinned .debs from the snapshot into <dir>.
download() {
  local dir="$1"
  curl -fsSL "${SNAPSHOT_BASE}/pool/universe/u/ucx/${LIBUCX_DEB}" -o "${dir}/${LIBUCX_DEB}"
  curl -fsSL "${SNAPSHOT_BASE}/pool/universe/m/mpich/${LIBMPICH_DEB}" -o "${dir}/${LIBMPICH_DEB}"
}

# verify <dir>
# Checks both .debs in <dir> against the embedded SHA256 sums. Fails on any
# mismatch or missing file, before install() touches the system.
verify() {
  local dir="$1"
  (
    cd "$dir" || exit 1
    sha256sum -c <<CHECKSUMS
${LIBUCX_SHA256}  ${LIBUCX_DEB}
${LIBMPICH_SHA256}  ${LIBMPICH_DEB}
CHECKSUMS
  )
}

# install <dir>
# Extracts both .debs already downloaded into <dir> and copies the UCX/MPICH
# shared libraries into the system library directory.
install() {
  local dir="$1"
  local extract_dir="${dir}/extract"
  mkdir -p "$extract_dir"
  dpkg-deb -x "${dir}/${LIBUCX_DEB}" "$extract_dir"
  dpkg-deb -x "${dir}/${LIBMPICH_DEB}" "$extract_dir"

  local libdir
  libdir="/usr/lib/$(arch)-linux-gnu"
  sudo cp -a "${extract_dir}${libdir}/ucx" "$libdir"
  sudo cp -a "${extract_dir}${libdir}"/libuc[mpst]*.so.0.*.* "$libdir"
  sudo cp -a "${extract_dir}${libdir}"/libuc[mpst]*.so.0 "$libdir"
  sudo cp -a "${extract_dir}${libdir}"/libmpi*.so.12.*.* "$libdir"
  sudo cp -a "${extract_dir}${libdir}"/libmpi*.so.12 "$libdir"
  sudo ldconfig
  echo "MPICH hotfix applied successfully"
}

main() {
  local build_arch
  build_arch=$(dpkg --print-architecture)
  if [[ "$build_arch" != "amd64" ]]; then
    echo "ERROR: mpich-hotfix.sh only supports amd64, found: $build_arch" >&2
    exit 1
  fi

  local dir
  dir=$(mktemp -d)
  trap 'rm -rf "$dir"' EXIT

  download "$dir"
  verify "$dir"
  install "$dir"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
