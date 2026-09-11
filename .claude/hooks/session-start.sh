#!/bin/bash
# Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
#
# SPDX-License-Identifier: BSD-3-Clause
#
# SessionStart hook for Claude Code on the web: installs all build dependencies
# through apt, exports the environment needed to run the MPI/OpenMP tests and
# configures + builds a development build (see CLAUDE.md). Idempotent.

set -euo pipefail

# only for remote (web) sessions
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi
export DEBIAN_FRONTEND=noninteractive

# --- apt packages -----------------------------------------------------------
# Third-party PPAs in the base image may be unreachable (403 / unsigned) and
# would make apt-get update fail; none of them is needed for PITTS.
for src in /etc/apt/sources.list.d/*; do
  if [ -f "$src" ] && grep -q "ppa.launchpadcontent.net" "$src"; then
    $SUDO mkdir -p /root/disabled-apt-sources
    $SUDO mv "$src" /root/disabled-apt-sources/
  fi
done

$SUDO apt-get update -qq
$SUDO apt-get install -y -qq --no-install-recommends \
  build-essential cmake ninja-build ccache gdb lld reuse \
  libopenmpi-dev openmpi-bin libopenblas-dev liblapack-dev \
  libeigen3-dev libcereal-dev pybind11-dev python3.12-dev python3-numpy

# --- environment for running the tests -------------------------------------
# OpenMPI refuses to run as root without these; passive OpenMP waiting and a
# single OpenBLAS thread avoid oversubscription stalls on small containers.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  cat >> "$CLAUDE_ENV_FILE" <<'EOF'
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMP_WAIT_POLICY=passive
export OPENBLAS_NUM_THREADS=1
EOF
fi

# --- configure and build ---------------------------------------------------
cd "$CLAUDE_PROJECT_DIR"
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DPYTHON_EXECUTABLE=/usr/bin/python3.12 \
  -DPITTS_DEVELOP_BUILD=On
cmake --build build -j"$(nproc)" --target all pitts_tests
ccache -s
