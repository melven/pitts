#!/bin/bash
# Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Consistency check for the per-directory CLAUDE.md files (documentation for coding agents):
#  * every file in src/, test/, test/itensor/, examples/ and cmake/ must be mentioned in the
#    CLAUDE.md of that directory (by its "topic" name, i.e. without _impl.hpp/.cpp/_pybind/_jlbind suffix),
#  * every concrete file name mentioned in those CLAUDE.md files must exist,
#  * every top-level directory must be mentioned in the root CLAUDE.md.
# Exits with 1 and lists the problems if anything is out of date. Used as Claude Code Stop hook.

set -uo pipefail
cd "$(dirname "$0")/.."

status=0
problem() { echo "CLAUDE.md check: $*"; status=1; }

# topic name of a source/test file: strip implementation / binding / extension suffixes
topic_of() {
  local f=$1
  f=${f%.cpp}; f=${f%.hpp}; f=${f%.py}; f=${f%.jl}
  f=${f%_impl}; f=${f%_pybind}; f=${f%_jlbind}
  echo "$f"
}

# --- files -> documentation ---------------------------------------------------------------
check_dir() {
  local dir=$1 doc=$2
  [ -f "$doc" ] || { problem "$doc is missing"; return; }
  for path in "$dir"/*; do
    [ -f "$path" ] || continue
    local f; f=$(basename "$path")
    case "$f" in CLAUDE.md|*.license|.gitignore) continue;; esac
    local t; t=$(topic_of "$f")
    if ! grep -qF -- "$t" "$doc"; then
      problem "$path is not mentioned in $doc (searched for '$t')"
    fi
  done
}

check_dir src      src/CLAUDE.md
check_dir test     test/CLAUDE.md
check_dir test/itensor test/CLAUDE.md
check_dir examples examples/CLAUDE.md
check_dir cmake    cmake/CLAUDE.md
[ -f bench/CLAUDE.md ] || problem "bench/CLAUDE.md is missing"

# --- documentation -> files ---------------------------------------------------------------
# concrete file names in backticks (no wildcards, no leading underscore abbreviations)
check_doc_refs() {
  local doc=$1; shift
  local dirs=("$@")
  grep -oE '`[A-Za-z][A-Za-z0-9_./-]*\.(hpp|cpp|py|jl|cmake|sh|in|txt)`' "$doc" \
    | tr -d '`' | sort -u | while read -r ref; do
      local found=0
      for d in "${dirs[@]}"; do
        [ -e "$d/$ref" ] && found=1 && break
      done
      [ -e "$ref" ] && found=1
      [ $found -eq 1 ] || problem "$doc mentions $ref which does not exist"
    done
}
# (runs in subshells because of the pipe -> collect status via output)
{
  check_doc_refs src/CLAUDE.md src
  check_doc_refs test/CLAUDE.md test test/itensor
  check_doc_refs examples/CLAUDE.md examples
  check_doc_refs cmake/CLAUDE.md cmake
} | tee /tmp/claude_md_refs.$$
if [ -s /tmp/claude_md_refs.$$ ]; then status=1; fi
rm -f /tmp/claude_md_refs.$$

# --- top-level directories -> root CLAUDE.md ---------------------------------------------
for path in */; do
  d=${path%/}
  case "$d" in build*|scripts) continue;; esac
  grep -qF -- "\`$d/\`" CLAUDE.md || problem "top-level directory $d/ is not mentioned in CLAUDE.md"
done

if [ $status -eq 0 ]; then
  echo "CLAUDE.md check: OK"
fi
exit $status
