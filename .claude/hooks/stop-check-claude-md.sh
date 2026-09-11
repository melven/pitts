#!/bin/bash
# Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Stop hook for Claude Code: refuse to end the turn while the per-directory CLAUDE.md files
# are out of sync with the repository contents (see scripts/check_claude_md.sh).
# Exit code 2 blocks stopping and feeds stderr back to the agent.

set -uo pipefail
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}"

# avoid an infinite loop if the hook itself keeps failing: Claude Code passes stop_hook_active
if [ -t 0 ]; then input=""; else input=$(cat); fi
if printf '%s' "$input" | grep -q '"stop_hook_active": *true'; then
  exit 0
fi

if ! out=$(./scripts/check_claude_md.sh 2>&1); then
  {
    echo "The CLAUDE.md documentation is out of sync with the repository. Update the affected CLAUDE.md"
    echo "tables (or the root CLAUDE.md layout table) before finishing, then commit:"
    echo "$out"
  } >&2
  exit 2
fi
exit 0
