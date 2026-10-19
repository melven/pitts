#!/bin/bash
# Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Stop hook for Claude Code: when the current state of src/ contains function or class definitions that do not
# exist on origin/master (compared by their heads, see definition-heads.awk; moved code is not new, untracked
# files are included), refuse to end the turn once and ask for a review of them (reuse of existing helpers,
# numerical soundness). Exit code 2 blocks stopping and feeds stderr back to the agent.
#
# Limitations: the review comes at the end of a turn, often after commit and push, so fixes become extra commits;
# reviewed definitions are remembered in .git/claude-reviewed-definitions, so a fresh clone (new cloud session)
# lists them again; definitions added while fixing after a block are only checked at the end of the next turn.

set -uo pipefail
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}"

# avoid an infinite loop: Claude Code passes stop_hook_active when a stop hook already blocked this turn
if [ -t 0 ]; then input=""; else input=$(cat); fi
if printf '%s' "$input" | grep -q '"stop_hook_active": *true'; then
  exit 0
fi

base=$(git merge-base HEAD origin/master 2>/dev/null) || exit 0
heads=.claude/hooks/definition-heads.awk
reviewed="$(git rev-parse --git-dir)/claude-reviewed-definitions"
touch "$reviewed"

# heads on the base (without the file name) and in the working tree ("<file>: <head>")
before=$(git ls-tree -r --name-only "$base" -- src | grep -E '\.(hpp|cpp)$' | while read -r f; do
           git show "$base:$f" | awk -v file="$f" -f "$heads"
         done | sed 's/^[^:]*: //' | sort -u)
after=$(for f in src/*.hpp src/*.cpp; do [ -f "$f" ] && awk -v file="$f" -f "$heads" "$f"; done | sort -u)

# new: the head exists on the base in no file, and it was not reported before
new=$(awk 'NR == FNR { known[$0] = 1; next } !(substr($0, index($0, ": ")+2) in known)' <(printf '%s\n' "$before") <(printf '%s\n' "$after") \
        | grep -v -x -F -f "$reviewed" | sed '/^$/d')
[ -z "$new" ] && exit 0
printf '%s\n' "$new" >> "$reviewed"

{
  echo "New definitions in src/ (not yet reviewed):"
  printf '%s\n' "$new" | sed 's/^/  /'
  echo
  echo "Before finishing, review each of them for reuse of existing helpers (\"Check existing functionality\" in"
  echo "src/CLAUDE.md) and for numerical soundness (\"Numerical soundness\" in DESIGN.md). Fix what you find; report"
  echo "remaining concerns, e.g. a numerically questionable step, to the user."
} >&2
exit 2
