#!/bin/bash
# Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Stop hook for Claude Code: when the changes in src/ (relative to origin/master, including uncommitted
# ones) add new function or class definitions, refuse to end the turn once and ask for a review of them:
# reuse of existing helpers and numerical soundness (see "Reuse and numerical soundness" in CLAUDE.md).
# Each definition is only reported once (remembered in .git/claude-reviewed-definitions).
# Exit code 2 blocks stopping and feeds stderr back to the agent.

set -uo pipefail
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}"

# avoid an infinite loop: Claude Code passes stop_hook_active when a stop hook already blocked this turn
if [ -t 0 ]; then input=""; else input=$(cat); fi
if printf '%s' "$input" | grep -q '"stop_hook_active": *true'; then
  exit 0
fi

base=$(git merge-base HEAD origin/master 2>/dev/null || git rev-parse HEAD 2>/dev/null) || exit 0
reviewed="$(git rev-parse --git-dir)/claude-reviewed-definitions"
touch "$reviewed"

# Added definitions: the head of a definition is followed by a line "{" (Allman style); git may align that "{"
# with an unchanged line, so an added hunk that ends like a definition head counts as well. Reported is the line
# with the opening parenthesis (or the class/struct line); control statements, lambdas, member initializers,
# comments and preprocessor lines are skipped.
definitions=$(git diff --unified=0 "$base" -- 'src/*.hpp' 'src/*.cpp' | awk '
  function endsHead() { return n > 0 && (buf[n] ~ /\)[ \t]*((const|final|override|noexcept)[ \t]*)*$/ || buf[n] ~ /^[ \t]*(class|struct)[ \t][^;]*$/) }
  function report(   i, head, initList, plain) {
    head = ""
    initList = 0
    for(i = n; i > 0 && i > n-40; i--)
    {
      if( buf[i] ~ /^[ \t]*(\/\/|#)/ )
        continue
      # constructor head "Name(...) :" ends an initializer list (member initializers end with an underscore)
      if( buf[i] ~ /\)[ \t]*:[ \t]*$/ )
      {
        head = buf[i]
        break
      }
      if( buf[i] ~ /^[ \t]*([A-Za-z_][A-Za-z0-9_]*_\(|:)/ )
        initList = 1
      if( initList )
        continue
      if( buf[i] ~ /^[ \t]*(class|struct)[ \t]/ || buf[i] ~ /\(/ )
      {
        head = buf[i]
        break
      }
    }
    plain = head
    gsub(/\[\[[^\]]*\]\]/, "", plain)   # attributes like [[nodiscard]] are no lambdas
    if( head != "" && plain !~ /^[ \t]*(if|for|while|switch|catch|else|do|return)[ \t(]/ && plain !~ /[=\[]/ )
    {
      gsub(/^[ \t]+|[ \t]+$/, "", head)
      print file ": " head
    }
    n = 0
  }
  /^\+\+\+ / { if( endsHead() ) report(); file = substr($0, 7); n = 0; next }
  /^@@/      { if( endsHead() ) report(); n = 0; next }
  /^\+/      {
    line = substr($0, 2)
    if( line ~ /^[ \t]*\{[ \t]*$/ )
    {
      if( n > 0 ) report()
      next
    }
    buf[++n] = line
    next
  }
  END { if( endsHead() ) report() }' | sort -u)

[ -z "$definitions" ] && exit 0
new=$(comm -23 <(printf '%s\n' "$definitions") <(sort -u "$reviewed"))
[ -z "$new" ] && exit 0
printf '%s\n' "$new" >> "$reviewed"

{
  echo "New definitions in src/ (not yet reviewed):"
  printf '%s\n' "$new" | sed 's/^/  /'
  echo
  echo "Before finishing, review each of them (see \"Reuse and numerical soundness\" in CLAUDE.md):"
  echo "  1. Reuse: is there already a helper for this (src/CLAUDE.md tables, the pitts_<type>_*.hpp headers of"
  echo "     the data types involved)? Use it instead. If it is a variant of an existing helper, generalize that"
  echo "     one instead of adding a similar second function; generally useful new helpers belong to the headers"
  echo "     of their data type, not into a single algorithm."
  echo "  2. Numerics: no explicit inverses or normal equations where a factorization/solve works, no avoidable"
  echo "     cancellation, scale-invariant (relative) tolerances, conditioning of every step considered."
  echo "Fix what you find; report remaining concerns (e.g. a numerically questionable step) to the user."
} >&2
exit 2
