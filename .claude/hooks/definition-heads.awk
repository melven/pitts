# Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Print the heads of the function and class definitions in a C++ source (Allman style: the head is followed by a
# line "{"), one per line as "<file>: <first line of the head>" (set file with -v file=...).
# Heuristic, used by stop-review-new-definitions.sh and tested by scripts/test_definition_heads.sh:
#  * the line before the "{" must end like a head: ")" (optionally followed by const/final/override/noexcept) or a
#    class/struct line; so blocks after "else", "do", "#pragma omp ..." or a statement are not reported;
#  * the head starts where its parenthesis opens (balance counted backwards over the lines, so multi-line parameter
#    lists with default arguments are handled); for a constructor with an initializer list, the head "Name(...) :"
#    above the member initializers (members end with an underscore) is reported;
#  * control statements and lambdas (a capture list "[...](" or "[...]<...>(") are skipped, attributes like [[nodiscard]]
#    are kept; one-line definitions ("... {return x_;}") are not reported.

function count(s, c,   tmp) { tmp = s; return gsub(c, "", tmp) }

function endsHead() { return n > 0 && buf[n] !~ /^[ \t]*(#|\/\/)/ && (buf[n] ~ /\)[ \t]*((const|final|override|noexcept)[ \t]*)*$/ || buf[n] ~ /^[ \t]*(class|struct)[ \t][^;]*$/) }

# first line of the statement that ends in line i (backwards until the parentheses are balanced)
function statementStart(i,   depth) {
  depth = 0
  for(; i > 0; i--)
  {
    depth += count(buf[i], "\\)") - count(buf[i], "\\(")
    if( depth <= 0 )
      return i
  }
  return 0
}

function report(   i, head, plain) {
  if( buf[n] ~ /^[ \t]*(class|struct)[ \t]/ )
    head = buf[n]
  else
  {
    i = statementStart(n)
    # member initializers: go up to the constructor head "Name(...) :"
    if( i > 0 && buf[i] ~ /^[ \t]*(:[ \t]*)?[A-Za-z_][A-Za-z0-9_]*_\(/ )
    {
      for(; i > 0 && buf[i] !~ /\)[ \t]*:[ \t]*$/; i--)
        ;
      i = i > 0 ? statementStart(i) : 0
    }
    head = i > 0 ? buf[i] : ""
  }
  plain = head
  gsub(/\[\[[^\]]*\]\]/, "", plain)   # attributes are no capture lists
  if( head != "" && plain !~ /^[ \t]*(if|for|while|switch|catch|else|do|return)[ \t(]/ && plain !~ /(^|[(,=])[ \t]*\[[^\]]*\][ \t]*(<[^>]*>)?[ \t]*\(/ )
  {
    gsub(/^[ \t]+|[ \t]+$/, "", head)
    print file ": " head
  }
}

/^[ \t]*\{[ \t]*$/ { if( endsHead() ) report(); n = 0; next }
{ buf[++n] = $0 }
