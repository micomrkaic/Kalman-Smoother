"""
Convert Octave-specific syntax to MATLAB-compatible syntax in .m files.

Transformations (applied in order):
  1. endif, endfor, endwhile, endswitch, endfunction, end_try_catch -> end
  2. printf( -> fprintf(
  3. != -> ~=
  4. !X  -> ~X       (where X is a letter, underscore, or '(')
  5. rows(EXPR) -> size(EXPR,1)
  6. columns(EXPR) -> size(EXPR,2)

The conversions are textual (regex-based) and DO NOT parse the file as
Octave/MATLAB source.  In practice this works fine for this package
because:
  * no `!` appears inside string literals (verified by grep);
  * no shell-style `!cmd` lines exist;
  * rows/columns are always called with a simple (parenthesizable) argument.

Word-boundary matching is used for keyword replacements so we don't
accidentally rewrite identifiers like `myendif`.

Usage:
    python3 octave_to_matlab.py FILE1.m FILE2.m ...
"""

# simksmoother --- a simultaneous Kalman smoother in sparse linear algebra form.
# Copyright (C) 2026 Mico Mrkaic.
#
# Produced under the guidance and direction of Mico Mrkaic, with the
# assistance of AI (Claude, Anthropic).
#
# This file is part of the simksmoother package.
#
# simksmoother is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; see the LICENSE file in the package root,
# or <https://www.gnu.org/licenses/>.
#

import re
import sys
from pathlib import Path


# Token-level keyword replacements (whole word).
KEYWORD_MAP = {
    "endif": "end",
    "endswitch": "end",
    "endfor": "end",
    "endwhile": "end",
    "endfunction": "end",
    "end_try_catch": "end",
}


def replace_balanced_paren(source: str, fname: str, replacement: str) -> str:
    """Replace fname(...) with `replacement(..., k)` style.  We need to
    find the matching close paren, since the argument may itself contain
    parentheses.  The `replacement` string is a literal substitution for
    `fname` -- e.g., 'size' for rows -- and we append `, 1` or `, 2` to
    the argument list."""
    # We'll handle this with a manual scan: find each occurrence of
    # `<word boundary>fname(`, then scan forward to find the matching `)`.
    pattern = re.compile(r"\b" + re.escape(fname) + r"\s*\(")
    out = []
    i = 0
    while True:
        m = pattern.search(source, i)
        if m is None:
            out.append(source[i:])
            break
        out.append(source[i:m.start()])
        # Now find the matching close paren starting after the '('.
        depth = 1
        j = m.end()
        while j < len(source) and depth > 0:
            c = source[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            j += 1
        if depth != 0:
            # Unbalanced; bail out gracefully.
            out.append(source[m.start():])
            break
        # source[m.end():j-1] is the inner argument; source[j-1] is ')'
        inner = source[m.end():j-1]
        out.append(replacement + "(" + inner + ")")
        i = j
    return "".join(out)


def convert(text: str) -> str:
    # 1. Keyword replacements.  These are whole-word, and the replacement
    #    is always `end`, so we can do them one regex at a time without
    #    worrying about cross-contamination.
    for kw, repl in KEYWORD_MAP.items():
        text = re.sub(r"\b" + kw + r"\b", repl, text)

    # 2. printf -> fprintf, but only at call sites (printf followed by
    #    optional space and `(`).  Use a negative lookbehind so we don't
    #    turn `fprintf` (already present) into `ffprintf`.
    text = re.sub(r"(?<![A-Za-z0-9_])printf(\s*\()", r"fprintf\1", text)

    # 3. != -> ~=   (do this BEFORE the unary `!` substitution so the
    #    pattern doesn't get mangled).
    text = text.replace("!=", "~=")

    # 4. Unary !X -> ~X, where X is a letter, underscore, or `(`.
    #    Negative lookbehind: not preceded by another `!`, to leave any
    #    accidental `!!` alone.
    text = re.sub(r"(?<![!])!(?=[A-Za-z_(])", "~", text)

    # 5. rows(EXPR) -> size(EXPR, 1)
    #    columns(EXPR) -> size(EXPR, 2)
    #    Use the balanced-paren replacement so EXPR can itself contain
    #    parentheses.
    text = _rename_rowscols(text, "rows", 1)
    text = _rename_rowscols(text, "columns", 2)

    # 6. Compound assignment operators (Octave/Python-style).  MATLAB does
    #    not accept `x += y`; it requires `x = x + y`.  We handle this
    #    line by line so we don't get confused by RHS expressions that
    #    span multiple statements.  The LHS can be an indexed expression
    #    like `h(idx(t))`; we match anything that isn't `=` or `;`.
    text = _expand_compound_assignments(text)

    return text


_COMPOUND_RE = re.compile(
    r"""^
        (\s*)            # 1: leading whitespace
        ([^;%=\n]+?)     # 2: LHS, no ';', '%', '=', or newline
        \s*
        ([+\-*/])=       # 3: the binary op before '='
        \s*
        (.+?)            # 4: RHS
        (\s*;?\s*)$      # 5: optional trailing ';' and whitespace
    """,
    re.MULTILINE | re.VERBOSE,
)


def _expand_compound_assignments(text: str) -> str:
    def _sub(m):
        lead, lhs, op, rhs, tail = m.groups()
        return f"{lead}{lhs} = {lhs} {op} {rhs}{tail}"
    return _COMPOUND_RE.sub(_sub, text)


def _rename_rowscols(text: str, fname: str, axis: int) -> str:
    """Replace `fname(EXPR)` with `size(EXPR, axis)`.  Manual paren scan
    so we handle nested parens in EXPR correctly."""
    pattern = re.compile(r"\b" + re.escape(fname) + r"\s*\(")
    out = []
    i = 0
    while True:
        m = pattern.search(text, i)
        if m is None:
            out.append(text[i:])
            break
        out.append(text[i:m.start()])
        depth = 1
        j = m.end()
        while j < len(text) and depth > 0:
            c = text[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            j += 1
        if depth != 0:
            out.append(text[m.start():])
            break
        inner = text[m.end():j-1]
        out.append(f"size({inner}, {axis})")
        i = j
    return "".join(out)


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    for arg in sys.argv[1:]:
        p = Path(arg)
        original = p.read_text()
        converted = convert(original)
        if converted != original:
            p.write_text(converted)
            print(f"  patched: {p}")
        else:
            print(f"  no change: {p}")


if __name__ == "__main__":
    main()
