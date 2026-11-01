#!/usr/bin/env python3
"""check-ffi-signatures.py: src/ffi.rs must declare the shims as csrc/ferrompi.h does.

src/ffi.rs declares every function of the C layer by hand, and nothing in the
build compares those declarations with the header: an extern with the wrong
parameter count, parameter type or return type still compiles and links, and
fails only at run time. This check reads both files and requires that

  - every ferrompi_* prototype in the header has an extern declaration in
    src/ffi.rs, by name or by #[link_name], unless UNBOUND_PROTOTYPES below
    lists it, and every extern declaration in src/ffi.rs has a prototype;
  - each pair has the same number of parameters, and each parameter and the
    return type correspond under C_TYPES and RUST_TYPES below, pointer level
    by pointer level and const by const.

Parameter names are not compared. #[cfg] attributes are ignored: the header
declares every shim unconditionally and the C layer always compiles it (the
MPI 4.0 fallbacks sit inside the function bodies), so a feature-gated extern
must match like any other.

Both readers accept only the plain style the two files use, and report what
they cannot read instead of skipping it: a ferrompi_ declaration that is not
`TYPE NAME(PARAMS);`, a type outside the tables below, or an extern whose
parameters they cannot split.

Usage: check-ffi-signatures.py [HEADER RUST]   (Python 3.10+, standard library only)
  With no arguments, checks the repository's csrc/ferrompi.h and src/ffi.rs.
  With two arguments, checks those files (used by the fixture test).

Exit codes: 0 the declarations match; 1 mismatches (printed as FILE:LINE:
message); 2 usage or read error.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import NamedTuple

SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[2]
HEADER = "csrc/ferrompi.h"
RUST = "src/ffi.rs"

# The C base types the header uses, each with the canonical name its Rust
# spelling must reduce to. `int` corresponds only to c_int, whose width is the
# platform's, never to i32.
C_TYPES = {
    "int": "c_int",
    "int32_t": "i32",
    "int64_t": "i64",
    "uint8_t": "u8",
    "double": "f64",
    "char": "c_char",
    "void": "c_void",
    "ferrompi_status": "FerrompiStatus",
}

# The Rust base types src/ffi.rs uses, each with its canonical name. int32_t
# and int64_t are src/ffi.rs's aliases of i32 and i64.
RUST_TYPES = {
    "c_int": "c_int",
    "int32_t": "i32",
    "i32": "i32",
    "int64_t": "i64",
    "i64": "i64",
    "u8": "u8",
    "c_double": "f64",
    "f64": "f64",
    "c_char": "c_char",
    "c_void": "c_void",
    "FerrompiStatus": "FerrompiStatus",
}

# Header prototypes that src/ffi.rs deliberately does not declare, each mapped
# to the reason it has no Rust caller.
UNBOUND_PROTOTYPES: dict[str, str] = {}

# A declaration's type as written, and its canonical type: None when it could
# not be read.
Typed = tuple[str, str | None]

# (file, line, message); line 0 when the message has no line.
Problem = tuple[str, int, str]


class Decl(NamedTuple):
    """A declaration: its function name, the line of that name, and the types
    of its parameters and return."""

    name: str
    line: int
    params: list[Typed]
    ret: Typed


PROTOTYPE = re.compile(
    r"(?:extern\s+)?(?P<ret>[^()]*?)\s*\b(?P<name>ferrompi_\w+)"
    r"\s*\((?P<params>[^()]*)\)",
    re.S,
)
LINK_NAME = re.compile(r'#\[\s*link_name\s*=\s*"([^"]*)"\s*\]')
RUST_PARAMETER = re.compile(r"(?:mut\s+)?[A-Za-z_]\w*\s*:\s*(.+)", re.S)


def strip_comments(text: str, char_literals: bool) -> str:
    """text with each comment replaced by a space and the newlines it held."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end < 0 else end
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            end = n if end < 0 else end + 2
            out.append(" " + "\n" * text.count("\n", i, end))
            i = end
        elif text[i] == '"' or (char_literals and text[i] == "'"):
            end = i + 1
            while end < n and text[end] != text[i]:
                end += 2 if text[end] == "\\" else 1
            out.append(text[i : end + 1])
            i = end + 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def blank_preprocessor(text: str) -> str:
    """text with each preprocessor directive, continuations included, emptied."""
    lines = text.split("\n")
    continued = False
    for k, line in enumerate(lines):
        if continued or line.lstrip().startswith("#"):
            continued = line.endswith("\\")
            lines[k] = ""
    return "\n".join(lines)


def line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def statement_start(text: str, pos: int) -> int:
    return max(text.rfind(c, 0, pos) for c in ";{}") + 1


def first_of(text: str, chars: str, pos: int) -> int | None:
    """The index of the first of chars in text at or after pos, or None."""
    found = [i for i in (text.find(c, pos) for c in chars) if i >= 0]
    return min(found) if found else None


def canonical(base: str, base_const: bool, pointer_consts: list[bool]) -> str:
    """The canonical Rust spelling of a base type behind pointers, innermost
    pointer first; each pointer's own const qualifies the pointer above it."""
    spelled = base
    pointee_const = base_const
    for const in pointer_consts:
        spelled = ("*const " if pointee_const else "*mut ") + spelled
        pointee_const = const
    return spelled


def c_type(text: str, named: bool) -> str | None:
    """The canonical type of a C parameter (named: drop its name) or return
    type, or None when it is not a C_TYPES type behind pointers."""
    tokens = re.findall(r"[A-Za-z_]\w*|\S", text)
    last = tokens[-1] if tokens else ""
    is_name = re.fullmatch(r"[A-Za-z_]\w*", last) and last not in [*C_TYPES, "const"]
    if named and len(tokens) > 1 and is_name:
        tokens.pop()
    base: str | None = None
    base_const, i = False, 0
    while i < len(tokens) and tokens[i] != "*":
        if tokens[i] == "const":
            base_const = True
        elif base is None and tokens[i] in C_TYPES:
            base = tokens[i]
        else:
            return None
        i += 1
    if base is None:
        return None
    pointer_consts: list[bool] = []
    while i < len(tokens):
        if tokens[i] != "*":
            return None
        i += 1
        const = False
        while i < len(tokens) and tokens[i] == "const":
            const, i = True, i + 1
        pointer_consts.append(const)
    if base == "void" and not pointer_consts:
        return "()"
    return canonical(C_TYPES[base], base_const, pointer_consts)


def rust_type(text: str) -> str | None:
    """The canonical type of a Rust parameter or return type, or None when it
    is not a RUST_TYPES type behind raw pointers."""
    text = text.strip()
    if text == "()":
        return "()"
    prefix = ""
    while True:
        m = re.match(r"\*\s*(const|mut)\s+", text)
        if not m:
            break
        prefix += "*" + m.group(1) + " "
        text = text[m.end() :]
    base = RUST_TYPES.get(text)
    return None if base is None else prefix + base


def split_parameters(text: str) -> list[str]:
    """The comma-separated items of a parameter list, at bracket depth 0."""
    items: list[str] = []
    depth, start = 0, 0
    for i, c in enumerate(text):
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == "," and depth == 0:
            items.append(text[start:i])
            start = i + 1
    items.append(text[start:])
    return [" ".join(item.split()) for item in items if item.strip()]


def matching_paren(text: str, open_pos: int) -> int | None:
    """The index of the parenthesis that closes the one at open_pos, or None."""
    depth = 0
    for i in range(open_pos, len(text)):
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        if depth == 0:
            return i
    return None


def unreadable(decl: Decl) -> bool:
    return decl.ret[1] is None or any(t is None for _, t in decl.params)


def read_header(text: str, path: str, problems: list[Problem]) -> dict[str, Decl]:
    """The ferrompi_* prototypes of a C header, by name."""
    text = blank_preprocessor(strip_comments(text, char_literals=True))
    protos: dict[str, Decl] = {}
    seen: set[int] = set()
    for m in re.finditer(r"\b(ferrompi_\w+)\s*\(", text):
        name, line = m.group(1), line_of(text, m.start())
        start = statement_start(text, m.start())
        if start in seen:
            continue
        seen.add(start)
        end = first_of(text, ";{}", m.start())
        p = None
        if end is not None and text[end] == ";":
            p = PROTOTYPE.fullmatch(text[start:end].strip())
        if p is None or p.group("name") != name or not p.group("ret"):
            message = f"{name}: cannot read this declaration as `TYPE NAME(PARAMS);`"
            problems.append((path, line, message))
            continue
        params = split_parameters(p.group("params"))
        if params == ["void"]:
            params = []
        ret = " ".join(p.group("ret").split())
        decl = Decl(
            name,
            line,
            [(t, c_type(t, named=True)) for t in params],
            (ret, c_type(ret, named=False)),
        )
        if unreadable(decl):
            types = ", ".join(C_TYPES)
            message = f"{name}: cannot read its types; the check reads {types}, "
            problems.append((path, line, message + "const and pointers"))
        protos.setdefault(name, decl)
    return protos


def read_rust(text: str, path: str, problems: list[Problem]) -> dict[str, list[Decl]]:
    """The extern declarations of a Rust file (every `fn` without a body), by
    the symbol they link to; a symbol may be declared more than once."""
    text = strip_comments(text, char_literals=False)
    externs: dict[str, list[Decl]] = {}
    for m in re.finditer(r"\bfn\s+([A-Za-z_]\w*)\s*\(", text):
        line = line_of(text, m.start())
        close = matching_paren(text, m.end() - 1)
        end = None if close is None else first_of(text, ";{", close + 1)
        if close is None or end is None:
            message = f"{m.group(1)}: cannot find the end of this declaration"
            problems.append((path, line, message))
            continue
        if text[end] == "{":
            continue
        link = LINK_NAME.search(text[statement_start(text, m.start()) : m.start()])
        name = link.group(1) if link else m.group(1)
        tail = " ".join(text[close + 1 : end].split())
        ret: Typed
        if not tail:
            ret = ("()", "()")
        elif tail.startswith("->"):
            ret = (tail[2:].strip(), rust_type(tail[2:]))
        else:
            ret = (tail, None)
        params: list[Typed] = []
        for item in split_parameters(text[m.end() : close]):
            typed = RUST_PARAMETER.fullmatch(item)
            params.append((item, rust_type(typed.group(1)) if typed else None))
        decl = Decl(name, line, params, ret)
        if unreadable(decl):
            types = ", ".join(RUST_TYPES)
            message = f"{name}: cannot read its types; the check reads {types}, "
            problems.append((path, line, message + "*const and *mut"))
        externs.setdefault(name, []).append(decl)
    return externs


def compare(proto: Decl, decl: Decl, header_path: str, rust_path: str) -> list[str]:
    """Each way decl, a declaration in rust_path, differs from proto."""
    where = f"{header_path}:{proto.line}"
    if len(decl.params) != len(proto.params):
        count = len(decl.params)
        plural = "" if count == 1 else "s"
        return [f"{count} parameter{plural}, {where} has {len(proto.params)}"]
    differences = [
        f"parameter {i}: {where} has `{c_text}`, {rust_path} has `{r_text}`"
        for i, ((c_text, c_canon), (r_text, r_canon)) in enumerate(
            zip(proto.params, decl.params, strict=True), 1
        )
        if c_canon != r_canon
    ]
    if proto.ret[1] != decl.ret[1]:
        differences.append(f"returns `{decl.ret[0]}`, {where} returns `{proto.ret[0]}`")
    return differences


def cross_check(
    protos: dict[str, Decl],
    externs: dict[str, list[Decl]],
    header_path: str,
    rust_path: str,
) -> list[Problem]:
    """Every mismatch between the prototypes and the extern declarations."""
    problems: list[Problem] = []
    if not protos:
        problems.append((header_path, 0, "no ferrompi_* prototypes found"))
    for name, decls in externs.items():
        proto = protos.get(name)
        for decl in decls:
            if proto is None:
                message = f"{name}: no prototype in {header_path}"
                problems.append((rust_path, decl.line, message))
            elif not unreadable(proto) and not unreadable(decl):
                for difference in compare(proto, decl, header_path, rust_path):
                    problems.append((rust_path, decl.line, f"{name}: {difference}"))
    for name, proto in protos.items():
        unbound = name in UNBOUND_PROTOTYPES
        if name in externs and unbound:
            message = f"{name}: has an extern in {rust_path}; unlist it in "
            problems.append((header_path, proto.line, message + "UNBOUND_PROTOTYPES"))
        elif name not in externs and not unbound:
            message = f"{name}: no extern declaration in {rust_path}"
            problems.append((header_path, proto.line, message))
    script = str(SCRIPT.relative_to(REPO_ROOT))
    for name in sorted(set(UNBOUND_PROTOTYPES) - set(protos)):
        message = (
            f"{name}: in UNBOUND_PROTOTYPES, but {header_path} does not declare it"
        )
        problems.append((script, 0, message))
    return problems


def main(argv: list[str]) -> int:
    if not argv:
        header_path, rust_path, root = HEADER, RUST, REPO_ROOT
    elif len(argv) == 2:
        header_path, rust_path, root = argv[0], argv[1], Path()
    else:
        print("usage: check-ffi-signatures.py [HEADER RUST]", file=sys.stderr)
        return 2
    try:
        header_text = (root / header_path).read_text(encoding="utf-8")
        rust_text = (root / rust_path).read_text(encoding="utf-8")
    except OSError as e:
        print(f"check-ffi-signatures.py: {e}", file=sys.stderr)
        return 2

    problems: list[Problem] = []
    protos = read_header(header_text, header_path, problems)
    externs = read_rust(rust_text, rust_path, problems)
    problems += cross_check(protos, externs, header_path, rust_path)

    if problems:
        print(f"FAIL: {rust_path} does not declare the shims as {header_path} does:")
        print()
        for path, line, message in sorted(problems):
            print(f"{path}:{line}: {message}" if line else f"{path}: {message}")
        print()
        print(f"Give each extern in {rust_path} its prototype's parameter count")
        print("and types: int is c_int, int32_t is int32_t or i32, const T* is")
        print("*const T, T* is *mut T. A prototype with no Rust caller by design")
        print("goes in UNBOUND_PROTOTYPES in .github/scripts/check-ffi-signatures.py,")
        print("with the reason.")
        return 1

    count = sum(len(decls) for decls in externs.values())
    print(
        f"OK: the {len(protos)} ferrompi_* prototypes in {header_path} match "
        f"the {count} extern declarations in {rust_path}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
