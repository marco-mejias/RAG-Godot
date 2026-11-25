"""Extracts structured facts from a Godot error log, without any LLM.

Understands the formats Godot 4 prints:

Output panel:
    SCRIPT ERROR: Invalid access to property or key 'position' on a base object of type 'null instance'.
              at: _process (res://player.gd:12)
    ERROR: Node not found: "Player" (relative to "/root/Main").
       at: get_node (scene/main/node.cpp:1792)

Debugger > Errors tab (copied):
    E 0:00:00:0612   player.gd:12 @ _ready(): Node not found: "Player" (relative to "/root/Main").
      <C++ Source>   scene/main/node.cpp:1792 @ get_node()

Debugger stop message / script editor:
    Invalid call. Nonexistent function 'jump' in base 'CharacterBody2D'.
    Line 5: Identifier "speed" not declared in the current scope.

These facts are given to the agent as hints, and they make better search
queries than the raw log (e.g. "CharacterBody2D" + "jump").
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

HEADER_RE = re.compile(
    r"^(?P<sev>USER SCRIPT ERROR|SCRIPT ERROR|USER ERROR|ERROR|USER WARNING|SCRIPT WARNING|WARNING):\s*(?P<msg>.+)$"
)
DEBUGGER_RE = re.compile(
    r"^(?P<sev>[EW])\s+\d+:\d{2}:\d{2}:\d+\s+(?P<script>\S+?):(?P<line>\d+)\s+@\s+(?P<func>[\w.]+)\(\):\s*(?P<msg>.+)$"
)
EDITOR_LINE_RE = re.compile(r"^Line (?P<line>\d+):\s*(?P<msg>.+)$")
AT_RE = re.compile(r"^at:\s*(?P<func>[\w:.<>]+)\s*\((?P<path>.+):(?P<line>\d+)\)")
CPP_SOURCE_RE = re.compile(r"^<C\+\+ Source>\s*(?P<path>\S+):(?P<line>\d+)\s*@\s*(?P<func>[\w:]+)\(\)")
STACK_RE = re.compile(r"^<Stack Trace>\s*(?P<script>\S+?):(?P<line>\d+)\s*@\s*(?P<func>[\w.]+)\(\)")
QUOTED_RE = re.compile(r"""['"]([^'"]{1,80})['"]""")

# First match wins, so the most specific categories go first.
CATEGORIES: list[tuple[str, re.Pattern]] = [
    ("null_instance", re.compile(r"null instance|null value|on a null|base 'Nil'|previously freed", re.I)),
    ("node_not_found", re.compile(r"Node not found|get_node.*not found", re.I)),
    ("parse_error", re.compile(r"Parse Error|Parser Error|Expected .* after|Unexpected ", re.I)),
    ("undeclared_identifier", re.compile(r"not declared in the current scope|not found in base self", re.I)),
    ("nonexistent_function", re.compile(r"Nonexistent function|Invalid call\.|Function .* not found", re.I)),
    ("invalid_property", re.compile(r"Invalid (get|set) index|Invalid access to property|Invalid assignment of property", re.I)),
    ("type_mismatch", re.compile(r"Invalid type|Cannot convert|value of type .* to a variable of type|Invalid operands|Invalid argument", re.I)),
    ("out_of_bounds", re.compile(r"out of bounds|Index .* out of", re.I)),
    ("signal", re.compile(r"signal", re.I)),
    ("resource_load", re.compile(r"Failed loading resource|Cannot open file|Resource file not found|load\(\)", re.I)),
]

SEVERITY = {"E": "error", "W": "warning"}


@dataclass
class GodotError:
    message: str
    severity: str = "error"                # error | warning | script_error | parse_error
    script: str | None = None              # res://player.gd
    line: int | None = None
    function: str | None = None            # function in the user's script
    engine_source: str | None = None       # scene/main/node.cpp:1792 (get_node)
    category: str | None = None
    classes: list[str] = field(default_factory=list)   # Godot classes mentioned
    names: list[str] = field(default_factory=list)     # methods, properties, identifiers, node names

    def summary(self) -> str:
        parts = [f"{self.severity}: {self.message}"]
        if self.script:
            where = f"{self.script}" + (f":{self.line}" if self.line else "")
            parts.append(f"in {where}" + (f" (function {self.function})" if self.function else ""))
        if self.engine_source:
            parts.append(f"raised by engine code {self.engine_source}")
        if self.category:
            parts.append(f"category: {self.category}")
        if self.classes:
            parts.append("classes: " + ", ".join(self.classes))
        if self.names:
            parts.append("names: " + ", ".join(self.names))
        return " | ".join(parts)

    def search_query(self) -> str:
        """A compact query for semantic search."""
        return " ".join([*self.classes, *self.names[:3], self.message])[:300]


def _looks_like_class(word: str, known_classes: set[str] | None) -> bool:
    if known_classes is not None:
        return word.lower() in known_classes
    return bool(re.fullmatch(r"[A-Z][A-Za-z0-9]*[a-z0-9][A-Za-z0-9]*", word))


def _enrich(err: GodotError, known_classes: set[str] | None) -> None:
    for name, pattern in CATEGORIES:
        if pattern.search(err.message):
            err.category = name
            break
    if err.severity != "parse_error" and re.search(r"Parse(r)? Error", err.message, re.I):
        err.severity = "parse_error"
    for quoted in QUOTED_RE.findall(err.message):
        quoted = quoted.strip()
        if quoted in ("null instance", "Nil") or " " in quoted:
            continue
        target = err.classes if _looks_like_class(quoted, known_classes) else err.names
        if quoted not in target:
            target.append(quoted)


def parse_godot_errors(text: str, known_classes: Iterable[str] | None = None) -> list[GodotError]:
    known = {c.lower() for c in known_classes} if known_classes is not None else None
    errors: list[GodotError] = []
    current: GodotError | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if m := HEADER_RE.match(line):
            sev = m.group("sev")
            severity = "warning" if "WARNING" in sev else "script_error" if "SCRIPT" in sev else "error"
            current = GodotError(m.group("msg").strip(), severity)
            errors.append(current)
        elif m := DEBUGGER_RE.match(line):
            current = GodotError(
                m.group("msg").strip(), SEVERITY[m.group("sev")],
                script=m.group("script"), line=int(m.group("line")), function=m.group("func"),
            )
            errors.append(current)
        elif m := EDITOR_LINE_RE.match(line):
            current = GodotError(m.group("msg").strip(), "script_error", line=int(m.group("line")))
            errors.append(current)
        elif current is not None and (m := AT_RE.match(line)):
            path, func, lineno = m.group("path"), m.group("func"), int(m.group("line"))
            if path.endswith((".gd", ".cs")):
                current.script, current.line = path, lineno
                if "::" not in func:     # "GDScript::reload" is the engine, not the user's function
                    current.function = func
            else:
                current.engine_source = f"{path}:{lineno} ({func})"
        elif current is not None and (m := CPP_SOURCE_RE.match(line)):
            current.engine_source = f"{m.group('path')}:{m.group('line')} ({m.group('func')})"
        elif current is not None and (m := STACK_RE.match(line)):
            current.script = current.script or m.group("script")
            current.line = current.line or int(m.group("line"))
            current.function = current.function or m.group("func")
        elif current is None or line.startswith(("Invalid", "Attempt", "Cannot", "Parse", "Parser", "Trying")):
            # A bare message, e.g. copied from the debugger's stop banner.
            current = GodotError(line)
            errors.append(current)

    for err in errors:
        _enrich(err, known)

    # The same error is often printed several times (e.g. every frame): keep one of each.
    unique: dict[tuple, GodotError] = {}
    for err in errors:
        unique.setdefault((err.message, err.script, err.line), err)
    return list(unique.values())
