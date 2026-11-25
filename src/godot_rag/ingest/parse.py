"""Converts a Godot docs .rst file into plain text split by sections.

We don't use docutils/Sphinx: the Godot docs rely on many custom Sphinx
directives and roles that a plain RST parser rejects. Instead we do what a
human reader would: recognise headings, keep the prose and the code, and
strip the markup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

ADORNMENT_CHARS = set("=-~^\"'`*+#:.")

# Directives whose whole block (options + indented content) is dropped.
SKIP_BLOCK_DIRECTIVES = {
    "toctree", "image", "figure", "raw", "video", "only", "rst-class",
    "highlight", "container", "meta", "index",
}
# Directives rendered as a short label, keeping their content.
LABEL_DIRECTIVES = {
    "note": "Note:", "warning": "Warning:", "tip": "Tip:", "important": "Important:",
    "seealso": "See also:", "caution": "Caution:", "danger": "Danger:",
    "attention": "Attention:", "hint": "Hint:", "deprecated": "Deprecated:",
    "admonition": "", "table": "", "list-table": "", "csv-table": "", "tabs": "",
}

DIRECTIVE_RE = re.compile(r"^(\s*)\.\.\s+([\w-]+)::\s*(.*)$")
LABEL_RE = re.compile(r"^\.\.\s+_([^:]+):\s*$")
COMMENT_RE = re.compile(r"^\.\.(\s|$)")
OPTION_RE = re.compile(r"^\s+:[\w-]+:")
ROLE_RE = re.compile(r":[\w-]+(?::[\w-]+)?:`([^`]+)`")
LINK_RE = re.compile(r"`([^`<]+?)\s*<([^>]+)>`__?")
SUBST_RE = re.compile(r"\|([\w-]+)\|")
# class_CharacterBody3D_method_move_and_slide -> ("method", "move_and_slide")
MEMBER_LABEL_RE = re.compile(
    r"^class_\w+?_(method|property|signal|theme_item|annotation|operator)_(\w+)$"
)


@dataclass
class Section:
    title: str
    path: list[str]          # breadcrumb: page title > ... > this section
    anchor: str              # HTML id on the published page ("" = top of page)
    text: str


@dataclass
class Page:
    rel_path: str            # e.g. "tutorials/physics/using_character_body_2d"
    title: str
    url: str
    kind: str                # "class" (API reference) or "manual"
    sections: list[Section] = field(default_factory=list)


def _is_adornment(line: str) -> bool:
    s = line.rstrip()
    return len(s) >= 3 and s[0] in ADORNMENT_CHARS and set(s) == {s[0]}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def clean_inline(text: str) -> str:
    """Strip inline RST markup, keeping the readable part."""

    def role(m: re.Match) -> str:
        inner = m.group(1)
        # :ref:`Text <target>` -> Text ; :ref:`target` -> target ; :ref:`~a.b.C` -> C
        if "<" in inner and inner.endswith(">"):
            inner = inner[: inner.index("<")].strip()
        return inner.lstrip("~")

    text = ROLE_RE.sub(role, text)
    text = LINK_RE.sub(lambda m: f"{m.group(1).strip()} ({m.group(2)})", text)
    text = SUBST_RE.sub(lambda m: m.group(1), text)
    text = text.replace("\\ ", "").replace("``", "`").replace("🔗", "")
    text = re.sub(r"\\(.)", r"\1", text)   # RST escapes: "value\: bool" -> "value: bool"
    text = re.sub(r"(?<![\w*])\*\*(.+?)\*\*", r"\1", text)
    return text


def _clean_body(lines: list[str]) -> str:
    out: list[str] = []
    skip_indent: int | None = None   # inside a dropped directive block
    for line in lines:
        indent = len(line) - len(line.lstrip())
        if skip_indent is not None:
            if not line.strip() or indent > skip_indent:
                continue
            skip_indent = None

        m = DIRECTIVE_RE.match(line)
        if m:
            lead, name, arg = len(m.group(1)), m.group(2), m.group(3).strip()
            if name in SKIP_BLOCK_DIRECTIVES:
                skip_indent = lead
            elif name in ("code-block", "code", "sourcecode", "code-tab"):
                # ".. code-tab:: gdscript GDScript" -> "[gdscript]"
                out.append(f"[{arg.split()[0] if arg else 'code'}]")
            elif name in LABEL_DIRECTIVES:
                label = LABEL_DIRECTIVES[name]
                out.append(" " * lead + (f"{label} {clean_inline(arg)}".strip() if label or arg else ""))
            else:
                out.append(clean_inline(arg))
            continue
        if OPTION_RE.match(line):
            continue
        if COMMENT_RE.match(line.strip()) or line.startswith(":github_url:"):
            continue
        stripped = line.strip()
        if stripped and set(stripped) <= set("+-=|") and len(stripped) >= 3:
            continue  # table borders and "----" separators
        if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
            cells = [clean_inline(c.strip()) for c in stripped.strip("|").split("|")]
            out.append(" — ".join(c for c in cells if c))
            continue
        out.append(clean_inline(line.rstrip()))

    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_rst(source: str, rel_path: str, base_url: str) -> Page:
    lines = source.splitlines()
    kind = "class" if rel_path.startswith("classes/") else "manual"
    url = f"{base_url}/{rel_path}.html"

    levels: list[tuple[str, bool]] = []   # heading styles in order of first use
    stack: list[str] = []                 # current breadcrumb
    sections: list[Section] = []
    body: list[str] = []
    current_title = ""
    current_path: list[str] = []
    current_anchor = ""
    pending_label = ""
    page_title = ""

    def flush() -> None:
        text = _clean_body(body)
        if text:
            sections.append(Section(current_title or page_title, current_path or [page_title], current_anchor, text))
        body.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        label = LABEL_RE.match(line.strip())
        if label:
            member = MEMBER_LABEL_RE.match(label.group(1)) if kind == "class" else None
            if member:
                # In the class reference every method/property/signal has its own label
                # and URL anchor: make it its own section so it becomes its own chunk.
                flush()
                current_title = member.group(2)
                current_path = list(stack) + [current_title]
                current_anchor = _slug(label.group(1))
                pending_label = ""
            else:
                pending_label = label.group(1)
            i += 1
            continue

        heading: str | None = None
        style: tuple[str, bool] | None = None
        # Overline + title + underline
        if _is_adornment(line) and nxt.strip() and i + 2 < len(lines) and _is_adornment(lines[i + 2]) \
                and lines[i + 2].strip()[0] == line.strip()[0]:
            heading, style, consumed = nxt.strip(), (line.strip()[0], True), 3
        # Title + underline (underline at least as long as the title)
        elif line.strip() and not line.startswith(" ") and _is_adornment(nxt) \
                and len(nxt.rstrip()) >= len(line.rstrip()) and not _is_adornment(line):
            heading, style, consumed = line.strip(), (nxt.strip()[0], False), 2

        if heading is None:
            # A label only names the heading right below it; any real content in between cancels it.
            if line.strip() and not line.strip().startswith(".. rst-class::"):
                pending_label = ""
            body.append(line)
            i += 1
            continue

        flush()
        if style not in levels:
            levels.append(style)
        depth = levels.index(style)
        heading = clean_inline(heading)
        stack[:] = stack[:depth] + [heading]
        if not page_title:
            page_title = heading
            current_anchor = ""
        else:
            # Sphinx turns an explicit label above a heading into its HTML id.
            current_anchor = _slug(pending_label) if pending_label else _slug(heading)
        current_title = heading
        current_path = list(stack)
        pending_label = ""
        i += consumed

    flush()
    return Page(rel_path, page_title or rel_path, url, kind, sections)


def parse_file(path: Path, docs_root: Path, base_url: str) -> Page:
    rel = path.relative_to(docs_root).with_suffix("").as_posix()
    return parse_rst(path.read_text(encoding="utf-8"), rel, base_url)
