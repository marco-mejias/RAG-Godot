"""Layer 1: check that the Godot APIs mentioned in an answer really exist in Godot 4.

No LLM involved. The class reference has one label per member
(`.. _class_Node_method_get_node:`) and an "Inherits:" line per class, so we can
build the full API table with regular expressions, then look up every class,
method and property that appears in the answer's code. It catches the most
common hallucinations: invented methods and Godot 3 names (KinematicBody2D,
instance(), yield...) that no longer exist.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

MEMBER_LABEL = re.compile(
    r"^\.\. _class_([A-Za-z0-9@_]+?)_(method|property|signal|constant|enum|theme_item|annotation|operator)_(\w+):",
    re.M,
)
INHERITS = re.compile(r"^\*\*Inherits:\*\*(.+)$", re.M)
REF_NAME = re.compile(r":ref:`([\w@]+)<class_")

# GDScript keywords and language-level names that are not in the class reference.
KEYWORDS = {
    "if", "elif", "else", "for", "while", "match", "break", "continue", "pass", "return", "class",
    "class_name", "extends", "is", "in", "as", "self", "signal", "func", "static", "const", "enum",
    "var", "await", "super", "not", "and", "or", "true", "false", "null", "new", "void", "breakpoint",
}
# Common Godot 3 names an LLM may still write, with their Godot 4 replacement.
GODOT3_RENAMES = {
    "KinematicBody2D": "CharacterBody2D", "KinematicBody": "CharacterBody3D", "Spatial": "Node3D",
    "Sprite": "Sprite2D", "AnimatedSprite": "AnimatedSprite2D", "Position2D": "Marker2D",
    "Position3D": "Marker3D", "MeshInstance": "MeshInstance3D", "Camera": "Camera3D", "Area": "Area3D",
    "RigidBody": "RigidBody3D", "StaticBody": "StaticBody3D", "CollisionShape": "CollisionShape3D",
    "RayCast": "RayCast3D", "Light2D": "PointLight2D", "VisibilityNotifier2D": "VisibleOnScreenNotifier2D",
    "Reference": "RefCounted", "File": "FileAccess", "Directory": "DirAccess", "GIProbe": "VoxelGI",
    "PoolStringArray": "PackedStringArray", "PoolVector2Array": "PackedVector2Array",
    "PoolIntArray": "PackedInt32Array", "PoolRealArray": "PackedFloat32Array", "PoolByteArray": "PackedByteArray",
    "instance": "instantiate", "yield": "await", "rand_range": "randf_range", "deg2rad": "deg_to_rad",
    "rad2deg": "rad_to_deg", "stepify": "snapped", "update": "queue_redraw", "linear_interpolate": "lerp",
    "empty": "is_empty", "change_scene": "change_scene_to_file", "funcref": "Callable",
    "is_network_master": "is_multiplayer_authority", "get_world": "get_world_3d",
}

DEFINES = re.compile(r"\b(?:func|var|const|signal|class_name|enum|class|for)\s+(\w+)")
PARAMS = re.compile(r"\bfunc\s+\w+\s*\(([^)]*)\)")


@dataclass
class ClassInfo:
    name: str
    parents: list[str]
    members: set[str]


@dataclass
class ApiIndex:
    classes: dict[str, ClassInfo]

    def __post_init__(self) -> None:
        self.all_members = set().union(*(c.members for c in self.classes.values())) if self.classes else set()
        self.globals = set()
        for scope in ("@globalscope", "@gdscript"):
            if scope in self.classes:
                self.globals |= self.classes[scope].members

    # --- lookups -----------------------------------------------------------------------
    def get(self, name: str) -> ClassInfo | None:
        return self.classes.get(name.lower())

    def members_of(self, name: str) -> set[str]:
        """Members of a class including the inherited ones."""
        info, out = self.get(name), set()
        while info:
            out |= info.members
            info = self.get(info.parents[0]) if info.parents else None
        return out

    # --- build / cache -----------------------------------------------------------------
    @classmethod
    def build(cls, classes_dir: Path) -> "ApiIndex":
        classes: dict[str, ClassInfo] = {}
        for path in classes_dir.glob("class_*.rst"):
            text = path.read_text(encoding="utf-8")
            title = next((l.strip() for l in text.splitlines()[:12] if l.strip() and not l.startswith((":", ".."))), "")
            name = title or path.stem.removeprefix("class_")
            inh = INHERITS.search(text)
            parents = REF_NAME.findall(inh.group(1)) if inh else []
            members = {m.group(3) for m in MEMBER_LABEL.finditer(text)}
            classes[name.lower()] = ClassInfo(name, parents, members)
        return cls(classes)

    def save(self, path: Path) -> None:
        data = {k: {"name": c.name, "parents": c.parents, "members": sorted(c.members)} for k, c in self.classes.items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ApiIndex":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls({k: ClassInfo(v["name"], v["parents"], set(v["members"])) for k, v in data.items()})

    @classmethod
    def from_settings(cls, settings) -> "ApiIndex":
        cache = settings.index_dir / "api_index.json"
        if cache.exists():
            return cls.load(cache)
        classes_dir = settings.raw_dir / "classes"
        if not classes_dir.exists():
            raise RuntimeError("The Godot docs are not downloaded yet. Run: python -m godot_rag ingest")
        index = cls.build(classes_dir)
        index.save(cache)
        return index


@dataclass
class ApiIssue:
    name: str               # what was written, e.g. "KinematicBody2D" or "CharacterBody2D.jump"
    kind: str               # "class" | "member" | "call"
    suggestion: str | None  # closest real name, if any

    def describe(self) -> str:
        what = {
            "class": "class not found in Godot 4",
            "member": "method/property not found in Godot 4",
            "call": "not a Godot function (fine if it's defined in your own code)",
        }[self.kind]
        bare = self.name.split(".")[-1].rstrip("()")
        if bare in GODOT3_RENAMES:
            return f"`{self.name}`: Godot 3 name, in Godot 4 it is `{GODOT3_RENAMES[bare]}`"
        hint = f" (did you mean `{self.suggestion}`?)" if self.suggestion else ""
        return f"`{self.name}`: {what}{hint}"


@dataclass
class ApiReport:
    checked: list[str] = field(default_factory=list)
    issues: list[ApiIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.kind != "call" for i in self.issues)

    def summary(self) -> str:
        if not self.checked:
            return "API check: no Godot APIs to check."
        if not self.issues:
            return f"API check: {len(self.checked)} Godot names checked, all exist in Godot 4."
        lines = [f"API check: {len(self.checked)} names checked, {len(self.issues)} not found:"]
        lines += [f"  - {i.describe()}" for i in self.issues]
        return "\n".join(lines)


def extract_code(text: str) -> list[str]:
    """GDScript code from fenced blocks and inline `code` (C# blocks are skipped)."""
    snippets = []
    for lang, body in re.findall(r"```([\w#+-]*)\n(.*?)```", text, flags=re.S):
        if lang.lower() not in ("csharp", "cs", "c#", "cpp", "c++", "bash", "sh", "json", "text"):
            snippets.append(body)
    without_blocks = re.sub(r"```.*?```", "", text, flags=re.S)
    snippets += re.findall(r"`([^`\n]+)`", without_blocks)
    return snippets


def _clean(code: str) -> str:
    code = re.sub(r"#.*", "", code)                           # comments
    code = re.sub(r"(\"[^\"\n]*\"|'[^'\n]*')", '""', code)   # strings ("node paths", signal names...)
    return re.sub(r"[$%][\w/.]+", "_node", code)              # $Path/To/Node, %UniqueNode


def _close(name: str, options) -> str | None:
    if name in GODOT3_RENAMES:
        return GODOT3_RENAMES[name]
    match = difflib.get_close_matches(name, list(options), n=1, cutoff=.75)
    return match[0] if match else None


def check_text(text: str, api: ApiIndex) -> ApiReport:
    report = ApiReport()
    seen: set[str] = set()
    code = "\n".join(_clean(s) for s in extract_code(text))
    defined = set(DEFINES.findall(code)) | KEYWORDS
    for params in PARAMS.findall(code):
        defined |= set(re.findall(r"(\w+)\s*(?::|=|,|$)", params))
    class_names = {c.name for c in api.classes.values()}

    def add(name: str, issue: ApiIssue | None) -> None:
        if name in seen:
            return
        seen.add(name)
        report.checked.append(name)
        if issue:
            report.issues.append(issue)

    # 1) class names in type positions: extends X, is X, as X, x: X, -> X, X.member, X(...)
    for name in re.findall(r"(?:\bextends|\bis|\bas|:|->)\s*([A-Z]\w*)|\b([A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9]*)(?=\s*[.(])", code):
        name = next(n for n in name if n)
        if name in defined or name.isupper():
            continue
        add(name, None if api.get(name) else ApiIssue(name, "class", _close(name, class_names)))
    # a lone class name in inline code, e.g. `CharacterBody2D`
    for snippet in extract_code(text):
        s = snippet.strip()
        if re.fullmatch(r"[A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9]*[23]?D?", s) and s not in defined:
            add(s, None if api.get(s) else ApiIssue(s, "class", _close(s, class_names)))

    # 2) Class.member: the member must exist in that class or its ancestors
    for cls_name, member in re.findall(r"\b([A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9]*)\.([A-Za-z_]\w*)", code):
        if not api.get(cls_name) or member in KEYWORDS:
            continue
        members = api.members_of(cls_name)
        key = f"{cls_name}.{member}"
        add(key, None if member in members else ApiIssue(key, "member", _close(member, members)))

    # 3) obj.member on anything else: must be a member of *some* Godot class
    for member in re.findall(r"(?<=[a-z0-9_\)\]])\.([a-z_]\w*)", code):
        if member in defined:
            continue
        add(f".{member}", None if member in api.all_members else
            ApiIssue(f".{member}", "member", _close(member, api.all_members)))

    # 4) bare calls foo(): Godot global/inherited function, or the user's own
    for name in re.findall(r"(?<![\w.$])([a-z_]\w*)\s*\(", code):
        if name in defined:
            continue
        known = name in api.all_members or name in api.globals
        kind = "member" if name in GODOT3_RENAMES else "call"     # a Godot 3 function is an error, not "maybe yours"
        add(f"{name}()", None if known else ApiIssue(f"{name}()", kind, _close(name, api.all_members)))
    return report
