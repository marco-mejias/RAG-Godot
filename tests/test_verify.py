"""Layer 1 (API check) with a tiny hand-made API table, and layer 2 (judge) with a scripted LLM."""

import json

from godot_rag.llm import LLM, Reply
from godot_rag.qa import answer_question
from godot_rag.retrieve import Hit
from godot_rag.verify.api_check import ApiIndex, ClassInfo, check_text

API = ApiIndex({
    "object": ClassInfo("Object", [], {"connect", "emit_signal"}),
    "node": ClassInfo("Node", ["Object"], {"get_node", "add_child", "queue_free", "get_tree"}),
    "characterbody2d": ClassInfo("CharacterBody2D", ["Node"], {"move_and_slide", "is_on_floor", "velocity"}),
    "vector2": ClassInfo("Vector2", [], {"x", "y"}),
    "packedscene": ClassInfo("PackedScene", [], {"instantiate"}),
    "@globalscope": ClassInfo("@GlobalScope", [], {"print"}),
    "@gdscript": ClassInfo("@GDScript", [], {"preload", "load"}),
})


def issues(text):
    return {i.name: i for i in check_text(text, API).issues}


def test_godot3_names_are_flagged_with_their_godot4_replacement():
    found = issues("```gdscript\nextends KinematicBody2D\nvar s = scene.instance()\nyield(get_tree(), \"idle_frame\")\n```")
    assert "Godot 3 name, in Godot 4 it is `CharacterBody2D`" in found["KinematicBody2D"].describe()
    assert "`instantiate`" in found[".instance"].describe()
    assert found["yield()"].kind == "member"          # a Godot 3 function is an error, not "maybe yours"


def test_inherited_members_pass_and_invented_ones_fail():
    found = issues("Call `CharacterBody2D.get_node` and `CharacterBody2D.jump`.")
    assert "CharacterBody2D.get_node" not in found          # inherited from Node
    assert found["CharacterBody2D.jump"].kind == "member"


def test_valid_code_has_no_issues_and_user_code_is_not_an_error():
    text = """```gdscript
extends CharacterBody2D
@onready var sprite: Sprite2D = $Sprite2D
func take_damage(amount):
    print("hit")
func _physics_process(delta):
    velocity.y += 10
    move_and_slide()
    if is_on_floor():
        take_damage(1)
        heal(2)
```"""
    report = check_text(text, ApiIndex({**API.classes, "sprite2d": ClassInfo("Sprite2D", ["Node"], set())}))
    names = {i.name: i for i in report.issues}
    assert "take_damage()" not in names                      # defined in the snippet
    assert names["heal()"].kind == "call"                    # unknown, but may be the user's own function
    assert report.ok                                         # "call" issues alone don't fail the check


def test_csharp_blocks_and_strings_are_ignored():
    text = '```csharp\nGetNode<Sprite2D>("A").Foo();\n```\n```gdscript\nprint("KinematicBody2D.fake()")\n```'
    assert check_text(text, API).issues == []


def test_scene_tree_drawings_file_names_and_node_names_are_not_errors():
    text = """```
Main
└─ HUD (MarginContainer)
   └─ HealthLabel (Label)
```
Attach `player.gd` and rename `HealthLabel`."""
    report = check_text(text, API)
    assert report.ok
    assert {i.kind for i in report.issues} <= {"own"}          # node names: a note, not an error


def test_class_name_comes_from_its_label_even_with_a_meta_block(tmp_path):
    (tmp_path / "class_label.rst").write_text(
        ":github_url: hide\n\n.. meta::\n\t:keywords: text\n\n.. _class_Label:\n\nLabel\n=====\n\n"
        "**Inherits:** :ref:`Control<class_Control>` **<** :ref:`Node<class_Node>`\n\n"
        ".. _class_Label_property_text:\n", encoding="utf-8")
    api = ApiIndex.build(tmp_path)
    assert api.get("Label").parents == ["Control", "Node"]
    assert api.get("Label").members == {"text"}


# --- layer 2 -------------------------------------------------------------------------------

def hit(section, body):
    return Hit(f"{section}\n\n{body}", section.split(" > ")[0], section, f"https://docs/{section}", "class", "p", .9)


class Retriever:
    def __init__(self):
        self.queries = []

    def search(self, query, k=None, where=None):
        self.queries.append(query)
        if query == "packedscene instantiate":
            return [hit("PackedScene > instantiate", "Creates the node hierarchy of the scene.")]
        return [hit("Node > add_child", "Adds a child node.")]


class ScriptedLLM(LLM):
    provider, model = "fake", "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def chat(self, system, messages, tools=None):
        self.prompts.append((system, messages[-1].content))
        return Reply(self.replies.pop(0), [])


def verdict(claims, corrected="", search_again=""):
    return json.dumps({"claims": claims, "corrected_answer": corrected, "search_again": search_again})


def test_judge_corrects_unsupported_claims_and_sees_the_api_check():
    generator = ScriptedLLM(["add child node", "Use `scene.instance()` then `add_child()` [1]."])
    judge_llm = ScriptedLLM([verdict(
        [{"claim": "Use scene.instance()", "verdict": "unsupported", "citations": [], "note": "Godot 3 name"},
         {"claim": "then add_child()", "verdict": "supported", "citations": [1], "note": ""}],
        corrected="Use `add_child()` to add the node [1].")])
    answer = answer_question("How do I add a scene?", Retriever(), generator, api=API, verifier=judge_llm)

    assert answer.draft.startswith("Use `scene.instance()`")
    assert answer.text == "Use `add_child()` to add the node [1]."      # the judge's correction wins
    assert answer.verdict.count("unsupported") == 1 and answer.verdict.needs_fix
    assert "Godot 3 name" in judge_llm.prompts[0][1]                      # layer 1 result reached the judge
    assert [n for n, _ in answer.sources] == [1]


def test_judge_can_trigger_one_extra_search_then_judges_again():
    generator = ScriptedLLM(["add scene", "Use add_child [1].", "Use instantiate [2] and add_child [1]."])
    judge_llm = ScriptedLLM([
        verdict([{"claim": "Use add_child", "verdict": "supported", "citations": [1], "note": ""}],
                corrected="", search_again="packedscene instantiate"),
        verdict([{"claim": "Use instantiate", "verdict": "supported", "citations": [2], "note": ""},
                 {"claim": "add_child", "verdict": "supported", "citations": [1], "note": ""}]),
    ])
    retriever = Retriever()
    answer = answer_question("How do I add a scene?", retriever, generator, api=API, verifier=judge_llm)

    assert retriever.queries == ["add scene", "packedscene instantiate"]
    assert len(answer.context) == 2 and answer.verdict.rounds == 2
    assert answer.text == "Use instantiate [2] and add_child [1]."      # the new draft, now fully supported
    assert not answer.verdict.needs_fix


def test_without_verifier_only_the_free_api_check_runs():
    generator = ScriptedLLM(["q", "Call `CharacterBody2D.jump()` [1]."])
    answer = answer_question("Jump?", Retriever(), generator, api=API)
    assert answer.verdict is None
    assert not answer.api_report.ok
