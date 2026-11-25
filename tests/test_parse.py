from godot_rag.ingest.chunk import chunk_page, split_text
from godot_rag.ingest.parse import clean_inline, parse_rst

BASE = "https://docs.godotengine.org/en/stable"

MANUAL = """\
.. _doc_using_character_body_2d:

Using CharacterBody2D/3D
========================

Introduction
------------

Godot offers several :ref:`collision objects <doc_physics_introduction>` like
:ref:`CharacterBody2D <class_CharacterBody2D>`.

.. note:: Read ``move_and_slide()`` first.

.. image:: img/body.webp
   :alt: a body

.. tabs::
 .. code-tab:: gdscript GDScript

    func _physics_process(delta):
        move_and_slide()

What is a character body?
~~~~~~~~~~~~~~~~~~~~~~~~~

It's a body moved by code.
"""

CLASS = """\
:github_url: hide

.. _class_Node:

Node
====

**Inherits:** :ref:`Object<class_Object>`

Base class for all scene objects.

Method Descriptions
-------------------

.. _class_Node_method_add_child:

.. rst-class:: classref-method

|void| **add_child**\\ (\\ node\\: :ref:`Node<class_Node>`\\ ) :ref:`🔗<class_Node_method_add_child>`

Adds a child node.

----

.. _class_Node_method_get_node:

.. rst-class:: classref-method

:ref:`Node<class_Node>` **get_node**\\ (\\ path\\: :ref:`NodePath<class_NodePath>`\\ ) |const|

Fetches a node.
"""


def test_clean_inline_strips_roles_and_literals():
    assert clean_inline(":ref:`collision objects <doc_physics>`") == "collision objects"
    assert clean_inline(":ref:`CharacterBody2D<class_CharacterBody2D>`") == "CharacterBody2D"
    assert clean_inline("call ``move_and_slide()``") == "call `move_and_slide()`"
    assert clean_inline("`Godot <https://godotengine.org>`__") == "Godot (https://godotengine.org)"


def test_manual_page_sections_and_anchors():
    page = parse_rst(MANUAL, "tutorials/physics/using_character_body_2d", BASE)
    assert page.title == "Using CharacterBody2D/3D"
    assert page.kind == "manual"
    intro, what = page.sections
    assert intro.path == ["Using CharacterBody2D/3D", "Introduction"]
    assert intro.anchor == "introduction"
    assert "collision objects like\nCharacterBody2D." in intro.text
    assert "Note: Read `move_and_slide()` first." in intro.text
    assert "[gdscript]" in intro.text and "move_and_slide()" in intro.text
    assert "img/body.webp" not in intro.text and ":alt:" not in intro.text
    assert what.path[-1] == "What is a character body?"
    assert what.anchor == "what-is-a-character-body"


def test_class_members_become_their_own_sections():
    page = parse_rst(CLASS, "classes/class_node", BASE)
    assert page.kind == "class"
    by_title = {s.title: s for s in page.sections}
    add_child = by_title["add_child"]
    assert add_child.path == ["Node", "Method Descriptions", "add_child"]
    assert add_child.anchor == "class-node-method-add-child"
    assert "void add_child(node: Node)" in add_child.text
    assert "🔗" not in add_child.text
    assert "Fetches a node." in by_title["get_node"].text
    assert "Adds a child node." not in by_title["get_node"].text


def test_chunk_urls_point_to_sections():
    page = parse_rst(CLASS, "classes/class_node", BASE)
    urls = {c.section: c.url for c in chunk_page(page, 1500, 200)}
    assert urls["Node > Method Descriptions > get_node"] == f"{BASE}/classes/class_node.html#class-node-method-get-node"
    assert urls["Node"] == f"{BASE}/classes/class_node.html"


def test_split_text_respects_size_and_overlaps():
    paragraphs = [f"paragraph {i} " + "x" * 80 for i in range(20)]
    chunks = split_text("\n\n".join(paragraphs), max_chars=300, overlap=100)
    assert all(len(c) <= 300 for c in chunks)
    # the last paragraph of one chunk is repeated at the start of the next
    for a, b in zip(chunks, chunks[1:]):
        assert a.split("\n\n")[-1] == b.split("\n\n")[0]
    assert "paragraph 19" in chunks[-1]
