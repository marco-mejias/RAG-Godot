from godot_rag.errors.parse_error import parse_godot_errors

KNOWN = {"node", "node2d", "characterbody2d", "label", "sprite2d"}


def test_output_panel_script_error_with_location():
    log = """\
SCRIPT ERROR: Invalid access to property or key 'position' on a base object of type 'null instance'.
          at: _process (res://player.gd:12)
"""
    [err] = parse_godot_errors(log, KNOWN)
    assert err.severity == "script_error"
    assert err.category == "null_instance"
    assert (err.script, err.line, err.function) == ("res://player.gd", 12, "_process")
    assert err.names == ["position"]


def test_engine_error_keeps_engine_source_and_node_name():
    log = """\
ERROR: Node not found: "Player" (relative to "/root/Main").
   at: get_node (scene/main/node.cpp:1792)
"""
    [err] = parse_godot_errors(log, KNOWN)
    assert err.category == "node_not_found"
    assert err.engine_source == "scene/main/node.cpp:1792 (get_node)"
    assert "Player" in err.names and err.classes == []   # a node name, not a class


def test_debugger_tab_format():
    log = """\
E 0:00:00:0612   player.gd:8 @ _ready(): Invalid call. Nonexistent function 'jump' in base 'CharacterBody2D'.
  <C++ Source>   core/variant/variant_call.cpp:1234 @ callp()
  <Stack Trace>  player.gd:8 @ _ready()
"""
    [err] = parse_godot_errors(log, KNOWN)
    assert err.severity == "error"
    assert err.category == "nonexistent_function"
    assert (err.script, err.line, err.function) == ("player.gd", 8, "_ready")
    assert err.classes == ["CharacterBody2D"] and err.names == ["jump"]


def test_parse_error_in_editor():
    [err] = parse_godot_errors('Line 5: Identifier "speed" not declared in the current scope.', KNOWN)
    assert err.category == "undeclared_identifier"
    assert err.line == 5 and err.names == ["speed"]


def test_bare_message_and_duplicates():
    log = "Attempt to call function 'set_text' in base 'null instance' on a null instance.\n" * 3
    errors = parse_godot_errors(log, KNOWN)
    assert len(errors) == 1
    assert errors[0].category == "null_instance"


def test_without_known_classes_uses_pascal_case_heuristic():
    [err] = parse_godot_errors("Invalid call. Nonexistent function 'jump' in base 'CharacterBody2D'.")
    assert err.classes == ["CharacterBody2D"]
