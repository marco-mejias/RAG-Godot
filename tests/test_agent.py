"""Tests the agent loop with a scripted fake LLM and an in-memory fake retriever (no network)."""

from godot_rag.errors.agent import ErrorAgent
from godot_rag.llm import LLM, Message, Reply, ToolCall
from godot_rag.qa import answer_question
from godot_rag.retrieve import Hit


def hit(section: str, body: str) -> Hit:
    url = "https://docs.godotengine.org/en/stable/" + section.lower().replace(" > ", "#").replace(" ", "-")
    return Hit(f"{section}\n\n{body}", section.split(" > ")[0], section, url, "class", "p", 0.9)


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, k=None, where=None):
        self.calls.append(("search", query))
        return [hit("Node > get_node", "Fetches a node. Fails if the node does not exist.")]

    def search_class(self, class_name, topic="", k=None):
        self.calls.append(("class", class_name, topic))
        if class_name != "Node":
            return []
        return [hit("Node > _ready", "Called when the node and its children entered the scene tree.")]


class ScriptedLLM(LLM):
    provider, model = "fake", "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen: list[list[Message]] = []

    def chat(self, system, messages, tools=None):
        self.seen.append(list(messages))
        return self.replies.pop(0)


def test_agent_runs_tools_then_answers_with_sources():
    llm = ScriptedLLM([
        Reply("", [ToolCall("1", "search_docs", {"query": "node not found"}),
                   ToolCall("2", "lookup_class", {"class_name": "Node", "topic": "_ready"})]),
        Reply("The node path is wrong [1]. Use it in _ready [2].", []),
    ])
    retriever = FakeRetriever()
    diagnosis = ErrorAgent(llm, retriever, max_steps=4).run('ERROR: Node not found: "Player".')

    assert retriever.calls == [("search", "node not found"), ("class", "Node", "_ready")]
    assert [n for n, _ in diagnosis.sources] == [1, 2]
    assert diagnosis.sources[1][1].section == "Node > _ready"
    assert len(diagnosis.steps) == 2
    # second LLM call saw both tool results, linked to their calls
    tool_msgs = [m for m in llm.seen[1] if m.role == "tool"]
    assert [m.tool_call_id for m in tool_msgs] == ["1", "2"]
    assert "[1] Node > get_node" in tool_msgs[0].content
    # the parsed hints were included in the first prompt
    assert "node_not_found" in llm.seen[0][0].content


def test_agent_repeated_excerpt_keeps_its_number_and_unknown_class_is_reported():
    llm = ScriptedLLM([
        Reply("", [ToolCall("a", "search_docs", {"query": "x"})]),
        Reply("", [ToolCall("b", "search_docs", {"query": "y"}),
                   ToolCall("c", "lookup_class", {"class_name": "Nope"})]),
        Reply("Answer [1].", []),
    ])
    agent = ErrorAgent(llm, FakeRetriever(), max_steps=4)
    diagnosis = agent.run("Invalid call.")
    results = [m.content for m in llm.seen[2] if m.role == "tool"]
    assert "already shown" in results[1]
    assert "No class named 'Nope'" in results[2]
    assert [n for n, _ in diagnosis.sources] == [1]


def test_agent_stops_at_step_limit_and_asks_for_final_answer():
    loop_forever = [Reply("", [ToolCall(str(i), "search_docs", {"query": "q"})]) for i in range(3)]
    llm = ScriptedLLM(loop_forever + [Reply("Best guess.", [])])
    diagnosis = ErrorAgent(llm, FakeRetriever(), max_steps=3).run("ERROR: something")
    assert diagnosis.text == "Best guess."
    assert "search limit" in llm.seen[-1][-1].content


def test_qa_keeps_only_cited_sources():
    llm = ScriptedLLM([Reply("Use get_node [1].", [])])
    answer = answer_question("How do I get a node?", FakeRetriever(), llm)
    assert [n for n, _ in answer.sources] == [1]
    assert "Question: How do I get a node?" in llm.seen[0][0].content
