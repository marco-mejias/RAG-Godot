# GODOT RAG 📚 - AI Documentation Assistant & Error-Explaining Agent

A command-line assistant for the **Godot 4** game engine, built on top of its official documentation.
Ask it a question in any language and it answers **citing the exact documentation section** each claim comes from. Paste an error from Godot's Output panel and an **AI agent** searches the docs on its own, then explains what happened, the likely cause and how to fix it.
Every answer goes through **three layers of verification**: a free check that every Godot API it mentions really exists, an optional second LLM (Claude) that fact-checks each claim against the cited sources, and an **MCP server** that plugs the whole thing into Claude Code and Claude Desktop.
Everything is built **from scratch, without a RAG framework** (no LangChain / LlamaIndex), and the language model is **swappable**: Gemini, Claude or a local model through Ollama. This project was a deep dive into Retrieval-Augmented Generation, embeddings and vector search, tool-calling agents, and how to keep an LLM grounded in real sources instead of letting it guess.

```text
> python -m godot_rag explain-error examples/null_instance.log --code examples/player.gd

Searches made by the agent:
  - search_docs(query='node not found get_node onready null instance')

**What happened**
The variable `sprite` is `null` when `_physics_process()` tries to set `sprite.flip_h` [1].

**Likely cause**
The shorthand `@onready var sprite = $Sprite` is equivalent to calling `get_node("Sprite")` [2].
If the child node has a different name (such as `Sprite2D`), `get_node()` fails and returns `null` [1].

**How to fix it**
@onready var sprite: Sprite2D = $Sprite2D      # match the exact name in the Scene dock
...

Sources:
  [1] Node > Method Descriptions > get_node
      https://docs.godotengine.org/en/stable/classes/class_node.html#class-node-method-get-node
  [2] Nodes and scene instances > Node paths > Syntactic sugar
      https://docs.godotengine.org/en/stable/tutorials/scripting/nodes_and_scene_instances.html#syntactic-sugar
```

## 🛠️ Technologies

* Python 3.11+
* Sentence-Transformers (`BAAI/bge-small-en-v1.5` embeddings, runs on CPU)
* ChromaDB (local vector database)
* Google Gemini API (`google-genai`)
* Anthropic Claude API (`anthropic`)
* Ollama (local open models)
* Model Context Protocol (MCP Python SDK)
* Structured outputs (JSON Schema) on all three providers
* Git sparse checkout
* pytest
* GitHub

---

## 🧠 How It Was Built

I chose the project and its main design options: indexing the Godot documentation, building the RAG pipeline by hand instead of using a framework, making the LLM provider swappable, and adding an agent for errors on top of plain Q&A. I tested it against real Godot questions and errors, and the problems found along the way shaped the final version, like Spanish questions finding nothing (solved with query rewriting) or the free Gemini models being overloaded (solved with a fallback chain).

I used **Claude (Anthropic's AI assistant, through Claude Code)** as a programming assistant: it wrote the code following those decisions and walked me through every part of it, and every change went through my review and testing.

---

## 💡 Core Elements

### 1. Documentation Ingestion (Parsing & Chunking)
The Godot docs are written in reStructuredText with many custom Sphinx extensions that standard parsers reject, so I wrote a dedicated parser that reads them the way a person would: it recognises headings, keeps prose and code, and strips the markup.
In the class reference, **every method, property and signal becomes its own chunk** with a link to its exact anchor (`...class_node.html#class-node-method-get-node`), so a search for one method finds that method, not a page with fifty others.
Chunks never cross a section boundary, are split on paragraphs (~1500 characters with 200 of overlap), and start with their breadcrumb (`CharacterBody2D > Method Descriptions > is_on_floor`), which gives the embedding model context. The result is **26,512 chunks** from 1,517 pages.

### 2. Semantic Search (Embeddings + Vector DB)
Each chunk is turned into a 384-number vector that represents its meaning, so two texts that say the same thing with different words end up close together. The vectors are normalized and stored in a local **ChromaDB** index with cosine distance.
Search can cover the whole documentation or be restricted to one class page, which the agent uses to check that a method really exists.

### 3. Grounded Answers with Citations (RAG)
For every question:
1. **Query rewriting:** the LLM turns the question into a short English search query (the embedding model only understands English, and short queries match better than conversational ones). So *"¿Cómo conecto una señal desde código?"* becomes `connect signal in GDScript`.
2. **Retrieval:** the 6 most similar chunks are fetched and numbered.
3. **Generation:** the LLM answers **only** from those excerpts, cites them as `[1]`, `[2]`..., replies in the user's language, and says so when the docs don't contain the answer instead of guessing.

Only the sources actually cited in the answer are shown at the end, each with its link.

### 4. Error-Explaining Agent (Tool Calling)
In plain Q&A the program decides what to search. Here **the LLM decides**: it gets two tools, `search_docs(query)` and `lookup_class(class_name, topic)`, and runs a loop of searching and reading (up to 6 rounds) before writing its diagnosis.
Before the agent starts, a deterministic **log parser** (regular expressions, no AI) reads Godot's three error formats (Output panel, debugger tab, script editor) and extracts the category (`null_instance`, `node_not_found`, `nonexistent_function`...), the classes and names involved, and the script and line. These facts are given to the agent as hints.
Every excerpt keeps a stable number for the whole conversation, so citations stay correct even after several searches.

### 5. Answer Verification in Three Layers
A cheap model writes the answer; the checking is done in layers, from free to thorough:

1. **API check (always on, free, no AI).** The class reference has one label per member and an "Inherits:" line per class, so the full Godot 4 API (1,063 classes, 16,000+ member names) is turned into a lookup table. Every class, method and property in the answer's code is checked against it, including inherited members. It catches the most common hallucinations: invented methods and **Godot 3 names** (`KinematicBody2D`, `.instance()`, `yield`), and suggests the Godot 4 replacement.
2. **LLM judge (`--verify`).** A second, stronger model (Claude by default) receives the question, the excerpts, the draft and the API check, and returns **structured JSON**: every claim marked as *supported*, *unsupported* or *contradicted* with the excerpts that back it, plus a corrected answer when needed. Checking a sentence against a paragraph is much easier than writing the answer, so a strong model as a judge is a cheap way to cut hallucinations.
3. **Corrective search.** If the judge reports that the excerpts miss something the question needs, it proposes a query; the system runs **one extra search**, writes a new draft with the added excerpts, and judges it again.

```text
API check: 11 Godot names checked, all exist in Godot 4.
Verification (gemini:gemini-3.5-flash): 6 claims - 6 supported, 0 unsupported, 0 contradicted
```

### 6. Claude Integration (MCP Server)
The same core is exposed as an **MCP server** (`godot_rag/mcp_server.py`), so Claude Code or Claude Desktop can use the Godot docs as tools while you work on a game. Here the roles flip: Claude is the LLM and decides when to search.

| Tool | What it does |
|---|---|
| `search_godot_docs(query)` | Semantic search over the whole documentation, with URLs. |
| `lookup_godot_class(class_name, topic)` | Searches one class page and returns its inheritance chain; answers "did you mean CharacterBody2D?" for Godot 3 names. |
| `check_godot_api(code)` | Layer 1 on any code, including **Claude's own** before it shows it to you. |
| `ask_godot_docs(question)` | A draft from the cheap generator plus its excerpts, which **Claude verifies** before answering (layer 2 done by Claude). |
| `parse_godot_error(log)` | The deterministic log parser. |

Inside Claude Code, Claude also sees your project files, so it can check the docs **and** your scenes (for example, the real name of a node) before proposing a fix.

---

## ⚙️ Architecture & Systems

The code follows one rule: **each module does one step of the pipeline**, and the LLM is hidden behind a small interface.

```text
INGEST (once):   godot-docs repo ──sparse clone──▶ .rst ──parse──▶ sections ──chunk──▶ embed ──▶ ChromaDB

ASK:             question ──rewrite──▶ English query ──search──▶ top-k excerpts ──▶ LLM ──▶ draft + [n] sources

EXPLAIN-ERROR:   error log ──regex parser──▶ hints ──▶ LLM ◀──┐
                                                       │      │ numbered excerpts
                                         tool calls: search_docs / lookup_class

VERIFY:          draft ──▶ API check (free) ──▶ judge LLM (--verify) ──▶ corrected answer
                                                     └─ missing info ─▶ extra search ─▶ new draft ─▶ judge

MCP:             Claude Code / Desktop ──tool calls──▶ mcp_server.py ──▶ same retrieval + checks
```

* **`ingest/`:** `download.py` (partial + sparse git clone: only the `.rst` sources, no images), `parse.py` (RST to sections with URL anchors), `chunk.py` (paragraph splitting with overlap).
* **`index.py` / `retrieve.py`:** Embeddings and the vector index, and similarity search over the whole docs or one class.
* **`llm/`:** A provider-neutral interface, `chat(system, messages, tools)` and `chat_json(system, messages, schema)`, with one adapter per provider. Each adapter translates roles, tool calls and tool results to its own SDK, uses the provider's native structured-output mode for JSON, and sends the provider's native response back unchanged (needed for Claude's and Gemini's thinking signatures).
* **`prompts.py`:** Every instruction given to the LLM, in one place.
* **`qa.py`:** Query rewriting, retrieval, generation, verification and citation extraction.
* **`errors/`:** `parse_error.py` (the log parser) and `agent.py` (the tool-calling loop, the same for every provider).
* **`verify/`:** `api_check.py` (layer 1, the API table) and `judge.py` (layer 2, the LLM judge).
* **`mcp_server.py`:** Layer 3, the MCP tools for Claude.
* **`cli.py`:** The `ingest`, `search`, `ask` and `explain-error` commands.

### Swappable LLM Providers

| Provider | Needs | Notes |
|---|---|---|
| `gemini` (default) | `GEMINI_API_KEY` ([free key](https://aistudio.google.com/apikey)) | Uses a **fallback chain** of models: if one is overloaded (503) or out of quota, the next one is tried. |
| `claude` | `ANTHROPIC_API_KEY` | Best at following the citation rules. Pay per use. |
| `ollama` | [Ollama](https://ollama.com) running + `ollama pull qwen3:8b` | Fully local and free; slower without a GPU. |

Choose it in `config.yaml`, with the `LLM_PROVIDER` environment variable, or per command with `--provider`.

### Tests
The tests cover RST parsing and chunking, the log parser, the API check, the judge (corrections and the extra search), and the whole agent loop using a **scripted fake LLM** and a fake retriever. They run offline in under a second, without spending any API quota. The common LLM interface is what makes swapping in a fake model possible. `tests/mcp_smoke.py` runs the MCP server end to end over the real protocol.

---

## 🧩 Design Decisions

* **No RAG framework:** Frameworks hide exactly what matters to tune: how text is split, what the prompt says, how citations are numbered. Writing it by hand keeps every step visible and explainable.
* **One chunk per class member:** The biggest boost to search precision. Method names are what people ask about.
* **Grounding over fluency:** The model must cite, and must say "I couldn't find it" when the excerpts don't answer the question.
* **Deterministic before generative:** The log parser is free, instant and predictable. The agent starts from reliable facts instead of reading a raw log.
* **Limits everywhere:** A maximum number of agent rounds, retries with backoff and request timeouts, so a bad run never loops or hangs.
* **Cheap generator, strong verifier:** Writing the answer is the hard part; checking it is easier and more reliable. A free model drafts and a stronger, different model judges, which costs much less than using the strong model for everything. Verification is opt-in (`--verify`) because it adds a call and a few seconds.
* **One core, two front doors:** A CLI pipeline I fully control (always cited, reproducible, can run locally) and an MCP server that puts the same retrieval inside the developer's assistant.

### Limitations & Next Steps
* Pure semantic search sometimes misses exact method names: a **hybrid search** (BM25 + embeddings) would fix it.
* The API check verifies names, not signatures: `move_and_slide(velocity)` (Godot 3 arguments) still passes.
* The judge only sees the retrieved excerpts: if the right one was never found, it can flag the gap but not know the answer.
* An **evaluation set** (questions with their expected sources) to measure retrieval hit rate, citation accuracy and hallucinations per provider.
* A multilingual embedding model, to avoid depending on query rewriting.
* A small web interface for demos.

---

## 🚀 Build & Installation Instructions

**Prerequisites:**
* Python 3.11 or newer
* Git
* A free Gemini API key (or a Claude key, or Ollama installed)

**Steps:**
1. Clone the repository and create a virtual environment:
   ```bash
   git clone https://github.com/marco-mejias/RAG-Godot.git
   cd RAG-Godot
   python -m venv .venv
   .venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
   pip install -e ".[dev]"
   ```
2. Copy `.env.example` to `.env` and paste your API key.
3. Build the index (downloads the docs, then embeds them; about 20 minutes on CPU, only once):
   ```bash
   python -m godot_rag ingest
   ```

**Usage:**
```bash
# see what the search finds (no LLM, free)
python -m godot_rag search "signal connect with arguments"

# ask in any language
python -m godot_rag ask "¿Cómo conecto una señal desde código?" --show-context

# explain an error: a log file, the error text itself, or '-' for stdin
python -m godot_rag explain-error examples/null_instance.log --code examples/player.gd
python -m godot_rag explain-error "Invalid call. Nonexistent function 'jump' in base 'CharacterBody2D'."
python -m godot_rag explain-error examples/node_not_found.log --parse-only

# fact-check the answer with a second model (Claude by default; needs ANTHROPIC_API_KEY)
python -m godot_rag ask "How do I instance a scene at runtime?" --verify --show-draft
python -m godot_rag ask "How do I instance a scene at runtime?" --verify --verifier gemini

# run the tests
pytest
```

**Connecting it to Claude (MCP):**

The install creates a `godot-rag-mcp` command inside `.venv`. Point your MCP client to it with its full path:

* **Claude Code:** copy `.mcp.json.example` to `.mcp.json` in the folder where you use Claude Code and fix the path, or run
  ```bash
  claude mcp add godot-docs -- C:\path\to\RAG-Godot\.venv\Scripts\godot-rag-mcp.exe
  ```
* **Claude Desktop:** add the same `"godot-docs"` entry from `.mcp.json.example` under `"mcpServers"` in `claude_desktop_config.json` (Settings → Developer → Edit Config) and restart the app.

The server loads the embedding model on the first call (a few seconds); after that each search takes milliseconds. `ask_godot_docs` uses the generator configured in `config.yaml` (Gemini by default), so it needs that provider's key in `.env`; the other tools need no API key at all.
