# Godot RAG Assistant

A command-line assistant for the **Godot 4** game engine, built on the official documentation:

- **`ask`** answers questions about Godot, citing the exact documentation section every claim comes from.
- **`explain-error`** takes an error from Godot's Output panel or debugger and explains what happened, the likely cause and how to fix it. An **agent** decides what to look up in the docs before answering.

It is built **from scratch, without a RAG framework** (no LangChain / LlamaIndex): every step (parsing, chunking, embedding, retrieval, prompting, the agent loop) is plain Python that can be read in a few minutes. The LLM is **swappable**: Gemini, Claude or a local model through Ollama.

What retrieval finds for a question (real output, before any LLM is involved):

```text
$ python -m godot_rag search "how to connect a signal from code with arguments" -k 4

[1] 0.790  Signal > Description
    https://docs.godotengine.org/en/stable/classes/class_signal.html#description
[2] 0.782  Using signals > Connecting a signal via code
    https://docs.godotengine.org/en/stable/getting_started/step_by_step/signals.html#connecting-a-signal-via-code
[3] 0.777  Using signals > Connecting a signal in the editor
    https://docs.godotengine.org/en/stable/getting_started/step_by_step/signals.html#connecting-a-signal-in-the-editor
[4] 0.773  Object > Method Descriptions > connect
    https://docs.godotengine.org/en/stable/classes/class_object.html#class-object-method-connect
```

`ask` gives these excerpts to the LLM and returns an answer that cites them as [1]..[4].

## How it works

### 1. Ingest (once)

```text
godot-docs repo ──sparse clone──▶ .rst files ──parse──▶ sections ──chunk──▶ ~26k chunks ──embed──▶ Chroma
```

| Step | File | What it does |
|---|---|---|
| Download | `ingest/download.py` | Partial + sparse git clone: fetches only the `.rst` sources, not the images. |
| Parse | `ingest/parse.py` | Turns reStructuredText into clean text split by headings. In the class reference, **every method, property and signal becomes its own section** with its own URL anchor. |
| Chunk | `ingest/chunk.py` | Splits long sections on paragraph boundaries (~1500 chars, 200 overlap). Each chunk starts with its breadcrumb (`CharacterBody2D > Method Descriptions > is_on_floor`), which gives the embedding useful context. |
| Embed + store | `index.py` | `BAAI/bge-small-en-v1.5` (runs on CPU) → normalized vectors → local Chroma database with cosine distance. |

### 2. Ask (RAG)

```text
question ──embed──▶ top-k similar chunks ──▶ prompt (numbered excerpts + question) ──▶ LLM ──▶ answer with [n] citations
```

The prompt (`prompts.py`) tells the model to answer **only** from the excerpts, cite them, and say so when the docs don't contain the answer instead of guessing. The CLI then lists only the sources the answer actually cited.

### 3. Explain an error (agent)

```text
error log ──regex parser──▶ hints (category, classes, names, script:line)
                                 │
                                 ▼
             ┌──────────▶ LLM ──── final answer (cause + fix + [n] sources)
             │             │
             │        tool calls: search_docs(query) / lookup_class(class, topic)
             │             │
             └── numbered excerpts ◀── retriever
```

- `errors/parse_error.py` understands Godot's log formats (Output panel, debugger tab, script editor) and extracts the category (`null_instance`, `node_not_found`, `nonexistent_function`...), the classes and names involved, and where it happened. No LLM needed: try it with `--parse-only`.
- `errors/agent.py` gives the LLM two tools and runs the tool-calling loop (max 6 rounds). Excerpts keep a stable number across the whole conversation so citations stay correct.
- The loop is **provider-neutral**: `llm/base.py` defines a tiny interface (`chat(system, messages, tools)`), and each provider file translates to its own SDK.

## Setup

Requires Python 3.11+ and git.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -e ".[dev]"
copy .env.example .env            # then paste your API key in .env
python -m godot_rag ingest        # ~1 min download + several minutes of embedding on CPU
```

### Choosing the LLM

Set `llm.provider` in `config.yaml`, the `LLM_PROVIDER` env variable, or `--provider` per command.

| Provider | Needs | Notes |
|---|---|---|
| `gemini` (default) | `GEMINI_API_KEY` ([free key](https://aistudio.google.com/apikey)) | Free tier is enough for this project. |
| `claude` | `ANTHROPIC_API_KEY` | Best at following citation rules. Pay per use. |
| `ollama` | [Ollama](https://ollama.com) running + `ollama pull qwen3:8b` | Fully local and free; slower without a GPU. |

Model names are in `config.yaml`.

## Usage

```bash
# retrieval only: see what the search finds (free, no LLM)
python -m godot_rag search "signal connect with arguments"

# questions (any language; the docs are in English)
python -m godot_rag ask "¿Cómo conecto una señal desde código?"
python -m godot_rag ask "How do I load a scene at runtime?" --show-context
python -m godot_rag ask "What is a Tween?" --provider claude

# errors: a log file, the text itself, or '-' for stdin
python -m godot_rag explain-error examples/null_instance.log --code examples/player.gd
python -m godot_rag explain-error "Invalid call. Nonexistent function 'jump' in base 'CharacterBody2D'."
python -m godot_rag explain-error examples/node_not_found.log --parse-only
```

## Tests

```bash
pytest
```

The tests cover RST parsing and chunking, the log parser, and the agent loop with a scripted fake LLM and fake retriever, so they run offline in under a second.

## Project structure

```text
config.yaml            all settings (docs branch, chunk size, top-k, models)
src/godot_rag/
  config.py            loads config.yaml + .env
  ingest/              download.py, parse.py, chunk.py
  index.py             embeddings + Chroma
  retrieve.py          similarity search (whole docs or one class)
  llm/                 base.py (interface), gemini.py, claude.py, ollama.py
  prompts.py           every prompt in one place
  qa.py                question answering
  errors/              parse_error.py (regex), agent.py (tool-calling loop)
  cli.py               commands
tests/
examples/              sample Godot errors and scripts
```

## Limitations and next steps

- Only the English docs of the latest stable Godot 4 release are indexed.
- Pure semantic search can miss exact identifiers; a hybrid search (BM25 + embeddings) would help with method names.
- No automatic evaluation yet. Next step: a set of questions with expected sources to measure retrieval hit rate, citation accuracy and hallucinations per provider.
- A small web UI (Streamlit) for demos.
