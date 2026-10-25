# LankaLaw Intelligence Engine: Project Guide

How to set up the project, start the server, load the knowledge base and use it. For the
design rationale and layer-by-layer notes see [README.md](README.md).

## 1. What this is

An agentic RAG system for Sri Lankan law. A question goes to a LangChain agent that decides
what to retrieve, searches a local vector store and a local knowledge graph, grades its own
coverage after every search round, and answers with numbered citations. Everything runs on
your machine except the LLM and embedding calls (Gemini, with Groq as fallback).

```
Browser UI  ──>  FastAPI (app/)  ──>  AgenticRAG (agent/rag)  ──>  tools
                       │                                           ├─ search_knowledge_base   -> ChromaDB (data/chroma)
                       │                                           └─ explore_knowledge_graph -> Kùzu    (data/graph)
                       └──>  IngestionService (ingestion/)  ──> chunks -> embeddings -> ChromaDB
                                                             └─> entities/relations (LLM) -> Kùzu
```

Three top-level packages:

| Package | Role |
|---|---|
| `agent/` | Reads the knowledge base. `agent/rag` is the agent; `agent/services/llm` the provider chain; `agent/services/db` the vector and graph stores. Never imports `ingestion/`. |
| `ingestion/` | The only writer to the knowledge base: loaders, chunker, graph extractor, `IngestionService`. |
| `app/` | FastAPI app, routes, UI (`app/static/index.html`), and `bootstrap.py` which wires the two sides together from `.env`. |

## 2. Prerequisites

- Python 3.13 (3.11+ should work; 3.13 is what the project is developed and tested on).
- A Gemini API key. Required: it is used for embeddings as well as the primary chat model.
- A Groq API key. Optional: enables cross-provider fallback when every Gemini model fails.
- No Docker, no database server. ChromaDB and Kùzu are embedded and store files under `data/`.
- Windows, macOS and Linux all work. Commands below are shown for a POSIX shell and PowerShell where they differ.

## 3. Install

```bash
git clone <repo-url> lankalaw
cd lankalaw

python -m venv .venv
source .venv/bin/activate          # PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

`requirements.txt` pulls in FastAPI, uvicorn, ChromaDB, Kùzu, LangChain 1.x, LangGraph,
google-genai, httpx, pypdf and pytest.

## 4. Configure `.env`

```bash
cp .env.example .env               # PowerShell: Copy-Item .env.example .env
```

Then fill in the keys. Everything else has a sensible default. The variables that matter:

| Variable | Default | What it does |
|---|---|---|
| `GEMINI_API_KEY` | (empty) | Required. Embeddings and primary chat model. |
| `GROQ_API_KEY` | (empty) | Optional. Enables Groq models as fallback. |
| `LLM_PROVIDER` | `gemini` | Which provider's models are tried first. |
| `GEMINI_MODEL`, `GEMINI_FALLBACK_MODELS` | `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite,gemini-3.5-flash` | Gemini chain order. |
| `GROQ_MODEL`, `GROQ_FALLBACK_MODELS` | `openai/gpt-oss-120b`, `openai/gpt-oss-20b` | Groq chain order. |
| `LLM_TIMEOUT_S`, `LLM_MAX_RETRIES_PER_MODEL` | `60`, `1` | Retry and timeout policy per model. |
| `CHROMA_PATH`, `RAG_COLLECTION` | `data/chroma`, `lankalaw_kb` | Where the vector store lives. |
| `EMBEDDING_MODEL`, `EMBEDDING_DIM` | `gemini-embedding-001`, `768` | Embedding model. Changing these requires re-seeding. |
| `RAG_TOP_K`, `RAG_MIN_SCORE` | `5`, `0.3` | Passages per search and the cosine cutoff. |
| `RAG_MAX_STEPS` | `6` | Model calls per question before the agent is forced to answer. |
| `RAG_REFLECT` | `true` | Reflect step after each search round (one extra model call per round). |
| `GRAPH_ENABLED`, `GRAPH_PATH` | `true`, `data/graph/lankalaw.kuzu` | Knowledge graph on or off, and its file. |
| `GRAPH_EXPAND_K` | `3` | Extra graph-linked passages added per search. |
| `INGEST_GRAPH_EXTRACT` | `true` | Run the LLM entity/relation extractor during ingestion. |
| `INGEST_CHUNK_SIZE`, `INGEST_CHUNK_OVERLAP`, `INGEST_MAX_UPLOAD_MB` | `1200`, `150`, `10` | Chunking and upload limits. |
| `APP_HOST`, `APP_PORT`, `APP_RELOAD` | `127.0.0.1`, `8080`, `true` | Server bind address and auto-reload. |

The `PINECONE_*`, `COHERE_*`, `LANGSMITH_*` and `PANDASAI_*` entries in `.env.example` are
placeholders for later phases and are not read by the current code.

## 5. Verify the install

The test suite is fully offline: fake embedder and LLM, real in-memory ChromaDB and Kùzu.

```bash
python -m pytest tests -q
```

Expect every test to pass in well under a minute. No API keys are needed for this step.

To check the real provider chain and keys:

```bash
python scripts/smoke_llm.py
```

It runs a plain completion, a tool call and a forced fallback, and prints which model
answered each.

## 6. Start the server

```bash
python -m app
```

This reads `APP_HOST`, `APP_PORT` and `APP_RELOAD` from `.env` and starts uvicorn. Open
<http://127.0.0.1:8080>. Port 8080 is the default because 8000 is often taken on
development machines.

Equivalent direct uvicorn command:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8080
```

On startup the app builds the services from `.env`: embedder, vector store, graph store,
provider chain, agent and ingestion service. If `GEMINI_API_KEY` is missing it fails fast
with a clear message.

Health check:

```bash
curl http://127.0.0.1:8080/api/health
```

```json
{"status":"ok","documents":13,"chain":["gemini:gemini-3.5-flash-lite", "..."],
 "cooldowns":{},"max_steps":6,"graph":{"chunks":12,"entities":78,"mentions":87,"relations":46}}
```

`documents` is the number of chunks in the vector store, `graph` is `null` when the graph
is disabled, and `cooldowns` lists models that recently failed and are temporarily demoted.

Interactive API docs are at <http://127.0.0.1:8080/docs>.

## 7. Seed the knowledge base

The repo ships 12 sample passages in `data/seed/sample_docs.json`. Load them with:

```bash
python scripts/seed_kb.py            # or: python scripts/seed_kb.py path/to/other.json
```

Each passage is embedded into ChromaDB and written to the graph, with entities and
relations extracted by the LLM. Seeding is idempotent: running it again replaces the same
ids.

**Single-writer rule.** Kùzu allows one process to hold the graph file. While the server is
running it owns the graph, so the seed script detects a live server (via `/api/health`) and
sends the passages to `POST /api/ingest/records` instead of opening the stores directly. You
can run the seed script with the server up or down; it picks the right path. If you ever see
`Could not set lock on file ... lankalaw.kuzu`, another process has the graph open: stop it,
or go through the API.

## 8. Using the UI

**Ask tab**

- Type a question or click an example. The answer streams in as the model writes it.
- The *Agent steps* panel shows each model call: searches issued (query, passages returned,
  how many were new), the model's one-line reasoning before each search, reflect steps that
  grade which parts of the question are answered and decide to search again or answer, and
  the final answer step with the passages it cited.
- *Sources* lists every passage retrieved, numbered to match the `[n]` citations. Passages
  found through the graph show `via graph: ...` instead of a similarity score. Click a
  citation to jump to its source.

**Ingest tab**

- *Add text*: paste a statute section or note with a source label. The label is the
  citation shown in answers and also the document identity, so reusing a label replaces
  the earlier version.
- *Upload files*: `.txt`, `.md`, `.pdf`; multiple files at once; optional metadata (act,
  year, topic). Files are chunked, embedded and graph-extracted.
- *Knowledge graph*: counts, entity search, and for each entity its relations and the
  passages that mention it.
- *Indexed documents*: everything in the store grouped by source, with delete.

## 9. API quick reference

| Method and path | Purpose |
|---|---|
| `GET /api/health` | Status, chunk count, model chain, cooldowns, graph stats |
| `POST /api/ask` | `{question, top_k?}` -> answer, sources, trace, token usage |
| `POST /api/ask/stream` | Same input, server-sent events: `token`, `model`, `tool`, `reflect`, `done`, `error` |
| `POST /api/search` | Retrieval only, no agent |
| `POST /api/ingest/text` | `{source, text, metadata?}` |
| `POST /api/ingest/files` | multipart `files[]`, optional `source` (single file) and `metadata` JSON |
| `POST /api/ingest/records` | Bulk pre-chunked passages `{records:[{id,text,metadata}]}` |
| `GET /api/ingest/documents` | Indexed documents grouped by source |
| `DELETE /api/ingest/documents/{doc_id}` | Remove a document from both stores |
| `GET /api/ingest/settings` | Chunking and upload limits |
| `GET /api/graph/stats` | Chunk, entity, mention and relation counts |
| `GET /api/graph/entities?q=&limit=` | Entity search, or the most-mentioned entities |
| `GET /api/graph/entities/{key}` | One entity's relations and passages |

Examples:

```bash
# ask
curl -s http://127.0.0.1:8080/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How long can a person be detained under the PTA after the 2022 amendment?"}'

# stream the same question as server-sent events
curl -N http://127.0.0.1:8080/api/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What did the 2022 PTA amendment change?"}'

# ingest a passage
curl -s http://127.0.0.1:8080/api/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"source": "RTI Act No. 12 of 2016, s. 25", "text": "A public authority shall ...", "metadata": {"act": "RTI Act", "year": 2016}}'

# browse the graph
curl -s "http://127.0.0.1:8080/api/graph/entities?q=terrorism"
```

## 10. Project layout

```
agent/
  rag/                 AgenticRAG (agent.py), tools, reflect step, retrievers, prompts, settings
  services/llm/        FallbackLLM provider chain (Gemini, Groq), LangChain adapter
  services/db/         vectorstore/ (ChromaDB), graphstore/ (Kùzu)
ingestion/             loaders, chunker, graph_extractor, IngestionService
app/
  __main__.py          python -m app launcher
  bootstrap.py         builds all services from .env (the only composition root)
  main.py, deps.py     FastAPI app factory and dependencies
  routes/              rag.py, ingest.py, graph.py
  schemas.py           request/response models
  static/index.html    the UI
data/
  seed/                sample passages
  chroma/, graph/      created at runtime; safe to delete to start over (then re-seed)
scripts/               seed_kb.py, smoke_llm.py
tests/                 offline test suite
```

## 11. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `GEMINI_API_KEY is required for embeddings` on startup | Key missing from `.env`. Add it; the server cannot start without it. |
| `Could not set lock on file ... lankalaw.kuzu` | Another process (usually the running server) holds the graph. Use the API or the seed script, or stop the other process. |
| Port already in use | Change `APP_PORT` in `.env`, or stop whatever is on 8080. |
| Answers say "I could not find anything relevant" | The store is empty or the question is far from any passage. Seed or ingest first; check `/api/health` `documents`. Lower `RAG_MIN_SCORE` if genuine matches are being cut. |
| `/api/graph/*` returns 404 "knowledge graph is disabled" | `GRAPH_ENABLED=false`. Set it to `true` and restart. |
| Graph counts are 0 after seeding with the server running | Seeding went direct while the server held the graph, or the seed ran before the graph was enabled. Re-run `scripts/seed_kb.py` with the server up. |
| Gemini 400 `missing thought_signature` | Gemini 3 requires function-call signatures to be echoed back. The provider does this; if you change message handling, keep `ToolCall.provider_data` intact. |
| Slow or expensive questions | Each search round costs one reflect call. Set `RAG_REFLECT=false` or lower `RAG_MAX_STEPS`. |
| Everything failing with 503 `all models failed` | Both providers down or keys invalid. Run `scripts/smoke_llm.py` to see per-model errors; check `cooldowns` in `/api/health`. |

To start over completely: stop the server, delete `data/chroma` and `data/graph`, start the
server and re-seed.
