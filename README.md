# LankaLaw Intelligence Engine

An agentic RAG system that maintains, investigates, and validates a changing legal
knowledge base for Sri Lankan law.

## Status

| Layer | State |
| --- | --- |
| LLM provider + model fallback (`agent/services/llm`) | done, tested live |
| Agentic RAG on local ChromaDB: LangChain agent with a search tool (`agent/rag`, `agent/services/db`) | done |
| Data ingestion: text + file upload, chunking, document management (`ingestion/`) | done, tested live |
| FastAPI backend + HTML UI with Ask / Ingest tabs (`app/`) | done |
| Agent loop (plan, retrieve, validate) | next |

## Setup

Step-by-step install, configuration, server start, seeding and troubleshooting are in
[PROJECT_GUIDE.md](PROJECT_GUIDE.md). Short version:

```bash
pip install -r requirements.txt
cp .env.example .env               # then fill in the keys
python -m pytest tests -q          # offline tests, no network
python scripts/seed_kb.py          # load the 12 sample passages into data/chroma
python -m app                      # open http://127.0.0.1:8080 (port from APP_PORT in .env)
```

## Layout (separation of concerns)

```
agent/            read-only consumer of the knowledge base
  services/          infrastructure, no RAG knowledge
    llm/               provider-agnostic chat + fallback chain    (Gemini, Groq)
    db/vectorstore/    VectorStore protocol (base.py) + ChromaVectorStore (chroma.py)
    db/graphstore/     GraphStore protocol + KuzuGraphStore (embedded Cypher graph DB, one file on disk)
  rag/
    config.py          RAG settings from .env
    schemas.py         Document, RetrievedChunk, RAGAnswer         (plain dataclasses)
    embeddings.py      Embedder protocol + GeminiEmbedder
    retriever.py       Retriever (vector) and GraphExpandingRetriever (vector + linked passages from the graph)
    tools.py           search_knowledge_base + explore_knowledge_graph tools, per-request RetrievalLog (stable citations)
    prompts.py         agent system prompt
    reflect.py         ReflectAfterRetrieval middleware: grade coverage after each search round, plan next move
    agent.py           AgenticRAG: LangChain create_agent loop over FallbackLLM; .ask() / .search()
  services/llm/langchain_adapter.py   FallbackLLM exposed as a LangChain BaseChatModel
ingestion/        the only writer to the knowledge base (isolated from agent/)
  config.py          chunk size/overlap, upload limit from .env  (INGEST_*)
  loaders.py         bytes -> text per file type                 (.txt, .md, .pdf)
  chunker.py         paragraph-aware chunking with overlap
  graph_extractor.py LLMGraphExtractor: chunk -> entities + relations (structured JSON) for the graph
  schemas.py         IngestResult, IndexedDocument
  service.py         IngestionService: ingest_text/ingest_file, list/delete documents
app/
  __main__.py        `python -m app` launcher (APP_HOST/APP_PORT from .env)
  bootstrap.py       composition root: builds the agent + ingestor sharing one embedder/store
  main.py            app factory, lifespan calls bootstrap
  deps.py            get_rag / get_ingestor dependencies
  schemas.py         pydantic request/response models
  routes/rag.py      POST /api/ask, POST /api/search, GET /api/health
  routes/graph.py    GET /api/graph/stats, /api/graph/entities[?q=], /api/graph/entities/{key}
  routes/ingest.py   POST /api/ingest/text, POST /api/ingest/files (multipart),
                     GET /api/ingest/documents, DELETE /api/ingest/documents/{doc_id},
                     GET /api/ingest/settings
  static/index.html  UI: Ask tab (cited answer + sources) and Ingest tab (upload/paste, manage docs)
data/seed/           sample passages loaded by scripts/seed_kb.py through the ingestion service
scripts/             seed_kb.py, smoke_llm.py
```

Each layer talks only to the one below it through a small interface, so the embedder or
the vector store can be swapped (e.g. a local sentence-transformers model) without
touching the retriever, agent, or API. `agent/` never imports `ingestion/`; both are
wired together only in `app/bootstrap.py`.

### API

```
GET    /api/health                      -> {documents, chain, cooldowns, max_steps, graph: {chunks, entities, mentions, relations} | null}
POST   /api/search                      {question, top_k?} -> ranked passages only
POST   /api/ask                         {question, top_k?} -> {answer, sources[], trace[], queries[], steps, provider, model, fell_back, tokens}
                                        trace[] = one entry per agent step: {n, kind: search|answer, query, citations[], new_passages, text, note, model, tokens}
POST   /api/ask/stream                  same input, server-sent events: token {n, text} | model {n, steps[]} | tool {n, step, sources[]} | reflect {n, step} | done {AskResponse} | error {status, detail}

GET    /api/ingest/settings             -> chunk_size, chunk_overlap, max_upload_mb, allowed_extensions
GET    /api/ingest/documents            -> indexed documents grouped by source, with chunk counts
POST   /api/ingest/text                 {source, text, metadata?} -> {doc_id, chunks, chars, replaced}
POST   /api/ingest/files                multipart: files[], source? (single file only), metadata? (JSON) -> {results[], errors[]}
POST   /api/ingest/records              {records: [{id, text, metadata}]} bulk pre-chunked load (seed format) -> {indexed, total_chunks, graph}
DELETE /api/ingest/documents/{doc_id}   -> {deleted_chunks}

GET    /api/graph/stats                 -> {chunks, entities, mentions, relations}
GET    /api/graph/entities?q=&limit=    -> entities matching q (or the most mentioned), with stats
GET    /api/graph/entities/{key}        -> {entities, triples[], passages[]} for one entity
```

### Knowledge graph (on-premises)

Beside the vector store there is an embedded graph database, [Kùzu](https://github.com/kuzudb/kuzu)
(Cypher, single file at `GRAPH_PATH`, no server). Ingestion writes every chunk into it with
its text, and an LLM extractor (`ingestion/graph_extractor.py`, structured JSON through the
fallback chain) adds the entities it mentions (acts, provisions, institutions, durations,
concepts) and subject-predicate-object relations between them (AMENDS, DEFINES, TIME_LIMIT...).
Re-ingesting or deleting a document removes its chunks and relations and prunes orphan
entities. Extraction is best-effort: a failed chunk keeps its text node and loses only links
(`graph_errors` in the ingest result).

Retrieval uses the graph in two ways:
- `GraphExpandingRetriever`: after vector search, passages linked to the top hits (shared
  entities, adjacent passage of the same document) are appended, up to `GRAPH_EXPAND_K`,
  marked `via: graph: ...` in the sources.
- `explore_knowledge_graph(entity)`: a second agent tool that returns an entity's relations
  and the passages mentioning it, numbered for citation, for "how does X relate to Y" questions.

The vector store is untouched by all of this; `GRAPH_ENABLED=false` removes the graph, the
tool and the `/api/graph` routes.

One operational rule: Kùzu allows a single writer process per database file. While
`python -m app` is running it holds the graph, so load data through the API (the UI, or
`POST /api/ingest/records`). `scripts/seed_kb.py` does this automatically when it finds the
server up, and opens the stores directly only when it is not. The Ingest tab has a graph browser (stats, entity search,
relations and passages per entity).

### Agentic loop

`/api/ask` runs a LangChain agent (`langchain.agents.create_agent`, LangGraph underneath)
whose only tool is `search_knowledge_base`. The model decides what to search, reads the
numbered passages, and may search again with new wording or for each part of a multi-part
question. After `RAG_MAX_STEPS` model calls the tools are withdrawn and it must answer.
After every round of search results a **reflect step** (`agent/rag/reflect.py`, a LangChain
middleware) makes a separate structured model call: it splits the question into parts, marks
each answered / partial / unanswered with the supporting passage numbers, and decides
`search` (with up to three new queries, never ones already run) or `answer`. The verdict is
shown as its own step, injected into the next model call as guidance, and on `answer` the
tools are withdrawn so the run ends. It is skipped once the step budget is spent, and a
failed or malformed verdict is logged and ignored rather than failing the run. Turn it off
with `RAG_REFLECT=false` to save one model call per search round.

Citation numbers are assigned on first retrieval and stay stable across searches, so
`sources[n-1]` is always what `[n]` in the answer refers to; `queries` lists the searches
the agent ran. If no search returned anything, the answer is a fixed "nothing found"
message rather than whatever the model said.

The model is the project's own `FallbackLLM` (Gemini -> Groq chain with retries and
cooldowns), wrapped as a LangChain chat model in `agent/services/llm/langchain_adapter.py`,
so provider fallback keeps working inside the agent loop.

Streaming: both providers stream tokens (`Provider.stream`), `FallbackLLM.stream` applies
the fallback policy until the first token is out (after that an error is raised rather than
silently restarting on another model), the adapter exposes it as LangChain `_stream`, and
`AgenticRAG.stream` turns the LangGraph `messages`/`updates` streams into `RAGEvent`s. The UI
uses `/api/ask/stream`: reasoning and answer text appear as typed, each search appears as a
pending step the moment the model issues it and fills in when its passages come back.
`/api/ask` is the same run collapsed to the final answer.

`RAG_TOP_K` and `RAG_MIN_SCORE` (cosine similarity cutoff) are in `.env`. Gemini
embeddings keep unrelated passages around 0.55 similarity, so the prompt also tells the
model to say when the passages do not answer the question.

Ingestion rules: the `source` label is the citation shown in answers and also the
document identity, so re-ingesting the same source replaces the previous version. Text is
split into chunks of `INGEST_CHUNK_SIZE` characters with `INGEST_CHUNK_OVERLAP` carried
over, never splitting a paragraph unless it alone exceeds the chunk size. Uploads are
capped at `INGEST_MAX_UPLOAD_MB`; a multi-file upload reports per-file errors instead of
failing the whole batch.

## LLM layer

```python
from agent.services.llm import FallbackLLM, Message, ToolSpec

llm = FallbackLLM.from_env()

# plain chat
print(llm.chat("Summarise Article 12 of the Constitution.", system="You are a Sri Lankan law assistant."))

# full control: tools, structured output, attempt log
resp = llm.invoke(
    [Message.user("Find PTA amendments since 2020.")],
    tools=[ToolSpec("search_statutes", "Search statutes.", {"type": "object", "properties": {"query": {"type": "string"}}})],
    temperature=0.0,
)
resp.tool_calls      # [ToolCall(id=..., name="search_statutes", arguments={...})]
resp.model           # which model actually answered
resp.attempts        # every model tried, with error kind and latency
await llm.ainvoke(...)  # async variant
```

### Fallback chain

The chain is built from `.env`:

```
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_FALLBACK_MODELS=gemini-3.1-flash-lite,gemini-3.5-flash
GROQ_MODEL=openai/gpt-oss-120b
GROQ_FALLBACK_MODELS=openai/gpt-oss-20b
```

gives `gemini-3.5-flash-lite -> gemini-3.1-flash-lite -> gemini-3.5-flash -> groq gpt-oss-120b -> groq gpt-oss-20b`.
Set `LLM_PROVIDER=groq` to flip the order. A provider with no key is left out.

What happens on an error, by kind:

| Error | HTTP | Action |
| --- | --- | --- |
| rate limit | 429, quota 403 | retry same model with backoff (honours `Retry-After`), then next model; 60 s cooldown |
| server / overloaded | 5xx | retry same model, then next; 30 s cooldown |
| timeout | network | retry same model, then next; 15 s cooldown |
| model unavailable | 404, decommissioned 400 | next model immediately; 10 min cooldown |
| auth | 401, bad-key 403 | skip every model of that provider; 1 h cooldown |
| content blocked | safety finish | next model immediately |
| bad request | other 400 | raise at once (another model would fail the same way) |

Models in cooldown are moved to the back of the chain, not dropped, so a chain where
everything is cooling still gets tried. When every model fails you get
`AllModelsFailedError` carrying the full attempt log.

Tuning knobs: `LLM_TIMEOUT_S`, `LLM_MAX_RETRIES_PER_MODEL`, `LLM_BACKOFF_BASE_S`, `LLM_BACKOFF_MAX_S`.
