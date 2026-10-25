"""Knowledge graph: Kùzu store, extraction during ingestion, graph-expanded retrieval, the
agent's graph tool, and the graph API. Real in-memory Kùzu + Chroma, fake embedder/LLM."""
from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.rag import AgenticRAG, Document, GraphExpandingRetriever, Retriever  # noqa: E402
from agent.rag.tools import GRAPH_TOOL, NO_ENTITY  # noqa: E402
from agent.services.db.graphstore import ChunkFacts, Entity, KuzuGraphStore, Relation, entity_key  # noqa: E402
from agent.services.db.vectorstore import ChromaVectorStore  # noqa: E402
from agent.services.llm import FallbackLLM, ModelRef, ProviderServerError  # noqa: E402
from agent.services.llm.providers.base import Provider  # noqa: E402
from agent.services.llm.types import ProviderResult, ToolCall, Usage  # noqa: E402
from ingestion import ExtractedFacts, IngestionService, IngestSettings  # noqa: E402
from tests.test_agentic_rag import DOCS, ScriptedProvider, WordEmbedder  # noqa: E402


# ----------------------------------------------------------------------- fakes
class KeywordExtractor:
    """Rule-based stand-in for the LLM extractor: entities from a keyword table."""

    TABLE = {
        "pta": Entity("Prevention of Terrorism Act", "act"),
        "detention": Entity("detention", "concept"),
        "rti": Entity("Right to Information Act", "act"),
        "information": Entity("information request", "concept"),
        "days": Entity("fourteen days", "duration"),
        "months": Entity("twelve months", "duration"),
        "article": Entity("Article 12", "provision"),
        "equality": Entity("equality", "concept"),
        "company": Entity("Companies Act", "act"),
    }

    def __init__(self):
        self.calls = 0
        self.fail = False

    def extract(self, text, *, source):
        self.calls += 1
        if self.fail:
            raise ProviderServerError("extractor down", provider="fake", model="x")
        words = {w.strip(".,:").lower() for w in text.split()}
        ents = [e for k, e in self.TABLE.items() if k in words]
        rels = []
        labels = [e.label for e in ents]
        if "Prevention of Terrorism Act" in labels and "twelve months" in labels:
            rels.append(Relation("Prevention of Terrorism Act", "TIME_LIMIT", "twelve months"))
        if "Right to Information Act" in labels and "fourteen days" in labels:
            rels.append(Relation("Right to Information Act", "TIME_LIMIT", "fourteen days"))
        return ExtractedFacts(entities=ents, relations=rels)


def facts(cid, doc, idx, text, ents, rels=()):
    return ChunkFacts(cid, doc, doc, idx, text, [Entity(e) for e in ents], list(rels))


# ------------------------------------------------------------------ graph store
@pytest.fixture
def g():
    return KuzuGraphStore(":memory:")


def test_store_upsert_is_idempotent_and_keeps_text(g):
    f = facts("a0", "A", 0, "section nine", ["PTA", "detention"], [Relation("PTA", "LIMITS", "detention")])
    g.upsert(f)
    g.upsert(f)
    assert g.stats() == {"chunks": 1, "entities": 2, "mentions": 2, "relations": 1}
    rec = g.get_chunks(["a0", "missing"])
    assert len(rec) == 1 and rec[0].text == "section nine" and rec[0].source == "A"


def test_store_neighbours_by_shared_entity_and_adjacency(g):
    g.upsert(facts("a0", "A", 0, "", ["PTA", "detention"]))
    g.upsert(facts("a1", "A", 1, "", ["minister"]))  # adjacent only
    g.upsert(facts("b0", "B", 0, "", ["detention", "RTI"]))  # shares 'detention'
    g.upsert(facts("c0", "C", 0, "", ["company"]))  # unrelated
    hits = g.neighbours(["a0"], limit=5)
    ids = {h.chunk_id: h for h in hits}
    assert set(ids) == {"a1", "b0"}
    assert ids["b0"].score == 1.0 and ids["b0"].via == ["detention"]
    assert ids["a1"].score == 0.5 and ids["a1"].via == ["adjacent passage"]
    assert g.neighbours([], limit=5) == [] and g.neighbours(["a0"], limit=1)[0].chunk_id == "b0"


def test_store_delete_document_prunes_orphans(g):
    g.upsert(facts("a0", "A", 0, "", ["PTA", "detention"], [Relation("PTA", "LIMITS", "detention")]))
    g.upsert(facts("b0", "B", 0, "", ["detention"]))
    assert g.delete_document("A") == 1
    s = g.stats()
    assert s == {"chunks": 1, "entities": 1, "mentions": 1, "relations": 0}  # 'PTA' pruned, 'detention' kept
    assert g.delete_document("nope") == 0


def test_store_entity_search_and_context(g):
    g.upsert(facts("a0", "A", 0, "pta text", ["Prevention of Terrorism Act", "detention"], [Relation("Prevention of Terrorism Act", "TIME_LIMIT", "twelve months")]))
    g.upsert(facts("b0", "B", 0, "rti text", ["detention"]))
    assert [e.label for e in g.search_entities("terrorism")] == ["Prevention of Terrorism Act"]
    assert [e.label for e in g.search_entities("DETENTION limit")][0] == "detention"  # falls back to word match
    assert g.search_entities("zzz") == [] and g.search_entities("  ") == [] and g.search_entities("2022") == []
    g.upsert(facts("c0", "C", 0, "", ["Personal Data Protection Act No. 9 of 2022"]))
    # word fallback ranks by matched words and drops entities sharing only weak tokens
    assert [e.label for e in g.search_entities("Prevention of Terrorism (Amendment) Act 2022")] == ["Prevention of Terrorism Act"]
    ctx = g.entity_context([entity_key("Prevention of Terrorism Act")])
    assert ctx.entities[0].mentions == 1
    assert [(t.subject, t.predicate, t.object) for t in ctx.triples] == [("Prevention of Terrorism Act", "TIME_LIMIT", "twelve months")]
    assert ctx.chunk_ids == ["a0"]
    top = g.top_entities(limit=1)
    assert top[0].label == "detention" and top[0].mentions == 2


# -------------------------------------------------------------------- ingestion
@pytest.fixture
def svc():
    store = ChromaVectorStore.ephemeral()
    graph = KuzuGraphStore(":memory:")
    ex = KeywordExtractor()
    s = IngestionService(WordEmbedder(), store, IngestSettings(chunk_size=50, chunk_overlap=0), graph=graph, extractor=ex)
    s._ex = ex
    return s


def test_ingest_writes_graph_and_replace_cleans_up(svc):
    r = svc.ingest_text("PTA detention may last twelve months.\n\nRTI information within fourteen days.", source="Mixed")
    assert r.chunks == 2 and r.entities >= 4 and r.relations == 2 and r.graph_errors == 0
    assert svc.graph.stats()["chunks"] == 2 and svc._ex.calls == 2
    r2 = svc.ingest_text("Only PTA detention here.", source="Mixed")  # replace: old chunks and their links go
    assert r2.replaced and svc.graph.stats()["chunks"] == 1
    assert svc.graph.search_entities("Right to Information") == []  # pruned with the old chunk


def test_ingest_graph_extraction_failure_is_best_effort(svc):
    svc._ex.fail = True
    r = svc.ingest_text("PTA detention twelve months.", source="X")
    assert r.graph_errors == 1 and r.entities == 0
    assert svc.graph.stats() == {"chunks": 1, "entities": 0, "mentions": 0, "relations": 0}  # text kept, no links
    assert svc.store.count() == 1


def test_ingest_graph_extraction_can_be_switched_off(svc):
    svc.settings.graph_extract = False
    r = svc.ingest_text("PTA detention twelve months.", source="X")
    assert r.entities == 0 and svc._ex.calls == 0 and svc.graph.stats()["chunks"] == 1


def test_delete_and_prechunked_touch_graph(svc):
    svc.index_prechunked([Document("c1", "Article 12 equality", {"source": "Const 12"})])
    assert svc.graph.stats()["chunks"] == 1 and svc.graph.search_entities("Article")[0].label == "Article 12"
    assert svc.delete_document("c1") == 1 and svc.graph.stats()["chunks"] == 0


# -------------------------------------------------------------------- retrieval
def seeded():
    """Vector store + graph holding DOCS, with the graph linking pta<->rti through 'detention'/'days'."""
    store = ChromaVectorStore.ephemeral("g_kb")
    emb = WordEmbedder()
    store.upsert(DOCS, emb.embed_documents([d.text for d in DOCS]))
    graph = KuzuGraphStore(":memory:")
    ex = KeywordExtractor()
    for d in DOCS:
        f = ex.extract(d.text, source=d.metadata["source"])
        graph.upsert(ChunkFacts(d.id, d.id, d.metadata["source"], 0, d.text, f.entities, f.relations))
    graph.upsert(ChunkFacts("co2", "co", "Companies Act", 1, "A company must file annual returns.", [Entity("Companies Act", "act")]))
    return store, emb, graph


def test_graph_expanding_retriever_adds_linked_passages():
    store, emb, graph = seeded()
    plain = Retriever(emb, store, top_k=1, min_score=0.1)
    hybrid = GraphExpandingRetriever(emb, store, graph, top_k=1, min_score=0.1, expand_k=2)
    assert [c.id for c in plain.retrieve("company shares")] == ["co"]
    got = hybrid.retrieve("company shares")
    assert [c.id for c in got] == ["co", "co2"]  # co2 shares 'Companies Act' and is the adjacent passage
    assert got[0].via is None and got[1].via.startswith("graph:") and "Companies Act" in got[1].via
    assert got[1].text == "A company must file annual returns." and got[1].source == "Companies Act"
    assert hybrid.retrieve("zebra") == []  # no seeds -> no expansion


# ------------------------------------------------------------------------ agent
class GraphProvider(Provider):
    """Calls explore_knowledge_graph(entity) first, then answers."""

    name = "fake"

    def __init__(self, entity: str):
        self.entity = entity
        self.calls = []

    def complete(self, model, messages, tools=None, json_schema=None, **kw):
        if json_schema is not None:
            return ProviderResult(text='{"parts":[{"part":"q","status":"answered","citations":[1]}],"decision":"answer","queries":[],"rationale":"ok"}', tool_calls=[], finish_reason="STOP", usage=Usage())
        self.calls.append(messages)
        names = [t.name for t in tools or []]
        if tools and not any(m.role == "tool" for m in messages):
            assert GRAPH_TOOL in names, names
            return ProviderResult(text="Checking the graph.", tool_calls=[ToolCall("g1", GRAPH_TOOL, {"entity": self.entity})], finish_reason="TOOL_CALLS", usage=Usage(5, 5, 10))
        return ProviderResult(text="Linked [1].", tool_calls=[], finish_reason="STOP", usage=Usage(5, 5, 10))

    async def acomplete(self, model, messages, **kw):
        return self.complete(model, messages, **kw)


def make_graph_rag(provider, **kw):
    store, emb, graph = seeded()
    llm = FallbackLLM([ModelRef("fake", "m")], {"fake": provider}, sleep=lambda s: None)
    rag = AgenticRAG(GraphExpandingRetriever(emb, store, graph, top_k=2, min_score=0.1), llm, graph=graph, max_steps=4, **kw)
    rag._provider = provider
    return rag


def test_agent_graph_tool_returns_relations_and_citable_passages():
    rag = make_graph_rag(GraphProvider("Prevention of Terrorism"))
    ans = rag.ask("What time limit does the PTA set?")
    tool_msg = next(m for m in rag._provider.calls[1] if m.role == "tool")
    assert tool_msg.name == GRAPH_TOOL
    assert "Prevention of Terrorism Act (act" in tool_msg.content
    assert "Prevention of Terrorism Act —TIME_LIMIT→ twelve months [1]" in tool_msg.content
    assert "[1] Source: PTA 2022" in tool_msg.content
    step = ans.trace[0]
    assert step.kind == "search" and step.tool == GRAPH_TOOL and step.query == "Prevention of Terrorism" and step.citations == [1]
    assert step.text == "Checking the graph."
    assert ans.sources[0].id == "pta" and ans.sources[0].via.startswith("graph:")
    assert ans.answer == "Linked [1]."


def test_agent_graph_tool_unknown_entity():
    rag = make_graph_rag(GraphProvider("Zebra Act"))
    ans = rag.ask("zebra?")
    tool_msg = next(m for m in rag._provider.calls[1] if m.role == "tool")
    assert tool_msg.content == NO_ENTITY and ans.trace[0].note == "no results"


def test_agent_without_graph_has_single_tool():
    rag = make_graph_rag(ScriptedProvider())
    assert [t.name for t in rag.tools] == ["search_knowledge_base", GRAPH_TOOL]
    store, emb, _ = seeded()
    llm = FallbackLLM([ModelRef("fake", "m")], {"fake": ScriptedProvider()}, sleep=lambda s: None)
    assert [t.name for t in AgenticRAG(Retriever(emb, store), llm).tools] == ["search_knowledge_base"]


# -------------------------------------------------------------------------- api
@pytest.fixture
def client():
    from app.main import create_app

    rag = make_graph_rag(GraphProvider("Right to Information"))
    svc = IngestionService(rag.retriever.embedder, rag.retriever.store, IngestSettings(), graph=rag.graph_store, extractor=KeywordExtractor())
    with TestClient(create_app(rag=rag, ingestor=svc)) as c:
        yield c


def test_health_and_graph_endpoints(client):
    h = client.get("/api/health").json()
    assert h["graph"]["chunks"] == 5 and h["graph"]["relations"] == 2
    ents = client.get("/api/graph/entities", params={"q": "information"}).json()
    assert ents["query"] == "information" and {e["label"] for e in ents["entities"]} >= {"Right to Information Act", "information request"}
    top = client.get("/api/graph/entities").json()
    assert top["stats"]["entities"] == h["graph"]["entities"] and len(top["entities"]) > 0
    key = next(e["key"] for e in ents["entities"] if e["label"] == "Right to Information Act")
    d = client.get(f"/api/graph/entities/{key}").json()
    assert d["triples"][0]["predicate"] == "TIME_LIMIT" and d["passages"][0]["chunk_id"] == "rti"
    assert client.get("/api/graph/entities/does-not-exist").status_code == 404


def test_ask_endpoint_reports_graph_sources(client):
    body = client.post("/api/ask", json={"question": "RTI?"}).json()
    assert body["trace"][0]["tool"] == GRAPH_TOOL and body["sources"][0]["via"].startswith("graph:")


def test_records_endpoint_bulk_loads_into_both_stores(client):
    before = client.get("/api/graph/stats").json()["chunks"]
    r = client.post("/api/ingest/records", json={"records": [
        {"id": "seed-1", "text": "PTA detention lasts twelve months.", "metadata": {"source": "Seed PTA"}},
        {"id": "seed-2", "text": "RTI information within fourteen days.", "metadata": {"source": "Seed RTI"}},
    ]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["indexed"] == 2 and body["graph"]["chunks"] == before + 2
    assert client.post("/api/ingest/records", json={"records": []}).status_code == 422
    docs = {d["doc_id"]: d for d in client.get("/api/ingest/documents").json()["documents"]}
    assert docs["seed-1"]["source"] == "Seed PTA"


def test_ingest_endpoint_reports_graph_counts(client):
    r = client.post("/api/ingest/text", json={"source": "New Act", "text": "PTA detention lasts twelve months."}).json()
    assert r["entities"] >= 3 and r["relations"] == 1 and r["graph_errors"] == 0
    assert client.get("/api/graph/stats").json()["chunks"] == 6


def test_graph_api_404_when_disabled():
    from app.main import create_app

    store, emb, _ = seeded()
    llm = FallbackLLM([ModelRef("fake", "m")], {"fake": ScriptedProvider()}, sleep=lambda s: None)
    rag = AgenticRAG(Retriever(emb, store), llm)
    svc = IngestionService(emb, store, IngestSettings())
    with TestClient(create_app(rag=rag, ingestor=svc)) as c:
        assert c.get("/api/health").json()["graph"] is None
        assert c.get("/api/graph/stats").status_code == 404
