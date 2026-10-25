"""Kùzu-backed GraphStore: an embedded, on-premises property graph queried with Cypher.

Schema
  (:Chunk {id, doc_id, source, chunk_index, text}) one node per ingested chunk, same ids as the vector store
  (:Entity {name, label, type})                    name = normalised key, label = display form
  (:Chunk)-[:MENTIONS]->(:Entity)
  (:Entity)-[:RELATED {predicate, doc_id, chunk_id}]->(:Entity)

The database is a single file; pass ":memory:" for tests.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import kuzu

from .types import ChunkFacts, ChunkRecord, EntityContext, EntityInfo, GraphHit, Triple, entity_key

_SCHEMA = [
    "CREATE NODE TABLE IF NOT EXISTS Entity(name STRING, label STRING, type STRING, PRIMARY KEY(name))",
    "CREATE NODE TABLE IF NOT EXISTS Chunk(id STRING, doc_id STRING, source STRING, chunk_index INT64, text STRING, PRIMARY KEY(id))",
    "CREATE REL TABLE IF NOT EXISTS MENTIONS(FROM Chunk TO Entity)",
    "CREATE REL TABLE IF NOT EXISTS RELATED(FROM Entity TO Entity, predicate STRING, doc_id STRING, chunk_id STRING)",
]


class KuzuGraphStore:
    def __init__(self, path: str | Path) -> None:
        p = str(path)
        if p != ":memory:":
            Path(p).parent.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(p)
        self._conn = kuzu.Connection(self._db)
        self._lock = threading.Lock()  # one connection, used from tool worker threads
        for stmt in _SCHEMA:
            self._run(stmt)

    # ------------------------------------------------------------------ helpers
    def _run(self, query: str, params: dict[str, Any] | None = None) -> list[list[Any]]:
        with self._lock:
            res = self._conn.execute(query, params or {})
            rows = []
            while res.has_next():
                rows.append(res.get_next())
            return rows

    # ------------------------------------------------------------------- writes
    def upsert(self, facts: ChunkFacts) -> None:
        self._run(
            "MERGE (k:Chunk {id: $id}) SET k.doc_id = $d, k.source = $s, k.chunk_index = $i, k.text = $t",
            {"id": facts.chunk_id, "d": facts.doc_id, "s": facts.source, "i": int(facts.chunk_index), "t": facts.text or ""},
        )
        for e in facts.entities:
            key = e.key
            if not key:
                continue
            self._run(
                "MERGE (e:Entity {name: $k}) ON CREATE SET e.label = $l, e.type = $t ON MATCH SET e.type = $t",
                {"k": key, "l": e.label.strip(), "t": e.type},
            )
            self._run(
                "MATCH (k:Chunk {id: $id}), (e:Entity {name: $k}) MERGE (k)-[:MENTIONS]->(e)",
                {"id": facts.chunk_id, "k": key},
            )
        for r in facts.relations:
            a, b = entity_key(r.subject), entity_key(r.object)
            if not a or not b or not r.predicate.strip():
                continue
            for key, label in ((a, r.subject), (b, r.object)):
                self._run(
                    "MERGE (e:Entity {name: $k}) ON CREATE SET e.label = $l, e.type = 'concept'",
                    {"k": key, "l": label.strip()},
                )
            self._run(
                "MATCH (a:Entity {name: $a}), (b:Entity {name: $b}) "
                "MERGE (a)-[r:RELATED {predicate: $p, chunk_id: $c}]->(b) SET r.doc_id = $d",
                {"a": a, "b": b, "p": r.predicate.strip().upper().replace(" ", "_"), "c": facts.chunk_id, "d": facts.doc_id},
            )

    def delete_document(self, doc_id: str) -> int:
        n = self._run("MATCH (k:Chunk {doc_id: $d}) RETURN count(k)", {"d": doc_id})[0][0]
        if n:
            self._run("MATCH (k:Chunk {doc_id: $d}) DETACH DELETE k", {"d": doc_id})
        self._run("MATCH ()-[r:RELATED {doc_id: $d}]->() DELETE r", {"d": doc_id})
        self._prune_orphans()
        return int(n)

    def _prune_orphans(self) -> None:
        self._run(
            "MATCH (e:Entity) OPTIONAL MATCH (e)<-[m:MENTIONS]-() OPTIONAL MATCH (e)-[r:RELATED]-() "
            "WITH e, count(m) AS cm, count(r) AS cr WHERE cm = 0 AND cr = 0 DELETE e"
        )

    # -------------------------------------------------------------------- reads
    def get_chunks(self, chunk_ids: list[str]) -> list[ChunkRecord]:
        if not chunk_ids:
            return []
        rows = self._run(
            "MATCH (k:Chunk) WHERE list_contains($ids, k.id) RETURN k.id, k.doc_id, k.source, k.chunk_index, k.text",
            {"ids": list(chunk_ids)},
        )
        by_id = {r[0]: ChunkRecord(chunk_id=r[0], doc_id=r[1], source=r[2], chunk_index=int(r[3] or 0), text=r[4] or "") for r in rows}
        return [by_id[c] for c in chunk_ids if c in by_id]

    def neighbours(self, chunk_ids: list[str], *, limit: int = 5) -> list[GraphHit]:
        if not chunk_ids or limit <= 0:
            return []
        hits: dict[str, GraphHit] = {}
        rows = self._run(
            "MATCH (s:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(k:Chunk) "
            "WHERE list_contains($ids, s.id) AND NOT list_contains($ids, k.id) "
            "RETURN k.id, count(DISTINCT e) AS shared, collect(DISTINCT e.label) AS via "
            "ORDER BY shared DESC, k.id LIMIT $lim",
            {"ids": list(chunk_ids), "lim": int(limit) * 3},
        )
        for cid, shared, via in rows:
            hits[cid] = GraphHit(chunk_id=cid, score=float(shared), via=list(via)[:5])
        rows = self._run(
            "MATCH (s:Chunk), (k:Chunk) WHERE list_contains($ids, s.id) AND NOT list_contains($ids, k.id) "
            "AND s.doc_id = k.doc_id AND abs(s.chunk_index - k.chunk_index) = 1 RETURN DISTINCT k.id",
            {"ids": list(chunk_ids)},
        )
        for (cid,) in rows:
            h = hits.get(cid)
            if h is None:
                hits[cid] = GraphHit(chunk_id=cid, score=0.5, via=["adjacent passage"])
            else:
                h.score += 0.5
                h.via.append("adjacent passage")
        return sorted(hits.values(), key=lambda h: (-h.score, h.chunk_id))[:limit]

    def _entity_rows(self, where: str, params: dict[str, Any], limit: int) -> list[EntityInfo]:
        rows = self._run(
            f"MATCH (e:Entity) {where} OPTIONAL MATCH (e)<-[m:MENTIONS]-() "
            "RETURN e.name, e.label, e.type, count(m) AS mentions ORDER BY mentions DESC, e.label LIMIT $lim",
            {**params, "lim": int(limit)},
        )
        return [EntityInfo(key=k, label=l, type=t, mentions=int(m)) for k, l, t, m in rows]

    def search_entities(self, text: str, *, limit: int = 10) -> list[EntityInfo]:
        text = text.strip()
        if not text:
            return []
        found = self._entity_rows("WHERE lower(e.label) CONTAINS lower($q)", {"q": text}, limit)
        if found:
            return found
        # Fall back to the query's significant words (alphabetic, 4+ chars: years and "Act" alone
        # would match half the graph). Rank by how many words an entity matches and keep only
        # entities matching at least half of them, so "Prevention of Terrorism (Amendment) Act
        # 2022" does not drag in every other act of 2022.
        words = {w.strip("(),.;:").lower() for w in text.split()}
        words = {w for w in words if len(w) >= 4 and w.isalpha()}
        if not words:
            return []
        matches: dict[str, int] = {}
        info: dict[str, EntityInfo] = {}
        for word in words:
            for e in self._entity_rows("WHERE lower(e.label) CONTAINS $q", {"q": word}, limit * 4):
                matches[e.key] = matches.get(e.key, 0) + 1
                info[e.key] = e
        needed = max(1, (len(words) + 1) // 2)
        keep = [info[k] for k, n in matches.items() if n >= needed]
        return sorted(keep, key=lambda e: (-matches[e.key], -e.mentions, e.label))[:limit]

    def entity_context(self, keys: list[str], *, limit_chunks: int = 10) -> EntityContext:
        if not keys:
            return EntityContext(entities=[], triples=[], chunk_ids=[])
        ents = self._entity_rows("WHERE list_contains($keys, e.name)", {"keys": list(keys)}, len(keys))
        rows = self._run(
            "MATCH (a:Entity)-[r:RELATED]->(b:Entity) WHERE list_contains($keys, a.name) OR list_contains($keys, b.name) "
            "RETURN a.label, r.predicate, b.label, r.chunk_id, r.doc_id ORDER BY a.label, r.predicate, b.label LIMIT 60",
            {"keys": list(keys)},
        )
        triples = [Triple(subject=a, predicate=p, object=b, chunk_id=c, doc_id=d) for a, p, b, c, d in rows]
        rows = self._run(
            "MATCH (k:Chunk)-[:MENTIONS]->(e:Entity) WHERE list_contains($keys, e.name) "
            "RETURN k.id, count(e) AS c ORDER BY c DESC, k.id LIMIT $lim",
            {"keys": list(keys), "lim": int(limit_chunks)},
        )
        chunk_ids = [cid for cid, _ in rows]
        for t in triples:  # a relation's own chunk is evidence too
            if t.chunk_id not in chunk_ids and len(chunk_ids) < limit_chunks:
                chunk_ids.append(t.chunk_id)
        return EntityContext(entities=ents, triples=triples, chunk_ids=chunk_ids)

    def top_entities(self, *, limit: int = 20) -> list[EntityInfo]:
        return self._entity_rows("", {}, limit)

    def stats(self) -> dict[str, int]:
        return {
            "chunks": int(self._run("MATCH (k:Chunk) RETURN count(k)")[0][0]),
            "entities": int(self._run("MATCH (e:Entity) RETURN count(e)")[0][0]),
            "mentions": int(self._run("MATCH ()-[m:MENTIONS]->() RETURN count(m)")[0][0]),
            "relations": int(self._run("MATCH ()-[r:RELATED]->() RETURN count(r)")[0][0]),
        }
