"""Read-only HTTP surface for the knowledge graph (what was extracted, how things connect)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from agent.services.db.graphstore import GraphStore

from ..deps import get_graph
from ..schemas import EntitiesResponse, EntityContextResponse, EntityOut, GraphPassageOut, GraphStatsOut, TripleOut

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("/stats", response_model=GraphStatsOut)
def stats(graph: GraphStore = Depends(get_graph)) -> GraphStatsOut:
    return GraphStatsOut(**graph.stats())


@router.get("/entities", response_model=EntitiesResponse)
def entities(
    q: str | None = Query(None, max_length=200, description="Substring of the entity label; omit for the most-mentioned entities"),
    limit: int = Query(25, ge=1, le=200),
    graph: GraphStore = Depends(get_graph),
) -> EntitiesResponse:
    found = graph.search_entities(q, limit=limit) if q else graph.top_entities(limit=limit)
    return EntitiesResponse(query=q, entities=[EntityOut.from_info(e) for e in found], stats=GraphStatsOut(**graph.stats()))


@router.get("/entities/{key}", response_model=EntityContextResponse)
def entity(key: str, limit: int = Query(10, ge=1, le=50), graph: GraphStore = Depends(get_graph)) -> EntityContextResponse:
    ctx = graph.entity_context([key], limit_chunks=limit)
    if not ctx.entities:
        raise HTTPException(status_code=404, detail=f"no entity with key {key!r}")
    return EntityContextResponse(
        entities=[EntityOut.from_info(e) for e in ctx.entities],
        triples=[TripleOut(subject=t.subject, predicate=t.predicate, object=t.object, chunk_id=t.chunk_id, doc_id=t.doc_id) for t in ctx.triples],
        passages=[GraphPassageOut.from_record(r) for r in graph.get_chunks(ctx.chunk_ids)],
    )
