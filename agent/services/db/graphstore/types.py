"""Plain types exchanged with a graph store. No database types leak out of the store."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_KEY = re.compile(r"[^a-z0-9]+")

ENTITY_TYPES = ("act", "provision", "institution", "person", "place", "concept", "duration", "other")


def entity_key(label: str) -> str:
    """Stable primary key for an entity label: lowercase, punctuation collapsed to single dashes."""
    return _KEY.sub("-", label.strip().lower()).strip("-")


@dataclass(frozen=True)
class Entity:
    label: str  # as written in the text, e.g. "Right to Information Act No. 12 of 2016"
    type: str = "concept"  # one of ENTITY_TYPES

    @property
    def key(self) -> str:
        return entity_key(self.label)


@dataclass(frozen=True)
class Relation:
    subject: str  # entity label
    predicate: str  # short upper-snake verb, e.g. AMENDS, DEFINES, TIME_LIMIT
    object: str  # entity label


@dataclass
class ChunkFacts:
    """Everything extracted from one chunk, ready to be written to the graph.

    The chunk text is stored too, so the graph is a self-contained retrieval store: passages it
    returns need no lookup in the vector store."""

    chunk_id: str
    doc_id: str
    source: str
    chunk_index: int
    text: str = ""
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)


@dataclass
class ChunkRecord:
    """A chunk as stored in the graph."""

    chunk_id: str
    doc_id: str
    source: str
    chunk_index: int
    text: str


@dataclass
class GraphHit:
    chunk_id: str
    score: float  # graph-side relevance (shared entities, adjacency); not a cosine
    via: list[str]  # why it was pulled in, e.g. entity labels shared with the seeds


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    chunk_id: str
    doc_id: str


@dataclass
class EntityInfo:
    key: str
    label: str
    type: str
    mentions: int


@dataclass
class EntityContext:
    entities: list[EntityInfo]
    triples: list[Triple]
    chunk_ids: list[str]  # chunks mentioning the entities, most relevant first
