"""Chunk text -> entities and relations for the knowledge graph.

One protocol, one LLM implementation (structured JSON output through `FallbackLLM`, so the
provider chain and retries apply). Extraction is best-effort: the caller decides what to do
when it fails for a chunk.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent.services.db.graphstore import ENTITY_TYPES, Entity, Relation
from agent.services.llm import FallbackLLM, Message

log = logging.getLogger("ingestion.graph")


@dataclass
class ExtractedFacts:
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)


class GraphExtractor(Protocol):
    def extract(self, text: str, *, source: str) -> ExtractedFacts: ...


EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Canonical name as written, e.g. 'Right to Information Act No. 12 of 2016'"},
                    "type": {"type": "string", "enum": list(ENTITY_TYPES)},
                },
                "required": ["name", "type"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string", "description": "UPPER_SNAKE verb, e.g. AMENDS, REPEALS, DEFINES, TIME_LIMIT, APPLIES_TO, EMPOWERS, PENALTY, PART_OF, REFERS_TO"},
                    "object": {"type": "string"},
                },
                "required": ["subject", "predicate", "object"],
            },
        },
    },
    "required": ["entities", "relations"],
}

EXTRACTION_SYSTEM = """You extract a knowledge graph from passages of Sri Lankan law. Output JSON only.

Entities (at most {max_entities}): the acts, provisions (a section or article of a named act,
written like "Section 9, Prevention of Terrorism Act No. 48 of 1979"), institutions (courts,
ministers, commissions, public authorities), persons, places, durations/time limits, and the
key legal concepts the passage is about. Use the full canonical name as written in the text,
including act number and year when present. Do not invent entities that are not in the text.

Relations (at most {max_relations}): facts stated in the passage as subject-predicate-object
between entities you listed. Predicates are short UPPER_SNAKE verbs such as AMENDS, REPEALS,
DEFINES, TIME_LIMIT, APPLIES_TO, EMPOWERS, REQUIRES, PENALTY, PART_OF, REFERS_TO, ESTABLISHES.
Prefer relations that connect acts, sections, institutions and durations; skip trivia."""


def _clean_label(value: Any) -> str:
    return " ".join(str(value or "").split()).strip(" .,;:")[:200]


class LLMGraphExtractor:
    def __init__(self, llm: FallbackLLM, *, max_entities: int = 12, max_relations: int = 10) -> None:
        self.llm = llm
        self.max_entities = max_entities
        self.max_relations = max_relations

    def extract(self, text: str, *, source: str) -> ExtractedFacts:
        """May raise LLMError (provider failure). Malformed output yields empty facts."""
        messages = [
            Message.system(EXTRACTION_SYSTEM.format(max_entities=self.max_entities, max_relations=self.max_relations)),
            Message.user(f"Source: {source}\n\nPassage:\n{text}"),
        ]
        resp = self.llm.invoke(messages, json_schema=EXTRACTION_SCHEMA, temperature=0.0)
        try:
            data = json.loads(resp.text)
        except json.JSONDecodeError:
            log.warning("graph extraction returned non-JSON for '%s': %.120s", source, resp.text)
            return ExtractedFacts()
        entities: list[Entity] = []
        seen: set[str] = set()
        for e in data.get("entities", []) if isinstance(data, dict) else []:
            if not isinstance(e, dict):
                continue
            name = _clean_label(e.get("name"))
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            etype = e.get("type") if e.get("type") in ENTITY_TYPES else "concept"
            entities.append(Entity(label=name, type=etype))
            if len(entities) >= self.max_entities:
                break
        relations: list[Relation] = []
        for r in data.get("relations", []) if isinstance(data, dict) else []:
            if not isinstance(r, dict):
                continue
            s, p, o = _clean_label(r.get("subject")), _clean_label(r.get("predicate")), _clean_label(r.get("object"))
            if s and p and o and s.lower() != o.lower():
                relations.append(Relation(subject=s, predicate=p.upper().replace(" ", "_"), object=o))
            if len(relations) >= self.max_relations:
                break
        return ExtractedFacts(entities=entities, relations=relations)


__all__ = ["EXTRACTION_SCHEMA", "ExtractedFacts", "GraphExtractor", "LLMGraphExtractor"]
