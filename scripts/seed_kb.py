"""Load the sample passages into the local stores. Idempotent (replaces by id).

    python scripts/seed_kb.py [path/to/docs.json]

Each entry: {"id": str, "text": str, "metadata": {...}}. Passages are indexed as-is
(one chunk each) through the ingestion service, so they land in the vector store AND the
knowledge graph (entities/relations extracted by the LLM) and show up in the Ingest tab.

If the app is running (APP_HOST:APP_PORT answers /api/health) the passages are sent to its
POST /api/ingest/records endpoint, because the embedded graph database (Kùzu) allows only
one writer process and the server already holds it. Otherwise the stores are opened directly.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # CHROMA_PATH / GRAPH_PATH in .env are relative to the project root


def _server_url() -> str:
    load_dotenv(ROOT / ".env", override=False)
    return f"http://{os.getenv('APP_HOST') or '127.0.0.1'}:{os.getenv('APP_PORT') or '8080'}"


def seed_via_api(base: str, records: list[dict]) -> None:
    print(f"server is running at {base}: loading through POST /api/ingest/records")
    r = httpx.post(f"{base}/api/ingest/records", json={"records": records}, timeout=600)
    if r.status_code != 200:
        sys.exit(f"server refused: {r.status_code} {r.text[:300]}")
    body = r.json()
    print(f"indexed {body['indexed']}; vector store now holds {body['total_chunks']} chunks", end="")
    g = body.get("graph")
    print(f"; graph: {g['chunks']} chunks, {g['entities']} entities, {g['relations']} relations" if g else "")


def seed_direct(records: list[dict]) -> None:
    from agent.rag import Document
    from app.bootstrap import build_services

    try:
        services = build_services(str(ROOT / ".env"))
    except ValueError as exc:
        sys.exit(str(exc))
    docs = [Document(**d) for d in records]
    print(f"indexing {len(docs)} passages directly (graph extraction {'on' if services.graph is not None else 'off'})")
    n = services.ingestor.index_prechunked(docs)
    print(f"indexed {n}; vector store now holds {services.store.count()} chunks", end="")
    if services.graph is not None:
        s = services.graph.stats()
        print(f"; graph: {s['chunks']} chunks, {s['entities']} entities, {s['relations']} relations")
    else:
        print()


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "seed" / "sample_docs.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    base = _server_url()
    try:
        alive = httpx.get(f"{base}/api/health", timeout=3).status_code == 200
    except httpx.HTTPError:
        alive = False
    if alive:
        seed_via_api(base, records)
    else:
        seed_direct(records)


if __name__ == "__main__":
    main()
