"""Kuzu-backed GraphRAG query service.

The extractor owns the typed Kuzu schema. This service provides a single
retrieval contract over that schema: lexical + dense candidate ranking,
evidence-aware relation expansion, and grounded answer metadata.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import kuzu
import numpy as np
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

try:
    from services.extractor.extractor.quality import lexical_score, normalize_text
except ImportError:
    # The service container copies only this directory into /app.
    _TOKEN_RE = re.compile(r"\w+", re.UNICODE)

    def normalize_text(value: str) -> str:
        return " ".join(_TOKEN_RE.findall((value or "").casefold()))

    def lexical_score(query: str, text: str) -> float:
        q = set(normalize_text(query).split())
        t = set(normalize_text(text).split())
        return len(q & t) / max(len(q), 1)


app = FastAPI(title="GraphRAG Service", version="0.3.0")

ENTITY_TABLES = [
    "Technology", "Organization", "Person", "Event", "Concept",
    "Benchmark", "Location", "Product", "Regulation",
]

_embedder: Optional[SentenceTransformer] = None
_embedder_failed = False
_snapshot: Optional[Dict[str, Any]] = None
_snapshot_signature: Optional[Tuple[int, int]] = None


def get_embedder() -> Optional[SentenceTransformer]:
    global _embedder, _embedder_failed
    if _embedder is not None or _embedder_failed:
        return _embedder
    model_name = os.getenv(
        "RETRIEVAL_EMBEDDING_MODEL",
        os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    )
    try:
        _embedder = SentenceTransformer(model_name)
    except Exception:
        # Lexical retrieval remains available if a model cannot be downloaded
        # or loaded. The health endpoint exposes this degraded state.
        _embedder_failed = True
    return _embedder


def _encode(texts: List[str]) -> Optional[np.ndarray]:
    embedder = get_embedder()
    if embedder is None or not texts:
        return None
    try:
        vectors = embedder.encode(texts, convert_to_numpy=True)
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-8)
    except Exception:
        return None


def _db_signature(db_path: str) -> Tuple[int, int]:
    path = Path(db_path)
    main_mtime = path.stat().st_mtime_ns if path.exists() else 0
    wal = Path(f"{db_path}.wal")
    wal_mtime = wal.stat().st_mtime_ns if wal.exists() else 0
    return main_mtime, wal_mtime


def _rows(conn: kuzu.Connection, query: str, params: Optional[dict] = None) -> List[tuple]:
    result = conn.execute(query, params) if params is not None else conn.execute(query)
    output = []
    while result.has_next():
        output.append(result.get_next())
    return output


def load_graph_snapshot(db_path: str) -> Dict[str, Any]:
    """Load a consistent read snapshot and build lightweight search vectors."""
    global _snapshot, _snapshot_signature
    signature = _db_signature(db_path)
    if _snapshot is not None and _snapshot_signature == signature:
        return _snapshot

    if not os.path.exists(db_path):
        return {
            "entities": [],
            "chunks": [],
            "assertions": [],
            "assertion_store_available": False,
        }

    db = kuzu.Database(db_path, read_only=True)
    conn = kuzu.Connection(db)
    entities: List[Dict[str, Any]] = []
    chunks: List[Dict[str, Any]] = []
    assertions: List[Dict[str, Any]] = []
    assertion_store_available = False

    for table in ENTITY_TABLES:
        try:
            try:
                entity_rows = _rows(
                    conn,
                    f"MATCH (c:Chunk)-[:MENTIONS_{table}]->(e:{table}) "
                    "RETURN DISTINCT e.id, e.name",
                )
            except Exception:
                entity_rows = _rows(conn, f"MATCH (e:{table}) RETURN e.id, e.name")
            for row in entity_rows:
                if row and row[0] and row[1]:
                    entities.append({"id": row[0], "name": row[1], "table": table})
        except Exception:
            continue

    try:
        for row in _rows(conn, "MATCH (c:Chunk) RETURN c.id, c.text, c.doc_id"):
            if row and row[0] and row[1]:
                chunks.append({"id": row[0], "text": row[1], "doc_id": row[2] or ""})
    except Exception:
        pass

    try:
        assertion_store_available = True
        for row in _rows(
            conn,
            """MATCH (a:RelationAssertion)
               RETURN a.id, a.source_id, a.source_table, a.predicate,
                      a.raw_predicate, a.target_id, a.target_table,
                      a.confidence, a.chunk_id, a.doc_id""",
        ):
            if row and row[0]:
                assertions.append({
                    "id": row[0],
                    "source_id": row[1],
                    "source_table": row[2],
                    "predicate": row[3],
                    "raw_predicate": row[4],
                    "target_id": row[5],
                    "target_table": row[6],
                    "confidence": float(row[7] or 0.0),
                    "chunk_id": row[8] or "",
                    "doc_id": row[9] or "",
                })
    except Exception:
        # RelationAssertion is added by the fixed extractor. Existing graphs
        # can still be queried through the typed-edge fallback below.
        pass

    entity_vectors = _encode([item["name"] for item in entities])
    chunk_vectors = _encode([item["text"] for item in chunks])
    _snapshot = {
        "entities": entities,
        "chunks": chunks,
        "assertions": assertions,
        "assertion_store_available": assertion_store_available,
        "entity_vectors": entity_vectors,
        "chunk_vectors": chunk_vectors,
    }
    _snapshot_signature = signature
    return _snapshot


def _rank_items(
    query: str,
    items: List[Dict[str, Any]],
    vectors: Optional[np.ndarray],
    text_key: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    if not items:
        return []
    query_vector = _encode([query])
    scored = []
    for index, item in enumerate(items):
        lexical = lexical_score(query, item.get(text_key, ""))
        dense = 0.0
        if query_vector is not None and vectors is not None and index < len(vectors):
            dense = float(np.dot(query_vector[0], vectors[index]))
        score = (0.60 * dense + 0.40 * lexical) if query_vector is not None else lexical
        scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    result = []
    for score, item in scored[:top_k]:
        copy = dict(item)
        copy["score"] = round(float(score), 6)
        result.append(copy)
    return result


def _node_key(table: str, entity_id: str) -> str:
    return f"{table}:{entity_id}"


def _legacy_edges(
    db_path: str,
    seed_entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Read old typed edges when the graph predates RelationAssertion."""
    if not seed_entities:
        return []
    db = kuzu.Database(db_path, read_only=True)
    conn = kuzu.Connection(db)
    edges = []
    seen = set()
    for entity in seed_entities:
        table = entity["table"]
        entity_id = entity["id"]
        for target_table in ENTITY_TABLES:
            rel_table = f"RELATES_{table}_{target_table}"
            try:
                rows = _rows(
                    conn,
                    f"MATCH (s:{table} {{id: $id}})-[r:{rel_table}]->(t:{target_table}) "
                    "RETURN s.id, s.name, r.type, r.confidence, t.id, t.name",
                    {"id": entity_id},
                )
                for row in rows:
                    key = (row[0], row[2], row[4])
                    if key in seen:
                        continue
                    seen.add(key)
                    edges.append({
                        "id": f"legacy:{table}:{row[0]}:{row[2]}:{target_table}:{row[4]}",
                        "source": _node_key(table, row[0]),
                        "source_name": row[1],
                        "source_table": table,
                        "target": _node_key(target_table, row[4]),
                        "target_name": row[5],
                        "target_table": target_table,
                        "predicate": row[2] or "RELATED_TO",
                        "confidence": float(row[3] or 0.0),
                        "chunk_id": "",
                        "doc_id": "",
                    })
            except Exception:
                continue
    return edges


def retrieve(query: str, db_path: str, top_k: int = 10, mode: str = "hybrid") -> Dict[str, Any]:
    snapshot = load_graph_snapshot(db_path)
    top_k = max(1, min(int(top_k), 50))
    mode = mode if mode in {"local", "global", "hybrid"} else "hybrid"

    ranked_entities = _rank_items(
        query, snapshot["entities"], snapshot.get("entity_vectors"), "name", top_k
    )
    ranked_chunks = _rank_items(
        query, snapshot["chunks"], snapshot.get("chunk_vectors"), "text", top_k * 2
    )

    seed_ids = {(item["table"], item["id"]) for item in ranked_entities}
    assertions = []
    for assertion in snapshot.get("assertions", []):
        source = (assertion["source_table"], assertion["source_id"])
        target = (assertion["target_table"], assertion["target_id"])
        if source in seed_ids or target in seed_ids:
            assertions.append(dict(assertion))

    edges = []
    for assertion in assertions:
        edges.append({
            "id": assertion["id"],
            "source": _node_key(assertion["source_table"], assertion["source_id"]),
            "source_name": next(
                (e["name"] for e in snapshot["entities"]
                 if e["table"] == assertion["source_table"] and e["id"] == assertion["source_id"]),
                assertion["source_id"],
            ),
            "source_table": assertion["source_table"],
            "target": _node_key(assertion["target_table"], assertion["target_id"]),
            "target_name": next(
                (e["name"] for e in snapshot["entities"]
                 if e["table"] == assertion["target_table"] and e["id"] == assertion["target_id"]),
                assertion["target_id"],
            ),
            "target_table": assertion["target_table"],
            "predicate": assertion["predicate"],
            "confidence": assertion["confidence"],
            "chunk_id": assertion["chunk_id"],
            "doc_id": assertion["doc_id"],
        })

    if not edges and not snapshot.get("assertion_store_available", False):
        edges = _legacy_edges(db_path, ranked_entities)

    edge_ids = {edge["source"] for edge in edges} | {edge["target"] for edge in edges}
    node_by_key = {
        _node_key(entity["table"], entity["id"]): entity
        for entity in snapshot["entities"]
    }
    nodes = []
    for entity in ranked_entities:
        nodes.append({
            "id": _node_key(entity["table"], entity["id"]),
            "name": entity["name"],
            "type": entity["table"],
            "score": entity["score"],
        })
    for key in edge_ids:
        entity = node_by_key.get(key)
        if entity and not any(node["id"] == key for node in nodes):
            nodes.append({
                "id": key,
                "name": entity["name"],
                "type": entity["table"],
                "score": 0.0,
            })

    if mode == "local":
        ranked_chunks = [
            chunk for chunk in ranked_chunks
            if any(edge.get("chunk_id") == chunk["id"] for edge in edges)
        ] or ranked_chunks[:top_k]
    elif mode == "global":
        edges = []
        nodes = []

    citations = [
        {
            "chunk_id": chunk["id"],
            "doc_id": chunk["doc_id"],
            "score": chunk["score"],
        }
        for chunk in ranked_chunks[:top_k]
    ]
    context_parts = []
    if nodes:
        context_parts.append(
            "ENTITIES:\n" + "\n".join(
                f"[{node['id']}] {node['name']} ({node['type']})"
                for node in nodes
            )
        )
    if edges:
        context_parts.append(
            "RELATIONS:\n" + "\n".join(
                f"[{edge['id']}] {edge['source_name']} --{edge['predicate']}--> "
                f"{edge['target_name']} (confidence={edge['confidence']:.2f}; "
                f"evidence={edge['chunk_id'] or 'unknown'})"
                for edge in edges[:top_k * 2]
            )
        )
    if ranked_chunks:
        context_parts.append(
            "SOURCE CHUNKS:\n" + "\n---\n".join(
                f"[C:{chunk['id']}] (document={chunk['doc_id']})\n{chunk['text']}"
                for chunk in ranked_chunks[:top_k]
            )
        )

    return {
        "context": "\n\n".join(context_parts) or "No evidence found.",
        "nodes": nodes,
        "edges": edges[:top_k * 2],
        "citations": citations,
        "ranked_chunks": ranked_chunks,
    }


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: str = "hybrid"
    top_k: int = Field(default=10, ge=1, le=50)


class QueryResponse(BaseModel):
    query: str
    mode: str
    result: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


def get_graph_context(query: str, db_path: str = "/database/db", top_k: int = 10) -> str:
    return retrieve(query, db_path, top_k=top_k, mode="hybrid")["context"]


@app.get("/health")
async def health_check():
    db_path = os.getenv("DB_PATH", "/database/db")
    snapshot = load_graph_snapshot(db_path)
    return {
        "status": "ok",
        "service": "graphrag",
        "version": "0.3.0",
        "entity_count": len(snapshot.get("entities", [])),
        "chunk_count": len(snapshot.get("chunks", [])),
        "relation_assertion_count": len(snapshot.get("assertions", [])),
        "embedding_status": "ready" if get_embedder() is not None else "lexical_fallback",
    }


@app.post("/query", response_model=QueryResponse)
async def handle_query(req: QueryRequest):
    try:
        db_path = os.getenv("DB_PATH", "/database/db")
        retrieval = retrieve(req.query, db_path, top_k=req.top_k, mode=req.mode)
        context = retrieval["context"]
        warnings: List[str] = []
        if get_embedder() is None:
            warnings.append("Dense embedding model unavailable; lexical retrieval was used.")

        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            result_text = "LLM is not configured. Retrieved evidence:\n\n" + context
        else:
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                timeout=60.0,
            )
            prompt = f"""Answer the user's question using only the evidence below.
Respond in the same language as the question. Cite supporting chunks using
[C:<chunk_id>]. If the evidence is insufficient, say so clearly instead of
guessing. Treat graph relations as claims that require their cited evidence.

EVIDENCE
========
{context}

QUESTION
========
{req.query}
"""
            try:
                completion = await client.chat.completions.create(
                    model=os.getenv("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2048,
                )
                result_text = (completion.choices[0].message.content or "").strip()
                if not result_text:
                    raise RuntimeError("empty model response")
            except Exception:
                warnings.append("Answer generation failed; returning retrieved evidence.")
                result_text = "Retrieved evidence:\n\n" + context

        return QueryResponse(
            query=req.query,
            mode=req.mode if req.mode in {"local", "global", "hybrid"} else "hybrid",
            result=result_text,
            citations=retrieval["citations"],
            nodes=retrieval["nodes"],
            edges=retrieval["edges"],
            warnings=warnings,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {type(exc).__name__}") from exc


@app.post("/invalidate-cache")
async def invalidate_cache():
    global _snapshot, _snapshot_signature, _embedder, _embedder_failed
    _snapshot = None
    _snapshot_signature = None
    _embedder = None
    _embedder_failed = False
    return {"status": "cache invalidated"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9621)
