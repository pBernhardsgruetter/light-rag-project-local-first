"""LightRAG Query Server with embedding-based graph traversal.

Instead of returning static LIMIT results, this server:
1. Embeds the query using SentenceTransformer
2. Finds the most relevant entities via cosine similarity
3. Traverses the graph 1-2 hops from those entities
4. Retrieves chunks that MENTION the found entities
5. Sends the structured context to an LLM for synthesis
"""
import os
import numpy as np
from typing import List, Dict, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from openai import AsyncOpenAI
import kuzu
from sentence_transformers import SentenceTransformer

app = FastAPI(title="LightRAG Service", version="0.2.0")

# ── Lazy-loaded globals ──
_embedder = None
_entity_cache = None  # List of (id, name, type, table, embedding)

ENTITY_TABLES = [
    "Technology", "Organization", "Person", "Event",
    "Concept", "Benchmark", "Location", "Product", "Regulation"
]


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        _embedder = SentenceTransformer(model_name)
    return _embedder


def load_entity_cache(db_path: str) -> List[Dict]:
    """Load all entities from all typed tables and compute embeddings."""
    global _entity_cache
    if _entity_cache is not None:
        return _entity_cache

    if not os.path.exists(db_path):
        _entity_cache = []
        return _entity_cache

    embedder = get_embedder()
    entities = []

    try:
        db = kuzu.Database(db_path, read_only=True)
        conn = kuzu.Connection(db)

        for table in ENTITY_TABLES:
            try:
                res = conn.execute(f"MATCH (e:{table}) RETURN e.id, e.name")
                while res.has_next():
                    row = res.get_next()
                    if row and row[0] and row[1]:
                        entities.append({
                            "id": row[0],
                            "name": row[1],
                            "table": table,
                        })
            except Exception:
                continue

        del conn
        del db
    except Exception:
        _entity_cache = []
        return _entity_cache

    if entities:
        names = [e["name"] for e in entities]
        embeddings = embedder.encode(names, convert_to_numpy=True)
        for ent, emb in zip(entities, embeddings):
            ent["embedding"] = emb

    _entity_cache = entities
    return _entity_cache


def find_relevant_entities(query: str, top_k: int = 10) -> List[Dict]:
    """Find top-k entities most similar to the query via cosine similarity."""
    db_path = os.getenv("DB_PATH", "/database/db")
    entities = load_entity_cache(db_path)
    if not entities:
        return []

    embedder = get_embedder()
    q_emb = embedder.encode([query], convert_to_numpy=True)[0]

    # Cosine similarity (embeddings are not normalized)
    scores = []
    for ent in entities:
        emb = ent.get("embedding")
        if emb is not None:
            sim = float(np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-8))
            scores.append((sim, ent))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [ent for _, ent in scores[:top_k]]


def get_graph_context(query: str, db_path: str = "/database/db") -> str:
    """Build rich context by traversing the graph from query-relevant entities."""
    if not os.path.exists(db_path):
        return "Keine Graph-Datenbank gefunden."

    # Step 1: Find relevant entities
    relevant = find_relevant_entities(query, top_k=10)
    if not relevant:
        return "Keine relevanten Entitäten gefunden."

    try:
        db = kuzu.Database(db_path, read_only=True)
        conn = kuzu.Connection(db)

        entity_lines = []
        relation_lines = []
        chunk_texts = []
        seen_chunks = set()

        for ent in relevant:
            eid = ent["id"]
            ename = ent["name"]
            etable = ent["table"]
            entity_lines.append(f"• {ename} ({etable})")

            # Step 2: Find relations FROM this entity
            for tgt_table in ENTITY_TABLES:
                rel_table = f"RELATES_{etable}_{tgt_table}"
                try:
                    res = conn.execute(
                        f"MATCH (s:{etable} {{id: $eid}})-[r:{rel_table}]->(t:{tgt_table}) "
                        f"RETURN s.name, r.type, t.name",
                        {"eid": eid}
                    )
                    while res.has_next():
                        row = res.get_next()
                        if row:
                            relation_lines.append(f"  {row[0]} —[{row[1]}]→ {row[2]}")
                except Exception:
                    continue

            # Step 3: Find relations TO this entity
            for src_table in ENTITY_TABLES:
                rel_table = f"RELATES_{src_table}_{etable}"
                try:
                    res = conn.execute(
                        f"MATCH (s:{src_table})-[r:{rel_table}]->(t:{etable} {{id: $eid}}) "
                        f"RETURN s.name, r.type, t.name",
                        {"eid": eid}
                    )
                    while res.has_next():
                        row = res.get_next()
                        if row:
                            relation_lines.append(f"  {row[0]} —[{row[1]}]→ {row[2]}")
                except Exception:
                    continue

            # Step 4: Find chunks that MENTION this entity
            mentions_rel = f"MENTIONS_{etable}"
            try:
                res = conn.execute(
                    f"MATCH (c:Chunk)-[:{mentions_rel}]->(e:{etable} {{id: $eid}}) "
                    f"RETURN c.id, c.text",
                    {"eid": eid}
                )
                while res.has_next():
                    row = res.get_next()
                    if row and row[0] not in seen_chunks:
                        seen_chunks.add(row[0])
                        chunk_texts.append(row[1])
            except Exception:
                continue

        # Deduplicate relations
        relation_lines = list(dict.fromkeys(relation_lines))

        # Build context
        parts = []
        if entity_lines:
            parts.append("RELEVANTE ENTITÄTEN:\n" + "\n".join(entity_lines))
        if relation_lines:
            parts.append("GRAPH-RELATIONEN:\n" + "\n".join(relation_lines))
        if chunk_texts:
            # Limit to top 8 chunks to stay within LLM context
            parts.append("QUELL-TEXTABSCHNITTE:\n" + "\n---\n".join(chunk_texts[:8]))

        return "\n\n".join(parts) if parts else "Graph DB ist noch leer."

    except Exception as e:
        return f"Fehler beim Abfragen des KuzuDB-Graphen: {e}"
    finally:
        try:
            del conn
            del db
        except Exception:
            pass


class QueryRequest(BaseModel):
    query: str
    mode: Optional[str] = "hybrid"
    top_k: Optional[int] = 10


class QueryResponse(BaseModel):
    query: str
    mode: str
    result: str


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "lightrag", "version": "0.2.0"}


@app.post("/query", response_model=QueryResponse)
async def handle_query(req: QueryRequest):
    try:
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        model_name = os.getenv("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")

        context = get_graph_context(req.query, db_path=os.getenv("DB_PATH", "/database/db"))

        if not api_key:
            result_text = f"Kontext aus KuzuDB:\n{context}"
        else:
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
            prompt = f"""Du bist ein Experte für Wissensgraph-Analyse. Beantworte die Frage ausführlich und präzise auf Deutsch.

Nutze dabei ALLE der folgenden Informationsquellen:
1. Die GRAPH-RELATIONEN zeigen dir strukturierte Beziehungen zwischen Entitäten
2. Die RELEVANTE ENTITÄTEN geben dir die Schlüsselkonzepte
3. Die QUELL-TEXTABSCHNITTE enthalten den Originaltext mit Details

Verknüpfe die Informationen aus dem Graphen mit dem Originaltext für eine vollständige Antwort.

KONTEXT AUS DEM WISSENSGRAPHEN:
---------------------------------
{context}

FRAGE:
{req.query}
"""
            try:
                completion = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2048
                )
                result_text = completion.choices[0].message.content.strip()
            except Exception as llm_err:
                result_text = (
                    f"[Hinweis: LLM API ({llm_err})]\n\n"
                    f"--- EXTRAHIERTER KUZUDB GRAPH-KONTEXT ---\n{context}"
                )

        return QueryResponse(query=req.query, mode=req.mode or "hybrid", result=result_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/invalidate-cache")
async def invalidate_cache():
    """Force reload of entity cache (call after re-ingestion)."""
    global _entity_cache
    _entity_cache = None
    return {"status": "cache invalidated"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9621)
