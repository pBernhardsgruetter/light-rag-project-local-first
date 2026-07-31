"""KuzuDB Knowledge Graph Ingestion with typed ontology schema.

Routes entities into typed node tables (Technology, Organization, Person, etc.)
and creates typed RELATES edges with cross-type support. Uses MENTIONS edges
from Chunk → typed entity tables and HAS_CHUNK from Document → Chunk.
"""
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Set
import kuzu
from sentence_transformers import SentenceTransformer
from .chunking import TextChunk
from .ontology import NODE_TABLES, INFRA_TABLES, ENTITY_TYPES, get_table_for_type


class KuzuIngestor:
    def __init__(self, db_path: Path, embedding_model: str = "all-MiniLM-L6-v2"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = kuzu.Database(str(self.db_path))
        self.conn = kuzu.Connection(self.db)
        self.embedder = SentenceTransformer(embedding_model)
        self._created_rel_tables: Set[str] = set()
        self._setup_schema()

    def _safe_execute(self, query: str, params: dict = None):
        """Execute a query, silently ignoring 'already exists' errors."""
        try:
            if params:
                self.conn.execute(query, params)
            else:
                self.conn.execute(query)
        except Exception:
            pass

    def _setup_schema(self):
        """Create all typed node tables, infrastructure tables, and relationship tables."""
        # Infrastructure tables
        for ddl in INFRA_TABLES.values():
            self._safe_execute(ddl)

        # Typed entity node tables
        for ddl in NODE_TABLES.values():
            self._safe_execute(ddl)

        # HAS_CHUNK: Document → Chunk
        self._safe_execute(
            "CREATE REL TABLE IF NOT EXISTS HAS_CHUNK(FROM Document TO Chunk)"
        )

        # MENTIONS: Chunk → each entity type
        for etype in ENTITY_TYPES:
            self._safe_execute(
                f"CREATE REL TABLE IF NOT EXISTS MENTIONS_{etype}"
                f"(FROM Chunk TO {etype})"
            )

        # RELATES: cross-product of all entity types
        for src_type in ENTITY_TYPES:
            for tgt_type in ENTITY_TYPES:
                rel_name = f"RELATES_{src_type}_{tgt_type}"
                self._safe_execute(
                    f"CREATE REL TABLE IF NOT EXISTS {rel_name}"
                    f"(FROM {src_type} TO {tgt_type}, "
                    f"type STRING, confidence DOUBLE)"
                )
                self._created_rel_tables.add(rel_name)

    def ingest_document(
        self,
        doc_path: Path,
        chunks: List[TextChunk],
        resolved_entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
    ):
        doc_id = chunks[0].doc_id if chunks else "doc_0"
        title = doc_path.stem
        ts = datetime.utcnow().isoformat()

        # ── Document ──
        self.conn.execute(
            "MERGE (d:Document {id: $id}) SET d.title = $title, d.source = $src, d.created_at = $ts",
            {"id": doc_id, "title": title, "src": str(doc_path), "ts": ts}
        )

        # ── Chunks & HAS_CHUNK ──
        for chunk in chunks:
            self.conn.execute(
                "MERGE (c:Chunk {id: $id}) SET c.text = $text, c.doc_id = $doc_id, c.start_idx = $s, c.end_idx = $e",
                {"id": chunk.id, "text": chunk.text, "doc_id": doc_id, "s": chunk.start_idx, "e": chunk.end_idx}
            )
            self.conn.execute(
                "MATCH (d:Document {id: $did}), (c:Chunk {id: $cid}) MERGE (d)-[:HAS_CHUNK]->(c)",
                {"did": doc_id, "cid": chunk.id}
            )

        # ── Entities into typed tables + MENTIONS ──
        inserted_entities: Set[str] = set()
        entity_id_to_table: Dict[str, str] = {}

        for ent in resolved_entities:
            eid = ent["entity_id"]
            table_name = get_table_for_type(ent["type"])
            entity_id_to_table[eid] = table_name

            if eid not in inserted_entities:
                self.conn.execute(
                    f"MERGE (e:{table_name} {{id: $id}}) SET e.name = $name",
                    {"id": eid, "name": ent["canonical_name"]}
                )
                inserted_entities.add(eid)

            # MENTIONS_<Type>: Chunk → Entity
            mentions_rel = f"MENTIONS_{table_name}"
            self.conn.execute(
                f"MATCH (c:Chunk {{id: $cid}}), (e:{table_name} {{id: $eid}}) "
                f"MERGE (c)-[:{mentions_rel}]->(e)",
                {"cid": ent["chunk_id"], "eid": eid}
            )

        # ── Relations with fuzzy matching ──
        ent_name_to_id: Dict[str, str] = {}
        ent_id_to_type: Dict[str, str] = {}

        for ent in resolved_entities:
            eid = ent["entity_id"]
            if "original_text" in ent:
                ent_name_to_id[ent["original_text"].lower()] = eid
            if "canonical_name" in ent:
                ent_name_to_id[ent["canonical_name"].lower()] = eid
            ent_id_to_type[eid] = get_table_for_type(ent["type"])

        for rel in relations:
            s_name = rel["source"].lower()
            t_name = rel["target"].lower()

            sid = ent_name_to_id.get(s_name)
            tid = ent_name_to_id.get(t_name)

            # Substring fallback
            if not sid:
                for k, v in ent_name_to_id.items():
                    if k in s_name or s_name in k:
                        sid = v
                        break
            if not tid:
                for k, v in ent_name_to_id.items():
                    if k in t_name or t_name in k:
                        tid = v
                        break

            if sid and tid and sid != tid:
                src_table = ent_id_to_type.get(sid, "Concept")
                tgt_table = ent_id_to_type.get(tid, "Concept")
                rel_table = f"RELATES_{src_table}_{tgt_table}"
                rel_type = rel.get("type", rel.get("relation", "RELATED_TO")).upper().replace(" ", "_")

                self.conn.execute(
                    f"MATCH (s:{src_table} {{id: $sid}}), (t:{tgt_table} {{id: $tid}}) "
                    f"MERGE (s)-[r:{rel_table} {{type: $type}}]->(t) "
                    f"SET r.confidence = $conf",
                    {"sid": sid, "tid": tid, "type": rel_type, "conf": float(rel.get("confidence", 0.8))}
                )

    def close(self):
        self.conn = None
        self.db = None
