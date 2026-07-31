import json
import os
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import kuzu
from lightrag.base import BaseGraphStorage

class KuzuStorage(BaseGraphStorage):
    def __init__(self, namespace: str, global_config: Dict[str, Any], db_path: str = "/database/db"):
        super().__init__(namespace, global_config)
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        if os.path.exists(db_path):
            self.db = kuzu.Database(db_path, read_only=True)
            self.conn = kuzu.Connection(self.db)
            self._ensure_schema()
        else:
            self.db = None
            self.conn = None

    def _ensure_schema(self):
        try:
            self.conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS _Entity(
                    _id STRING, entity_name STRING, entity_type STRING,
                    description STRING, source_id STRING,
                    file_path STRING, properties STRING, PRIMARY KEY(_id)
                )
            """)
        except Exception:
            pass

        try:
            self.conn.execute("""
                CREATE REL TABLE IF NOT EXISTS _Relates(
                    FROM _Entity TO _Entity,
                    weight DOUBLE, description STRING,
                    source_id STRING, file_path STRING, properties STRING
                )
            """)
        except Exception:
            pass

    async def has_node(self, node_id: str) -> bool:
        result = self.conn.execute("MATCH (e:_Entity {_id: $id}) RETURN count(e) as c", {"id": node_id})
        if result.has_next():
            return result.get_next()[0] > 0
        return False

    async def has_edge(self, source_id: str, target_id: str) -> bool:
        result = self.conn.execute(
            "MATCH (s:_Entity {_id: $sid})-[r:_Relates]->(t:_Entity {_id: $tid}) RETURN count(r) as c",
            {"sid": source_id, "tid": target_id}
        )
        if result.has_next():
            return result.get_next()[0] > 0
        return False

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        result = self.conn.execute("MATCH (e:_Entity {_id: $id}) RETURN e", {"id": node_id})
        if result.has_next():
            row = result.get_next()[0]
            props = json.loads(row.get("properties", "{}")) if isinstance(row, dict) and "properties" in row else {}
            return {
                "_id": node_id,
                "entity_name": row.get("entity_name", ""),
                "entity_type": row.get("entity_type", "Unknown"),
                "description": row.get("description", ""),
                "source_id": row.get("source_id", ""),
                "file_path": row.get("file_path", ""),
                "properties": props
            }
        return None

    async def get_edge(self, source_id: str, target_id: str) -> Optional[Dict[str, Any]]:
        result = self.conn.execute(
            "MATCH (s:_Entity {_id: $sid})-[r:_Relates]->(t:_Entity {_id: $tid}) RETURN r",
            {"sid": source_id, "tid": target_id}
        )
        if result.has_next():
            row = result.get_next()[0]
            props = json.loads(row.get("properties", "{}")) if isinstance(row, dict) and "properties" in row else {}
            return {
                "weight": row.get("weight", 1.0),
                "description": row.get("description", ""),
                "source_id": row.get("source_id", ""),
                "file_path": row.get("file_path", ""),
                "properties": props
            }
        return None

    async def get_node_edges(self, source_id: str) -> Optional[List[tuple]]:
        result = self.conn.execute(
            "MATCH (s:_Entity {_id: $sid})-[r:_Relates]->(t:_Entity) RETURN s._id, t._id",
            {"sid": source_id}
        )
        edges = []
        while result.has_next():
            row = result.get_next()
            edges.append((row[0], row[1]))
        return edges

    async def upsert_node(self, node_id: str, node_data: Dict[str, Any]) -> None:
        props = json.dumps(node_data.get("properties", {}))
        self.conn.execute(
            """
            MERGE (e:_Entity {_id: $id})
            SET e.entity_name = $name,
                e.entity_type = $type,
                e.description = $desc,
                e.source_id = $src,
                e.file_path = $fp,
                e.properties = $props
            """,
            {
                "id": node_id,
                "name": node_data.get("entity_name", node_id),
                "type": node_data.get("entity_type", "Unknown"),
                "desc": node_data.get("description", ""),
                "src": node_data.get("source_id", ""),
                "fp": node_data.get("file_path", ""),
                "props": props
            }
        )

    async def upsert_edge(self, source_id: str, target_id: str, edge_data: Dict[str, Any]) -> None:
        props = json.dumps(edge_data.get("properties", {}))
        self.conn.execute(
            """
            MATCH (s:_Entity {_id: $sid}), (t:_Entity {_id: $tid})
            MERGE (s)-[r:_Relates]->(t)
            SET r.weight = $w,
                r.description = $desc,
                r.source_id = $src,
                r.file_path = $fp,
                r.properties = $props
            """,
            {
                "sid": source_id,
                "tid": target_id,
                "w": float(edge_data.get("weight", 1.0)),
                "desc": edge_data.get("description", ""),
                "src": edge_data.get("source_id", ""),
                "fp": edge_data.get("file_path", ""),
                "props": props
            }
        )

    async def node_degrees(self, node_ids: List[str]) -> Dict[str, int]:
        if not node_ids:
            return {}
        degrees = {}
        for nid in node_ids:
            res = self.conn.execute(
                "MATCH (e:_Entity {_id: $id})-[r:_Relates]-() RETURN count(r)",
                {"id": nid}
            )
            if res.has_next():
                degrees[nid] = res.get_next()[0]
            else:
                degrees[nid] = 0
        return degrees

    async def embed_nodes(self, algorithm: str) -> None:
        pass

    async def delete_node(self, node_id: str) -> None:
        self.conn.execute("MATCH (e:_Entity {_id: $id}) DETACH DELETE e", {"id": node_id})
