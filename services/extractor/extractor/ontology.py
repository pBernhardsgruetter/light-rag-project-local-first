"""Ontology schema definitions for the Knowledge Graph.

Defines typed node tables and relationship tables for KuzuDB,
enabling structured multi-hop traversal and ontology-aware queries.
"""

from typing import Dict, List

# ─────────────────────────────────────────────────
# Typed Node Tables
# ─────────────────────────────────────────────────
NODE_TABLES: Dict[str, str] = {
    "Technology": """
        CREATE NODE TABLE IF NOT EXISTS Technology(
            id STRING, name STRING, description STRING,
            PRIMARY KEY(id)
        )
    """,
    "Organization": """
        CREATE NODE TABLE IF NOT EXISTS Organization(
            id STRING, name STRING, description STRING,
            PRIMARY KEY(id)
        )
    """,
    "Person": """
        CREATE NODE TABLE IF NOT EXISTS Person(
            id STRING, name STRING, affiliation STRING,
            PRIMARY KEY(id)
        )
    """,
    "Event": """
        CREATE NODE TABLE IF NOT EXISTS Event(
            id STRING, name STRING, year STRING,
            PRIMARY KEY(id)
        )
    """,
    "Concept": """
        CREATE NODE TABLE IF NOT EXISTS Concept(
            id STRING, name STRING, description STRING,
            PRIMARY KEY(id)
        )
    """,
    "Benchmark": """
        CREATE NODE TABLE IF NOT EXISTS Benchmark(
            id STRING, name STRING, metric STRING,
            PRIMARY KEY(id)
        )
    """,
    "Location": """
        CREATE NODE TABLE IF NOT EXISTS Location(
            id STRING, name STRING,
            PRIMARY KEY(id)
        )
    """,
    "Product": """
        CREATE NODE TABLE IF NOT EXISTS Product(
            id STRING, name STRING, description STRING,
            PRIMARY KEY(id)
        )
    """,
    "Regulation": """
        CREATE NODE TABLE IF NOT EXISTS Regulation(
            id STRING, name STRING, description STRING,
            PRIMARY KEY(id)
        )
    """,
}

# Infrastructure tables (always present)
INFRA_TABLES: Dict[str, str] = {
    "Document": """
        CREATE NODE TABLE IF NOT EXISTS Document(
            id STRING, title STRING, source STRING,
            created_at STRING, PRIMARY KEY(id)
        )
    """,
    "Chunk": """
        CREATE NODE TABLE IF NOT EXISTS Chunk(
            id STRING, text STRING, doc_id STRING,
            start_idx INT64, end_idx INT64, PRIMARY KEY(id)
        )
    """,
    # Relation assertions make provenance first-class without forcing a
    # migration of the original typed relationship tables. Each assertion is
    # tied to the chunk that produced it and can be independently deleted or
    # displayed as evidence.
    "RelationAssertion": """
        CREATE NODE TABLE IF NOT EXISTS RelationAssertion(
            id STRING, source_id STRING, source_table STRING,
            predicate STRING, raw_predicate STRING,
            target_id STRING, target_table STRING,
            confidence DOUBLE, chunk_id STRING, doc_id STRING,
            PRIMARY KEY(id)
        )
    """,
}

# ─────────────────────────────────────────────────
# Typed Relationship Tables
# ─────────────────────────────────────────────────

# All entity type names for generating cross-type REL tables
ENTITY_TYPES: List[str] = list(NODE_TABLES.keys())

# Map GLiNER/extractor type labels → KuzuDB table names
TYPE_LABEL_MAP: Dict[str, str] = {
    "technology": "Technology",
    "organization": "Organization",
    "person": "Person",
    "event": "Event",
    "concept": "Concept",
    "benchmark": "Benchmark",
    "location": "Location",
    "product": "Product",
    "regulation": "Regulation",
}

def get_table_for_type(entity_type: str) -> str:
    """Map an entity type label to its KuzuDB table name."""
    return TYPE_LABEL_MAP.get(entity_type.lower(), "Concept")
