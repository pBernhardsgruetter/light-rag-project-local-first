# 📋 **Agenten-Plan: Autonomer Aufbau eines GraphRAG-Systems mit LightRAG**

> **Ziel**: Vollständig automatisierter Aufbau einer GraphRAG-Pipeline mit Knowledge Graph Konstruktion, LightRAG Retrieval, und ThinkingCap (via OpenRouter) bis lokale Infrastruktur bereitsteht. Docker-basiert, iterativ, selbstvalidierend.

---

## 🎯 **Projekt-Übersicht**

| Aspekt | Entscheidung |
|--------|--------------|
| **LLM (Interim)** | OpenRouter: `nvidia/nemotron-3-ultra-550b-a55b:free` (bzw. konfigurierbar) |
| **LLL (Final)** | Lokales ThinkingCap-Qwen3.6-27B (Ollama/vLLM) |
| **Embeddings** | `bge-m3` (lokal via Ollama/OPEA) |
| **NER** | GLiNER-large-v2.1 (Zero-Shot, GPU-Batch) |
| **RE** | REBEL-large (BART, OpenIE) |
| **Entity Resolution** | bge-m3 Embeddings + FAISS/HNSW + LLM-Verifikation |
| **Graph DB** | KuzuDB (embedded, Cypher, schnelle Metadata-Filter) |
| **Retrieval Framework** | LightRAG (Dual-Level, Incremental, Graph+Vector) |
| **Orchestrierung** | Docker Compose + Python Agents |
| **Validierung** | Automatische Tests nach jedem Schritt (Ingestion → Retrieval → QA) |

---

## 🏗️ **Architektur**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DOCKER COMPOSE STACK                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  EXTRACTOR   │  │  GRAPH_DB    │  │  LIGHTRAG    │  │  VALIDATOR   │   │
│  │  (GPU)       │  │  (KuzuDB)    │  │  (Retrieval) │  │  (Tests)     │   │
│  │              │  │              │  │              │  │              │   │
│  │ GLiNER       │  │ Cypher       │  │ Dual-Level   │  │ Ingestion    │   │
│  │ REBEL        │  │ Metadata     │  │ Graph+Vector │  │ Retrieval    │   │
│  │ EntityRes    │  │ ACID         │  │ Incremental  │  │ QA-Pairs     │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│         │                 │                 │                 │            │
│         └─────────────────┼─────────────────┼─────────────────┘            │
│                           ▼                 ▼                              │
│                    ┌──────────────┐  ┌──────────────┐                     │
│                    │  SHARED VOL  │  │  OPENROUTER  │                     │
│                    │  (Graph,     │  │  (LLM API)   │                     │
│                    │   Index,     │  │              │                     │
│                    │   Chunks)    │  │  ThinkingCap │                     │
│                    └──────────────┘  └──────────────┘                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 **Docker Compose Struktur**

```yaml
# docker-compose.yml
version: '3.8'

services:
  # ─────────────────────────────────────────────────────────────
  # EXTRACTOR SERVICE (GPU) - Batch Processing
  # ─────────────────────────────────────────────────────────────
  extractor:
    build:
      context: ./services/extractor
      dockerfile: Dockerfile.gpu
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - BATCH_SIZE=32
      - GLINER_MODEL=urchade/gliner_large-v2.1
      - REBEL_MODEL=Babelscape/rebel-large
      - EMBEDDING_MODEL=BAAI/bge-m3
      - OPENROUTER_API_KEY=${OPEN...KEY}
      - OPENROUTER_MODEL=${OPENROUTER_MODEL:-nvidia/nemotron-3-ultra-550b-a55b:free}
    volumes:
      - ./data:/data
      - ./models:/models
      - ./output:/output
      - ./config:/config
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: python -m extractor.pipeline --config /config/extractor.yaml

  # ─────────────────────────────────────────────────────────────
  # KUZUDB GRAPH DATABASE (Embedded, via Python API)
  # ─────────────────────────────────────────────────────────────
  kuzu:
    image: kuzudb/kuzu:latest
    volumes:
      - ./data/kuzu:/database
    ports:
      - "8000:8000"  # REST API (optional)
    command: ["--read-only=false"]

  # ─────────────────────────────────────────────────────────────
  # LIGHTRAG SERVICE (Retrieval + Generation)
  # ─────────────────────────────────────────────────────────────
  lightrag:
    build:
      context: ./services/lightrag
      dockerfile: Dockerfile
    environment:
      - WORKING_DIR=/workspace
      - LLM_BINDING=openrouter
      - LLM_MODEL=${OPENROUTER_MODEL}
      - LLM_API_KEY=${OPEN...KEY}
      - EMBEDDING_BINDING=ollama
      - EMBEDDING_MODEL=bge-m3
      - EMBEDDING_HOST=http://ollama:11434
      - GRAPH_STORAGE=KuzuStorage
      - GRAPH_HOST=kuzu
      - VECTOR_STORAGE=NanoVectorDBStorage
    volumes:
      - ./data/lightrag:/workspace
      - ./data/kuzu:/database
    ports:
      - "9621:9621"
    depends_on:
      - kuzu
      - ollama

  # ─────────────────────────────────────────────────────────────
  # OLLAMA (Local Embeddings + Future Local LLM)
  # ─────────────────────────────────────────────────────────────
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ./data/ollama:/root/.ollama
    ports:
      - "11434:11434"
    entrypoint: >
      /bin/sh -c "
        ollama serve &
        sleep 5 &&
        ollama pull bge-m3 &&
        ollama pull nomic-embed-text &&
        wait
      "

  # ─────────────────────────────────────────────────────────────
  # VALIDATOR (Automatische Tests nach jedem Iterationsschritt)
  # ─────────────────────────────────────────────────────────────
  validator:
    build:
      context: ./services/validator
      dockerfile: Dockerfile
    environment:
      - LIGHTRAG_ENDPOINT=http://lightrag:9621
      - KUZU_ENDPOINT=http://kuzu:8000
      - OPENROUTER_API_KEY=${OPEN...KEY}
      - TEST_DATASET=/data/test_qa.jsonl
    volumes:
      - ./data:/data
      - ./tests:/tests
    depends_on:
      - lightrag
      - kuzu
    command: python -m validator.run --continuous --interval 60
```

---

## 🔧 **Service-Implementierungen**

### **1. Extractor Service** (`services/extractor/`)

```dockerfile
# Dockerfile.gpu
FROM nvidia/cuda:12.4-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3.11-venv git && \
    python3.11 -m venv /venv && \
    /venv/bin/pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

WORKDIR /app
COPY requirements.txt .
RUN /venv/bin/pip install --no-cache-dir -r requirements.txt

COPY . .
ENTRYPOINT ["/venv/bin/python", "-m", "extractor.pipeline"]
```

```python
# extractor/pipeline.py
"""
Autonome KG-Extraktions-Pipeline:
1. Chunking (semantisch, overlap)
2. NER (GLiNER, zero-shot, batched)
3. RE (REBEL, batched)
4. Entity Resolution (Embeddings + Clustering + LLM-Verify)
5. Schema Alignment (LLM once)
6. KuzuDB Ingestion (Cypher, batched, transactional)
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import yaml
from gliner import GLiNER
from transformers import pipeline
import kuzu
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import AsyncOpenAI

@dataclass
class Config:
    input_dir: Path
    output_dir: Path
    batch_size: int
    gliner_model: str
    rebel_model: str
    embedding_model: str
    openrouter_key: str
    openrouter_model: str
    kuzu_path: Path
    entity_types: List[str]
    relation_types: List[str]

class ExtractorPipeline:
    def __init__(self, config: Config):
        self.cfg = config
        self._init_models()
        self._init_db()
        self._init_openrouter()

    def _init_models(self):
        self.ner = GLiNER.from_pretrained(self.cfg.gliner_model)
        self.re = pipeline("text2text-generation",
                          model=self.cfg.rebel_model,
                          device=0, batch_size=self.cfg.batch_size)
        self.embedder = SentenceTransformer(self.cfg.embedding_model)

    def _init_db(self):
        self.db = kuzu.Database(str(self.cfg.kuzu_path))
        self.conn = kuzu.Connection(self.db)
        self._setup_schema()

    def _setup_schema(self):
        # Node tables
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Document(
                id STRING, title STRING, source STRING,
                created_at TIMESTAMP, metadata MAP, PRIMARY KEY(id)
            )
        """)
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Entity(
                id STRING, name STRING, type STRING,
                properties MAP, embedding BLOB, PRIMARY KEY(id)
            )
        """)
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Chunk(
                id STRING, text STRING, doc_id STRING,
                start_idx INT64, end_idx INT64, PRIMARY KEY(id)
            )
        """)
        # Relationship tables
        self.conn.execute("CREATE REL TABLE IF NOT EXISTS HAS_CHUNK(FROM Document TO Chunk)")
        self.conn.execute("CREATE REL TABLE IF NOT EXISTS MENTIONS(FROM Chunk TO Entity)")
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS RELATES(
                FROM Entity TO Entity,
                type STRING, confidence FLOAT, properties MAP
            )
        """)
        # Indices for metadata filtering
        self.conn.execute("CREATE INDEX ON Document(metadata)")
        self.conn.execute("CREATE INDEX ON Entity(type)")

    def _init_openrouter(self):
        self.llm = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.cfg.openrouter_key
        )

    async def run(self):
        """Haupt-Pipeline: inkrementell, checkpoint-basiert"""
        checkpoint = self._load_checkpoint()

        for doc_path in self._get_documents():
            if doc_path in checkpoint.processed:
                continue

            # 1. Chunking
            chunks = self._semantic_chunk(doc_path)

            # 2. NER (batched)
            entities = await self._extract_entities_batched(chunks)

            # 3. RE (batched)
            relations = await self._extract_relations_batched(chunks)

            # 4. Entity Resolution
            resolved_entities = await self._resolve_entities(entities)

            # 5. Schema Alignment (nur bei neuen Types)
            await self._align_schema(resolved_entities, relations)

            # 6. Ingestion
            self._ingest_to_kuzu(doc_path, chunks, resolved_entities, relations)

            checkpoint.mark_processed(doc_path)
            self._save_checkpoint(checkpoint)

        # Final: LightRAG Sync triggern
        self._trigger_lightrag_sync()

    async def _extract_entities_batched(self, chunks) -> List[Dict]:
        texts = [c.text for c in chunks]
        results = []
        for i in range(0, len(texts), self.cfg.batch_size):
            batch = texts[i:i+self.cfg.batch_size]
            preds = self.ner.predict_entities(batch, labels=self.cfg.entity_types)
            for j, ents in enumerate(preds):
                for e in ents:
                    results.append({
                        "chunk_id": chunks[i+j].id,
                        "text": e["text"],
                        "label": e["label"],
                        "start": e["start"],
                        "end": e["end"],
                        "score": e["score"]
                    })
        return results

    async def _resolve_entities(self, entities: List[Dict]) -> List[Dict]:
        """Embedding-basiert + LLM-Verifikation für Edge Cases"""
        # 1. Embeddings
        texts = [e["text"] for e in entities]
        embs = self.embedder.encode(texts, batch_size=64, show_progress_bar=True)

        # 2. FAISS Clustering
        index = faiss.IndexFlatIP(embs.shape[1])
        faiss.normalize_L2(embs)
        index.add(embs)

        # Self-similarity search
        D, I = index.search(embs, k=10)

        # 3. Clusters bilden (Threshold 0.85)
        clusters = self._cluster_entities(D, I, threshold=0.85)

        # 4. LLM-Verifikation für ambige Clusters
        verified = await self._llm_verify_clusters(clusters, entities)

        # 5. Canonical Entities erstellen
        return self._create_canonical_entities(verified)

    async def _llm_verify_clusters(self, clusters, entities):
        """Nur für Clusters mit >1 Entity und niedriger Confidence"""
        prompts = []
        for cluster in clusters:
            if len(cluster) <= 1: continue
            ent_texts = [entities[i]["text"] for i in cluster]
            prompt = f"""Sind folgende Entities identisch (coreferent)?
Entities: {ent_texts}
Antworte nur: JA/NEIN + kurze Begründung"""
            prompts.append(prompt)

        if not prompts: return clusters

        responses = await asyncio.gather(*[
            self.llm.chat.completions.create(
                model=self.cfg.openrouter_model,
                messages=[{"role": "user", "content": p}],
                temperature=0.1, max_tokens=100
            ) for p in prompts
        ])

        # Merge basierend auf LLM-Entscheidung
        return self._merge_clusters(clusters, responses)

    def _ingest_to_kuzu(self, doc_path, chunks, entities, relations):
        """Transaktionale Bulk-Ingestion"""
        self.conn.execute("BEGIN TRANSACTION")
        try:
            # Document
            doc_id = self._hash(doc_path)
            self.conn.execute("""
                CREATE (d:Document {id: $id, title: $title, source: $source,
                                   created_at: $ts, metadata: $meta})
            """, {"id": doc_id, "title": Path(doc_path).stem,
                  "source": str(doc_path), "ts": self._now(), "meta": {}})

            # Chunks
            for chunk in chunks:
                self.conn.execute("""
                    CREATE (c:Chunk {id: $id, text: $text, doc_id: $doc_id,
                                     start_idx: $s, end_idx: $e})
                    CREATE (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c)
                """, {"id": chunk.id, "text": chunk.text, "doc_id": doc_id,
                      "s": chunk.start, "e": chunk.end})

            # Entities
            for ent in entities:
                emb = self.embedder.encode([ent["canonical_name"]])[0]
                self.conn.execute("""
                    CREATE (e:Entity {id: $id, name: $name, type: $type,
                                      properties: $props, embedding: $emb})
                """, {"id": ent["id"], "name": ent["canonical_name"],
                      "type": ent["type"], "props": ent["properties"],
                      "emb": emb.tobytes()})

            # Mentions
            for mention in entities:
                self.conn.execute("""
                    MATCH (c:Chunk {id: $cid}), (e:Entity {id: $eid})
                    CREATE (c)-[:MENTIONS]->(e)
                """, {"cid": mention["chunk_id"], "eid": mention["entity_id"]})

            # Relations
            for rel in relations:
                self.conn.execute("""
                    MATCH (s:Entity {id: $sid}), (t:Entity {id: $tid})
                    CREATE (s)-[:RELATES {type: $type, confidence: $conf,
                                          properties: $props}]->(t)
                """, {"sid": rel["source_id"], "tid": rel["target_id"],
                      "type": rel["type"], "conf": rel["confidence"],
                      "props": rel.get("properties", {})})

            self.conn.execute("COMMIT")
        except Exception as e:
            self.conn.execute("ROLLBACK")
            raise
```

### **2. LightRAG Service mit KuzuStorage** (`services/lightrag/`)

```python
# services/lightrag/kuzu_storage.py
"""KuzuDB GraphStorage Implementation für LightRAG"""

from lightrag.base import BaseGraphStorage
from lightrag.utils import logger
import kuzu
import json
from typing import Dict, List, Any, Optional

class KuzuStorage(BaseGraphStorage):
    def __init__(self, namespace: str, global_config: Dict, db_path: str = "/database"):
        super().__init__(namespace, global_config)
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)
        self._ensure_schema()

    def _ensure_schema(self):
        # LightRAG expects: _id, entity_name, entity_type, description, source_id, ...
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS _Entity(
                _id STRING, entity_name STRING, entity_type STRING,
                description STRING, source_id STRING,
                file_path STRING, properties MAP, PRIMARY KEY(_id)
            )
        """)
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS _Relates(
                FROM _Entity TO _Entity,
                weight DOUBLE, description STRING,
                source_id STRING, file_path STRING, properties MAP
            )
        """)

    async def upsert_node(self, node_id: str, node_data: Dict) -> None:
        props = {
            "_id": node_id,
            "entity_name": node_data.get("entity_name", ""),
            "entity_type": node_data.get("entity_type", "Unknown"),
            "description": node_data.get("description", ""),
            "source_id": node_data.get("source_id", ""),
            "file_path": node_data.get("file_path", ""),
            "properties": json.dumps(node_data.get("properties", {}))
        }
        self.conn.execute("""
            MERGE (e:_Entity {_id: $id})
            SET e.entity_name = $entity_name,
                e.entity_type = $entity_type,
                e.description = $description,
                e.source_id = $source_id,
                e.file_path = $file_path,
                e.properties = $properties
        """, props)

    async def upsert_edge(self, source_id: str, target_id: str, edge_data: Dict) -> None:
        self.conn.execute("""
            MATCH (s:_Entity {_id: $sid}), (t:_Entity {_id: $tid})
            MERGE (s)-[r:_Relates]->(t)
            SET r.weight = $weight,
                r.description = $description,
                r.source_id = $source_id,
                r.file_path = $file_path,
                r.properties = $properties
        """, {
            "sid": source_id, "tid": target_id,
            "weight": edge_data.get("weight", 1.0),
            "description": edge_data.get("description", ""),
            "source_id": edge_data.get("source_id", ""),
            "file_path": edge_data.get("file_path", ""),
            "properties": json.dumps(edge_data.get("properties", {}))
        })

    async def get_node(self, node_id: str) -> Optional[Dict]:
        result = self.conn.execute("MATCH (e:_Entity {_id: $id}) RETURN e", {"id": node_id})
        if result.has_next():
            row = result.get_next()
            return self._row_to_dict(row)
        return None

    async def get_edge(self, source_id: str, target_id: str) -> Optional[Dict]:
        result = self.conn.execute("""
            MATCH (s:_Entity {_id: $sid})-[r:_Relates]->(t:_Entity {_id: $tid})
            RETURN r
        """, {"sid": source_id, "tid": target_id})
        if result.has_next():
            return self._edge_to_dict(result.get_next())
        return None

    async def node_degrees(self, node_ids: List[str]) -> Dict[str, int]:
        if not node_ids: return {}
        placeholders = ",".join([f"'{nid}'" for nid in node_ids])
        result = self.conn.execute(f"""
            MATCH (e:_Entity)-[r:_Relates]-()
            WHERE e._id IN [{placeholders}]
            RETURN e._id, count(r) as deg
        """)
        return {row[0]: row[1] for row in result}

    # ... weitere Interface-Methoden (delete, query, etc.)
```

```yaml
# services/lightrag/config.yaml
working_dir: /workspace
llm_binding: openrouter
llm_model: ${OPENROUTER_MODEL}
llm_api_key: ${OPENROUTER_API_KEY}
llm_temperature: 0.1
llm_max_tokens: 4096

embedding_binding: ollama
embedding_model: bge-m3
embedding_host: http://ollama:11434
embedding_dim: 1024
max_token_size: 8192

graph_storage: KuzuStorage
graph_storage_kwargs:
  db_path: /database

vector_storage: NanoVectorDBStorage
vector_storage_kwargs:
  namespace: lightrag_vectors

kv_storage: JsonKVStorage
kv_storage_kwargs:
  namespace: lightrag_kv

# Dual-Level Retrieval Config
retrieve_mode: hybrid  # low + high level
top_k: 30
max_token_for_text_unit: 4000
max_token_for_global_context: 8000
max_token_for_local_context: 8000

# Incremental Updates
enable_incremental: true
force_llm_summary: false
```

### **3. Validator Service** (`services/validator/`)

```python
# services/validator/run.py
"""Kontinuierliche Validierung: Ingestion → Retrieval → QA"""

import asyncio
import json
from pathlib import Path
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, ollama_embedding
from lightrag.utils import EmbeddingFunc

class GraphRAGValidator:
    def __init__(self, config):
        self.cfg = config
        self.rag = self._init_rag()
        self.test_cases = self._load_test_cases()

    def _init_rag(self):
        return LightRAG(
            working_dir=self.cfg.workspace,
            llm_model_func=openai_complete_if_cache,
            llm_model_name=self.cfg.llm_model,
            llm_api_key=self.cfg.api_key,
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=lambda texts: ollama_embedding(texts,
                    host="http://ollama:11434", model="bge-m3")
            ),
            graph_storage="KuzuStorage",
            vector_storage="NanoVectorDBStorage"
        )

    async def validate_ingestion(self) -> Dict:
        """Prüft: Sind alle Dokumente im Graph?"""
        # KuzuDB direkt abfragen
        # Entity/Chunk/Document Counts
        pass

    async def validate_retrieval(self) -> Dict:
        """Führt Test-Queries aus, prüft Recall/Precision"""
        results = []
        for tc in self.test_cases:
            param = QueryParam(mode="hybrid", top_k=20)
            answer = await self.rag.aquery(tc["question"], param=param)

            # Evaluation via LLM-as-Judge
            score = await self._llm_judge(tc["question"], tc["expected"], answer)
            results.append({
                "question": tc["question"],
                "expected": tc["expected"],
                "actual": answer,
                "score": score
            })
        return {
            "avg_score": sum(r["score"] for r in results) / len(results),
            "details": results
        }

    async def _llm_judge(self, question, expected, actual) -> float:
        prompt = f"""Bewerte die Antwort auf Skala 0-1:
Frage: {question}
Erwartet: {expected}
Erhalten: {actual}
Kriterien: Richtigkeit, Vollständigkeit, Relevanz
Antworte nur mit Zahl (z.B. 0.85)"""
        resp = await self.llm.chat.completions.create(
            model=self.cfg.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=10
        )
        return float(resp.choices[0].message.content.strip())

    async def run_continuous(self, interval: int = 60):
        while True:
            print(f"[{datetime.now()}] Starting validation cycle...")
            ing = await self.validate_ingestion()
            ret = await self.validate_retrieval()

            report = {
                "timestamp": datetime.now().isoformat(),
                "ingestion": ing,
                "retrieval": ret,
                "status": "PASS" if ret["avg_score"] > 0.7 else "FAIL"
            }

            self._save_report(report)

            if report["status"] == "PASS":
                print("✅ All validations passed!")
            else:
                print(f"❌ Validation failed: {ret['avg_score']:.2f}")

            await asyncio.sleep(interval)
```

---

## 🔄 **Iterations-Schema (Agent Loop)**

```python
# agent/orchestrator.py
"""Haupt-Agent: Iteriert bis Ingestion + Retrieval funktionieren"""

class GraphRAGAgent:
    def __init__(self, config):
        self.cfg = config
        self.iteration = 0
        self.max_iterations = 20
        self.success_criteria = {
            "ingestion_complete": False,
            "retrieval_score": 0.0,
            "target_score": 0.75
        }

    async def run(self):
        while self.iteration < self.max_iterations:
            self.iteration += 1
            print(f"\n{'='*60}")
            print(f"ITERATION {self.iteration}/{self.max_iterations}")
            print(f"{'='*60}")

            # Phase 1: Extraction (falls neue Dokumente)
            if self._has_new_documents():
                await self._run_extraction()

            # Phase 2: LightRAG Sync
            await self._sync_lightrag()

            # Phase 3: Validation
            report = await self._validate()

            # Phase 4: Analyse & Fix
            if not self._check_success(report):
                await self._diagnose_and_fix(report)
            else:
                print("🎉 SUCCESS: All criteria met!")
                break

            await asyncio.sleep(10)  # Breathing room

    async def _diagnose_and_fix(self, report):
        """LLM-basierte Diagnose und Auto-Fix"""
        issues = self._extract_issues(report)

        for issue in issues:
            fix_plan = await self._llm_plan_fix(issue)
            await self._execute_fix(fix_plan)
            # Re-validate specific issue
            await self._validate_specific(issue)

    async def _llm_plan_fix(self, issue) -> Dict:
        prompt = f"""Du bist ein GraphRAG-Experte. Diagnose und Fix-Plan:

Problem: {issue['description']}
Kontext:
- Extraction: {issue.get('extraction_stats', {})}
- Retrieval Score: {issue.get('retrieval_score', 0)}
- Fehlgeschlagene Queries: {issue.get('failed_queries', [])}

Erstelle einen konkreten Fix-Plan (JSON):
{{
  "action": "rechunk|reextract|reembed|reschema|retune",
  "params": {{...}},
  "reasoning": "..."
}}"""
        resp = await self.llm.chat.completions.create(
            model=self.cfg.planner_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=1000
        )
        return json.loads(resp.choices[0].message.content)
```

---

## 📋 **Test-Datensatz für Validierung** (`data/test_qa.jsonl`)

```jsonl
{"question": "Welche Entities vom Typ 'Person' werden im Dokument 'report_2024.pdf' erwähnt?", "expected": "Liste von Personen mit Quelle", "type": "entity_lookup", "metadata_filter": {"source": "report_2024.pdf", "entity_type": "Person"}}
{"question": "Wie hängen 'EV-Adoption' und 'Luftqualität' zusammen?", "expected": "Kausale Kette: EV → weniger Emissionen → bessere Luftqualität", "type": "multi_hop_reasoning"}
{"question": "Welche Organisationen arbeiten an 'Batterietechnologie'?", "expected": "Organisationen mit Relation zu Batterietechnologie", "type": "relation_query", "metadata_filter": {"relation_type": "WORKS_ON"}}
{"question": "Zusammenfassung aller Dokumente aus Q1 2024 zum Thema 'Regulierung'", "expected": "Thematische Zusammenfassung mit Quellen", "type": "global_summary", "metadata_filter": {"date": {">=": "2024-01-01", "<=": "2024-03-31"}, "category": "Regulierung"}}
```

---

## 🚀 **Start-Befehle**

```bash
# 1. Repository klonen & Setup
git clone <dein-repo> graphrag-system
cd graphrag-system

# 2. Env-Datei
cp .env.example .env
# EDIT: OPENROUTER_API_KEY, OPENROUTER_MODEL

# 3. Build & Start
docker compose build
docker compose up -d

# 4. Logs verfolgen
docker compose logs -f extractor lightrag validator

# 5. Test-Queries manuell
curl -X POST http://localhost:9621/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Deine Frage", "param": {"mode": "hybrid"}}'

# 6. Status prüfen
docker compose exec validator python -c "
from validator.run import GraphRAGValidator
import asyncio
v = GraphRAGValidator(config)
print(asyncio.run(v.validate_retrieval()))
"
```

---

## 📊 **Erfolgs-Kriterien (Definition of Done)**

| Kriterium | Target | Messung |
|-----------|--------|---------|
| **Ingestion Complete** | 100% Docs verarbeitet | `validator.validate_ingestion()` |
| **Entity Coverage** | > 90% erwartete Entities | Manuell annotierter Test-Set |
| **Relation Coverage** | > 80% erwartete Relationen | Manuell annotierter Test-Set |
| **Retrieval Score (LLM-Judge)** | > 0.75 | `validator.validate_retrieval()` |
| **Metadata Filter Precision** | > 0.9 | Filter-Test-Queries |
| **Multi-Hop Accuracy** | > 0.7 | 2-3 Hop Test-Queries |
| **Latency (P95)** | < 3s | Query-End-to-End |
| **Incremental Update** | < 30s pro Doc | Neues Doc → Query verfügbar |

---

## 🔧 **Migration zu lokalem ThinkingCap (Später)**

```yaml
# docker-compose.override.yml (für lokalen Switch)
services:
  lightrag:
    environment:
      - LLM_BINDING=ollama
      - LLM_MODEL=thinkingcap-qwen3.6-27b:q4_k_m
      - LLM_HOST=http://ollama:11434
      - LLM_TEMPERATURE=0.1
      - LLM_MAX_TOKENS=***

  ollama:
    entrypoint: >
      /bin/sh -c "
        ollama serve &
        sleep 5 &&
        ollama pull bge-m3 &&
        ollama pull hf.co/bartowski/ThinkingCap-Qwen3.6-27B-GGUF:Q4_K_M &&
        wait
      "
```

```bash
# Switch Command
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d lightrag ollama
```

---

## 📁 **Projektstruktur (Final)**

```
graphrag-system/
├── docker-compose.yml
├── docker-compose.override.yml
├── .env.example
├── .env
├── config/
│   ├── extractor.yaml
│   ├── lightrag.yaml
│   └── validator.yaml
├── data/
│   ├── input/           # Roh-Dokumente (PDF, TXT, MD, ...)
│   ├── test_qa.jsonl    # Validierungs-Set
│   ├── kuzu/            # KuzuDB Files
│   ├── lightrag/        # LightRAG Workspace
│   └── ollama/          # Ollama Models
├── models/              # Cache für GLiNER, REBEL, bge-m3
├── output/              # Extractor Checkpoints, Reports
├── services/
│   ├── extractor/
│   │   ├── Dockerfile.gpu
│   │   ├── requirements.txt
│   │   └── extractor/
│   │       ├── __init__.py
│   │       ├── pipeline.py
│   │       ├── chunking.py
│   │       ├── ner.py
│   │       ├── re.py
│   │       ├── resolution.py
│   │       └── ingestion.py
│   ├── lightrag/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── kuzu_storage.py
│   └── validator/
│       ├── Dockerfile
│       ├── requirements.txt
│       └── validator/
│           ├── __init__.py
│           └── run.py
├── agent/
│   └── orchestrator.py
├── tests/
│   ├── unit/
│   └── integration/
└── README.md
```

---

## ✅ **Nächste Schritte für dich**

1. **Repository erstellen** mit obiger Struktur
2. **`.env` befüllen** mit OpenRouter Key
3. **`docker compose build`** → erste Images bauen
4. **Test-Dokumente** in `data/input/` legen
5. **`docker compose up -d`** → Agent startet Iteration 1
6. **Logs beobachten** → `docker compose logs -f extractor validator`
7. **Bei Bedarf**: `config/extractor.yaml` anpassen (Entity-Types, Chunk-Size, etc.)

Der Agent läuft **autonom**, validiert **kontinuierlich**, und **iteriert** bis die Retrieval-Qualität passt. Sobald du lokale GPU hast, **ein Config-Switch** → alles lokal.