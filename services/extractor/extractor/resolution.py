import asyncio
import hashlib
from typing import List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
from .quality import normalize_text

class EntityResolver:
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-m3",
        similarity_threshold: float = 0.85,
        openrouter_key: str = "",
        openrouter_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    ):
        self.similarity_threshold = similarity_threshold
        self.embedder = SentenceTransformer(embedding_model)
        self.openrouter_key = openrouter_key
        self.openrouter_model = openrouter_model
        if openrouter_key:
            self.llm = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key
            )
        else:
            self.llm = None

    def _cluster_entities(
        self,
        embeddings: np.ndarray,
        entities: List[Dict[str, Any]],
        threshold: float,
    ) -> List[List[int]]:
        """Cluster only same-type mentions using symmetric connectivity."""
        n = len(entities)
        parent = list(range(n))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        similarities = embeddings @ embeddings.T
        for i in range(n):
            for j in range(i + 1, n):
                if entities[i]["label"].casefold() != entities[j]["label"].casefold():
                    continue
                if similarities[i, j] >= threshold:
                    union(i, j)

        clusters: Dict[int, List[int]] = {}
        for index in range(n):
            clusters.setdefault(find(index), []).append(index)
        return list(clusters.values())

    async def resolve_entities(
        self,
        entities: List[Dict[str, Any]],
        existing_entities: List[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not entities:
            return []

        texts = [e["text"] for e in entities]
        embeddings = self.embedder.encode(texts, batch_size=64, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)

        clusters = self._cluster_entities(embeddings, entities, self.similarity_threshold)
        resolved = []

        existing_by_key = {
            (normalize_text(item["canonical_name"]), item["type"].casefold()): item
            for item in (existing_entities or [])
        }
        existing_items = list(existing_entities or [])
        existing_embeddings = None
        if existing_items:
            existing_texts = [item["canonical_name"] for item in existing_items]
            existing_embeddings = self.embedder.encode(
                existing_texts,
                batch_size=64,
                convert_to_numpy=True,
            )
            faiss.normalize_L2(existing_embeddings)

        for cluster in clusters:
            cluster_ents = [entities[idx] for idx in cluster]
            entity_type = max(
                (ent["label"] for ent in cluster_ents),
                key=lambda label: sum(1 for ent in cluster_ents if ent["label"] == label),
            )
            canonical_name = max(
                (ent["text"] for ent in cluster_ents),
                key=lambda value: (len(normalize_text(value)), value.casefold()),
            )

            existing = existing_by_key.get((normalize_text(canonical_name), entity_type.casefold()))
            if existing is None and existing_embeddings is not None:
                cluster_embedding = embeddings[cluster].mean(axis=0, keepdims=True)
                faiss.normalize_L2(cluster_embedding)
                scores = existing_embeddings @ cluster_embedding[0]
                best = int(np.argmax(scores))
                if (
                    float(scores[best]) >= max(self.similarity_threshold, 0.92)
                    and existing_items[best]["type"].casefold() == entity_type.casefold()
                ):
                    existing = existing_items[best]

            if existing:
                ent_id = existing["entity_id"]
                canonical_name = existing["canonical_name"]
            else:
                stable_key = f"{normalize_text(canonical_name)}_{entity_type.casefold()}"
                ent_id = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:16]

            for ent in cluster_ents:
                resolved.append({
                    "entity_id": ent_id,
                    "canonical_name": canonical_name,
                    "type": entity_type,
                    "chunk_id": ent["chunk_id"],
                    "doc_id": ent["doc_id"],
                    "original_text": ent["text"],
                    "mention_score": float(ent.get("score", 1.0)),
                    "start": ent.get("start"),
                    "end": ent.get("end"),
                    "properties": {}
                })

        return resolved
