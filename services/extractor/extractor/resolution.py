import asyncio
import hashlib
from typing import List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI

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

    def _cluster_entities(self, D: np.ndarray, I: np.ndarray, threshold: float) -> List[List[int]]:
        n = D.shape[0]
        visited = set()
        clusters = []

        for i in range(n):
            if i in visited:
                continue
            cluster = [i]
            visited.add(i)
            for neighbor_idx, sim in zip(I[i], D[i]):
                if neighbor_idx != i and sim >= threshold and neighbor_idx not in visited:
                    cluster.append(neighbor_idx)
                    visited.add(neighbor_idx)
            clusters.append(cluster)

        return clusters

    async def resolve_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not entities:
            return []

        texts = [e["text"] for e in entities]
        embeddings = self.embedder.encode(texts, batch_size=64, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        k = min(10, len(texts))
        D, I = index.search(embeddings, k)

        clusters = self._cluster_entities(D, I, self.similarity_threshold)
        resolved = []

        for cluster in clusters:
            cluster_ents = [entities[idx] for idx in cluster]
            canonical_name = cluster_ents[0]["text"]
            entity_type = cluster_ents[0]["label"]
            ent_id = hashlib.sha256(f"{canonical_name}_{entity_type}".encode("utf-8")).hexdigest()[:16]

            for ent in cluster_ents:
                resolved.append({
                    "entity_id": ent_id,
                    "canonical_name": canonical_name,
                    "type": entity_type,
                    "chunk_id": ent["chunk_id"],
                    "doc_id": ent["doc_id"],
                    "original_text": ent["text"],
                    "properties": {}
                })

        return resolved
