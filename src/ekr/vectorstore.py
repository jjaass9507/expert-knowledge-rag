"""向量化 pipeline —— 核准卡片 → embedding → 向量庫。

與 LLM/ASR 對稱的 adapter 模式：
- Embedding：正式用 OpenAI 相容 /embeddings（如本地 bge-m3 推論伺服器），測試用 StubEmbedding。
- VectorStore：正式用 Qdrant（延遲匯入），測試用 InMemoryStore。
組 embedding 文字採【標題】【類型】【內容】【適用範圍】；metadata 含標籤/知識類型/適用範圍/
信心等級/更新人/最後更新，供檢索時做 filter。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Protocol

import requests

from .models import KnowledgeCard


@dataclass
class Hit:
    """一筆檢索結果。"""

    id: str
    score: float
    metadata: dict


def build_embedding_text(card: KnowledgeCard) -> str:
    return (
        f"【標題】{card.標題}\n"
        f"【類型】{card.知識類型.value}\n"
        f"【內容】{card.內容}\n"
        f"【適用範圍】{card.適用範圍}"
    )


def build_metadata(card: KnowledgeCard) -> dict:
    return {
        "標籤": card.標籤,
        "知識類型": card.知識類型.value,
        "適用範圍": card.適用範圍,
        "信心等級": card.信心等級.value,
        "更新人": card.更新人,
        "最後更新": card.最後更新,
        "標題": card.標題,
        "內容": card.內容,
    }


# --- Embedding ---
class Embedding(Protocol):
    def embed(self, text: str) -> list[float]: ...


class ApiEmbedding:
    def __init__(self, url: str, model: str, api_key: str = "", timeout: int = 60):
        self.url = url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def embed(self, text: str) -> list[float]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(
            self.url,
            json={"model": self.model, "input": text},
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


class StubEmbedding:
    """測試用：以字元碼產生固定維度的決定性向量（非語意，僅供 pipeline 驗證）。"""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for i, ch in enumerate(text):
            v[i % self.dim] += ord(ch) % 17
        return v


# --- VectorStore ---
class VectorStore(Protocol):
    def upsert(self, card_id: str, vector: list[float], metadata: dict) -> None: ...

    def search(
        self, vector: list[float], top_k: int = 5, 知識類型: str | None = None
    ) -> list[Hit]: ...


class QdrantStore:
    def __init__(self, url: str, collection: str, dim: int):
        from qdrant_client import QdrantClient  # 延遲匯入
        from qdrant_client.models import Distance, VectorParams

        self.collection = collection
        self.client = QdrantClient(url=url)
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, card_id: str, vector: list[float], metadata: dict) -> None:
        from qdrant_client.models import PointStruct

        self.client.upsert(
            self.collection,
            points=[PointStruct(id=_uuid_from_id(card_id), vector=vector, payload={**metadata, "id": card_id})],
        )

    def search(
        self, vector: list[float], top_k: int = 5, 知識類型: str | None = None
    ) -> list[Hit]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        flt = None
        if 知識類型:
            flt = Filter(
                must=[FieldCondition(key="知識類型", match=MatchValue(value=知識類型))]
            )
        results = self.client.search(
            self.collection, query_vector=vector, limit=top_k, query_filter=flt
        )
        return [
            Hit(id=r.payload.get("id", str(r.id)), score=r.score, metadata=r.payload)
            for r in results
        ]


class InMemoryStore:
    """測試用：保存 upsert 過的向量與 metadata，並做 cosine 檢索。"""

    def __init__(self):
        self.points: dict[str, tuple[list[float], dict]] = {}

    def upsert(self, card_id: str, vector: list[float], metadata: dict) -> None:
        self.points[card_id] = (vector, metadata)

    def search(
        self, vector: list[float], top_k: int = 5, 知識類型: str | None = None
    ) -> list[Hit]:
        hits = []
        for cid, (vec, md) in self.points.items():
            if 知識類型 and md.get("知識類型") != 知識類型:
                continue
            hits.append(Hit(id=cid, score=_cosine(vector, vec), metadata=md))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def index_card(card: KnowledgeCard, embedding: Embedding, store: VectorStore) -> None:
    text = build_embedding_text(card)
    vector = embedding.embed(text)
    store.upsert(card.id, vector, build_metadata(card))


def retrieve(
    query: str,
    embedding: Embedding,
    store: VectorStore,
    top_k: int = 5,
    知識類型: str | None = None,
) -> list[Hit]:
    """RAG 檢索：把查詢字串轉向量後在知識庫搜尋；可依知識類型做 metadata filter。

    供 8-step Agent Framework 的 Tier 2 檢索節點呼叫（例如異常診斷場景傳 知識類型='診斷'）。
    """
    return store.search(embedding.embed(query), top_k=top_k, 知識類型=知識類型)


def _uuid_from_id(card_id: str) -> str:
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, card_id))


def reindex_jsonl(path, embedding: Embedding, store: VectorStore) -> int:
    """從 approved cards.jsonl 回填/重建向量庫，回傳索引筆數。"""
    import json

    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            index_card(KnowledgeCard(**json.loads(line)), embedding, store)
            n += 1
    return n


def build_embedding_store() -> tuple[Embedding, VectorStore]:
    """依環境變數建立 (embedding, store)，供 indexer / reindex / search 共用。"""
    embedding = ApiEmbedding(
        url=os.environ["EMBEDDING_API_URL"],
        model=os.environ.get("EMBEDDING_MODEL", "bge-m3"),
        api_key=os.environ.get("EMBEDDING_API_KEY", ""),
    )
    store = QdrantStore(
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        collection=os.environ.get("QDRANT_COLLECTION", "knowledge_cards"),
        dim=int(os.environ.get("EMBEDDING_DIM", "1024")),
    )
    return embedding, store


def indexer_from_env():
    """回傳一個 on_approve(card) 函式；EKR_VECTOR=off 時回傳 None（不向量化）。"""
    if os.environ.get("EKR_VECTOR", "off").lower() == "off":
        return None
    embedding, store = build_embedding_store()
    return lambda card: index_card(card, embedding, store)
