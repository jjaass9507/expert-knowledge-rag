"""向量化 pipeline —— 核准卡片 → embedding → 向量庫。

與 LLM/ASR 對稱的 adapter 模式：
- Embedding：正式用 OpenAI 相容 /embeddings（如本地 bge-m3 推論伺服器），測試用 StubEmbedding。
- VectorStore：正式用 Qdrant（延遲匯入），測試用 InMemoryStore。
組 embedding 文字採【標題】【類型】【內容】【適用範圍】；metadata 含標籤/知識類型/適用範圍/
信心等級/更新人/最後更新，供檢索時做 filter。
"""

from __future__ import annotations

import os
from typing import Protocol

import requests

from .models import KnowledgeCard


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


class InMemoryStore:
    """測試用：保存 upsert 過的向量與 metadata。"""

    def __init__(self):
        self.points: dict[str, tuple[list[float], dict]] = {}

    def upsert(self, card_id: str, vector: list[float], metadata: dict) -> None:
        self.points[card_id] = (vector, metadata)


def index_card(card: KnowledgeCard, embedding: Embedding, store: VectorStore) -> None:
    text = build_embedding_text(card)
    vector = embedding.embed(text)
    store.upsert(card.id, vector, build_metadata(card))


def _uuid_from_id(card_id: str) -> str:
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, card_id))


def indexer_from_env():
    """回傳一個 on_approve(card) 函式；EKR_VECTOR=off 時回傳 None（不向量化）。"""
    if os.environ.get("EKR_VECTOR", "off").lower() == "off":
        return None
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
    return lambda card: index_card(card, embedding, store)
