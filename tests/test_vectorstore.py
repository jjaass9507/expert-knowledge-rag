from ekr.models import KnowledgeCard
from ekr.storage import Storage
from ekr.vectorstore import (
    InMemoryStore,
    StubEmbedding,
    build_embedding_text,
    build_metadata,
    index_card,
    reindex_jsonl,
    retrieve,
)

CARD = KnowledgeCard(
    id="KB-vec01",
    標題="電流偏高",
    內容="壓力正常但電流偏高。",
    標籤=["電流", "壓縮機"],
    知識類型="診斷",
    適用範圍="冰水主機",
    信心等級="中",
    原始逐字稿="壓力正常但電流飄高",
    更新人="王技師",
    最後更新="2026-06-15",
)


def test_embedding_text_has_all_sections():
    text = build_embedding_text(CARD)
    assert "【標題】電流偏高" in text
    assert "【類型】診斷" in text
    assert "【內容】" in text
    assert "【適用範圍】冰水主機" in text


def test_metadata_includes_filter_fields():
    md = build_metadata(CARD)
    assert md["知識類型"] == "診斷"
    assert md["信心等級"] == "中"
    assert md["標籤"] == ["電流", "壓縮機"]


def test_index_card_upserts_vector_and_metadata():
    store = InMemoryStore()
    index_card(CARD, StubEmbedding(dim=8), store)
    assert CARD.id in store.points
    vector, md = store.points[CARD.id]
    assert len(vector) == 8
    assert md["知識類型"] == "診斷"


class FakeEmbed:
    """決定性語意替身：依關鍵字回傳正交向量，讓相似度排序可預期。"""

    def embed(self, text):
        if "電流" in text:
            return [1.0, 0.0]
        if "抽真空" in text:
            return [0.0, 1.0]
        return [0.5, 0.5]


def test_retrieve_returns_most_similar_first():
    embed = FakeEmbed()
    store = InMemoryStore()
    diag = CARD  # 內容含「電流」
    sop = KnowledgeCard(
        id="KB-sop01",
        標題="抽真空標準步驟",
        內容="先接真空泵，抽至 500 micron 並維持 30 分鐘。",
        標籤=["抽真空"],
        知識類型="SOP",
        適用範圍="",
        信心等級="高",
        原始逐字稿="抽真空要抽到 500 micron",
        更新人="李技師",
        最後更新="2026-06-15",
    )
    index_card(diag, embed, store)
    index_card(sop, embed, store)

    hits = retrieve("電流偏高壓力正常", embed, store, top_k=2)
    assert hits[0].id == diag.id  # 最相近者排第一
    assert hits[0].score > hits[1].score


def test_retrieve_filters_by_knowledge_type():
    embed = StubEmbedding(dim=8)
    store = InMemoryStore()
    index_card(CARD, embed, store)  # 診斷
    hits = retrieve("任意查詢", embed, store, 知識類型="SOP")
    assert hits == []  # 無 SOP 卡片
    hits = retrieve("任意查詢", embed, store, 知識類型="診斷")
    assert len(hits) == 1 and hits[0].metadata["知識類型"] == "診斷"


def test_approve_triggers_on_approve_hook(tmp_path):
    vec = InMemoryStore()
    embed = StubEmbedding(dim=8)
    store = Storage(
        db_path=tmp_path / "cards.db",
        approved_dir=tmp_path / "approved",
        on_approve=lambda card: index_card(card, embed, vec),
    )
    store.insert_pending(CARD)
    store.approve(CARD.id)
    assert CARD.id in vec.points  # 核准即被索引
    store.close()


def test_reindex_jsonl_backfills_store(tmp_path):
    # 先核准一張卡片以產生 cards.jsonl
    store = Storage(db_path=tmp_path / "cards.db", approved_dir=tmp_path / "approved")
    store.insert_pending(CARD)
    store.approve(CARD.id)
    store.close()

    vec = InMemoryStore()
    n = reindex_jsonl(tmp_path / "approved" / "cards.jsonl", StubEmbedding(dim=8), vec)
    assert n == 1
    assert CARD.id in vec.points
