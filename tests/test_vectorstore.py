from ekr.models import KnowledgeCard
from ekr.storage import Storage
from ekr.vectorstore import (
    InMemoryStore,
    StubEmbedding,
    build_embedding_text,
    build_metadata,
    index_card,
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
