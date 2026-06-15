import json

import pytest

from ekr.models import KnowledgeCard
from ekr.storage import Storage

CARD = KnowledgeCard(
    id="KB-test01",
    標題="電流偏高",
    內容="壓力正常但電流偏高。",
    標籤=["電流", "壓縮機"],
    知識類型="診斷",
    適用範圍="",
    信心等級="中",
    原始逐字稿="壓力正常但電流飄高",
    更新人="王技師",
    最後更新="2026-06-15",
)


@pytest.fixture
def store(tmp_path):
    s = Storage(db_path=tmp_path / "cards.db", approved_dir=tmp_path / "approved")
    yield s
    s.close()


def test_insert_and_list_pending(store):
    store.insert_pending(CARD)
    pending = store.list_by_status("pending")
    assert len(pending) == 1
    assert pending[0].標題 == "電流偏高"
    assert pending[0].標籤 == ["電流", "壓縮機"]


def test_update_fields(store):
    store.insert_pending(CARD)
    store.update_fields(CARD.id, 標題="新標題", 標籤=["A", "B"])
    got = store.get(CARD.id)
    assert got.標題 == "新標題"
    assert got.標籤 == ["A", "B"]


def test_approve_writes_jsonl_and_yaml(store, tmp_path):
    store.insert_pending(CARD)
    store.approve(CARD.id)
    assert store.get(CARD.id) not in store.list_by_status("pending")

    jsonl = tmp_path / "approved" / "cards.jsonl"
    lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["標題"] == "電流偏高"

    yaml_file = tmp_path / "approved" / "yaml" / "KB-test01.yaml"
    assert yaml_file.exists()
    assert "電流偏高" in yaml_file.read_text(encoding="utf-8")

    assert len(store.list_by_status("approved")) == 1


def test_reject(store):
    store.insert_pending(CARD)
    store.reject(CARD.id)
    assert store.list_by_status("pending") == []
    assert len(store.list_by_status("rejected")) == 1
