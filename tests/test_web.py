import pytest

from ekr.storage import Storage
from ekr.web.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("EKR_LLM", "stub")
    store = Storage(db_path=tmp_path / "cards.db", approved_dir=tmp_path / "approved")
    yield create_app(store).test_client(), store
    store.close()


def test_submit_then_approve_appears_in_library(client):
    c, store = client
    r = c.post("/submit", data={"逐字稿": "壓力正常但電流飄高", "更新人": "王技師"})
    assert r.status_code == 302
    cid = r.headers["Location"].split("/")[-1]

    # 待校稿清單看得到
    assert cid in c.get("/").get_data(as_text=True)

    # 核准
    c.post(
        f"/review/{cid}/approve",
        data={
            "標題": "電流偏高研判",
            "內容": "壓力正常但電流偏高",
            "標籤": "電流, 壓縮機",
            "知識類型": "診斷",
            "適用範圍": "RTHD",
            "信心等級": "中",
        },
    )

    # 知識卡目錄與唯讀詳情看得到
    lib = c.get("/library").get_data(as_text=True)
    assert "電流偏高研判" in lib
    detail = c.get(f"/library/{cid}").get_data(as_text=True)
    assert "RTHD" in detail


def test_card_detail_404(client):
    c, _ = client
    assert c.get("/library/不存在").status_code == 404


def _approve_one(c):
    cid = c.post(
        "/submit", data={"逐字稿": "壓力正常但電流飄高", "更新人": "王技師"}
    ).headers["Location"].split("/")[-1]
    c.post(
        f"/review/{cid}/approve",
        data={
            "標題": "電流偏高研判", "內容": "壓力正常但電流偏高", "重點": "要點一\n要點二",
            "標籤": "電流", "知識類型": "診斷", "大分類": "冰水主機",
            "適用範圍": "", "信心等級": "中",
        },
    )
    return cid


def test_edit_approved_card(client):
    c, store = client
    cid = _approve_one(c)
    r = c.post(
        f"/library/{cid}/edit",
        data={
            "標題": "改後標題", "內容": "x", "重點": "a\nb", "標籤": "電流",
            "知識類型": "診斷", "大分類": "空壓機", "適用範圍": "", "信心等級": "高",
        },
    )
    assert r.status_code == 302
    got = store.get(cid)
    assert got.標題 == "改後標題" and got.大分類 == "空壓機" and got.重點 == ["a", "b"]


def test_delete_approved_card(client):
    c, store = client
    cid = _approve_one(c)
    assert c.post(f"/library/{cid}/delete").status_code == 302
    assert store.get(cid) is None


def test_keyword_search_finds_card(client):
    c, _ = client
    _approve_one(c)
    body = c.get("/search?q=電流偏高").get_data(as_text=True)
    assert "電流偏高研判" in body
