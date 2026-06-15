import io

from ekr.asr import StubASR
from ekr.storage import Storage
from ekr.web.app import create_app


def test_stub_asr_returns_text():
    assert "電流" in StubASR("電流偏高").transcribe("x.wav")


def test_submit_audio_upload_flows_through_structuring(tmp_path, monkeypatch):
    """上傳音檔 → ASR 轉文字 → 結構化 → 進待審佇列。"""
    monkeypatch.setenv("EKR_LLM", "stub")
    monkeypatch.setenv("EKR_ASR", "stub")
    store = Storage(db_path=tmp_path / "cards.db", approved_dir=tmp_path / "approved")
    client = create_app(store).test_client()

    data = {
        "更新人": "王技師",
        "音檔": (io.BytesIO(b"fake audio bytes"), "memo.wav"),
    }
    r = client.post(
        "/submit", data=data, content_type="multipart/form-data", follow_redirects=False
    )
    assert r.status_code == 302
    pending = store.list_by_status("pending")
    assert len(pending) == 1
    # StubASR 的逐字稿應被保留為 provenance
    assert pending[0].原始逐字稿.startswith("壓力都正常")
    store.close()
