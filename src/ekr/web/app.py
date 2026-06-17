"""Flask 審核介面 —— 「EKR Intelligence」校稿台。

路由薄薄一層覆在 storage / structurer / vectorstore 之上：
清單、提交、校對審核、核准/退回/重整理、定稿卡編輯/刪除、知識檢索。
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, url_for

from ..asr import from_env as asr_from_env
from ..llm import available_backends, build_llm
from ..models import EQUIPMENT_CATEGORIES, Confidence, KnowledgeType
from ..storage import Storage
from ..structurer import refine_card, structure_transcript, structure_transcripts

load_dotenv()


def _transcribe_upload(file_storage) -> str:
    """把上傳音檔存到暫存檔後轉文字。"""
    import os
    import tempfile

    suffix = os.path.splitext(file_storage.filename)[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        file_storage.save(path)
        return asr_from_env().transcribe(path).strip()
    finally:
        os.remove(path)


def create_app(storage: Storage | None = None) -> Flask:
    app = Flask(__name__)
    searcher = None
    if storage is None:
        from ..vectorstore import hooks_from_env

        on_approve, on_delete, searcher = hooks_from_env()
        storage = Storage(on_approve=on_approve, on_delete=on_delete)
    store = storage

    KNOWLEDGE_TYPES = [t.value for t in KnowledgeType]
    CONFIDENCES = [c.value for c in Confidence]

    def _selected_backend() -> str:
        ids = [b[0] for b in available_backends()]
        cur = store.get_setting("llm_backend") or os.environ.get("EKR_LLM", "pensieve")
        return cur if cur in ids else (ids[0] if ids else "pensieve")

    def current_llm():
        return build_llm(_selected_backend())

    @app.context_processor
    def inject_globals():
        # 側邊欄計數、下拉選項與 LLM 後端選擇，所有頁面共用。
        return {
            "nav_pending": len(store.list_by_status("pending")),
            "nav_approved": len(store.list_by_status("approved")),
            "knowledge_types": KNOWLEDGE_TYPES,
            "confidences": CONFIDENCES,
            "categories": EQUIPMENT_CATEGORIES,
            "llm_backends": available_backends(),
            "llm_current": _selected_backend(),
        }

    @app.route("/settings/llm", methods=["POST"])
    def set_llm():
        backend = request.form.get("backend", "")
        if backend in [b[0] for b in available_backends()]:
            store.set_setting("llm_backend", backend)
        nxt = request.form.get("next", "")
        return redirect(nxt if nxt.startswith("/") else url_for("index"))

    @app.route("/")
    def index():
        pending = store.list_by_status("pending")
        approved = store.list_by_status("approved")
        return render_template(
            "list.html", pending=pending, approved_count=len(approved)
        )

    @app.route("/library")
    def library():
        return render_template("library.html", cards=store.list_by_status("approved"))

    @app.route("/library/<card_id>")
    def card_detail(card_id):
        card = store.get(card_id)
        if card is None:
            return "找不到此卡片", 404
        return render_template("card.html", card=card)

    @app.route("/library/<card_id>/edit", methods=["GET", "POST"])
    def edit_card(card_id):
        card = store.get(card_id)
        if card is None:
            return "找不到此卡片", 404
        if request.method == "POST":
            _save_edits(store, card_id, request.form)
            store.update_card(card_id)  # 重新驗證並同步 YAML/JSONL/向量
            return redirect(url_for("card_detail", card_id=card_id))
        return render_template("edit.html", card=card)

    @app.route("/library/<card_id>/delete", methods=["POST"])
    def delete_card(card_id):
        store.delete(card_id)
        return redirect(url_for("library"))

    @app.route("/search")
    def search():
        q = request.args.get("q", "").strip()
        知識類型 = request.args.get("type", "").strip() or None
        results = _run_search(store, searcher, q, 知識類型) if q else None
        return render_template(
            "search.html", q=q, 知識類型=知識類型 or "", results=results,
            semantic=searcher is not None,
        )

    @app.route("/submit", methods=["GET", "POST"])
    def submit():
        if request.method == "POST":
            transcript = request.form.get("逐字稿", "").strip()
            更新人 = request.form.get("更新人", "").strip() or "匿名"
            # 語音輸入：若上傳音檔則先轉文字，再走同一結構化流程
            audio = request.files.get("音檔")
            if audio and audio.filename:
                try:
                    transcript = _transcribe_upload(audio)
                except Exception as e:  # noqa: BLE001
                    return render_template(
                        "submit.html", error=f"語音轉文字失敗：{e}", 更新人=更新人
                    )
            if not transcript:
                return render_template(
                    "submit.html", error="請輸入逐字稿、描述內容，或上傳音檔", 更新人=更新人
                )
            try:
                cards = structure_transcripts(transcript, current_llm(), 更新人)
            except Exception as e:  # noqa: BLE001 — 結構化失敗如實回報給審核者
                return render_template(
                    "submit.html", error=f"結構化失敗：{e}",
                    逐字稿=transcript, 更新人=更新人,
                )
            for card in cards:
                store.insert_pending(card)
            # 單一知識點 → 直接進校稿台；多張 → 回待校稿清單逐一校對
            if len(cards) == 1:
                return redirect(url_for("review", card_id=cards[0].id))
            return redirect(url_for("index"))
        return render_template("submit.html")

    @app.route("/review/<card_id>")
    def review(card_id):
        card = store.get(card_id)
        if card is None:
            return "找不到此卡片", 404
        return render_template("review.html", card=card)

    @app.route("/review/<card_id>/approve", methods=["POST"])
    def approve(card_id):
        _save_edits(store, card_id, request.form)
        store.approve(card_id)
        return redirect(url_for("index"))

    @app.route("/review/<card_id>/reject", methods=["POST"])
    def reject(card_id):
        store.reject(card_id)
        return redirect(url_for("index"))

    @app.route("/review/<card_id>/restructure", methods=["POST"])
    def restructure(card_id):
        card = store.get(card_id)
        if card is None:
            return "找不到此卡片", 404
        # 先存下審核者當前的編輯，再依補充說明對話式重整理
        _save_edits(store, card_id, request.form)
        card = store.get(card_id)
        feedback = request.form.get("補充說明", "").strip()
        try:
            if feedback:
                new = refine_card(card, feedback, current_llm())
            else:
                new = structure_transcript(card.原始逐字稿, current_llm(), card.更新人)
        except Exception as e:  # noqa: BLE001
            return render_template(
                "review.html", card=card, error=f"重新整理失敗：{e}"
            )
        store.update_fields(
            card_id,
            標題=new.標題, 內容=new.內容, 重點=new.重點, 可回答問題=new.可回答問題,
            標籤=new.標籤, 知識類型=new.知識類型.value, 大分類=new.大分類,
            適用範圍=new.適用範圍, 信心等級=new.信心等級.value,
        )
        return redirect(url_for("review", card_id=card_id))

    return app


def _save_edits(store: Storage, card_id: str, form) -> None:
    標籤 = [t.strip() for t in form.get("標籤", "").split(",") if t.strip()]
    重點 = [r.strip() for r in form.get("重點", "").splitlines() if r.strip()]
    可回答問題 = [q.strip() for q in form.get("可回答問題", "").splitlines() if q.strip()]
    store.update_fields(
        card_id,
        標題=form.get("標題", "").strip(),
        內容=form.get("內容", "").strip(),
        重點=重點,
        可回答問題=可回答問題,
        標籤=標籤,
        知識類型=form.get("知識類型", ""),
        大分類=form.get("大分類", "").strip(),
        適用範圍=form.get("適用範圍", "").strip(),
        信心等級=form.get("信心等級", ""),
    )


def _run_search(store: Storage, searcher, q: str, 知識類型: str | None) -> list[dict]:
    """檢索：有向量化用語意檢索，否則對定稿卡片做關鍵字搜尋。回傳統一格式。"""
    if searcher is not None:
        hits = searcher(q, top_k=8, 知識類型=知識類型)
        return [{"id": h.id, "score": h.score, **h.metadata} for h in hits]
    out = []
    for c in store.list_by_status("approved"):
        if 知識類型 and c.知識類型.value != 知識類型:
            continue
        hay = " ".join([c.標題, c.內容, " ".join(c.標籤), " ".join(c.重點), c.大分類])
        if q in hay:
            out.append({
                "id": c.id, "score": None, "標題": c.標題, "內容": c.內容,
                "知識類型": c.知識類型.value, "大分類": c.大分類,
                "信心等級": c.信心等級.value, "重點": c.重點,
            })
    return out


if __name__ == "__main__":
    create_app().run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
