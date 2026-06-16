"""Flask 審核介面 —— 「校稿台 / 編輯室」。

路由薄薄一層覆在 storage / structurer 之上：清單、提交、左右校對審核、核准/退回/重整理。
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, url_for

from ..asr import from_env as asr_from_env
from ..llm import from_env
from ..models import Confidence, KnowledgeType
from ..storage import Storage
from ..structurer import refine_card, structure_transcript

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
    if storage is None:
        from ..vectorstore import indexer_from_env

        storage = Storage(on_approve=indexer_from_env())
    store = storage

    KNOWLEDGE_TYPES = [t.value for t in KnowledgeType]
    CONFIDENCES = [c.value for c in Confidence]

    @app.context_processor
    def inject_nav():
        # 側邊欄計數，所有頁面共用。
        return {
            "nav_pending": len(store.list_by_status("pending")),
            "nav_approved": len(store.list_by_status("approved")),
        }

    @app.route("/")
    def index():
        pending = store.list_by_status("pending")
        approved = store.list_by_status("approved")
        return render_template(
            "list.html", pending=pending, approved_count=len(approved)
        )

    @app.route("/library")
    def library():
        approved = store.list_by_status("approved")
        return render_template("library.html", cards=approved)

    @app.route("/library/<card_id>")
    def card_detail(card_id):
        card = store.get(card_id)
        if card is None:
            return "找不到此卡片", 404
        return render_template("card.html", card=card)

    @app.route("/submit", methods=["GET", "POST"])
    def submit():
        if request.method == "POST":
            transcript = request.form.get("逐字稿", "").strip()
            更新人 = request.form.get("更新人", "").strip() or "匿名"
            # 語音輸入（Phase 2）：若上傳音檔則先轉文字，再走同一結構化流程
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
                card = structure_transcript(transcript, from_env(), 更新人)
            except Exception as e:  # noqa: BLE001 — 結構化失敗如實回報給審核者
                return render_template(
                    "submit.html",
                    error=f"結構化失敗：{e}",
                    逐字稿=transcript,
                    更新人=更新人,
                )
            store.insert_pending(card)
            return redirect(url_for("review", card_id=card.id))
        return render_template("submit.html")

    @app.route("/review/<card_id>")
    def review(card_id):
        card = store.get(card_id)
        if card is None:
            return "找不到此卡片", 404
        return render_template(
            "review.html",
            card=card,
            knowledge_types=KNOWLEDGE_TYPES,
            confidences=CONFIDENCES,
        )

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
        # 先存下審核者當前的編輯，再依補充說明對話式重整理（Phase 4）
        _save_edits(store, card_id, request.form)
        card = store.get(card_id)
        feedback = request.form.get("補充說明", "").strip()
        try:
            if feedback:
                new = refine_card(card, feedback, from_env())
            else:
                new = structure_transcript(card.原始逐字稿, from_env(), card.更新人)
        except Exception as e:  # noqa: BLE001
            return render_template(
                "review.html",
                card=card,
                knowledge_types=KNOWLEDGE_TYPES,
                confidences=CONFIDENCES,
                error=f"重新整理失敗：{e}",
            )
        store.update_fields(
            card_id,
            標題=new.標題,
            內容=new.內容,
            標籤=new.標籤,
            知識類型=new.知識類型.value,
            適用範圍=new.適用範圍,
            信心等級=new.信心等級.value,
        )
        return redirect(url_for("review", card_id=card_id))

    return app


def _save_edits(store: Storage, card_id: str, form) -> None:
    標籤 = [t.strip() for t in form.get("標籤", "").split(",") if t.strip()]
    store.update_fields(
        card_id,
        標題=form.get("標題", "").strip(),
        內容=form.get("內容", "").strip(),
        標籤=標籤,
        知識類型=form.get("知識類型", ""),
        適用範圍=form.get("適用範圍", "").strip(),
        信心等級=form.get("信心等級", ""),
    )


if __name__ == "__main__":
    create_app().run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
