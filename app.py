# app.py
import os
import io
import json
import time
import uuid
import math
import base64
import datetime
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from flask import Flask, request, abort, send_from_directory
from PIL import Image

# LINE SDK
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageSendMessage
)

# OpenAI SDK (v1 風格)
from openai import OpenAI
client = None  # 稍後 init

# ========= 基礎設定 =========
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")  # 對外可存取的 base URL
PORT = int(os.getenv("PORT", "5000"))

# 目錄
DATA_DIR = "data"
LOG_DIR = os.path.join(DATA_DIR, "logs")
KB_DIR = os.path.join(DATA_DIR, "kb")
INDEX_PATH = os.path.join(DATA_DIR, "index.jsonl")
STATIC_DIR = "static"
IMG_DIR = os.path.join(STATIC_DIR, "images")

# 確保目錄存在
for p in [DATA_DIR, LOG_DIR, KB_DIR, STATIC_DIR, IMG_DIR]:
    os.makedirs(p, exist_ok=True)

# Flask 與 LINE 初始化
app = Flask(__name__)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# OpenAI 初始化
client = OpenAI(api_key=OPENAI_API_KEY)

# ========= 實用工具 =========
def now_ts() -> str:
    """回傳台灣時區時間字串 YYYY-MM-DD HH:MM:SS"""
    # 簡化起見，使用系統時間；如需更精準可用 pytz
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def save_chat_log(user_id: str, role: str, content: str) -> None:
    """將每則對話存文字檔，一行一則，方便查核與審計"""
    fname = os.path.join(LOG_DIR, f"{user_id}_{datetime.date.today()}.log")
    with open(fname, "a", encoding="utf-8") as f:
        f.write(f"[{now_ts()}] {role.upper()}: {content}\n")

def load_recent_dialogue(user_id: str, max_lines: int = 20) -> List[str]:
    """
    讀取最近一天的聊天紀錄若干行，做為上下文回憶。
    如需跨日、跨檔案，可自行加強。
    """
    fname = os.path.join(LOG_DIR, f"{user_id}_{datetime.date.today()}.log")
    if not os.path.exists(fname):
        return []
    with open(fname, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [ln.strip() for ln in lines[-max_lines:]]

def cosine(a: List[float], b: List[float]) -> float:
    """計算兩個向量的 cosine 相似度"""
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def embed_text(text: str) -> List[float]:
    """
    呼叫 OpenAI Embeddings，把文字轉成向量。
    使用 text-embedding-3-small 以節省成本，已足夠 RAG。
    """
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding

def append_index(record: Dict[str, Any]) -> None:
    """將新索引項目追加寫入 JSONL 檔"""
    with open(INDEX_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_index() -> List[Dict[str, Any]]:
    """讀取整個索引為 list"""
    if not os.path.exists(INDEX_PATH):
        return []
    out = []
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def search_kb(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    在本地 KB 索引做相似度檢索，回傳最接近的片段清單。
    """
    qv = embed_text(query)
    idx = load_index()
    scored = []
    for rec in idx:
        sim = cosine(qv, rec["embedding"])
        scored.append((sim, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [rec for sim, rec in scored[:top_k] if sim > 0.2]  # 簡單門檻

def ensure_user_memo_file(user_id: str) -> str:
    """每位使用者一個 memo 檔，存持久備忘錄"""
    path = os.path.join(KB_DIR, f"memo_{user_id}.txt")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("")  # 建空檔
    return path

def read_user_memo(user_id: str) -> str:
    """讀出使用者備忘錄內容"""
    path = ensure_user_memo_file(user_id)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def append_user_memo(user_id: str, text: str) -> None:
    """將重要資訊寫入使用者備忘錄，並同步更新索引"""
    path = ensure_user_memo_file(user_id)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{now_ts()}] {text}\n")
    # 寫入索引，讓 memo 也能被檢索到
    emb = embed_text(text)
    append_index({
        "id": str(uuid.uuid4()),
        "type": "memo",
        "user_id": user_id,
        "title": "user_memo",
        "chunk": text,
        "embedding": emb
    })

def write_kb_and_index(title: str, content: str, user_id: str = "global") -> None:
    """
    把新知寫進 KB 檔，並切 chunk 做向量索引。
    簡化策略：每 600~800 字元切一段。
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c for c in title if c.isalnum() or c in ("_", "-", " ")).strip()
    kb_path = os.path.join(KB_DIR, f"{ts}_{safe_title or 'kb'}.txt")

    with open(kb_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{content}\n")

    # 切片
    chunk_size = 800
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    for ch in chunks:
        emb = embed_text(ch)
        append_index({
            "id": str(uuid.uuid4()),
            "type": "kb",
            "user_id": user_id,   # 若想區分私有 vs 全域，可改策略
            "title": title,
            "chunk": ch,
            "embedding": emb
        })

def render_chat_context(history_lines: List[str]) -> List[Dict[str, str]]:
    """
    將文字檔形式的歷史對話轉為 OpenAI Chat messages。
    範例行： [2025-10-29 14:05:01] USER: 你好
    """
    msgs = []
    for ln in history_lines:
        # 粗略解析
        if "] USER:" in ln:
            content = ln.split("] USER:", 1)[1].strip()
            msgs.append({"role": "user", "content": content})
        elif "] ASSISTANT:" in ln:
            content = ln.split("] ASSISTANT:", 1)[1].strip()
            msgs.append({"role": "assistant", "content": content})
    return msgs[-10:]  # 控制訊息量

def build_system_prompt(user_memo: str, kb_snippets: List[Dict[str, Any]]) -> str:
    """
    將使用者備忘錄與 KB 片段組成 System Prompt，讓模型在回覆時使用。
    """
    guide = []
    guide.append("你是嚴謹且有記憶力的助理。回覆時優先根據下列知識與備忘錄。")
    if user_memo.strip():
        guide.append("\n[使用者備忘錄]\n" + user_memo.strip())
    if kb_snippets:
        guide.append("\n[知識庫片段]")
        for i, rec in enumerate(kb_snippets, 1):
            guide.append(f"({i}) 來源:{rec.get('title','KB')} 內容:{rec['chunk'][:500]}")
    guide.append("\n若找不到答案，才根據一般常識回覆。回答務必簡潔、準確。")
    return "\n".join(guide)

def chat_reply(user_id: str, user_text: str) -> str:
    """
    一般聊天流程：
    1) 取出備忘錄與 KB 檢索片段
    2) 組合 system + 歷史對話 + 當前訊息
    3) 呼叫 Chat 產生回覆
    """
    memo = read_user_memo(user_id)
    kb_hits = search_kb(user_text, top_k=5)
    system_prompt = build_system_prompt(memo, kb_hits)

    history_msgs = render_chat_context(load_recent_dialogue(user_id, max_lines=20))
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_msgs)
    messages.append({"role": "user", "content": user_text})

    resp = client.chat.completions.create(
        model="gpt-4o-mini",   # 可換成你可用的模型
        temperature=0.3,
        messages=messages,
    )
    return resp.choices[0].message.content.strip()

def generate_image_to_file(prompt: str) -> Tuple[str, str]:
    """
    產圖並落地成本地檔案，回傳 (public_url, local_path)。
    說明：
    - OpenAI Images API 回傳 base64 或 URL。這裡選 base64 更保險，再自行寫檔。
    - LINE 顯示需要可公開的 URL，所以需要 PUBLIC_BASE_URL 指向你的伺服器。
    """
    # 生成 1024x1024 圖片
    img_resp = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024",
        response_format="b64_json"
    )
    b64 = img_resp.data[0].b64_json
    img_bytes = base64.b64decode(b64)
    im = Image.open(io.BytesIO(img_bytes))

    # 存檔
    fname = f"{int(time.time())}_{uuid.uuid4().hex}.png"
    fpath = os.path.join(IMG_DIR, fname)
    im.save(fpath, format="PNG")

    if not PUBLIC_BASE_URL:
        raise RuntimeError("PUBLIC_BASE_URL 未設定，無法提供對外圖片 URL")

    public_url = f"{PUBLIC_BASE_URL}/static/images/{fname}"
    return public_url, fpath

# ========= LINE Webhook =========
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    user_id = event.source.user_id
    text = event.message.text.strip()

    # 1) 紀錄使用者輸入
    save_chat_log(user_id, "user", text)

    try:
        # 2) 指令路由
        if text.startswith("!learn"):
            """
            使用方式：
            !learn 標題
            內容內容內容...
            """
            parts = text.split("\n", 1)
            first = parts[0]
            title = first.replace("!learn", "", 1).strip() or "untitled"
            content = parts[1].strip() if len(parts) > 1 else ""
            if not content:
                reply = "請在下一行提供要學習的內容。範例：\n!learn T300RS 故障\n檢查USB連接、更新韌體、切PS模式..."
            else:
                write_kb_and_index(title, content, user_id="global")
                reply = f"已學會：{title}（內容已寫入本地 KB 並完成索引）"

            save_chat_log(user_id, "assistant", reply)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
            return

        if text.startswith("!memo"):
            """
            使用方式：
            !memo 我的稱呼是小王；偏好中文；常用單位TWD...
            存入個人備忘錄，作為持久記憶。
            """
            mem = text.replace("!memo", "", 1).strip()
            if not mem:
                reply = "請在 !memo 後面加要記住的內容。"
            else:
                append_user_memo(user_id, mem)
                reply = "已更新你的備忘錄，之後回覆會參考這些資訊。"

            save_chat_log(user_id, "assistant", reply)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
            return

        if text.startswith("!draw"):
            """
            使用方式：
            !draw 幫我畫一隻戴安全帽的烏龜，在工廠裡巡檢
            """
            prompt = text.replace("!draw", "", 1).strip()
            if not prompt:
                reply = "請在 !draw 後面描述你想畫的內容。"
                save_chat_log(user_id, "assistant", reply)
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
                return

            public_url, _ = generate_image_to_file(prompt)
            # 回傳圖片訊息給 LINE，用相同 URL 做 original 與 preview
            line_bot_api.reply_message(
                event.reply_token,
                ImageSendMessage(original_content_url=public_url, preview_image_url=public_url)
            )
            save_chat_log(user_id, "assistant", f"[image] {public_url}")
            return

        # 3) 一般聊天：做 KB 檢索增強
        reply_text = chat_reply(user_id, text)

        # 4) 存回覆並回傳
        save_chat_log(user_id, "assistant", reply_text)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

    except Exception as e:
        # 兜底錯誤處理，避免 Webhook 超時
        err = f"系統錯誤：{e}"
        save_chat_log(user_id, "assistant", err)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=err))

# 提供靜態圖片（若未用 Nginx 可走 Flask 靜態）
@app.route("/static/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(IMG_DIR, filename)

if __name__ == "__main__":
    # 開發模式啟動
    app.run(host="0.0.0.0", port=PORT, debug=True)
