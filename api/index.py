import os
from datetime import datetime, timezone

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import PlainTextResponse
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Index
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool

from openai import OpenAI

# ======= 基礎設定 =======
app = FastAPI()

# 讀取環境變數（不要把金鑰寫死在程式碼裡）
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    # 在 Vercel 冷啟動時提示；實際運行時仍可回 500
    pass

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# OpenAI 用新版 SDK
oai_client = OpenAI(api_key=OPENAI_API_KEY)

# ======= 資料庫（建議使用雲端 Postgres，如 Neon/Supabase/Vercel Postgres）=======
# DATABASE_URL 例：postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DBNAME
DB_URL = os.getenv("DATABASE_URL", "")

Base = declarative_base()

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)          # "user" 或 "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

Index("idx_user_time", ChatMessage.user_id, ChatMessage.created_at)

if DB_URL:
    engine = create_engine(DB_URL, poolclass=NullPool, echo=False, future=True)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
else:
    # 無資料庫時，降級為「記憶體模式」（僅當前執行個體生效，serverless 會重置，不建議正式用）
    engine = None
    SessionLocal = None

# ======= OpenAI 生成函式 =======
def build_history_for_user(db_sess, user_id: str, limit: int = 12):
    history = []
    if db_sess is None:
        return history
    msgs = (
        db_sess.query(ChatMessage)
        .filter(ChatMessage.user_id == user_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
        .all()
    )
    msgs = list(reversed(msgs))
    for m in msgs:
        history.append({"role": "assistant" if m.role == "assistant" else "user", "content": m.content})
    return history

def get_gpt_response(user_id: str, user_message: str, db_sess):
    system_prompt = (
        "You are a helpful LINE bot. Keep answers concise and useful. "
        "If the user asks for code, provide clear, runnable snippets."
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(build_history_for_user(db_sess, user_id, limit=12))
    messages.append({"role": "user", "content": user_message})

    resp = oai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=600,
    )
    return resp.choices[0].message.content.strip()

# ======= Health Check =======
@app.get("/")
@app.get("/health")
def health():
    return PlainTextResponse("OK", 200)

# ======= LINE Webhook (Vercel 以 /api/index.py 對應；我們在 vercel.json 裡把 /callback 導過來) =======
@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    body_text = body.decode("utf-8")
    try:
        handler.handle(body_text, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return PlainTextResponse("OK")

# ======= LINE 事件處理 =======
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id
    user_text = event.message.text

    db = SessionLocal() if SessionLocal else None
    try:
        # Step 1: 儲存 user 訊息（若無 DB 則略過）
        if db:
            db.add(ChatMessage(user_id=user_id, role="user", content=user_text))
            db.commit()

        # Step 2: OpenAI 回覆
        reply_text = get_gpt_response(user_id, user_text, db)

        # Step 3: 回覆 LINE
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply_text[:5000])  # LINE 單則訊息長度限制
        )

        # Step 4: 儲存 assistant 訊息
        if db:
            db.add(ChatMessage(user_id=user_id, role="assistant", content=reply_text))
            db.commit()
    except Exception as e:
        # 失敗時仍嘗試回覆
        try:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="抱歉，系統忙碌，請稍後再試。")
            )
        except Exception:
            pass
    finally:
        if db:
            db.close()
