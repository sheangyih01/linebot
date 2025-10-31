from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage
import openai, json, datetime
from linebot import LineBotApi, WebhookHandler
import firebase_admin
from firebase_admin import credentials, firestore, storage


# 載入環境變數（若使用 .env 檔）
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
line_bot_api = "A8aPv13zEB5MSBRZS5m34Q3ozFFOX66sI8D0IdFFl09bXHssUljyIl/IJ6v4L6/Dxb7qKlsXDE60N2HmYQumh7tHXCRs3bComJQuDyvmaTV/cja5ffcOnCWKa/JwA3hd4ZzYRtjQ9k8gHQvxkqlOowdB04t89/1O/w1cDnyilFU="
handler = "8775b088f58f1b63d1c5c77d16461887"
openai.api_key  = "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"
cred = credentials.Certificate("linebot-9a1b6-firebase-adminsdk-fbsvc-20eca79fa9.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'https://linebot-9a1b6-default-rtdb.firebaseio.com/'})
db = firestore.client()
bucket = storage.bucket()

@app.route("/api/callback", methods=['POST'])
def callback():
    # 驗證 LINE 簽章
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)   # 交給 handler 處理事件
    except:
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    user_id = event.source.user_id
    user_message = event.message.text
    reply_token = event.reply_token

    # 保存使用者訊息到 Firestore
    db.collection("conversations").add({
        "userId": user_id,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "role": "user",
        "content": user_message
    })

    # 建立對話歷史上下文（讀取最近的幾則對話）
    # 這裡簡化示意：取得該用戶最近 5 條訊息
    history = []
    docs = db.collection("conversations")\
             .where("userId", "==", user_id)\
             .order_by("timestamp", direction=firestore.Query.DESCENDING)\
             .limit(5).stream()
    for doc in reversed(list(docs)):
        data = doc.to_dict()
        history.append({"role": data["role"], "content": data["content"]})

    # 若有預先儲存的知識（如 FAQ），也可插入系統提示
    faq_doc = db.collection("knowledge").document("faq").get()
    if faq_doc.exists:
        faq_content = faq_doc.to_dict().get("text", "")
        history.insert(0, {"role": "system", "content": faq_content})

    # 呼叫 OpenAI 的對話完成 API，傳入對話上下文
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=history + [{"role": "user", "content": user_message}]
    )
    ai_reply = response.choices[0].message.content

    # 保存機器人回覆到 Firestore
    db.collection("conversations").add({
        "userId": user_id,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "role": "assistant",
        "content": ai_reply
    })

    # 回傳訊息給 LINE
    line_bot_api.reply_message(
        reply_token,
        TextSendMessage(text=ai_reply)
    )
@handler.add(MessageEvent, message=ImageMessage)

def handle_image(event):
    user_id = event.source.user_id
    reply_token = event.reply_token
    message_id = event.message.id

    # 取得圖片二進位資料
    message_content = line_bot_api.get_message_content(message_id)
    filename = f"{user_id}_{message_id}.jpg"
    with open(filename, 'wb') as fd:
        for chunk in message_content.iter_content():
            fd.write(chunk)

    # 將圖片上傳到 Firebase Storage
    blob = bucket.blob(f"images/{filename}")
    blob.upload_from_filename(filename, content_type='image/jpeg')
    image_url = blob.public_url  # 儲存後可取得公開 URL

    # 儲存圖片訊息紀錄到 Firestore
    db.collection("conversations").add({
        "userId": user_id,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "role": "assistant",
        "content": image_url
    })

    # 回覆提示文字給使用者
    line_bot_api.reply_message(
        reply_token,
        TextSendMessage(text="圖片已收到並儲存囉！")
    )
