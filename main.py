import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

API_KEY = os.environ.get("GEMINI_API_KEY")  # ключ хранится ТОЛЬКО на сервере
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

SYSTEM = (
    "Ты — эмпатичный ИИ-консультант для студентов (проект MindGuard). "
    "Отвечай по-русски, тепло и по-разному каждый раз, коротко (до 120 слов). "
    "Анализируй эмоции и смысл сообщения. Если у человека всё хорошо — искренне радуйся вместе с ним. "
    "Если стресс или выгорание — сначала отрази чувства, назови возможный триггер, дай 1–2 маленьких шага (КПТ, заземление) "
    "и мягко предложи обратиться к психологу университета: Айгерим Маратовна, Satbayev University, "
    "ГМК-218, тел. +7 702 672 9961. Используй гендерно-нейтральные окончания со скобками "
    "(например, «прошел(-а)», «устал(-а)»). Ты не врач и не ставишь диагнозов. "
    "При мыслях о самоповреждении или суициде тепло отреагируй и напомни: 150 (телефон доверия), 112 (служба спасения)."
)

app = FastAPI(title="MindGuard API")


class Chat(BaseModel):
    message: str
    history: list[dict] = []


@app.post("/api/chat")
async def chat(body: Chat):
    if not API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY не задан на сервере")
    contents = []
    for h in body.history[-6:]:
        contents.append({"role": "user", "parts": [{"text": str(h.get("q", ""))}]})
        contents.append({"role": "model", "parts": [{"text": str(h.get("a", ""))}]})
    contents.append({"role": "user", "parts": [{"text": body.message[:2000]}]})
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.9, "maxOutputTokens": 500},
    }
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(URL, json=payload, headers={"x-goog-api-key": API_KEY})
    if r.status_code != 200:
        raise HTTPException(502, "Ошибка Gemini API")
    try:
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise HTTPException(502, "Пустой ответ модели")
    return {"reply": text}


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))
