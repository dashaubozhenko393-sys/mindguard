import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import asyncio

API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()  # ключ только на сервере
BASE = "https://generativelanguage.googleapis.com/v1beta"
FIRST = [m for m in [os.environ.get("GEMINI_MODEL"), "gemini-flash-latest",
                     "gemini-3.8-flash", "gemini-flash-lite-latest"] if m]
WORKING = None   # запоминаем рабочую модель
DISCOVERED = []  # модели, найденные у самого Google

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


async def discover(c):
    """Спрашиваем у Google, какие flash-модели доступны этому ключу."""
    global DISCOVERED
    if DISCOVERED:
        return DISCOVERED
    try:
        r = await c.get(f"{BASE}/models?pageSize=100", headers={"x-goog-api-key": API_KEY})
        for m in r.json().get("models", []):
            n = m.get("name", "").replace("models/", "")
            bad = ("image", "tts", "live", "audio", "embedding", "robotics", "computer")
            if "flash" in n and "generateContent" in m.get("supportedGenerationMethods", []) \
                    and not any(x in n for x in bad):
                DISCOVERED.append(n)
        DISCOVERED.sort(reverse=True)
    except Exception as e:
        print("DISCOVER FAIL", e, flush=True)
    return DISCOVERED


async def ask(payload):
    global WORKING
    attempts = []
    async with httpx.AsyncClient(timeout=30) as c:
        names = ([WORKING] if WORKING else []) + FIRST
        tried = set()
        for phase in (0, 1):
            if phase == 1:
                names = await discover(c)
            for model in names:
                if model in tried:
                    continue
                tried.add(model)
                for _ in range(2):  # при 503/429 пробуем ещё раз
                    r = await c.post(f"{BASE}/models/{model}:generateContent", json=payload,
                                     headers={"x-goog-api-key": API_KEY})
                    if r.status_code in (503, 429):
                        await asyncio.sleep(1.5)
                        continue
                    break
                if r.status_code == 200:
                    try:
                        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                        WORKING = model
                        return text, attempts
                    except (KeyError, IndexError):
                        attempts.append({"model": model, "status": 200, "err": "пустой ответ"})
                        continue
                err = r.text[:160].replace("\n", " ")
                attempts.append({"model": model, "status": r.status_code, "err": err})
                print("GEMINI FAIL", model, r.status_code, err, flush=True)
    return None, attempts


@app.post("/api/chat")
async def chat(body: Chat):
    if not API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY не задан")
    contents = []
    for h in body.history[-6:]:
        contents.append({"role": "user", "parts": [{"text": str(h.get("q", ""))}]})
        contents.append({"role": "model", "parts": [{"text": str(h.get("a", ""))}]})
    contents.append({"role": "user", "parts": [{"text": body.message[:2000]}]})
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.9, "maxOutputTokens": 600},
    }
    text, attempts = await ask(payload)
    if text is None:
        raise HTTPException(502, "Gemini недоступен")
    return {"reply": text}


@app.get("/api/test")
async def test():
    """Открой в браузере /api/test — покажет, работает ли ключ."""
    payload = {"contents": [{"role": "user", "parts": [{"text": "Скажи «привет» одним словом"}]}]}
    text, attempts = await ask(payload) if API_KEY else (None, [])
    return {"ok": text is not None, "reply": text, "key_set": bool(API_KEY),
            "key_len": len(API_KEY), "key_start": API_KEY[:3], "attempts": attempts}


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))
