import os
import random
import asyncio

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()  # ключ только на сервере
BASE = "https://generativelanguage.googleapis.com/v1beta"
FIRST = [m for m in [os.environ.get("GEMINI_MODEL"), "gemini-flash-latest",
                     "gemini-flash-lite-latest"] if m]
WORKING = None   # запоминаем рабочую модель
DISCOVERED = []  # модели, найденные у самого Google

SYSTEM = (
    "Ты — эмпатичный ИИ-консультант для студентов (проект MindGuard). "
    "Отвечай по-русски, тепло, живым человеческим языком, коротко (до 120 слов). "
    "Каждый ответ должен отличаться от предыдущих: меняй начало, структуру, длину и тон, "
    "не повторяй одни и те же фразы-заготовки и эмодзи. "
    "Анализируй эмоции и смысл сообщения. Если у человека всё хорошо — искренне радуйся вместе с ним "
    "и поддержи разговор (спроси, что порадовало), без советов про стресс. "
    "Если стресс, тревога или выгорание — сначала отрази чувства, назови возможный триггер, "
    "дай 1–2 маленьких конкретных шага (КПТ, заземление, дыхание) "
    "Пол собеседника неизвестен: используй гендерно-нейтральные окончания со скобками "
    "(например, «прошел(-а)», «устал(-а)», «рад(-а)», «один(-а)»), но не перегружай текст. "
    "Ты не врач и не ставишь диагнозов. "
    "При мыслях о самоповреждении или суициде тепло отреагируй, скажи, что человек важен, "
    "и напомни: 150 (телефон доверия), 112 (служба спасения), 103 (скорая)."
)

# Случайный «настрой» ответа — добавляется к каждому запросу, чтобы ответы не повторялись
STYLES = [
    "Начни с короткой фразы, которая отражает чувство человека.",
    "Начни с вопроса, который помогает лучше понять ситуацию.",
    "Ответь спокойно и просто, как близкий друг.",
    "Начни с поддержки, потом дай один конкретный маленький шаг.",
    "Используй образ или мягкую метафору, но коротко.",
    "Ответь бодро и тепло, с лёгкой улыбкой в тексте.",
    "Ответь вдумчиво, без эмодзи.",
    "Начни не со слов «Понимаю» и «Привет», выбери необычное начало.",
]

SAFETY = [{"category": c, "threshold": "BLOCK_ONLY_HIGH"} for c in (
    "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")]

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


def extract(data):
    """Достаём текст из ответа Gemini (все части, без служебных)."""
    cand = data["candidates"][0]
    parts = cand.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
    if not text:
        raise KeyError("empty")
    return text


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
                r = None
                for _ in range(2):  # при 503/429 пробуем ещё раз
                    try:
                        r = await c.post(f"{BASE}/models/{model}:generateContent", json=payload,
                                         headers={"x-goog-api-key": API_KEY})
                    except httpx.HTTPError as e:
                        attempts.append({"model": model, "status": 0, "err": str(e)[:120]})
                        r = None
                        break
                    if r.status_code in (503, 429):
                        await asyncio.sleep(1.5)
                        continue
                    break
                if r is None:
                    continue
                if r.status_code == 200:
                    try:
                        text = extract(r.json())
                        WORKING = model
                        return text, attempts
                    except (KeyError, IndexError, ValueError):
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
        contents.append({"role": "user", "parts": [{"text": str(h.get("q", ""))[:2000]}]})
        contents.append({"role": "model", "parts": [{"text": str(h.get("a", ""))[:2000]}]})
    contents.append({"role": "user", "parts": [{"text": body.message[:2000]}]})
    system = SYSTEM + " Настрой для этого ответа: " + random.choice(STYLES)
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": contents,
        # запас по токенам, чтобы «размышления» модели не обрезали ответ
        "generationConfig": {"temperature": 1.0, "topP": 0.95, "maxOutputTokens": 1500},
        "safetySettings": SAFETY,
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


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/")
def index():
    # no-cache: после обновления файла браузер сразу берёт новую версию
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"),
                        headers={"Cache-Control": "no-cache, must-revalidate"})
