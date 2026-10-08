import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()  # ключ только на сервере
MODELS = [m for m in [os.environ.get("GEMINI_MODEL"), "gemini-2.5-flash",
                      "gemini-2.0-flash", "gemini-flash-latest"] if m]
WORKING = None  # запоминаем рабочую связку (api, model)

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


def targets():
    combos = [(api, m) for api in ("studio", "vertex") for m in MODELS]
    if WORKING in combos:
        combos.remove(WORKING)
        combos.insert(0, WORKING)
    return combos


async def ask(payload):
    """Пробует AI Studio и Vertex (ключи формата AQ.… работают через Vertex)."""
    global WORKING
    attempts = []
    async with httpx.AsyncClient(timeout=30) as c:
        for api, model in targets():
            if api == "studio":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                r = await c.post(url, json=payload, headers={"x-goog-api-key": API_KEY})
            else:
                url = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent"
                r = await c.post(url, json=payload, params={"key": API_KEY})
            if r.status_code == 200:
                try:
                    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                    WORKING = (api, model)
                    return text, attempts
                except (KeyError, IndexError):
                    attempts.append({"api": api, "model": model, "status": 200, "err": "пустой ответ"})
                    continue
            err = r.text[:160].replace("\n", " ")
            attempts.append({"api": api, "model": model, "status": r.status_code, "err": err})
            print("GEMINI FAIL", api, model, r.status_code, err, flush=True)
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
