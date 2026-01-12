from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import google.generativeai as genai
from prompt import SYSTEM_PROMPT
from fastapi.staticfiles import StaticFiles
import whisper
from fastapi import UploadFile, File
from gtts import gTTS
import uuid
import os

# CONFIGURE GEMINI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash-lite")

whisper_model = whisper.load_model("base")

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message")

    prompt = f"""
    {SYSTEM_PROMPT}

    User: {user_message}
    Assistant:
    """

    response = model.generate_content(prompt)

    return JSONResponse({"reply": response.text})


@app.post("/voice-chat")
async def voice_chat(file: UploadFile = File(...)):

    # Save uploaded audio
    audio_path = f"static/{uuid.uuid4()}.webm"
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    # VOICE → TEXT (STT)
    result = whisper_model.transcribe(audio_path)
    user_text = result["text"]

    # AI RESPONSE
    prompt = f"""
    {SYSTEM_PROMPT}

    User: {user_text}
    Assistant:
    """

    response = model.generate_content(prompt)
    reply_text = response.text

    # TEXT → VOICE (TTS)
    audio_reply_path = f"static/audio/{uuid.uuid4()}.mp3"
    tts = gTTS(reply_text)
    tts.save(audio_reply_path)

    return {
        "reply_text": reply_text,
        "audio_reply": audio_reply_path
    }
