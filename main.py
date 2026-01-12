from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import google.generativeai as genai
from prompt import SYSTEM_PROMPT
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from fastapi import UploadFile, File
from gtts import gTTS
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os

# Ensure directories exist
os.makedirs("static/audio", exist_ok=True)

# CONFIGURE GEMINI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-pro")

whisper_model = WhisperModel("base")

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins (OK for MVP)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health-check")
def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message")

    prompt = f"""
    {SYSTEM_PROMPT}

    User: {user_message}
    Assistant:
    """

    try:
     response = model.generate_content(prompt)
     return JSONResponse({"reply": response.text})
    except Exception as e:
     return JSONResponse(
        {"error": str(e)},
        status_code=500
    )



@app.post("/voice-chat")
async def voice_chat(file: UploadFile = File(...)):

    # Save uploaded audio
    audio_path = f"static/{uuid.uuid4()}.webm"
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    # VOICE → TEXT (STT)
    segments, info = whisper_model.transcribe(audio_path)
    user_text = "".join(segment.text for segment in segments)


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
