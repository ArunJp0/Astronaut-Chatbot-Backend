from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import google.generativeai as genai
from prompt import SYSTEM_PROMPT
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from fastapi import UploadFile, File
from gtts import gTTS
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Header, HTTPException
from supabase import create_client
import uuid
import os

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def verify_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing auth token")

    token = authorization.replace("Bearer ", "")
    user = supabase.auth.get_user(token)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")

    return user

# Ensure directories exist
os.makedirs("static/audio", exist_ok=True)

# CONFIGURE GEMINI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

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
async def chat(request: Request, user=Depends(verify_user)):
    data = await request.json()
    user_message = data.get("message")

    MAX_MESSAGE_LENGTH = 500

    if not user_message:
        return JSONResponse(
            {"error": "Message is required"},
            status_code=400
        )

    if len(user_message) > MAX_MESSAGE_LENGTH:
        return JSONResponse(
            {"error": "Message too long. Please keep it under 500 characters."},
            status_code=400
        )

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
