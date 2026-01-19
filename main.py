from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import google.generativeai as genai
from prompt import SYSTEM_PROMPT
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os


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
async def chat(request: Request):
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
        print(response)

        reply = ""
        if response.candidates and response.candidates[0].content.parts:
           reply = response.candidates[0].content.parts[0].text

        return JSONResponse({"reply": reply})
    
    except Exception as e:
      return JSONResponse(
        {"error": str(e)},
        status_code=500
    )




@app.post("/voice-chat")
async def voice_chat(file: UploadFile = File(...)):

    audio_path = f"temp_{uuid.uuid4()}.webm"
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    # VOICE → TEXT
    segments, info = whisper_model.transcribe(audio_path)
    user_text = "".join(segment.text for segment in segments)

    # Cleanup audio file
    if os.path.exists(audio_path):
        os.remove(audio_path)

    # AI RESPONSE
    prompt = f"""
    {SYSTEM_PROMPT}

    User: {user_text}
    Assistant:
    """

    response = model.generate_content(prompt)
    reply_text = ""
    if response.candidates and response.candidates[0].content.parts:
        reply_text = response.candidates[0].content.parts[0].text

    return {
        "user_text": user_text,
        "reply_text": reply_text
    }

