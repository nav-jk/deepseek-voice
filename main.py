from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os, time, tempfile
import httpx
from gtts import gTTS
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
print("🔑 OpenRouter Key Loaded?", bool(OPENROUTER_API_KEY))
PUBLIC_AUDIO_BASE_URL = "https://agrivoice-2-ws-2a-8000.ml.iit-ropar.truefoundry.cloud"

# FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# DeepSeek client via OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://agrikart.ai",
        "X-Title": "AgriKart VoiceBot"
    }
)


SYSTEM_PROMPT_TEMPLATE = """
You are an assistant designed to help Indian farmers by answering their agricultural questions. These questions are often spoken in local languages and converted to text, so they may contain spelling or grammar mistakes. Your job is to understand the question and respond appropriately.

Respond in {lang_name}.

Instructions:
- Give short answers, ideally 1 to 3 lines.
- Use very simple language that even a village farmer can understand.
- Focus only on the question — don’t add extra facts unless necessary.
- Don’t repeat the question in your answer.
- Do not mention AI or voice input in the response.

Add this line at the end of your answer and choose the language of this line according to {lang_name}:
“This is an AI-generated response. Please confirm with local experts or call Kisan Call Centre (KCC).”
"यह एक एआई द्वारा जनित उत्तर है। कृपया स्थानीय विशेषज्ञों से पुष्टि करें या किसान कॉल सेंटर (KCC) पर कॉल करें।"
"""

LANG_MAP = {
    "hi": "Hindi", "en": "English", "ta": "Tamil", "ml": "Malayalam",
    "te": "Telugu", "mr": "Marathi", "kn": "Kannada", "bn": "Bengali"
}

# 🔍 Context fetch from Serper
async def fetch_search_snippets(query: str):
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post("https://google.serper.dev/search", headers=headers, json={"q": query, "gl": "in"})
            data = res.json()
            snippets = [item.get("snippet", "") for item in data.get("organic", [])[:5]]
            print("🔍 Search snippets:", snippets)
            return snippets
    except Exception as e:
        print("❌ Serper fetch error:", e)
        return []

# 🤖 DeepSeek response
async def call_deepseek_with_context(question: str, context_snippets: list[str], lang_name: str):
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(lang_name=lang_name)
    prompt = "\n".join(context_snippets) + f"\n\nFarmer's Question: {question}"
    try:
        completion = client.chat.completions.create(
            model="deepseek/deepseek-r1-0528:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        print("✅ DeepSeek response generated")
        return completion.choices[0].message.content
    except Exception as e:
        print("❌ DeepSeek failed:", e)
        return "Sorry, I'm unable to answer your question at the moment."


# 🔊 TTS with gTTS
def generate_tts(text: str, lang: str) -> str | None:
    try:
        tts = gTTS(text=text, lang=lang)
        filename = f"{int(time.time())}_{lang}.mp3"
        audio_dir = "static/audio"
        os.makedirs(audio_dir, exist_ok=True)
        path = os.path.join(audio_dir, filename)
        tts.save(path)
        print("🔊 Audio saved at:", path)
        return f"{PUBLIC_AUDIO_BASE_URL}/static/audio/{filename}"
    except Exception as e:
        print("TTS error:", e)
        return None

# 🎙️ Main API Endpoint
@app.post("/chat/")
async def chat(file: UploadFile = File(...), lang: str = Form(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(await file.read())
            audio_path = tmp.name
            print("📥 Audio uploaded to:", audio_path)

        # Send audio to transcription API
        transcription_api_url = "https://agrivoice-api-ws-2a-8000.ml.iit-ropar.truefoundry.cloud/chat/"
        async with httpx.AsyncClient() as client:
            with open(audio_path, "rb") as f:
                files = {"file": (file.filename, f, file.content_type)}
                response = await client.post(transcription_api_url, files=files)
        os.remove(audio_path)

        if response.status_code != 200:
            raise Exception(f"Transcription API failed: {response.status_code} {response.text}")

        transcribed = response.json()
        question = transcribed.get("transcription", "")
        print("🗣 Transcription:", question)

    except Exception as e:
        print("❌ Transcription failed:", e)
        return JSONResponse({"error": f"Transcription failed: {str(e)}"}, status_code=500)

    # Search + LLM
    context = await fetch_search_snippets(question)
    lang_name = LANG_MAP.get(lang, "Hindi")
    reply = await call_deepseek_with_context(question, context, lang_name)
    print("🤖 Final reply:", reply)

    # TTS
    audio_url = generate_tts(reply, lang)

    return JSONResponse({
        "language": lang,
        "transcription": question,
        "response": reply,
        "audio_url": audio_url
    })
