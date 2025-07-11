from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os, time, tempfile, json, requests
import httpx
from gtts import gTTS
from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
print("🔑 OpenRouter Key Loaded?", bool(OPENROUTER_API_KEY))

PUBLIC_AUDIO_BASE_URL = "https://agrivoice-2-ws-2a-8000.ml.iit-ropar.truefoundry.cloud"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

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

async def refine_query_with_deepseek(question: str) -> str:
    system_prompt = "You are an AI that rewrites noisy farmer questions into clean search queries relevant to Indian agriculture."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://agrikart.ai",
        "X-Title": "AgriKart VoiceBot"
    }

    payload = {
        "model": "deepseek/deepseek-r1:free",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Refine this into a clean Google search query: {question}"}
        ]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        refined = response.json()["choices"][0]["message"]["content"]
        print("🔍 Refined search query:", refined)
        return refined.strip()
    except Exception as e:
        print("❌ DeepSeek refine failed:", e)
        return question

async def search_tnau(query: str, max_results=5) -> list[str]:
    """
    Use an external API endpoint to fetch search snippets from agritech.tnau.ac.in.
    This replaces direct Google API call.
    """
    TNAU_SEARCH_API = "http://localhost:8001/search/"  # 🔁 Replace with deployed URL if needed

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(TNAU_SEARCH_API, json={"query": query, "num_results": max_results})
            response.raise_for_status()
            data = response.json()
            snippets = data.get("snippets", [])
            print("📄 TNAU Search Snippets:", snippets)
            return snippets
    except Exception as e:
        print("❌ TNAU Search API Error:", e)
        return []


async def call_deepseek_with_context(question: str, context_snippets: list[str], lang_name: str):
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(lang_name=lang_name)
    prompt = "\n".join(context_snippets) + f"\n\nFarmer's Question: {question}"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://agrikart.ai",
        "X-Title": "AgriKart VoiceBot"
    }

    payload = {
        "model": "deepseek/deepseek-r1:free",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        completion = response.json()
        print("✅ DeepSeek response generated")
        return completion["choices"][0]["message"]["content"]
    except Exception as e:
        print("❌ DeepSeek failed:", e)
        return "Sorry, I'm unable to answer your question at the moment."

def convert_ogg_to_mp3(input_path: str, output_path: str):
    try:
        sound = AudioSegment.from_ogg(input_path)
        sound.export(output_path, format="mp3")
        print("🎵 Converted OGG to MP3")
    except Exception as e:
        print("Conversion error:", e)

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

@app.post("/chat/")
async def chat(file: UploadFile = File(...), lang: str = Form(...)):
    try:
        suffix = ".ogg" if file.filename.endswith(".ogg") else ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            raw_audio_path = tmp.name
            print("📥 Audio uploaded to:", raw_audio_path)

        if raw_audio_path.endswith(".ogg"):
            mp3_audio_path = raw_audio_path.replace(".ogg", ".mp3")
            convert_ogg_to_mp3(raw_audio_path, mp3_audio_path)
            os.remove(raw_audio_path)
        else:
            mp3_audio_path = raw_audio_path

        transcription_api_url = "https://agrivoice-api-ws-2a-8000.ml.iit-ropar.truefoundry.cloud/chat/"
        async with httpx.AsyncClient() as client:
            with open(mp3_audio_path, "rb") as f:
                files = {"file": (file.filename, f, "audio/mpeg")}
                response = await client.post(transcription_api_url, files=files)
        os.remove(mp3_audio_path)

        if response.status_code != 200:
            raise Exception(f"Transcription API failed: {response.status_code} {response.text}")

        transcribed = response.json()
        question = transcribed.get("transcription", "")
        print("🗣 Transcription:", question)

    except Exception as e:
        print("❌ Transcription failed:", e)
        return JSONResponse({"error": f"Transcription failed: {str(e)}"}, status_code=500)

    # ✨ Refine query
    refined_query = await refine_query_with_deepseek(question)
    # ✨ Search TNAU with refined query
    context = await search_tnau(refined_query)

    # ✨ Generate reply
    lang_name = LANG_MAP.get(lang, "Hindi")
    reply = await call_deepseek_with_context(question, context, lang_name)

    # ✨ Sanitize reply
    reply = reply.strip()
    if len(reply) > 1000:
        reply = reply[:1000] + "..."
    if not reply.replace(" ", "").isprintable():
        print("⚠️ Bad reply detected:", repr(reply))
        reply = "Sorry, I couldn't process your question. Please try again."

    print("🤖 Final reply:", repr(reply))

    audio_url = generate_tts(reply, lang)

    return JSONResponse({
        "language": lang,
        "transcription": question,
        "response": reply,
        "audio_url": audio_url
    })
