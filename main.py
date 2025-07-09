from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os, time, tempfile, torch
import httpx
from gtts import gTTS
from openai import OpenAI
from dotenv import load_dotenv
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq

# Load env variables
load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PUBLIC_AUDIO_BASE_URL = "http://localhost:8000"

# FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Whisper setup
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
asr_model_id = "openai/whisper-large-v3"

asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    asr_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
).to(device)
asr_processor = AutoProcessor.from_pretrained(asr_model_id)

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=asr_model,
    tokenizer=asr_processor.tokenizer,
    feature_extractor=asr_processor.feature_extractor,
    chunk_length_s=30,
    batch_size=16,
    torch_dtype=torch_dtype,
    device=device
)

print("✅ Whisper ready")

# DeepSeek client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# System prompt (template)
SYSTEM_PROMPT_TEMPLATE = """
You are an assistant designed to help Indian farmers by answering their agricultural questions. These questions are often spoken in local languages and converted to text, so they may contain spelling or grammar mistakes. Your job is to understand the question and respond appropriately.

Respond in {lang_name}.

Instructions:
- Give short answers, ideally 1 to 3 lines.
- Use very simple language that even a village farmer can understand.
- Focus only on the question — don’t add extra facts unless necessary.
- Don’t repeat the question in your answer.
- Do not mention AI or voice input in the response.

Add this line at the end of your answer:
“This is an AI-generated response. Please confirm with local experts or call Kisan Call Centre (KCC).”
"""

# 🌐 Web search
async def fetch_search_snippets(query: str):
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    body = {"q": query, "gl": "in"}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post("https://google.serper.dev/search", headers=headers, json=body)
            data = response.json()
            return [item.get("snippet", "") for item in data.get("organic", [])[:5]]
    except Exception as e:
        print(f"❌ Serper failed: {e}")
        return []

# 🤖 DeepSeek
async def call_deepseek_with_context(question: str, context_snippets: list[str], lang_name: str):
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(lang_name=lang_name)
    prompt = "\n".join(context_snippets) + f"\n\nFarmer's Question: {question}"
    try:
        completion = client.chat.completions.create(
            model="deepseek/deepseek-r1-0528:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            extra_headers={
                "HTTP-Referer": "https://agrikart.ai",
                "X-Title": "AgriKart VoiceBot"
            }
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"❌ DeepSeek failed: {e}")
        return "Sorry, I'm unable to answer your question at the moment."

# 🔊 gTTS TTS
def generate_tts(text: str, lang: str) -> str | None:
    try:
        tts = gTTS(text=text, lang=lang)
        filename = f"{int(time.time())}_{lang}.mp3"
        audio_dir = "static/audio"
        os.makedirs(audio_dir, exist_ok=True)
        filepath = os.path.join(audio_dir, filename)
        tts.save(filepath)
        return f"{PUBLIC_AUDIO_BASE_URL}/static/audio/{filename}"
    except Exception as exc:
        print(f"TTS error: {exc}")
        return None

# 🌐 Language Map
LANG_MAP = {
    "hi": "Hindi",
    "en": "English",
    "ta": "Tamil",
    "ml": "Malayalam",
    "te": "Telugu",
    "mr": "Marathi",
    "kn": "Kannada",
    "bn": "Bengali",
}

# 🎙️ API endpoint: POST /chat/
@app.post("/chat/")
async def chat(file: UploadFile = File(...), lang: str = Form(...)):
    # Save uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        audio_path = tmp.name

    # Transcription
    forced_ids = asr_processor.get_decoder_prompt_ids(language=lang, task="transcribe")
    asr_result = asr_pipe(audio_path, generate_kwargs={"forced_decoder_ids": forced_ids})
    os.remove(audio_path)
    question = asr_result["text"]
    print(f"🗣 Transcription: {question}")

    # Search context
    context = await fetch_search_snippets(question)
    print(f"🔍 Context: {context}")

    # DeepSeek response
    lang_name = LANG_MAP.get(lang, "Hindi")
    reply = await call_deepseek_with_context(question, context, lang_name)
    print(f"🤖 Reply: {reply}")

    # gTTS audio
    audio_url = generate_tts(reply, lang)

    return JSONResponse({
        "language": lang,
        "transcription": question,
        "response": reply,
        "audio_url": audio_url
    })
