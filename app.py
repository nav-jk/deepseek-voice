from fastapi import FastAPI, UploadFile, File
import torch
import tempfile
import os
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline
)
from openai import OpenAI
import httpx

# ✅ Load TrueFoundry-injected environment variables
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ✅ FastAPI app
app = FastAPI()

# ✅ DeepSeek via OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ✅ Device and dtype
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# ✅ Whisper model loading
asr_model_id = "openai/whisper-large-v3"
asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    asr_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
).to(device)
asr_processor = AutoProcessor.from_pretrained(asr_model_id)
forced_decoder_ids = asr_processor.get_decoder_prompt_ids(language="en", task="transcribe")

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=asr_model,
    tokenizer=asr_processor.tokenizer,
    feature_extractor=asr_processor.feature_extractor,
    chunk_length_s=30,
    batch_size=16,
    torch_dtype=torch_dtype,
    device=device,
    generate_kwargs={"forced_decoder_ids": forced_decoder_ids}
)
print("✅ Whisper model ready")

# 🔍 Internet search (via Serper.dev)
async def fetch_search_snippets(query):
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    body = {"q": query}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post("https://google.serper.dev/search", headers=headers, json=body)
            data = response.json()
            return [item.get("snippet", "") for item in data.get("organic", [])[:5]]
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return []

# 🧠 DeepSeek LLM call
async def call_deepseek_with_context(user_question, search_snippets):
    context_text = "\n".join(search_snippets)
    system_prompt = (
        "You are an agriculture expert assistant. Use the following information from the internet "
        "to help answer farmer queries in simple language. If unsure, say so."
    )
    prompt = f"{context_text}\n\nFarmer's Question: {user_question}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    try:
        completion = client.chat.completions.create(
            model="deepseek/deepseek-r1-0528:free",
            messages=messages,
            extra_headers={
                "HTTP-Referer": "https://agrikart.ai",
                "X-Title": "AgriKart VoiceBot"
            }
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"❌ DeepSeek failed: {e}")
        return "Sorry, I'm unable to provide an answer at the moment."

# 🎙️ Main /chat endpoint
@app.post("/chat/")
async def chat(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        audio_path = tmp.name

    asr_result = asr_pipe(audio_path)
    os.remove(audio_path)

    question = asr_result["text"]
    print(f"🗣 Transcribed: {question}")

    search_snippets = await fetch_search_snippets(question)
    print(f"🔎 Context Snippets: {search_snippets}")

    response = await call_deepseek_with_context(question, search_snippets)
    print(f"🤖 Response: {response}")

    return {
        "transcription": question,
        "response": response
    }
