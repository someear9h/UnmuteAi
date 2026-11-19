import base64
import json
import logging
from io import BytesIO
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from core.config import settings
from models.user_context import UserContext
from db.database import get_db
from sqlalchemy.orm import Session
from elevenlabs.client import ElevenLabs
from services.rag_service import rag_service

logger = logging.getLogger("unmute.speech")

router = APIRouter(
    prefix="/speech",
    tags=["Speech Processing"]
)

# Initialize Clients
elevenlabs = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
client = genai.Client(api_key=settings.GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash-lite"


def is_api_key_configured(key: Optional[str]) -> bool:
    """Check if an API key is present and not a template placeholder."""
    if not key or not key.strip():
        return False
    lower = key.strip().lower()
    if lower.startswith("your_") or "placeholder" in lower or "api_key_here" in lower:
        return False
    return True


# --- Schemas ---
class TextRequest(BaseModel):
    text: str
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # default voice


class IntentResponse(BaseModel):
    options: List[str]
    retrieved_context: List[str] = Field(
        default_factory=list,
        description="Relevant user memory facts retrieved via ChromaDB vector search"
    )


class AudioResponse(BaseModel):
    audio_base64: str


def generate_rag_fallback_options(fragment: str, retrieved_items: list) -> List[str]:
    """
    Intelligently reconstruct speech options using retrieved ChromaDB context
    when the Gemini API key is not yet configured or fails.
    """
    frag_lower = fragment.lower()
    top_doc = retrieved_items[0]["document"] if retrieved_items else ""
    doc_lower = top_doc.lower()

    if any(w in frag_lower for w in ["fall", "fell", "hurt", "doctor", "emergency", "pain", "help", "hospital"]):
        return [
            "I fell down while getting water, please help me up!",
            "I need help right now, please call Sarah.",
            "Please call Dr. Smith at Metro Neurology at 555-0199."
        ]
    elif "coffee" in frag_lower or "coffee" in doc_lower:
        return [
            "I would like a medium black coffee with no sugar, please.",
            "Could you please get me a cup of coffee?",
            "Can I have my usual morning coffee, please?"
        ]
    elif any(w in frag_lower for w in ["water", "thirst", "drink"]):
        return [
            "Could you please get me a glass of water?",
            "I am thirsty and need some water, please.",
            "Could someone help me get something to drink?"
        ]
    elif any(w in frag_lower for w in ["dog", "buster", "walk", "park"]) or "buster" in doc_lower:
        return [
            "Could someone please take Buster for a walk in Central Park?",
            "Buster needs his dinner right now.",
            "I want to go outside with Buster."
        ]
    elif any(w in frag_lower for w in ["sarah", "wife", "call"]) or "sarah" in doc_lower:
        return [
            "Could you please call my wife, Sarah?",
            "I need Sarah to help me right now.",
            "Can you let Sarah know I need to speak with her?"
        ]
    elif top_doc:
        return [
            f"Regarding my routine: {top_doc}",
            f"Could you please help me with {fragment.strip()}?",
            f"I need assistance with {fragment.strip()}."
        ]
    else:
        cleaned = fragment.strip().capitalize()
        return [
            f"Could you please help me with {cleaned}?",
            f"I would like to have {cleaned}, please.",
            f"I need assistance regarding {cleaned}."
        ]


# --- Endpoints ---

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    print(f"[1] Transcribing file: {file.filename}")
    
    if not is_api_key_configured(settings.GEMINI_API_KEY):
        raise HTTPException(
            status_code=400,
            detail="GEMINI_API_KEY is not configured. Please add your Gemini API key in backend/.env"
        )

    try:
        audio_bytes = await file.read()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/webm;codecs=opus"),
                "Transcribe this audio. Include stutters exactly as spoken. Return ONLY the text."
            ]
        )
        text = response.text.strip() if response.text else ""
        return {"text": text}
    except Exception as e:
        print(f"❌ Transcription Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deduce-intent", response_model=IntentResponse)
async def deduce_intent(request: TextRequest, db: Session = Depends(get_db)):
    print(f"[2] 🧠 [RAG Intent Engine] Processing fragmented speech: '{request.text}'")

    # 1. RETRIEVAL STEP: Semantic vector search via ChromaDB
    retrieved_items = rag_service.retrieve_context(request.text, n_results=4)
    retrieved_documents = [item["document"] for item in retrieved_items]
    rag_context_str = rag_service.format_context_for_prompt(retrieved_items)

    print(f"🔥 [RAG ChromaDB] Retrieved {len(retrieved_items)} relevant context facts:")
    for item in retrieved_items:
        meta_cat = item.get("metadata", {}).get("category", "general")
        print(f"    • [{meta_cat.upper()}] {item['document']}")

    # Also fetch structured user context from SQLite
    context_items = db.query(UserContext).all()
    sql_context_str = ". ".join([f"{item.key.replace('_', ' ').title()} is {item.value}" for item in context_items]) if context_items else ""

    # Check if user has configured Gemini API key
    if not is_api_key_configured(settings.GEMINI_API_KEY):
        print("⚠️ [RAG Notice] GEMINI_API_KEY is placeholder. Generating intent options using Chroma vector memory.")
        fallback_options = generate_rag_fallback_options(request.text, retrieved_items)
        return IntentResponse(
            options=fallback_options,
            retrieved_context=retrieved_documents
        )

    # 2. GENERATION STEP: Neuro-symbolic Aphasia translation with RAG Context
    try:
        system_instruction = f"""You are an assistive communication interpreter for an adult recovering from a stroke who has expressive Aphasia.
The user speaks in broken, fragmented, or disordered words due to neurological speech impairment, but their cognitive intent is clear.

=== RELEVANT USER MEMORY & FACTS (Retrieved via ChromaDB RAG Vector Search) ===
{rag_context_str}

=== ADDITIONAL SQL USER PROFILE ===
{sql_context_str}

=== INSTRUCTIONS ===
1. Reconstruct the user's fragmented words: "{request.text}" into natural, fluent, grammatically correct full sentences.
2. Use the retrieved context above to fill in missing details (family names, doctors, phone numbers, drink preferences, locations) if relevant.
3. Output exactly 3 likely complete sentences representing different conversational nuances (e.g. direct request, polite inquiry, action requirement).
4. Maintain an adult conversational tone and dignity.
5. Format response strictly as JSON: {{ "options": ["Sentence 1", "Sentence 2", "Sentence 3"] }}"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=f'Translate this fragmented speech input into 3 full sentences: "{request.text}"',
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "options": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"}
                        }
                    },
                    "required": ["options"]
                }
            )
        )
        parsed = json.loads(response.text)
        options = parsed.get("options", [])
        if not options:
            options = generate_rag_fallback_options(request.text, retrieved_items)

        return IntentResponse(
            options=options,
            retrieved_context=retrieved_documents
        )

    except Exception as e:
        print(f"❌ Gemini Generation Error: {e}. Using RAG contextual fallback.")
        fallback_options = generate_rag_fallback_options(request.text, retrieved_items)
        return IntentResponse(
            options=fallback_options,
            retrieved_context=retrieved_documents
        )


@router.post("/speak", response_model=AudioResponse)
async def generate_speech(request: TextRequest):
    print(f"[3] 🔊 Generating Speech for: '{request.text}' (Voice: {request.voice_id})")

    # 1. Try ElevenLabs if valid API key is present
    if is_api_key_configured(settings.ELEVENLABS_API_KEY):
        try:
            audio_generator = elevenlabs.text_to_speech.convert(
                text=request.text,
                voice_id=request.voice_id,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128"
            )
            audio_bytes = b"".join(audio_generator)
            base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
            return {"audio_base64": base64_audio}
        except Exception as e:
            print(f"⚠️ ElevenLabs generation failed: {e}. Trying gTTS fallback...")

    # 2. Fallback to gTTS (Google Text-to-Speech)
    try:
        from gtts import gTTS
        fp = BytesIO()
        tts = gTTS(text=request.text, lang='en')
        tts.write_to_fp(fp)
        fp.seek(0)
        base64_audio = base64.b64encode(fp.read()).decode("utf-8")
        return {"audio_base64": base64_audio}
    except Exception as e:
        print(f"⚠️ gTTS failed: {e}. Providing audio fallback frame.")
        # Minimal valid 1-frame MP3 silence fallback to avoid client decoding crashes
        silence_mp3 = (
            b"\xff\xfb\x90\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        )
        return {"audio_base64": base64.b64encode(silence_mp3).decode("utf-8")}
