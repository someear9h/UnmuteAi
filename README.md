# 🎙️ Unmute AI

### 🧐 The Problem

For millions of stroke survivors and people with Aphasia, speaking is a struggle. Their thoughts are clear, but their words come out fragmented, stuttered, or broken. Standard tools like Siri or Google Assistant fail because they transcribe the *error* ("W-w-water... hurt") instead of the *intent*. This leaves users feeling isolated and "muted" by society.

### 💡 The Solution

**Unmute AI** is a smart communication prosthetic. It listens to broken speech, understands the context behind it, and instantly reconstructs it into a fluent, grammatically correct sentence. It then speaks that sentence out loud using a natural, human-sounding voice that carries the right emotion.

### 🚀 Key Features

* **🧠 Neuro-Symbolic Intent Reconstruction:** Turns fragmented input (e.g., *"Kitchen... water... fall"*) into clear requests (*"I fell in the kitchen while getting water, please help!"*).
* **📚 ChromaDB Vector RAG Memory:** Uses **ChromaDB** as an embedded vector database to index and semantically retrieve personal patient memories (emergency contacts, family members, daily habits, medical history, allergies) in real-time.
* **🗣️ Non-Robotic TTS:** Uses **ElevenLabs** (with gTTS fallback) to generate high-quality, human-like audio with selectable Voice Personas.
* **💾 Hybrid Memory Store:** Seamlessly synchronizes SQLite relational data with the Chroma vector space for low-latency similarity retrieval.

---

### 🛠️ Architecture & Tech Stack

1. **Frontend:** React + TypeScript + Vite.
2. **Backend:** FastAPI (Python 3.12+).
3. **Vector Database / RAG:** ChromaDB (`chromadb`) with local ONNX embeddings and persistent storage.
4. **LLM Reasoning:** Google Gemini 2.5 Flash-Lite with RAG-injected contextual prompts.
5. **Speech Synthesis:** ElevenLabs API with gTTS fallback.
6. **Relational Database:** SQLite (`test.db`) for user settings and key-value context.

---

### ⚙️ Quickstart & Testing

#### 1. Configure Environment Variables
Edit `backend/.env` (or copy from `backend/.env.example`):
```bash
cp backend/.env.example backend/.env
```
Inside `backend/.env`:
```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
ELEVENLABS_API_KEY=YOUR_ELEVENLABS_API_KEY_HERE
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=unmute_user_context
DATABASE_URL=sqlite:///./test.db
ALLOWED_ORIGINS=*
```

#### 2. Start the Backend Server
Using the virtual environment inside `backend/`:
```bash
cd backend
./venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
*(Or activate the virtualenv first: `source backend/venv/bin/activate && uvicorn main:app --reload`)*

Once running, visit:
- **Interactive Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check & Chroma Stats:** [http://localhost:8000/](http://localhost:8000/)

#### 3. Test RAG Vector Search & Intent Reconstruction
You can test the RAG endpoints via curl:

**Semantic Vector Search Query:**
```bash
curl -X POST http://localhost:8000/api/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "coffee preference", "top_k": 2}'
```

**Intent Reconstruction with RAG:**
```bash
curl -X POST http://localhost:8000/api/speech/deduce-intent \
  -H "Content-Type: application/json" \
  -d '{"text": "coffee... morning... black"}'
```

**List ChromaDB Documents:**
```bash
curl http://localhost:8000/api/rag/documents
```

---

### 💜 Social Impact

This project falls under **AI for Social Good** and **Assistive Tech**. By moving beyond simple transcription and focusing on *intent*, we are giving agency and dignity back to those who have lost their voice.
