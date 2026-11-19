# 🎓 Unmute AI — 10-Minute Viva & Oral Exam Master Guide

> **Target Reading Time:** ~10 Minutes  
> **Purpose:** Comprehensive, high-yield technical and conceptual preparation for viva, project defense, and oral evaluation.

---

## 📌 Table of Contents
1. [Executive Summary & The Core Problem](#1-executive-summary--the-core-problem)
2. [The Unmute AI Solution](#2-the-unmute-ai-solution)
3. [System Architecture & End-to-End Pipeline](#3-system-architecture--end-to-end-pipeline)
4. [Deep Dive: RAG & ChromaDB Vector Memory](#4-deep-dive-rag--chromadb-vector-memory)
5. [Technical Stack Justification (Why X over Y?)](#5-technical-stack-justification-why-x-over-y)
6. [Key Engineering Innovations](#6-key-engineering-innovations)
7. [Anticipated Viva Questions & Model Answers](#7-anticipated-viva-questions--model-answers)
8. [Future Enhancements & Scalability](#8-future-enhancements--scalability)

---

## 1. Executive Summary & The Core Problem

### The Clinical Context
Millions of individuals worldwide live with **Expressive Aphasia** (commonly known as *Broca's Aphasia*), typically triggered by an ischemic stroke, traumatic brain injury (TBI), or neurodegenerative conditions. 

- **Key Clinical Insight:** In expressive aphasia, the patient's **cognitive reasoning and comprehension remain largely intact**. They know *exactly* what they want to say, but the motor-speech production pathway in the brain is disrupted. 
- Words come out **fragmented, telegraphic, phonetically broken, or stuttered** (e.g., *"Kitchen... water... fall"* or *"Coffee... morning... black"*).

### Why Traditional Tech Fails
Standard voice assistants (Apple Siri, Google Assistant, Amazon Alexa) and generic Speech-to-Text (STT) models:
1. **Transcribe the acoustic error**, not the cognitive intent (e.g., typing *"W-w-water... hurt"* and returning *"I didn't understand that"*).
2. **Lack personal episodic memory**: They have no knowledge of the speaker's family, medical history, home environment, or daily habits.
3. **Sound robotic or childish**: Off-the-shelf assistive AAC (Augmentative and Alternative Communication) devices often use robotic voices that strip adult patients of their dignity and personality.

---

## 2. The Unmute AI Solution

**Unmute AI** is a **Neuro-Symbolic Communication Prosthetic**. It acts as an intelligent intermediary between broken verbal expression and fluent social communication:

1. **Acoustic Ingestion:** Captures broken, stuttered vocal input directly from the user.
2. **Intent Deduction via RAG:** Queries a local **ChromaDB vector database** storing the patient's personal knowledge graph (caregiver contacts, medical conditions, medication schedules, drink preferences, home address).
3. **LLM Reconstruction:** Utilizes **Google Gemini 2.5 Flash-Lite** with structured prompt engineering to reconstruct the fragment into 3 distinct, contextually accurate, fluent sentences.
4. **Voice Restoration:** Converts the user's selected choice into emotional, human-grade audio using **ElevenLabs TTS** with customizable Voice Personas, restoring their adult agency.

---

## 3. System Architecture & End-to-End Pipeline

```
 [User Speaks Broken Fragment]
             │
             ▼
┌─────────────────────────┐
│   React + Vite Frontend │ ─── Audio captured as WebM (Opus codec)
└────────────┬────────────┘
             │ HTTP POST (Base64 / Multipart)
             ▼
┌─────────────────────────┐
│     FastAPI Backend     │
├─────────────────────────┤
│ 1. Transcription        │ ─── Gemini 2.5 Flash-Lite (preserves spoken tokens)
│                         │
│ 2. Vector Retrieval     │ ─── ChromaDB: Semantic search (all-MiniLM-L6-v2)
│    (RAG Engine)         │     Retrieves top-k relevant patient memories
│                         │
│ 3. Intent Reasoning     │ ─── Gemini 2.5: Injects RAG context into prompt;
│                         │     Returns 3 structured options (JSON schema)
│                         │
│ 4. Voice Synthesis      │ ─── ElevenLabs API (with fallback to gTTS)
│                         │     Generates Base64 MP3 stream
└────────────┬────────────┘
             │ JSON Response + Base64 Audio
             ▼
┌─────────────────────────┐
│   React UI Playback     │ ─── Native Web Audio API decodes & speaks aloud
└─────────────────────────┘
```

### Detailed 6-Step Pipeline Walkthrough:
1. **Audio Acquisition:** React's `MediaRecorder` API captures browser microphone input encoded as `audio/webm;codecs=opus`.
2. **Transcription:** Sent to `/api/speech/transcribe`. Gemini transcribes raw utterances without prematurely autocorrecting stuttered phonemes.
3. **Vector Context Retrieval:** The fragmented text is passed to ChromaDB via `/api/speech/deduce-intent`. Chroma computes cosine similarity embeddings and retrieves the top 3–4 most relevant patient facts.
4. **Context Injection & Reasoning:** The system prompt injects retrieved facts:
   - *Example Query:* `"chest... doctor... hurry"`
   - *Retrieved Memory:* `[MEDICAL] Neurologist Dr. Smith at Metro Neurology: 555-0199` & `[EMERGENCY] Call Sarah and 911 immediately if chest pain occurs.`
5. **Structured Candidate Generation:** Gemini produces strictly structured JSON containing 3 conversational variations (e.g., Direct Statement, Polite Inquiry, Urgent Action).
6. **Selection & Vocalization:** The patient taps their desired card; `/api/speech/speak` generates emotional, natural speech delivered through the browser.

---

## 4. Deep Dive: RAG & ChromaDB Vector Memory

### What is RAG and Why is it Essential Here?
**Retrieval-Augmented Generation (RAG)** dynamically enriches an LLM's prompt with external, query-relevant knowledge retrieved from a specialized database before generating an answer.

In Unmute AI, RAG solves three fundamental limitations of pure LLMs:
1. **Eliminating Hallucinations:** Prevents the LLM from inventing random names, addresses, or medical conditions.
2. **Patient Personalization:** The LLM does not know that the user's wife is "Sarah" or that their dog is "Buster" unless this context is retrieved at query time.
3. **Token Efficiency & Latency:** Instead of dumping an entire 50-page patient medical chart and biography into every prompt (which inflates cost and slows inference), RAG retrieves *only* the 3 or 4 relevant context chunks needed for that specific sentence.

### Why ChromaDB?
- **Embedded & Serverless:** ChromaDB runs directly within the FastAPI Python process via `chromadb.PersistentClient`. It requires zero external Docker daemon or hosted cloud cluster for basic operation.
- **On-Disk Persistence:** Data persists locally in `backend/chroma_db/`, surviving server restarts.
- **Local Embedding Execution:** Uses ONNX Runtime with `all-MiniLM-L6-v2` locally to convert text queries into 384-dimensional dense vectors with sub-10ms latency and zero API cost.
- **Hybrid Memory Synchronization:** SQLite stores relational user settings, while Chroma stores semantic unstructured patient memories. When the system boots, SQL records are automatically synchronized and indexed in Chroma.

---

## 5. Technical Stack Justification (Why X over Y?)

| Component | Selected Technology | Why Selected Over Alternatives |
| :--- | :--- | :--- |
| **Backend Framework** | **FastAPI (Python 3.12)** | Asynchronous non-blocking I/O (`async`/`await`), native Pydantic typing, auto-generated OpenAPI/Swagger interactive documentation, significantly lower latency than Flask or Django. |
| **Vector Database** | **ChromaDB** | In-process native execution, local disk persistence, zero external cloud dependencies (crucial for patient medical privacy), easier integration than Pinecone, FAISS, or Milvus. |
| **Core LLM** | **Google Gemini 2.5 Flash-Lite** | Ultra-low latency (<500ms time-to-first-token), native multimodal audio-to-text processing, strict JSON schema output enforcement, cost-efficient compared to GPT-4o. |
| **Voice Synthesis** | **ElevenLabs API** | High emotional cadence, natural pause and breath modeling, selectable personas (The Professional, The Anchor) restoring adult dignity. |
| **Audio Fallback** | **gTTS (Google TTS)** | Offline/fail-safe resilience ensuring the patient is never left unable to communicate if API credits or internet drop out. |
| **Frontend** | **React 19 + TypeScript + Vite** | Instant HMR, strict type safety for audio buffers, accessible single-tap UI designed for motor-impaired stroke survivors. |

---

## 6. Key Engineering Innovations

1. **Neuro-Symbolic Bridging:** Combines symbolic structured facts (names, phone numbers, addresses from SQLite/Chroma) with neural generative modeling (Gemini 2.5) to produce factually grounded sentences.
2. **Graceful Degradation (Fail-Safe Architecture):**
   - If external Gemini API keys are absent or network is degraded, the backend uses a rule-guided RAG fallback synthesis engine to deduce intent directly from ChromaDB matches.
   - If ElevenLabs fails or runs out of credits, speech synthesis automatically falls back to `gTTS`.
3. **Latency Masking UI:** Interactive audio-wave animations and optimistic state transitions reassure the patient that their input was heard and is being processed.
4. **Emergency SOS Bypass:** A persistent, high-contrast SOS button allows one-tap broadcasting of emergency medical information without needing to speak or navigate menus.

---

## 7. Anticipated Viva Questions & Model Answers

### Q1: What is Expressive Aphasia, and why can't existing voice assistants help?
> **Model Answer:** Expressive Aphasia is a neurological condition where speech production is broken or stuttered, even though cognitive comprehension is intact. Mainstream assistants like Siri fail because their acoustic models are trained on fluent speech; they interpret phonetic stutters as errors and reject the input. Unmute AI treats broken speech as semantic clues, using RAG to reconstruct the underlying intent.

### Q2: How does the RAG search work in your ChromaDB implementation?
> **Model Answer:** When the user utters a fragment (e.g., *"water... fall... hurt"*), ChromaDB converts this text into a 384-dimensional dense vector using an ONNX-optimized embedding model. It calculates the cosine distance between this query vector and our stored patient context vectors. It retrieves the closest matches—such as the patient's emergency protocol and doctor's contact—and formats them into the prompt for the LLM.

### Q3: Why did you choose ChromaDB over cloud vector stores like Pinecone?
> **Model Answer:** Three reasons:
> 1. **Privacy & HIPAA compliance:** Health and personal caregiver details remain local on the patient's device rather than being broadcast to third-party vector clouds.
> 2. **Latency:** Chroma runs in-process with zero network overhead for retrieval.
> 3. **Cost & Reliability:** Chroma requires no paid subscription or always-on internet connectivity for embedding generation.

### Q4: How do you prevent the LLM from hallucinating wrong personal details?
> **Model Answer:** We use two safeguards: First, **Grounding via RAG**: We explicitly instruct the LLM in its system prompt to restrict missing biographical details (names, contacts, addresses) to the retrieved context chunks. Second, **Structured Output Schema**: We enforce strict JSON generation with `response_schema`, preventing conversational deviations and hallucinated conversational filler.

### Q5: What is the purpose of SQLite if you already have ChromaDB?
> **Model Answer:** We use a **hybrid architecture**. SQLite handles relational, strictly typed configuration data (key-value user settings, persistent IDs). ChromaDB handles high-dimensional semantic similarity search. On startup, SQLite records are automatically ingested into ChromaDB, giving us the benefits of both structured relational integrity and semantic vector search.

### Q6: What happens if the patient speaks a phrase that matches no context in ChromaDB?
> **Model Answer:** Vector search will still return the closest semantic vectors with higher distance values, but our prompt instructions specify that context should only be used *if relevant*. If the input is completely general (e.g., *"Sun... bright... outside"*), Gemini relies on general linguistic reasoning to generate sentences like *"It is very sunny outside today"* without forcing irrelevant patient facts.

### Q7: How do you evaluate the quality of reconstructed sentences?
> **Model Answer:** In production and user studies, assistive AAC systems are evaluated using three metrics:
> 1. **Intent Accuracy:** Does the reconstructed sentence reflect the user's intended meaning?
> 2. **Keystroke/Effort Reduction:** How many words or taps did the user save compared to typing on an alphabet board?
> 3. **Latency:** Time from speech end to audio playback (targeting under 1.5 seconds for natural dialogue).

---

## 8. Future Enhancements & Scalability

If asked *"What would you build next?"*, highlight these planned milestones:

1. **On-Device Edge Inference:** Deploying a quantized SLM (Small Language Model like Gemma 2B or Llama-3-8B) with local Whisper STT directly on a Raspberry Pi or mobile phone for complete offline capability.
2. **Continuous Active Learning:** Allowing the patient or caregiver to tap a *"Save Fact"* button after a conversation, automatically appending new episodic memories to ChromaDB in real time.
3. **Wearable IoT Trigger (Fall Detection):** Pairing the system with smartwatch accelerometer data so that upon a detected fall, Unmute AI automatically primes the emergency medical context in ChromaDB.
4. **Multilingual Aphasia Support:** Supporting bilingual stroke survivors who often code-switch between their native tongue and English during recovery.

---

### 💡 Quick Viva Tip Sheet
- **Keywords to mention:** *Neuro-Symbolic AI, Expressive Aphasia, Assistive Tech, Vector Embeddings, Cosine Distance, Retrieval-Augmented Generation (RAG), Latency Masking, Dense Retrieval.*
- **If asked to demo:** Use the fragment `"coffee... morning... black"` or `"kitchen... water... fall"`, show Swagger UI at `/docs`, execute `/api/rag/query`, and demonstrate how ChromaDB context is passed into `/api/speech/deduce-intent`.
