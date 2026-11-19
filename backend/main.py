from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models.user_context import UserContext
from core.config import settings
from db.database import SessionLocal, create_tables
from services.rag_service import rag_service

# Import speech and rag routers
from routers import speech, rag

# Initialize SQL tables
create_tables()

app = FastAPI(
    title="Unmute AI API",
    description="Neuro-Symbolic Aphasia Translation Engine with ChromaDB RAG Vector Memory",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Handle CORS using settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    db = SessionLocal()
    try:
        # Seed default SQL user context facts
        db.query(UserContext).delete()
        sql_facts = [
            UserContext(key="spouse_name", value="Sarah"),
            UserContext(key="emergency_contact", value="Dr. Smith at 555-0199"),
            UserContext(key="dog_name", value="Buster"),
            UserContext(key="coffee_preference", value="medium black coffee with no sugar"),
            UserContext(key="home_address", value="42 Wallaby Way, Sydney"),
        ]
        db.add_all(sql_facts)
        db.commit()

        # Synchronize SQL facts with ChromaDB vector store
        rag_service.sync_from_sqlite(sql_facts)
        
        doc_count = rag_service.collection.count()
        print(f"✅ User context seeded in SQLite & synchronized with ChromaDB Vector Store ({doc_count} memories indexed)")
    finally:
        db.close()

# Root health check endpoint
@app.get("/")
def root():
    return {
        "app": "Unmute AI API",
        "status": "online",
        "rag_vector_db": "ChromaDB",
        "collection": settings.CHROMA_COLLECTION_NAME,
        "indexed_memories": rag_service.collection.count(),
        "docs_url": "/docs"
    }

# Include routers
app.include_router(speech.router, prefix=settings.API_PREFIX)
app.include_router(rag.router, prefix=settings.API_PREFIX)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
