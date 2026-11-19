from sqlalchemy.orm import Session
from db.database import SessionLocal, engine, Base
from models.user_context import UserContext
from services.rag_service import rag_service

# Create tables
Base.metadata.create_all(bind=engine)

def seed_data():
    db = SessionLocal()
    try:
        # 1. Clear old SQLite data
        db.query(UserContext).delete()
        
        # 2. Demo User Context Facts
        facts = [
            UserContext(key="spouse_name", value="Sarah"),
            UserContext(key="home_address", value="42 Wallaby Way, Sydney"),
            UserContext(key="dog_name", value="Buster"),
            UserContext(key="emergency_contact", value="Dr. Smith at 555-0199"),
            UserContext(key="coffee_preference", value="medium black coffee with no sugar")
        ]
        
        db.add_all(facts)
        db.commit()
        print("✅ SQLite database seeded with User Context!")

        # 3. Reset and seed Chroma Vector DB
        rag_service.reset_collection(seed=True)
        rag_service.sync_from_sqlite(facts)
        count = rag_service.collection.count()
        print(f"✅ Chroma Vector Database seeded with {count} rich patient memories for RAG!")
        
    finally:
        db.close()

if __name__ == "__main__":
    seed_data()
