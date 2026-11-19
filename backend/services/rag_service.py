import os
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from core.config import settings, BASE_DIR

logger = logging.getLogger("unmute.rag")

# Default comprehensive profile facts for an Aphasia patient to seed vector DB
DEFAULT_PATIENT_CONTEXT = [
    {
        "id": "ctx_spouse_sarah",
        "text": "My spouse's name is Sarah. She is my primary caregiver and emergency contact.",
        "metadata": {"category": "family", "key": "spouse_name", "importance": "high"}
    },
    {
        "id": "ctx_daughter_emily",
        "text": "My daughter's name is Emily. She is 10 years old and finishes school at 3:30 PM.",
        "metadata": {"category": "family", "key": "daughter_name", "importance": "medium"}
    },
    {
        "id": "ctx_pet_buster",
        "text": "My dog is a Golden Retriever named Buster. He loves morning walks in Central Park and needs his dinner at 6 PM.",
        "metadata": {"category": "pets", "key": "dog_name", "importance": "medium"}
    },
    {
        "id": "ctx_doctor_smith",
        "text": "My neurologist is Dr. Smith at Metro Neurology. His office phone number is 555-0199.",
        "metadata": {"category": "medical", "key": "doctor_contact", "importance": "critical"}
    },
    {
        "id": "ctx_medical_condition",
        "text": "I survived an ischemic stroke resulting in expressive aphasia. My comprehension is intact, but producing speech is difficult.",
        "metadata": {"category": "medical", "key": "diagnosis", "importance": "critical"}
    },
    {
        "id": "ctx_medication_allergy",
        "text": "I take daily blood thinners (Apixaban) every evening at 8:00 PM. I am severely allergic to Penicillin.",
        "metadata": {"category": "medical", "key": "allergies_meds", "importance": "critical"}
    },
    {
        "id": "ctx_home_address",
        "text": "My home address is 42 Wallaby Way, Sydney. The spare key is kept under the front porch flower pot.",
        "metadata": {"category": "location", "key": "home_address", "importance": "high"}
    },
    {
        "id": "ctx_coffee_preference",
        "text": "My preferred coffee order is a medium black coffee with no sugar. In the evening, I drink decaf chamomile tea.",
        "metadata": {"category": "preference", "key": "coffee_preference", "importance": "medium"}
    },
    {
        "id": "ctx_emergency_fall",
        "text": "In case of a fall, sudden chest pain, or severe dizziness, please call 911 immediately and notify Sarah.",
        "metadata": {"category": "emergency", "key": "fall_protocol", "importance": "critical"}
    },
    {
        "id": "ctx_groceries",
        "text": "For groceries, I regularly buy oat milk, whole wheat bread, fresh bananas, and low-sodium vegetable soup.",
        "metadata": {"category": "preference", "key": "grocery_items", "importance": "low"}
    },
    {
        "id": "ctx_assistance_mobility",
        "text": "I sometimes need physical assistance walking up stairs or standing up quickly due to occasional balance instability.",
        "metadata": {"category": "mobility", "key": "mobility_assistance", "importance": "high"}
    }
]


class RAGService:
    def __init__(self):
        # Resolve persistence directory relative to backend base dir if not absolute
        persist_dir = Path(settings.CHROMA_PERSIST_DIRECTORY)
        if not persist_dir.is_absolute():
            persist_dir = BASE_DIR / persist_dir
        
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir = str(persist_dir)
        self.collection_name = settings.CHROMA_COLLECTION_NAME

        logger.info(f"Initializing ChromaDB PersistentClient at: {self.persist_dir}")
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Unmute AI Aphasia Patient Context Vector Store"}
        )

        # Seed initial knowledge if empty
        if self.collection.count() == 0:
            self.seed_defaults()

    def seed_defaults(self):
        """Seed default patient facts into the Chroma collection."""
        logger.info("Seeding Chroma vector database with default patient context...")
        ids = [item["id"] for item in DEFAULT_PATIENT_CONTEXT]
        documents = [item["text"] for item in DEFAULT_PATIENT_CONTEXT]
        metadatas = [item["metadata"] for item in DEFAULT_PATIENT_CONTEXT]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"✅ Chroma vector DB seeded with {len(ids)} context documents.")

    def retrieve_context(self, query: str, n_results: int = 4) -> List[Dict[str, Any]]:
        """
        Query Chroma vector DB using semantic similarity search.
        Returns top matching user context items with documents and metadata.
        """
        total_docs = self.collection.count()
        if total_docs == 0:
            return []

        # Chroma throws if n_results > count
        k = min(n_results, total_docs)
        if not query or not query.strip():
            # If query is empty, return top k general items
            all_items = self.collection.get(limit=k)
            results = []
            for doc_id, doc, meta in zip(all_items["ids"], all_items["documents"], all_items["metadatas"]):
                results.append({
                    "id": doc_id,
                    "document": doc,
                    "metadata": meta or {},
                    "distance": None
                })
            return results

        query_res = self.collection.query(
            query_texts=[query],
            n_results=k
        )

        documents = query_res.get("documents", [[]])[0]
        metadatas = query_res.get("metadatas", [[]])[0]
        ids = query_res.get("ids", [[]])[0]
        distances = query_res.get("distances", [[]])[0] if "distances" in query_res else [None] * len(documents)

        items = []
        for doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            items.append({
                "id": doc_id,
                "document": doc,
                "metadata": meta or {},
                "distance": dist
            })

        return items

    def format_context_for_prompt(self, context_items: List[Dict[str, Any]]) -> str:
        """Format retrieved vector chunks into an enriched prompt section."""
        if not context_items:
            return "No specific personal memory found."

        lines = []
        for item in context_items:
            meta = item.get("metadata", {})
            category = meta.get("category", "General").capitalize()
            lines.append(f"- [{category}] {item['document']}")
        return "\n".join(lines)

    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None, doc_id: Optional[str] = None) -> str:
        """Add or update a context document in ChromaDB."""
        if not doc_id:
            doc_id = f"custom_{uuid.uuid4().hex[:8]}"
        
        self.collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata or {"source": "manual", "category": "custom"}]
        )
        return doc_id

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all indexed documents in the collection."""
        res = self.collection.get()
        items = []
        for doc_id, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
            items.append({
                "id": doc_id,
                "document": doc,
                "metadata": meta or {}
            })
        return items

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by id."""
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False

    def reset_collection(self, seed: bool = True):
        """Reset the Chroma collection and optionally re-seed."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Unmute AI Aphasia Patient Context Vector Store"}
        )
        if seed:
            self.seed_defaults()

    def sync_from_sqlite(self, user_contexts: list):
        """Sync SQL UserContext records into ChromaDB vector store."""
        if not user_contexts:
            return
        
        ids = []
        docs = []
        metas = []
        for item in user_contexts:
            formatted_key = item.key.replace("_", " ").title()
            doc_text = f"My {formatted_key} is {item.value}."
            ids.append(f"sqlite_{item.key}")
            docs.append(doc_text)
            metas.append({
                "source": "sqlite",
                "key": item.key,
                "category": "profile"
            })
        
        self.collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas
        )
        logger.info(f"Synced {len(ids)} items from SQLite to Chroma vector DB.")


# Global singleton instance
rag_service = RAGService()

