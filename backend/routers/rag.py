from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from services.rag_service import rag_service

router = APIRouter(
    prefix="/rag",
    tags=["RAG & ChromaDB Management"]
)

# --- Schemas ---
class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="The query or fragmented speech text to search in ChromaDB")
    top_k: int = Field(default=4, ge=1, le=10, description="Number of context documents to retrieve")

class DocumentItem(BaseModel):
    id: str
    document: str
    metadata: Dict[str, Any] = {}
    distance: Optional[float] = None

class RAGQueryResponse(BaseModel):
    query: str
    retrieved_count: int
    matches: List[DocumentItem]
    formatted_prompt_context: str

class AddDocumentRequest(BaseModel):
    text: str = Field(..., description="The context fact or patient memory to store")
    category: str = Field(default="custom", description="Category: medical, family, preference, emergency, etc.")
    metadata: Optional[Dict[str, Any]] = None

class DocumentActionResponse(BaseModel):
    id: Optional[str] = None
    message: str
    status: str = "success"


# --- Endpoints ---

@router.post("/query", response_model=RAGQueryResponse)
async def query_vector_context(request: RAGQueryRequest):
    """
    Perform semantic vector similarity search against ChromaDB.
    Returns the most relevant context facts for the input text.
    """
    try:
        results = rag_service.retrieve_context(request.query, n_results=request.top_k)
        formatted = rag_service.format_context_for_prompt(results)
        return RAGQueryResponse(
            query=request.query,
            retrieved_count=len(results),
            matches=[DocumentItem(**r) for r in results],
            formatted_prompt_context=formatted
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ChromaDB Query Error: {str(e)}")


@router.get("/documents")
async def list_all_documents():
    """List all context memories currently indexed in ChromaDB."""
    try:
        docs = rag_service.list_documents()
        return {
            "total_documents": len(docs),
            "collection": rag_service.collection_name,
            "documents": docs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.post("/documents", response_model=DocumentActionResponse)
async def add_context_document(request: AddDocumentRequest):
    """Add a new patient fact or memory to ChromaDB."""
    try:
        meta = request.metadata or {}
        meta["category"] = request.category
        doc_id = rag_service.add_document(text=request.text, metadata=meta)
        return DocumentActionResponse(
            id=doc_id,
            message="Context document indexed into ChromaDB successfully.",
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add document: {str(e)}")


@router.delete("/documents/{doc_id}", response_model=DocumentActionResponse)
async def delete_context_document(doc_id: str):
    """Delete a context document from ChromaDB by ID."""
    success = rag_service.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found or could not be deleted")
    return DocumentActionResponse(
        id=doc_id,
        message=f"Document {doc_id} deleted from ChromaDB.",
        status="success"
    )


@router.post("/reset", response_model=DocumentActionResponse)
async def reset_rag_database():
    """Reset ChromaDB and re-seed with default Aphasia patient profile facts."""
    try:
        rag_service.reset_collection(seed=True)
        count = rag_service.collection.count()
        return DocumentActionResponse(
            message=f"Chroma collection reset and re-seeded with {count} default facts.",
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset ChromaDB: {str(e)}")

