from app.services.embeddings import generate_embeddings
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.chunking import chunk_text
from app.services.text_extraction import extract_text_from_pdf
from pydantic import BaseModel
from app.services.vector_store import store_chunks, search_chunks
from app.services.llm import generate_answer

import os
import uuid
import shutil



router = APIRouter(
    prefix="/documents",
    tags=["Documents"]
)
class SearchRequest(BaseModel):
    query: str
    n_results: int = 3

class AskRequest(BaseModel):
    query: str
    n_results: int = 3

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    # 1. Check file type
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    # 2. Generate unique document ID
    unique_id = str(uuid.uuid4())

    original_filename = file.filename

    # 3. Create file path
    file_path = os.path.join(
        UPLOAD_FOLDER,
        f"{unique_id}_{original_filename}"
    )

    # 4. Save uploaded PDF
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 5. Extract text from PDF
    pages = extract_text_from_pdf(file_path)

   # 6. Create meaningful chunks
    chunks = chunk_text(
    pages,
    document_id=unique_id,
    filename=original_filename
)


# 7. Generate embeddings for all chunks
    texts = [chunk["text"] for chunk in chunks]

    embeddings = generate_embeddings(texts)


# 8. Store chunks and embeddings in ChromaDB
    stored_count = store_chunks(
    chunks,
    embeddings
)


# 9. Return result
    return {
    "message": "Document uploaded, text extracted, chunked and embedded successfully",
    "document_id": unique_id,
    "filename": original_filename,
    "chunks_created": len(chunks),
    "chunks_stored": stored_count
}
@router.post("/search")
async def search_documents(request: SearchRequest):

    # Convert user's question into an embedding
    query_embedding = generate_embeddings(
        [request.query]
    )[0]

    # Search ChromaDB
    results = search_chunks(
        query_embedding,
        request.n_results
    )

    return {
        "query": request.query,
        "results": results
    }

@router.post("/ask")
async def ask_document(request: AskRequest):

    # 1. Convert the user's question into an embedding
    query_embedding = generate_embeddings(
        [request.query]
    )[0]

    # 2. Retrieve relevant chunks from ChromaDB
    results = search_chunks(
        query_embedding,
        request.n_results
    )

    # 3. Extract the retrieved documents
    retrieved_documents = results["documents"][0]

    # 4. Combine the chunks into one context
    context = "\n\n---\n\n".join(
        retrieved_documents
    )

    # 5. Send question + retrieved context to Gemini
    answer = generate_answer(
        request.query,
        context
    )

    # 6. Return answer and retrieved sources
    return {
        "question": request.query,
        "answer": answer,
        "sources": results["metadatas"][0]
    }