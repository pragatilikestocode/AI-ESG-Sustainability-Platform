import chromadb


# Create persistent ChromaDB client
client = chromadb.PersistentClient(
    path="./chroma_db"
)


# Create or get our collection
collection = client.get_or_create_collection(
    name="esg_documents"
)


def store_chunks(chunks, embeddings):
    """
    Store document chunks and their embeddings in ChromaDB.
    """

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:

        ids.append(chunk["chunk_id"])

        documents.append(chunk["text"])

        metadatas.append({
            "document_id": chunk["document_id"],
            "filename": chunk["filename"],
            "page": chunk["page"]
        })

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)

def search_chunks(query_embedding, n_results=3):
    """
    Search ChromaDB for the most relevant chunks.
    """

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results