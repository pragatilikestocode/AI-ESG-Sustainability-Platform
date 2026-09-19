import chromadb
from sentence_transformers import SentenceTransformer


# 1. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 2. Connect to our persistent ChromaDB
client = chromadb.PersistentClient(
    path="./chroma_db"
)


# 3. Get our collection
collection = client.get_or_create_collection(
    name="esg_documents"
)


# 4. Our test documents
documents = [
    "The company will reduce greenhouse gas emissions.",
    "The organization plans to lower its carbon output.",
    "Employees receive mandatory workplace safety training."
]


# 5. Generate embeddings
embeddings = model.encode(documents).tolist()


# 6. Store documents
collection.upsert(
    ids=["doc1", "doc2", "doc3"],
    documents=documents,
    embeddings=embeddings
)


# 7. User's question
question = "What is the company doing about carbon emissions?"


# 8. Convert question into an embedding
question_embedding = model.encode(question).tolist()


# 9. Search ChromaDB
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=2
)


# 10. Display results
print("\nQUESTION:")
print(question)

print("\nRETRIEVED DOCUMENTS:")

for document in results["documents"][0]:
    print("-", document)