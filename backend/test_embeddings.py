from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2")


sentences = [
    "The company will reduce greenhouse gas emissions.",
    "The organization plans to lower its carbon output.",
    "Employees receive mandatory workplace safety training."
]


embeddings = model.encode(sentences)


similarity = cosine_similarity(embeddings)


print(similarity)