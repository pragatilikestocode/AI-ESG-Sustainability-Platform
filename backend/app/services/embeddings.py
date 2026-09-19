from sentence_transformers import SentenceTransformer


# Load the embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(texts):
    """
    Convert a list of text chunks into embedding vectors.
    """

    embeddings = model.encode(texts)

    return embeddings.tolist()