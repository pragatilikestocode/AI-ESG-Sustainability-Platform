import os

from dotenv import load_dotenv
from google import genai


# Load variables from .env
load_dotenv()


# Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")


# Create Gemini client
client = genai.Client(api_key=api_key)


def generate_answer(question, context):
    """
    Generate an answer using Gemini
    based only on the provided context.
    """

    prompt = f"""
You are an ESG document assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context,
say that the information was not found in the provided document.

Context:
{context}

Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text