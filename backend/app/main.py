from app.routers import documents
from fastapi import FastAPI

app = FastAPI(
    title="AI ESG Sustainability Platform",
    description="Document Intelligence Module for ESG Analysis",
    version="1.0.0"
)
app.include_router(documents.router)

@app.get("/")
def home():
    return {
        "message": "AI ESG Document Intelligence API is running!"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }