from fastapi import FastAPI
from pydantic import BaseModel
from src.service.ai_service import generate_quiz
from src.config.settings import settings

app = FastAPI(title="NIEdu AI Backend")

class QuizRequest(BaseModel):
    content: str

@app.get("/")
def root():
    return {"message": "NIEdu AI running", "db": settings.DB_HOST}

@app.post("/api/quiz")
def quiz(req: QuizRequest):
    quiz_text = generate_quiz(req.content)
    return {"quiz": quiz_text}
