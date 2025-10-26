from fastapi import FastAPI
from pydantic import BaseModel
from src.quiz.CURRENT_AFFAIRS_n import generate_background_card_api
from src.config.settings import settings

app = FastAPI(title="NIEdu AI Backend")

class QuizRequest(BaseModel):
    content: str

@app.get("/")
def root():
    return {"message": "NIEdu AI running", "db": settings.DB_HOST}

@app.post("/api/quiz")
def quiz(req: QuizRequest):
    result = generate_background_card_api()  # AI 로직 호출
    return {"quiz": result}