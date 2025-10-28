# src/api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.service.ai_service import generate_quiz
from src.config.settings import settings
from src.api.models.course_wrapper import CourseWrapper

app = FastAPI(title="NIEdu AI Backend")

# === 요청 모델 ===
class QuizRequest(BaseModel):
    content: str

@app.get("/")
def root():
    return {"message": "NIEdu AI running", "db": settings.DB_HOST}

# AI 서비스 전용 prefix 적용
@app.post("/api/ai/quiz")
def quiz(req: QuizRequest):
    quiz_text = generate_quiz(req.content)
    return {"quiz": quiz_text}

# === Course 구조 샘플 (기존과 동일) ===
@app.get("/api/courses/sample", response_model=CourseWrapper)
def sample_course():
    """코스 구조 샘플 응답"""
    return {
        "courses": [
            {
                "courseName": "정치적 책임과 신뢰의 재조명",
                "topic": {"name": "Politics"},
                "subTopic": {"name": "정당"},
                "subTags": [{"name": "정치개혁"}, {"name": "책임"}],
                "sessions": [
                    {
                        "newsRef": {
                            "headline": "검찰청 폐지안 통과 논란",
                            "publisher": "연합뉴스",
                            "publishedAt": "2025-10-28T12:00:00Z",
                            "sourceUrl": "https://example.com",
                            "thumbnailUrl": "https://example.com/thumb.jpg"
                        },
                        "quizzes": [
                            {
                                "level": "N",
                                "steps": [
                                    {"stepOrder": 1, "contentType": "SUMMARY_READING", "contents": []},
                                    {"stepOrder": 5, "contentType": "MULTIPLE_CHOICE", "contents": []}
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }