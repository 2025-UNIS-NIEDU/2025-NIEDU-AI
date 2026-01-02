from fastapi import APIRouter, HTTPException

from src.quiz.completion_feedback import KoreanQuizEvaluator, QuizRequest, QuizResponse

router = APIRouter(prefix="/api/quiz", tags=["Quiz"])


@router.post("/feedback", response_model=QuizResponse)
async def get_feedback_quiz(request: QuizRequest) -> QuizResponse:
    evaluator = KoreanQuizEvaluator()
    try:
        return await evaluator.solve_feedback_quiz(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Quiz evaluation failed: {exc}") from exc
