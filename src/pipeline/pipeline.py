# === src/pipeline/pipeline.py ===

# --- 코스 생성 관련 ---
from course.news_api import fetch_news
from course.rag_builder import build_rag_data
from course.course_generator import generate_course_for_topic  
from course.course_refiner import refine_course_structure

# --- 퀴즈 모듈 import ---
from quiz.summary_reading import generate_summary_reading_quiz
from quiz.term import generate_term_quiz
from quiz.current_affairs import generate_current_affairs_quiz
from quiz.ox import generate_ox_quiz
from quiz.multi import generate_multi_choice_quiz
from quiz.short import generate_short_quiz
from quiz.completion import generate_completion_quiz
from quiz.completion_feedback import generate_completion_feedback_quiz
from quiz.reflect import generate_reflect_quiz

# --- Wrapper import ---
from wrapper.course_wrapper import build_course_package  

from sentence_transformers import SentenceTransformer
import time


def run_learning_pipeline(topic: str):
    """
    뉴스 → RAG → 코스 생성(LLM 호출 포함) → 세션별 퀴즈 생성 → 코스 패키지 통합
    순서: summary_reading → term → current_affairs → ox → multi → short → completion → completion_feedback → reflection
    """
    print(f"\n=== [{topic}] 파이프라인 시작 ===")

    # 1️. 임베딩 모델 로드
    embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

    # 2️. 뉴스 수집 + RAG + 코스 생성 (LLM 호출 포함)
    course_data = generate_course_for_topic(topic, embedding_model=embedding_model)
    if not course_data:
        print(f"[{topic}] 코스 생성 실패 → 스킵")
        return None

    # 3️. 세션별 퀴즈 생성
    for session in course_data:
        session["quizzes"] = {
            "summary_reading": generate_summary_reading_quiz(session),
            "term": generate_term_quiz(session),
            "current_affairs": generate_current_affairs_quiz(session),
            "ox": generate_ox_quiz(session),
            "multi": generate_multi_choice_quiz(session),
            "short": generate_short_quiz(session),
            "completion": generate_completion_quiz(session),
            "completion_feedback": generate_completion_feedback_quiz(session),
            "reflection": generate_reflect_quiz(session),
        }

    # 4️. 통합 패키지 생성 (wrapper 호출)
    build_course_package(topic, course_data)

    print(f"=== [{topic}] 파이프라인 완료 ===\n")
    return course_data