# === src/pipeline/pipeline.py ===

# --- 코스 생성 관련 ---
from course.news_api import fetch_news
from course.rag_builder import build_rag_data
from course.course_generator import generate_all_courses
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
from datetime import datetime
import time

def run_learning_pipeline():
    """
    뉴스 → RAG → 코스 생성(LLM 호출 포함) → 세션별 퀴즈 생성 → 코스 패키지 통합 → 코스 정제
    순서: summary_reading → term → current_affairs → ox → multi → short → completion → completion_feedback → reflection
    """
    print("\n=== 전체 학습 파이프라인 시작 ===")

    # 1️ 임베딩 모델 로드
    embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

    # 2️ 전체 토픽 순회
    TOPICS = ["politics", "economy", "society", "world"]
    date_to = datetime.now().strftime("%Y-%m-%d")
    date_from = date_to  # 하루치만 수집

    for topic in TOPICS:
        print(f"\n--- [{topic}] 처리 시작 ---")

        # 2-1. 뉴스 수집
        fetch_news(topic=topic, date_from=date_from, date_to=date_to, max_samples=100)

        # 2-2. RAG DB 구축
        build_rag_data(topic, embedding_model=embedding_model)

        # 2-3. 코스 생성
        course_data = generate_all_courses(topic, embedding_model=embedding_model)
        if not course_data:
            print(f"[{topic}] 코스 생성 실패 → 스킵")
            continue

        # 2-4. 세션별 퀴즈 생성
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

        # 2-5. 코스 패키지 통합
        build_course_package(topic, course_data)

        # 2-6. 코스 정제
        refine_course_structure(topic)

        print(f"=== [{topic}] 파이프라인 완료 ===")

    print("\n=== 전체 파이프라인 완료 ===")