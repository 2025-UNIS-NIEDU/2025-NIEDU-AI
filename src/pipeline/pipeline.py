# === src/pipeline/pipeline.py ===

import sys, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# --- 코스 생성 관련 ---
from course.news_api import fetch_news
from course.rag_builder import build_rag_data
from course.course_generator import generate_all_courses
from course.course_refiner import refine_course_structure

# --- 퀴즈 생성 관련 ---
from quiz.select_session import select_session
from quiz.article_reading import generate_article_reading_quiz
from quiz.summary_reading import generate_summary_reading_quiz
from quiz.term import generate_term_quiz
from quiz.current_affairs import generate_current_affairs_quiz
from quiz.ox import generate_ox_quiz
from quiz.multi import generate_multi_choice_quiz
from quiz.short import generate_short_quiz
from quiz.completion import generate_completion_quiz
from quiz.completion_feedback import generate_completion_feedback_quiz
from quiz.reflect import generate_reflect_quiz

# --- Wrapper ---
from wrapper.course_wrapper import build_course_packages

from sentence_transformers import SentenceTransformer
from datetime import datetime


def run_learning_pipeline():
    """
    전체 자동 학습 파이프라인
    뉴스 → RAG → 코스 생성 → 정제 → 퀴즈 생성 → 패키징
    (최종 JSON 패키지 반환)
    """
    print("\n=== 전체 학습 파이프라인 시작 ===")

    embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

    fetch_news()
    build_rag_data()
    generate_all_courses()
    refine_course_structure()

    print("\n=== 퀴즈 생성 단계 시작 ===")

    selected_session = select_session()
    print(f"\n[선택된 세션 정보]\n{json.dumps(selected_session, ensure_ascii=False, indent=2)}\n")

    generate_article_reading_quiz(selected_session)
    generate_summary_reading_quiz(selected_session)
    generate_term_quiz(selected_session)
    generate_current_affairs_quiz(selected_session)
    generate_ox_quiz(selected_session)
    generate_multi_choice_quiz(selected_session)
    generate_short_quiz(selected_session)
    generate_completion_quiz(selected_session)
    generate_reflect_quiz(selected_session)

    print("\n=== 퀴즈 생성 완료 → 코스 패키징 시작 ===")

    package_data = build_course_packages()  

    print("\n=== 전체 파이프라인 완료 ===")
    return package_data  

if __name__ == "__main__":
    run_learning_pipeline()