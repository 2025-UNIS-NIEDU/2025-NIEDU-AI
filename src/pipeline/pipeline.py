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

    # 코스 생성 파트
    fetch_news()
    build_rag_data()
    generate_all_courses()
    refine_course_structure()

    print("\n=== 퀴즈 생성 단계 시작 ===")

    # 모든 세션 자동 불러오기
    all_sessions = select_session()
    print(f"총 {len(all_sessions)}개 세션을 처리합니다.")

    # 모든 세션에 대해 퀴즈 생성
    for session in all_sessions:
        print(f"\n 세션 처리 중: {session['courseName']} / {session.get('headline', '')}")

        generate_article_reading_quiz(session)
        generate_summary_reading_quiz(session)
        generate_term_quiz(session)
        generate_current_affairs_quiz(session)
        generate_ox_quiz(session)
        generate_multi_choice_quiz(session)
        generate_short_quiz(session)
        generate_completion_quiz(session)
        generate_reflect_quiz(session)

    # 코스 + 퀴즈 통합 패키징
    print("\n=== 퀴즈 생성 완료 → 코스 패키징 시작 ===")
    package_data = build_course_packages()  

    print("\n=== 전체 파이프라인 완료 ===")
    return package_data  

if __name__ == "__main__":
    run_learning_pipeline()