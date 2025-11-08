# === src/pipeline/pipeline.py ===

import sys, json, logging, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv

# === 환경 변수 로드 ===
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH, override=True)

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

# === 로깅 설정 ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def run_learning_pipeline():
    """
    전체 자동 학습 파이프라인
    뉴스 → RAG → 코스 생성 → 정제 → 퀴즈 생성 → 패키징
    (최종 JSON 패키지 반환)
    """
    logger.info("=== 전체 학습 파이프라인 시작 ===")

    embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

    # 코스 생성 파트
    logger.info("뉴스 수집 및 RAG 데이터 빌드 시작")
    fetch_news()
    build_rag_data()
    generate_all_courses()
    refine_course_structure()
    logger.info("코스 생성 및 정제 완료")

    # 퀴즈 생성
    logger.info("=== 퀴즈 생성 단계 시작 ===")
    all_sessions = select_session()
    logger.info(f"총 {len(all_sessions)}개 세션을 처리합니다.")

    for session in all_sessions:
        course_name = session.get("courseName", "")
        headline = session.get("headline", "")
        logger.info(f"세션 처리 중 → {course_name} / {headline}")

        try:
            generate_article_reading_quiz(session)
            generate_summary_reading_quiz(session)
            generate_term_quiz(session)
            generate_current_affairs_quiz(session)
            generate_ox_quiz(session)
            generate_multi_choice_quiz(session)
            generate_short_quiz(session)
            generate_completion_quiz(session)
            generate_reflect_quiz(session)
        except Exception as e:
            logger.error(f"세션 처리 중 오류 발생 ({course_name}): {e}", exc_info=True)
            continue

    # 코스 + 퀴즈 통합 패키징
    logger.info("=== 퀴즈 생성 완료 → 코스 패키징 시작 ===")
    try:
        package_data = build_course_packages()
        logger.info("코스 패키징 완료")
    except Exception as e:
        logger.error(f"코스 패키징 중 오류 발생: {e}", exc_info=True)
        package_data = None

    logger.info("=== 전체 파이프라인 완료 ===")
    return package_data


if __name__ == "__main__":
    run_learning_pipeline()