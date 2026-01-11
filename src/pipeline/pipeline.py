# === src/pipeline/pipeline.py ===

import sys, logging
from pathlib import Path
from dotenv import load_dotenv

# === 경로 설정 ===
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR / "src"))

ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH, override=True)

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

# === 로깅 설정 ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_learning_pipeline():
    logger.info("=== START LEARNING PIPELINE ===")

    # fetch_news()
    # generate_all_courses()
    # refine_course_structure()
    logger.info("Course generation step skipped (already exists)")

    logger.info("=== START QUIZ GENERATION (courseId=1, sessionId=1) ===")

    sessions = select_session()
    logger.info(f"총 세션 수: {len(sessions)}")

    selected_by_topic = {}

    for s in sessions:
        if s.get("courseId") != 1:
            continue
        if s.get("sessionId") != 1:
            continue

        topic = s.get("topic")
        if topic not in selected_by_topic:
            selected_by_topic[topic] = s
            logger.info(
                f"선택됨 → topic={topic}, courseId=1, sessionId=1"
            )

    logger.info(f"총 퀴즈 생성 대상 세션 수: {len(selected_by_topic)}")

    # --------------------------------------------------
    # 3. 퀴즈 실제 생성
    # --------------------------------------------------
    for topic, session in selected_by_topic.items():
        logger.info(f"퀴즈 생성 시작 → [{topic}] {session.get('headline')}")

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

            logger.info(f"퀴즈 생성 완료 → [{topic}]")

        except Exception:
            logger.exception(f"퀴즈 생성 실패 → [{topic}]")

    # --------------------------------------------------
    # 4. 패키징
    # --------------------------------------------------
    logger.info("=== START COURSE PACKAGING ===")
    build_course_packages()
    logger.info("=== PIPELINE PROCESS COMPLETED ===")


if __name__ == "__main__":
    run_learning_pipeline()