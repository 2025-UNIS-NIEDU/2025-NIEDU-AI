# === src/quiz/article_reading.py ===
import os, json, logging
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from quiz.select_session import select_session

logger = logging.getLogger(__name__)

def generate_article_reading_quiz(selected_session=None):
    """
    ARTICLE_READING 단계별(I, E) 메타데이터 JSON 생성기
    """
    # === 환경 변수 로드 ===
    BASE_DIR = Path(__file__).resolve().parents[2]
    ENV_PATH = BASE_DIR / ".env"
    load_dotenv(ENV_PATH, override=True)

    # === 세션 선택 ===
    if selected_session is None:
        selected_session = select_session()

    topic = selected_session["topic"]
    course_id = selected_session["courseId"]
    session_id = selected_session.get("sessionId")
    headline = selected_session.get("headline", "")
    publisher = selected_session.get("publisher", "")
    published_at = selected_session.get("publishedAt", "")
    source_url = selected_session.get("sourceUrl", "")
    thumbnail_url = selected_session.get("thumbnailUrl", "")

    # === 메타데이터 구조 ===
    article_metadata = {
        "contentType": "ARTICLE_READING",
        "contents": [
            {
                "thumbnailUrl": thumbnail_url,
                "headline": headline,
                "publisher": publisher,
                "publishedAt": published_at,
                "sourceUrl": source_url
            }
        ]
    }

    # === 저장 경로 설정 ===
    QUIZ_DIR = BASE_DIR / "data" / "quiz"
    QUIZ_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    # === I / E 단계별 저장 ===
    for level in ["I", "E"]:
        save_path = QUIZ_DIR / f"{topic}_{course_id}_{session_id}_ARTICLE_READING_{level}_{today}.json"

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump([article_metadata], f, ensure_ascii=False, indent=2)
            logger.info(f"[{topic}] {level} 단계 ARTICLE_READING 저장 완료 → {save_path.name}")
        except Exception as e:
            logger.error(f"[{topic}] {level} 단계 파일 저장 실패: {e}", exc_info=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    generate_article_reading_quiz()