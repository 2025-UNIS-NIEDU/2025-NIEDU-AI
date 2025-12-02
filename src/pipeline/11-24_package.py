# === src/pipeline/refine_and_package.py ===

import sys, logging
from pathlib import Path

# 상위 디렉토리 import
sys.path.append(str(Path(__file__).resolve().parents[1]))

from quiz.summary_reading import generate_summary_reading_quiz
from quiz.multi import generate_multi_choice_quiz
from quiz.short import generate_short_quiz
from wrapper.course_wrapper import build_course_packages

# === 1) 날짜 강제 오버라이드 (refiner + wrapper 모두) ===
from datetime import datetime
import course.course_refiner as cr
import wrapper.course_wrapper as cw
import quiz.summary_reading as sr
import quiz.multi as ab
import quiz.short as cd

FORCED_DATE = "2025-11-24"

class FakeDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime.fromisoformat(FORCED_DATE)

# refine에서 쓰는 datetime 패치
cr.datetime = FakeDateTime

# package에서 쓰는 datetime 패치
cw.datetime = FakeDateTime

# 3개 quiz 에서 쓰는 datetime 패치
sr.datetime = FakeDateTime
ab.datetime = FakeDateTime
cd.datetime = FakeDateTime

# === 로깅 설정 ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d",
)
logger = logging.getLogger(__name__)

def refine_and_package_only():
    logger.info("=== world 토픽의 모든 코스의 session_1 SUMMARY_READING 재생성 + 전체 패키징 시작 ===")
    logger.info(f"강제 날짜 = {FORCED_DATE}")

    target_topic = "world"
    target_course_id = 1
    target_session_id = 1

    try:
        from quiz.select_session import select_session
        all_sessions = select_session(today=FORCED_DATE)

        # [1] world 토픽의 session_1 전부 찾기
        target_sessions = []
        for s in all_sessions:
            if s.get("topic") == target_topic and s.get("courseId") == target_course_id and s.get("sessionId") == target_session_id:
                target_sessions.append(s)

        if not target_sessions:
            raise ValueError("world 토픽에 해당하는 session_1 세션을 찾을 수 없습니다.")

        logger.info(f"[1] 총 {len(target_sessions)}개의 world/*/session_1 세션 발견")


        # [2] 모든 세션에 대해 3가지 퀴즈 재생성
        for idx, session in enumerate(target_sessions, start=1):
            cid = session.get("courseId")
            sid = session.get("sessionId")

            logger.info(f"[2] ({idx}/{len(target_sessions)}) world 코스 {cid} session {sid} 재생성 시작")
            
            generate_summary_reading_quiz(selected_session=session)

            logger.info(f"[2] world 코스 {cid} session {sid} 재생성 완료")

        # [3] 전체 패키징
        logger.info("[3] 전체 패키징 실행")
        package_data = build_course_packages()
        logger.info("[3] 패키징 완료")

        return package_data

    except Exception as e:
        logger.error(f"[REFINE+PACKAGE] 오류 발생: {e}", exc_info=True)
        return None
    
if __name__ == "__main__":
    refine_and_package_only()